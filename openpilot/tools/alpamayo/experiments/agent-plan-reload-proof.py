"""Exact address enumeration for captured global stores and scratch read coverage.

Never infer initialized bytes from ProgramInfo.outs. Unknown addressing fails closed.
Reads conservatively require the entire argument view to have been initialized.
"""
import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np
from tinygrad import Device
from tinygrad.dtype import AddrSpace,dtypes,Invalid
from tinygrad.uop.ops import Ops

EXP=Path(__file__).resolve().parent
WORK=Path('/mnt/nvme-500/tmp/alpamayo-speed-20261116')

def local(path):
  spec=importlib.util.spec_from_file_location(Path(path).stem.replace('-','_'),path)
  module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
  return module

class Unproved(ValueError): pass

def dependencies(u):
  if u.op in (Ops.SPECIAL,Ops.RANGE) or (u.op is Ops.PARAM and u.addrspace is AddrSpace.ALU): return {u}
  if u.op is Ops.AFTER: return dependencies(u.src[0])
  return set().union(*(dependencies(s) for s in u.src)) if u.src else set()

def evaluate(u,env,cache=None):
  if cache is None: cache={}
  if u in cache: return cache[u]
  if u in env: return env[u]
  if u.op is Ops.AFTER: result=evaluate(u.src[0],env,cache)
  elif u.op is Ops.CONST:
    if u.arg is Invalid or type(u.arg) not in (int,bool): raise Unproved('Noninteger address constant')
    result=np.asarray(u.arg,dtype=np.int64)
  elif u.op is Ops.CAST:
    result=evaluate(u.src[0],env,cache)
    if not (dtypes.is_int(u.dtype) or u.dtype==dtypes.bool): raise Unproved('Noninteger address cast')
    if np.any(result<u.dtype.min) or np.any(result>u.dtype.max): raise Unproved('Wrapping address cast')
  else:
    functions={Ops.ADD:np.add,Ops.MUL:np.multiply,Ops.SHL:np.left_shift,Ops.SHR:np.right_shift,
      Ops.AND:np.bitwise_and,Ops.OR:np.bitwise_or,Ops.XOR:np.bitwise_xor,Ops.CMPLT:np.less,
      Ops.CMPEQ:np.equal,Ops.CMPNE:np.not_equal,Ops.WHERE:np.where,Ops.DIV:np.floor_divide,Ops.MOD:np.mod}
    if u.op not in functions: raise Unproved('Address depends on '+u.op.name)
    values=[evaluate(s,env,cache) for s in u.src]
    if u.op in (Ops.DIV,Ops.MOD) and (np.any(values[0]<0) or np.any(values[1]<=0)):
      raise Unproved('Unreviewed signed division semantics')
    result=functions[u.op](*values)
    if dtypes.is_int(u.dtype) and (np.any(result<u.dtype.min) or np.any(result>u.dtype.max)):
      raise Unproved('Address arithmetic overflow')
  cache[u]=result
  return result

def pointer(u):
  """Return parameter, byte-offset expression, and contiguous vector width."""
  width=None
  if u.op is Ops.CAST:
    # Address CAST changes pointer element type. The scalar base index retains its byte scale.
    width=u.dtype.itemsize;u=u.src[0]
  if u.op is Ops.SHRINK:
    if u.src[0].op is not Ops.PARAM or u.src[0].addrspace is not AddrSpace.GLOBAL:
      raise Unproved('Unreviewed vector pointer base')
    count=int(evaluate(u.src[2],{}))
    return u.src[0],u.src[1],count*u.dtype.itemsize
  if u.op is not Ops.INDEX or len(u.src)!=2: raise Unproved('Unreviewed global pointer')
  base=u.src[0]
  while base.op is Ops.AFTER: base=base.src[0]
  if base.op is not Ops.PARAM or base.addrspace is not AddrSpace.GLOBAL: raise Unproved('Nonparameter global pointer')
  return base,u.src[1],width or base.dtype.itemsize

def instruction_rows(program):
  scope=[];rows=[]
  for index,u in enumerate(program.src[1].src):
    if u.op in (Ops.IF,Ops.RANGE): scope.append(u)
    elif u.op in (Ops.END,Ops.ENDIF):
      if not scope: raise Unproved('Unbalanced kernel scope')
      opening=scope.pop()
      if (u.op is Ops.END)!=(opening.op is Ops.RANGE): raise Unproved('Mismatched kernel scope')
    elif u.op in (Ops.LOAD,Ops.STORE):
      if u.src[0].buf_uop.op is not Ops.PARAM: continue
      if len(u.src) not in ((2,) if u.op is Ops.STORE else (1,3)):
        raise Unproved(f'Unreviewed memory operands: {program.key.hex()} index {index} {u.op.name} '+str([(s.op.name,str(s.arg)) for s in u.src]))
      param,address,width=pointer(u.src[0])
      if u.op is Ops.STORE:
        actual=int(math.prod(u.src[1].shape))*u.src[1].dtype.itemsize
        if actual!=width: raise Unproved(f'Store vector width differs: {actual}!={width}')
      else:
        actual=int(math.prod(u.shape))*u.dtype.itemsize
        if actual!=width: raise Unproved(f'Load vector width differs: {actual}!={width}')
      rows.append((index,u,param,address,width,tuple(scope)))
  if scope: raise Unproved('Unclosed kernel scope')
  # Custom functions must be pure AMD lane operations. No hidden global memory is admitted.
  for u in program.src[1].src:
    if u.op is Ops.CALL:
      if u.body.op is not Ops.CUSTOM_FUNCTION or u.body.arg.name!='llvm.amdgcn.ds.swizzle':
        raise Unproved('Unreviewed external kernel function')
      if any(s.addrspace not in (None,AddrSpace.ALU) for s in u.src[1:]):
        raise Unproved('Lane operation received a memory pointer')
  return rows

def enumerate_store(program,row,limit=8_000_000):
  index,store,param,address,width,scope=row
  gates=[s.src[0] for s in scope if s.op is Ops.IF]
  if store.op is Ops.LOAD and len(store.src)==3: gates.append(store.src[2])
  roots=[address,*gates]
  axes=set().union(*(dependencies(u) for u in roots))
  # A loop with an index-independent store must still have at least one iteration.
  axes.update(s for s in scope if s.op is Ops.RANGE)
  ordered=sorted(axes,key=lambda u:u.key)
  env={};sizes=[]
  for axis in ordered:
    if axis.op is Ops.PARAM: raise Unproved('Store address or predicate depends on symbolic position')
    size=int(evaluate(axis.src[0],{}))
    if size<=0: raise Unproved('Empty or dynamic loop')
    if axis.op is Ops.SPECIAL:
      expected=(program.arg.local_size if axis.arg.startswith('l') else program.arg.global_size)[int(axis.arg[-1])]
      if size!=expected: raise Unproved('Launch dimension differs from address domain')
    sizes.append(size)
  if math.prod(sizes)>limit:
    raise Unproved(f'Exact address enumeration bound exceeded: {program.key.hex()} instruction {index} sizes {sizes}')
  for i,(axis,size) in enumerate(zip(ordered,sizes,strict=True)):
    env[axis]=np.arange(size,dtype=np.int64).reshape((1,)*i+(size,)+(1,)*(len(sizes)-i-1))
  mask=np.asarray(True)
  for gate in gates: mask=np.logical_and(mask,evaluate(gate,env))
  indices,active=np.broadcast_arrays(evaluate(address,env),mask)
  indices=indices[active.astype(bool)]
  if indices.size==0: return np.zeros(param.max_numel()*param.dtype.itemsize,dtype=np.bool_),dict(index=index,active=0)
  offsets=indices*param.dtype.itemsize
  nbytes=param.max_numel()*param.dtype.itemsize
  if offsets.min()<0 or offsets.max()+width>nbytes: raise Unproved('Global memory access exceeds its declared view')
  written=np.zeros(nbytes,dtype=np.bool_)
  for byte in range(width): written[offsets+byte]=True
  return written,dict(index=index,param=param.arg.slot,width=width,enumerated_domain=math.prod(sizes),
    active=int(indices.size),bytes=int(written.sum()),masked=bool(gates),address_key=address.key.hex(),
    gate_keys=[u.key.hex() for u in gates])

def span(argument):
  if argument['kind']=='scratch': return argument['name'],0,None
  if argument['kind']!='expression': return None
  if argument['op']=='BITCAST': return span(argument['src'][0])
  if argument['op']=='SHRINK':
    base=span(argument['src'][0])
    if base is None: return None
    values=argument['src'][1:]
    if any(v!={'kind':'expression','op':'CONST','arg':v.get('arg'),'src':[]} or type(v['arg']) is not int for v in values):
      raise Unproved('Dynamic outer scratch slice')
    start,count=(v['arg'] for v in values)
    if base[2] is not None and (start<0 or start+count>base[2]): raise Unproved('Nested scratch slice exceeds view')
    return base[0],base[1]+start,count
  raise Unproved('Unreviewed scratch view')

def prove(document,programs):
  accesses={};witnesses={};external_partial=[]
  scratch_reads={(c['program_key'],slot) for c in document['calls'] for slot,arg in enumerate(c['arguments']) if span(arg) is not None}
  read_bounds=[]
  for key,program in programs.items():
    rows=instruction_rows(program);writes={};reads=set();evidence=[]
    for row in rows:
      _,op,param,*_=row
      if op.op is Ops.LOAD:
        reads.add(param.arg.slot)
        if (key,param.arg.slot) in scratch_reads:
          _,witness=enumerate_store(program,row)
          read_bounds.append(dict(program_key=key,**witness))
        continue
      try: mask,witness=enumerate_store(program,row)
      except Unproved as error:
        # Symbolic KV writes are not scratch initialization claims. All instances
        # must bind this argument to caller-owned KV, checked below per call.
        external_partial.append(dict(program_key=key,param=param.arg.slot,reason=str(error)))
        continue
      writes[param.arg.slot]=writes.get(param.arg.slot,np.zeros_like(mask))|mask;evidence.append(witness)
    accesses[key]=(reads,writes,rows);witnesses[key]=evidence
  arena=document['registry']['scratch']
  if len(arena)!=1 or arena[0]['name']!='scratch.0' or arena[0]['dtype']!='i8': raise Unproved('Unexpected scratch arenas')
  initialized=np.zeros(arena[0]['nbytes'],dtype=np.bool_);last=np.full(len(initialized),-1,dtype=np.int32)
  checks=[];output_full={}
  for call in document['calls']:
    key=call['program_key'];reads,writes,rows=accesses[key]
    params={p.arg.slot:p for p in programs[key].arg.params if p.addrspace is AddrSpace.GLOBAL}
    for _,instruction,param,*_ in rows:
      if instruction.op is not Ops.STORE: continue
      argument=call['arguments'][param.arg.slot]
      if argument['kind']=='input': raise Unproved('A numerical kernel writes an explicit current input')
      if argument['kind']=='buffer' and not all(n.startswith(('kv/','return/')) for n in argument['owners']):
        raise Unproved('A numerical kernel writes an immutable owner')
    for item in (r for r in external_partial if r['program_key']==key):
      arg=call['arguments'][item['param']]
      if arg['kind']!='buffer' or not arg['owners'] or not all(n.startswith('kv/') for n in arg['owners']):
        raise Unproved('Unproved store targets scratch/output/non-KV owner: '+str(call['index']))
    for slot in reads:
      arg=call['arguments'][slot];region=span(arg)
      if region is None: continue
      _,offset,size=region
      expected=params[slot].max_numel()*params[slot].dtype.itemsize
      if size!=expected or offset<0 or offset+size>len(initialized): raise Unproved('Scratch argument ABI extent differs')
      if not initialized[offset:offset+size].all():
        missing=np.flatnonzero(~initialized[offset:offset+size])+offset
        raise Unproved(f'call {call["index"]} arg {slot}: {len(missing)} unreadied bytes, first {int(missing[0])}')
      checks.append(dict(call=call['index'],slot=slot,offset=offset,nbytes=size,
        producer_calls=np.unique(last[offset:offset+size]).tolist()))
    for slot,mask in writes.items():
      arg=call['arguments'][slot];region=span(arg)
      if region is None:
        if arg['kind']=='buffer' and any(n.startswith('return/') for n in arg['owners']):
          if arg['nbytes']!=len(mask) or not mask.all(): raise Unproved('Returned buffer is not completely overwritten')
          for owner in arg['owners']: output_full[owner]=call['index']
        continue
      _,offset,size=region
      if size!=len(mask) or offset<0 or offset+size>len(initialized): raise Unproved('Scratch store ABI extent differs')
      initialized[offset:offset+size]|=mask
      view=last[offset:offset+size];view[mask]=call['index']
  if set(output_full)!={'return/logits','return/token'}: raise Unproved('Both returned allocations need complete writes')
  return dict(schema=1,proved=True,proof_scope='Actual global scratch reads and per-byte writes; conservative full-view read coverage',
    calls=len(document['calls']),programs=len(programs),scratch_bytes=len(initialized),initialized_bytes=int(initialized.sum()),
    scratch_read_checks=checks,scratch_instruction_bounds=read_bounds,store_witnesses=witnesses,external_partial_stores=external_partial,
    full_output_writes=output_full,immutable_owners_and_explicit_inputs_not_written=True,
    source='Instruction LINEAR addresses, IF predicates, loop/thread domains, and actual arena views')

def main():
  parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
  if not args.output.resolve().is_relative_to(WORK) or args.output.exists(): raise ValueError('Use a new WORK output')
  inspect=local(EXP/'agent-plan-reload-inspect.py')
  def forbidden(*args,**kwargs): raise AssertionError('No device access')
  with patch.object(type(Device),'__getitem__',forbidden):
    document,programs=inspect.load()
    try: report=prove(document,programs)
    except Unproved as error: report=dict(proved=False,error=str(error))
    report.update(capture_sha256=inspect.CAPTURE_SHA,devices=list(Device._opened_devices),gpu_executed=False)
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('scratch_read_checks','scratch_instruction_bounds','store_witnesses')}))
    if not report['proved']: raise SystemExit(1)

if __name__=='__main__': main()
