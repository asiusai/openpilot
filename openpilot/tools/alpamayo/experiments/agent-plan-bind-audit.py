"""Explicit ownership and pre-HCQ call audit for one actual count1 decoder plan.

The output is diagnostic metadata, not permission to execute a persisted plan.
Unknown owners and unproved scratch initialization remain explicit blockers.
"""
from collections import Counter
from contextlib import contextmanager
import hashlib
from functools import lru_cache
import importlib.util
import json
from pathlib import Path
import sys
from unittest.mock import patch

from tinygrad.device import Buffer
from tinygrad.dtype import AddrSpace
from tinygrad.uop.ops import Ops, UOp

ROOT=Path(__file__).resolve().parent


def local(path):
  spec=importlib.util.spec_from_file_location(Path(path).stem.replace('-','_'),path)
  value=importlib.util.module_from_spec(spec);sys.modules[spec.name]=value;spec.loader.exec_module(value)
  return value


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@lru_cache(None)
def codec_module(): return local(ROOT/'agent-plan-bind-codec.py')


def storage_of(u):
  # Reading Buffer ownership/offset metadata never calls allocate, host, or get_storage.
  buf=u.buffer
  if not isinstance(buf,Buffer) or buf.options.external_ptr is not None: raise ValueError('Unsupported external or multi-device buffer')
  return buf.base,buf.offset,buf.nbytes


class Registry:
  def __init__(self):
    self.rows=[];self.bases={};self.owners={};self.scratch={};self.unknown={}

  def base_id(self,base):
    if base not in self.bases: self.bases[base]='storage.'+str(len(self.bases))
    return self.bases[base]

  def add(self,name,tensor,role,source):
    if name in self.owners: raise ValueError('Duplicate owner name')
    base=tensor.uop.base
    if base.op is not Ops.BUFFER: raise ValueError('Named tensor is not backed by a held buffer: '+name)
    owner,offset,nbytes=storage_of(base)
    row=dict(name=name,role=role,source=source,storage=self.base_id(owner),offset_bytes=offset,nbytes=nbytes,
             dtype=tensor.dtype.name,shape=list(tensor.shape),base_dtype=base.dtype.name,
             layout_key=tensor.uop.substitute({base:UOp(Ops.NOOP)}).key.hex(),
             storage_bytes=owner.nbytes,device=owner.device)
    self.rows.append(row);self.owners[name]=(owner,offset,nbytes,row)

  def buffer(self,u):
    base,offset,nbytes=storage_of(u)
    owners=[r['name'] for b,start,size,r in self.owners.values() if b is base and start<=offset and offset+nbytes<=start+size]
    if not owners:
      key=(base,offset,nbytes,u.dtype)
      if key not in self.unknown:
        self.unknown[key]=dict(name='unresolved.'+str(len(self.unknown)),storage=self.base_id(base),offset_bytes=offset,
          nbytes=nbytes,dtype=u.dtype.name,device=u.device,reason='No explicit model/input/return owner covers this captured view')
      names=[self.unknown[key]['name']]
    else: names=owners
    return dict(kind='buffer',owners=names,storage=self.base_id(base),offset_bytes=offset,nbytes=nbytes,dtype=u.dtype.name)

  def expression(self,u):
    if u.op is Ops.BUFFER: return self.buffer(u)
    if u.op is Ops.ALLOC:
      if u.addrspace is not AddrSpace.GLOBAL or u.arg.name is not None or u.arg.spec is not None:
        raise ValueError('Unreviewed outer allocation or runtime resource')
      if u not in self.scratch:
        self.scratch[u]=dict(name='scratch.'+str(len(self.scratch)),nbytes=u.max_numel()*u.dtype.itemsize,
                            dtype=u.dtype.name,device=u.device,initialization_proved=False)
      return dict(kind='scratch',name=self.scratch[u]['name'])
    if u.op is Ops.PARAM:
      if u.is_variable:
        if u.is_bound_var: raise ValueError('Capture-time variable value survived into the persistent boundary')
        return dict(kind='variable',name=u.expr,min=u.vmin,max=u.vmax,dtype=u.dtype.name)
      if u.addrspace is not AddrSpace.GLOBAL or u.arg.slot not in (0,1,2): raise ValueError('Unexpected input parameter')
      return dict(kind='input',slot=u.arg.slot,dtype=u.dtype.name,nbytes=u.max_numel()*u.dtype.itemsize,device=u.device)
    if u.op not in (Ops.BITCAST,Ops.SHRINK,Ops.RESHAPE,Ops.CONST,Ops.STACK,Ops.CAST,Ops.ADD,Ops.MUL):
      raise ValueError('Unreviewed outer view/expression '+u.op.name)
    encoder=codec_module().Encoder();argument=encoder.value(u.arg)
    if encoder.nodes: raise ValueError('Outer expression contains an unreviewed nested argument graph')
    return dict(kind='expression',op=u.op.name,arg=argument,src=[self.expression(s) for s in u.src])

  def describe(self):
    return dict(owners=self.rows,storages=[dict(name=name,nbytes=base.nbytes,device=base.device) for base,name in self.bases.items()],
                scratch=list(self.scratch.values()),unresolved=list(self.unknown.values()))


def validate_binding_contract(expected,actual):
  """No rebinding is approved unless complete ownership and alias metadata match."""
  if expected!=actual: raise ValueError('Ownership, source, view, ABI, or alias contract changed')
  if actual['unresolved']: raise ValueError('Unresolved owners prevent rebinding')
  if any(not row['initialization_proved'] for row in actual['scratch']):
    raise ValueError('Scratch initialization is unproved; persistent execution is forbidden')


def model_registry(model,inputs,ret,*,bulk_index,vocab,checkpoint):
  index=json.loads(Path(bulk_index).read_text());index_sha=sha(bulk_index)
  records={(r['kind'],r['name']):r for r in index['records']}
  registry=Registry()
  for name,tensor in model.weights.items():
    if not (name.startswith('vlm.model.language_model.') or name=='vlm.lm_head.weight'): continue
    record=next((records[(kind,name)] for kind in ('fp8','bf16') if (kind,name) in records),None)
    source=dict(loader='verified benchmark-fast bulk or original factory restoration',bundle_sha256=index['sha256'],
      index_sha256=index_sha,record=record,checkpoint_config_sha256=sha(Path(checkpoint)/'config.json'))
    if name=='vlm.lm_head.weight': source.update(transform='original reduced-vocab row gather and last-row padding',vocab_sha256=sha(vocab))
    registry.add('weight/'+name,tensor,'immutable_weight',source)
  for name,tensor in model.weight_scales.items():
    if not (name.startswith('vlm.model.language_model.') or name=='vlm.lm_head.weight'): continue
    registry.add('scale/'+name,tensor,'immutable_weight_scale',dict(bundle_sha256=index['sha256'],index_sha256=index_sha,record=records.get(('scale',name))))
  registry.add('mapping/output_token_mapping',model.output_token_mapping,'immutable_token_mapping',
               dict(vocab_sha256=sha(vocab),transform='original allowed/sorted/unique vocabulary with last-row padding'))
  for i,tensor in enumerate(model.cache):
    registry.add('kv/'+str(i),tensor,'caller_mutable_kv',dict(initialization='fresh verified current prefill; never serialize contents'))
  for name,tensor in zip(('input/logits','input/cos','input/sin'),inputs,strict=True):
    registry.add(name,tensor,'caller_current_input',dict(initialization='same current input owner; no stored tensor values'))
  for name,tensor in zip(('return/logits','return/token'),ret,strict=True):
    registry.add(name,tensor,'plan_output',dict(initialization='must prove complete writes before reading after reload'))
  return registry


@contextmanager
def capture_boundary(input_uops):
  """Observe one normal capture after memory planning and before HCQ compilation."""
  import tinygrad.engine.jit as jit_module
  import tinygrad.engine.realize as realize
  original=jit_module.compile_linear
  record={}
  def compile_one(linear,*args,**kwargs):
    # CapturedJit also compiles an empty staging-copy linear for its first
    # non-aliased replay. It is outside the observed numerical plan boundary.
    if not linear.src: return original(linear,*args,**kwargs)
    if record: raise ValueError('Expected exactly one count1 JIT compilation boundary')
    if kwargs.get('input_uops')!=input_uops or kwargs.get('cache') is not False:
      raise ValueError('Unexpected JIT compilation inputs')
    if args or kwargs.get('beam',0)!=0: raise ValueError('This diagnostic does not change optimization policy')
    compiled=realize.lower_and_compile(linear)
    record['linear']=compiled
    result=original(compiled,**kwargs)
    record['hcq']=result
    return result
  with patch.object(jit_module,'compile_linear',compile_one): yield record


def export_capture(record,jit,registry,directory,expected):
  codec=codec_module()
  directory=Path(directory);directory.mkdir()
  programs={};calls=[];variables={}
  linear=record['linear']
  for index,original in enumerate(linear.src):
    call=original.without_after
    if call.body.op is not Ops.PROGRAM: raise ValueError('Unreviewed non-PROGRAM call at '+str(index))
    prg=call.body;key=prg.key.hex()
    if key not in programs:
      programs[key]=codec.save(prg,directory/'programs')
      if sum(p.stat().st_size for p in (directory/'programs').iterdir())>256*1024**2:
        raise ValueError('Diagnostic program artifact limit exceeded')
    arguments=[registry.expression(u) for u in call.src[1:]]
    for u in call.src[1:]:
      for v in u.variables(): variables[v.expr]=dict(min=v.vmin,max=v.vmax,dtype=v.dtype.name)
    calls.append(dict(index=index,program_key=key,arguments=arguments,outputs=list(prg.arg.outs),inputs=list(prg.arg.ins),
      name=prg.src[0].arg.name,ordered_dependencies=[u.op.name for u in original.src[1:]] if original.op is Ops.AFTER else []))
  # Record only structural aliases. No Buffer addresses or raw Buffer/UOp identities leave the process.
  aliases=[]
  for alias_inputs,(_,staging) in jit.captured._linears.items():
    aliases.append(dict(inputs=[registry.buffer(u) for u in alias_inputs],staging_count=len(staging)))
  info=[dict(view_key=v.key.hex(),variable_keys=[u.key.hex() for u in vs],dtype=str(dt),device=dev)
        for v,vs,dt,dev in jit.captured.expected_input_info]
  actual=[dict(device='AMD',name=c['name'],program_key=c['program_key'],output_indices=c['outputs'],input_indices=c['inputs']) for c in calls]
  identity=(actual==expected['kernels'] and info==expected['input_contract'] and jit.captured.expected_names==expected['input_names']
            and len(aliases)==expected['linked_alias_variants'])
  ownership=registry.describe()
  blockers=[]
  if not identity: blockers.append('Captured numerical programs/input contracts/alias count differ from the qualified count1 plan')
  if ownership['unresolved']: blockers.append('Captured buffers lack explicit owners')
  if ownership['scratch']: blockers.append('Per-byte scratch initialization across masked stores and arena reuse is not proved')
  blockers.append('Fresh-process AMD allocation, relocation, owner rebinding, and output-alias staging have not been tested')
  report=dict(schema=1,kind='actual-count1-pre-hcq-diagnostic',executable=False,qualified=False,programs=programs,calls=calls,
    registry=ownership,variables=variables,input_contract=info,input_names=jit.captured.expected_names,aliases=aliases,
    qualified_numerical_identities_equal=identity,expected_count=len(expected['kernels']),call_count=len(calls),
    blockers=blockers,content_policy='No weights, inputs, KV bytes, scratch bytes, predictions, or device addresses are serialized',
    scratch_warning='ProgramInfo output indices declare writes; they do not prove complete byte initialization',
    source_sha256={p.name:sha(p) for p in ROOT.glob('agent-plan-bind-*.py')},
    file_limit_bytes=256*1024**2)
  path=directory/'audit.json'
  with path.open('x') as stream: json.dump(report,stream,indent=2,allow_nan=False);stream.write('\n')
  total=sum(p.stat().st_size for p in directory.rglob('*') if p.is_file())
  if total>report['file_limit_bytes']: raise ValueError('Diagnostic artifact size exceeds its bound')
  return dict(audit=str(path),audit_sha256=sha(path),bytes=total,programs=len(programs),calls=len(calls),
              ownership_roles=dict(Counter(r['role'] for r in ownership['owners'])),unresolved=len(ownership['unresolved']),
              scratch_arenas=len(ownership['scratch']),qualified_numerical_identities_equal=identity,blockers=blockers)
