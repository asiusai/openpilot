"""CPU validation of actual family loaders and all production mutation/poison paths."""
import argparse
import ast
import copy
from dataclasses import replace
import gc
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch

EXP=Path(__file__).resolve().parent;WORK=Path('/mnt/nvme-500/tmp/alpamayo-speed-20261116')
TG=Path('/tmp/opencode/alpamayo-asius-20261114/tinygrad');SOURCE=WORK/'agent-paced-dynamic-source-cpu02/alpamayo'
sys.dont_write_bytecode=True;sys.path[:0]=[str(TG),str(SOURCE),str(SOURCE/'experiments')]

def local(path):
  import importlib.util
  spec=importlib.util.spec_from_file_location('family_reload_check_'+Path(path).stem.replace('-','_'),path)
  module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
  return module

def reject(fn):
  try: fn()
  except (ValueError,AssertionError,KeyError): return
  raise AssertionError('Negative control was accepted')

def main():
  parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
  contract=local(EXP/'agent-plan-family-reload-contract.py');family=contract.family();family.fresh(args.output)
  for key in ('TMPDIR','CACHEDB','XDG_CACHE_HOME'):
    if not Path(os.environ.get(key,'')).resolve().is_relative_to(WORK): raise ValueError('All caches must remain on WORK')
  for name in contract.SOURCES: ast.parse((EXP/name).read_text())
  from tinygrad import Tensor,Device,dtypes,TinyJit
  from tinygrad.uop.ops import UOp,Ops
  from tinygrad.device import Buffer
  from tinygrad.helpers import Context
  import numpy as np
  original_getitem=type(Device).__getitem__
  def forbid(*a,**kw): raise AssertionError('Offline construction opened a device')
  def cpu(self,key):
    if key is None or key.split(':')[0] not in ('CPU','NPY','PYTHON'): raise AssertionError('Hardware access forbidden')
    return original_getitem(self,key)
  report=dict(gpu_executed=False,passed=False,persistent_GPU_execution=False,actual_members=[],cpu_mutations=[])
  with patch.object(type(Device),'__getitem__',forbid):
    files,members=contract.evidence();protocol=dict(counts=[2,3,4,5,6],artifacts=files,members=members)
    loader=local(EXP/'agent-plan-family-reload-loader.py');diagnostic=local(EXP/'agent-plan-family-reload-diagnostic.py')
    metadata=local(EXP/'agent-plan-reload-check.py')
    if any(getattr(loader.Prepared,n) is not getattr(loader.base.Prepared,n) for n in ('argument','guard','compile_runtime')):
      raise AssertionError('Qualified argument/guard/runtime methods changed')
    for count in protocol['counts']:
      doc,programs,proof,views=contract.read_member(count,protocol,reprove=True)
      model,inputs,options=metadata.metadata_model();model2,inputs2,options2=metadata.metadata_model()
      prepared=loader.Prepared(doc,programs,proof,model,inputs,**options,count=count,protocol=protocol)
      other=loader.Prepared(doc,programs,proof,model2,inputs2,**options2,count=count,protocol=protocol)
      if prepared.registry.describe()!=other.registry.describe(): raise AssertionError('Fresh rebinding differs')
      if any(a is b for a,b in zip(prepared.owner_bases.values(),other.owner_bases.values(),strict=True)):
        raise AssertionError('Fresh metadata owners share process storage')
      if [c.body.key.hex() for c in prepared.linear.src]!=[c['program_key'] for c in doc['calls']]: raise AssertionError('PROGRAM sequence differs')
      fake=SimpleNamespace(captured=SimpleNamespace(ret=other.ret,expected_input_info=other.expected_info,expected_names=[0,1,2]))
      diagnostic.compare_original(contract,model2,inputs2,fake,{'linear':other.linear},other,options2)
      reject(lambda:prepared.compile_runtime())
      wrong=copy.deepcopy(proof);wrong['initialized_bytes']-=1
      reject(lambda:loader.Prepared(doc,programs,wrong,model,inputs,**options,count=count,protocol=protocol))
      wrongdoc=copy.deepcopy(doc);wrongdoc['calls'][0]['arguments'][0]['nbytes']=0
      reject(lambda:loader.Prepared(wrongdoc,programs,proof,model,inputs,**options,count=count,protocol=protocol))
      key=doc['calls'][0]['program_key'];p=programs[key]
      replacements=[p.replace(arg=replace(p.arg,outs=())),p.replace(arg=replace(p.arg,target=replace(p.arg.target,interface=''))),
        p.replace(src=(*p.src[:-1],p.src[-1].replace(arg=p.src[-1].arg[:-1]+bytes([p.src[-1].arg[-1]^1]))))]
      for changed in replacements:
        bad=dict(programs);bad[key]=changed
        reject(lambda:loader.Prepared(doc,bad,proof,model,inputs,**options,count=count,protocol=protocol))
      previous=model.cache[0];model.cache[0]=model2.cache[0];reject(prepared.guard);model.cache[0]=previous
      owners=diagnostic.ClosedOwners(model);weight=next(iter(model.weights));previous=model.weights[weight]
      model.weights[weight]=previous.reshape(-1);reject(owners.guard);model.weights[weight]=previous;owners.guard()
      seen=[];prepared.jit=lambda *values:seen.append(values)
      limits=family.bounds(count);prepared(*inputs,limits['block_position']['max'],limits['block_step']['max'])
      if [v.vmax for v in seen[-1][-2:]]!=[2304-count,64-count]: raise AssertionError('Symbolic upper bounds changed')
      reject(lambda:prepared(*inputs,2305-count,0));reject(lambda:prepared(*inputs,2027,65-count))
      report['actual_members'].append(dict(count=count,calls=len(doc['calls']),programs=len(programs),
        fresh_owner_rebinding=True,program_identities_exact=True,proof_recomputed_exact=True,output_views=views,
        scratch_bytes=proof['scratch_bytes'],initialized_bytes=proof['initialized_bytes'],scratch_reads=len(proof['scratch_read_checks'])))
      print(json.dumps({'CPU_actual_family_member_proved':count}),flush=True)
    report['all_five_proofs_recomputed_exact']=True
  # The original production BlockDecoder controls count, variables, and concatenation.
  # Only neural arithmetic is replaced by an explicit CPU fixture.
  tree=ast.parse((SOURCE/'experiments/block-decode.py').read_text())
  cls=copy.deepcopy(next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='BlockDecoder'))
  cls.body=[n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='decode']
  namespace=dict(Tensor=Tensor,TinyJit=TinyJit,UOp=UOp)
  exec(compile(ast.Module(body=[cls],type_ignores=[]),'<original-BlockDecoder-CPU-fixture>','exec'),namespace)
  writes=[]
  with Context(DEV='CPU:LLVM'),patch.object(type(Device),'__getitem__',cpu):
    first=Tensor((np.arange(2112,dtype=np.float32)%33-16).reshape(1,1,2112),device='CPU').cast(dtypes.bfloat16).realize()
    cos=Tensor(np.arange(64,dtype=np.float32).reshape(1,1,64,1),device='CPU').cast(dtypes.bfloat16).realize();sin=cos.clone().realize()
    for count in range(2,7):
      cache=[Tensor.zeros(2,1,1,2304,1,device='CPU',dtype=dtypes.bfloat16).contiguous().realize() for _ in range(28)]
      mapping=Tensor(np.arange(7,dtype=np.int32),device='CPU').realize()
      model=SimpleNamespace(cache=cache,weights={'probe':first},weight_scales={},output_token_mapping=mapping,
        synchronize=lambda:Device['CPU'].synchronize())
      snapshots=[t.clone().realize() for t in cache];owners=diagnostic.ClosedOwners(model)
      def greedy(logits,c,s,pos,step):
        y=(logits+c[:,:,step:step+1,:].reshape(1,1,1)+1).cast(dtypes.bfloat16).realize()
        values=y[:,:,:1].reshape(1,1,1,1,1).expand(2,1,1,1,1)
        for tensor in model.cache:
          store=tensor[:,:,:,pos:pos+1,:].uop.store(values.uop);Tensor(tensor.uop.after(store)).realize()
        return y,logits.argmax(axis=-1).float().reshape(1,1).realize()
      model._greedy_decode=greedy
      def reset():
        for a,b in zip(cache,snapshots,strict=True): a.assign(b).realize()
      arena_bytes=members[str(count)]['capture']['contract']['scratch_bytes'];arenas=[]
      def make_decoder():
        decoder=namespace['BlockDecoder']();decoder.model=model;decoder.jits={};raw=decoder.decode
        def invoke(x,c,s,p,st,n):
          result=raw(x,c,s,p,st,n);jit=decoder.jits[(c.shape,n,2304)]
          if jit.captured:
            for alias,(linear,staging) in list(jit.captured._linears.items()):
              if any(u.op is Ops.BUFFER and u.buffer.nbytes==arena_bytes for u in linear.toposort()): continue
              arena=Buffer('CPU',arena_bytes,preallocate=True);arenas.append(arena)
              ref=UOp.from_buffer(arena,dtypes.int8)
              # Test-only arena metadata; the production poison code is unchanged.
              jit.captured._linears[alias]=(linear.replace(src=(linear.src[0].after(ref),*linear.src[1:])),staging)
          return result
        decoder.decode=invoke
        for _ in range(3): reset();Tensor.realize(*decoder.decode(first,cos,sin,2027,0,count))
        return decoder,decoder.jits[(cos.shape,count,2304)]
      original_decoder,original=make_decoder();candidate_decoder,candidate=make_decoder()
      class Loaded:
        def __init__(self): self.jit,self.ret,self.proof=candidate,candidate.captured.ret,{'scratch_bytes':arena_bytes}
        def __call__(self,x,c,s,p,st): return candidate_decoder.decode(x,c,s,p,st,count)
      mutation=diagnostic.Mutations(contract,model,original_decoder,Loaded(),original,(first,cos,sin),first,snapshots,2027,count)
      previous_profile=sys.getprofile()
      def observe(frame,event,arg):
        if event!='return' or frame.f_code is not mutation.poison.__code__: return
        target,staging,pattern=(frame.f_locals[k] for k in ('target','staging','pattern'))
        if bytes(target.host[:target.nbytes])!=bytes([pattern])*target.nbytes: raise AssertionError('Incomplete production poison write')
        if not staging.is_allocated() or staging.nbytes!=target.nbytes: raise AssertionError('Staging lifetime/extent changed')
        writes.append(dict(count=count,bytes=target.nbytes,pattern=pattern,staging_alive_after_sync=True))
      row={'count':count,'cases':[]}
      try: sys.setprofile(observe);mutation.exercise(row,lambda name:None)
      finally: sys.setprofile(previous_profile)
      if len(row['cases'])!=6: raise AssertionError('Incomplete production mutation cases')
      alias=next(k for k in original.captured._linears if k);saved=original.captured._linears[alias]
      try:
        original.captured._linears[alias]=(saved[0],{})
        reject(lambda:mutation.run('original',True,first,2027,0,False,0xA5))
      finally: original.captured._linears[alias]=saved
      linear,staging=original.captured._linears[()]
      ref=next(u for u in linear.toposort() if u.op is Ops.BUFFER and u.buffer.nbytes==arena_bytes)
      try:
        original.captured._linears[()]=(linear.substitute({ref:UOp.new_buffer('CPU',arena_bytes-1,dtypes.int8)}),staging)
        reject(lambda:mutation.active_storage(original,first,2027,0))
      finally: original.captured._linears[()]=(linear,staging)
      reject(lambda:mutation.run('original',False,first,2027,0,False,0x5A))
      mutation.reset();mutation.same(model.cache,snapshots);owners.guard()
      report['cpu_mutations'].append(row)
      print(json.dumps({'CPU_production_mutations_passed':count,'cases':6}),flush=True)
      del mutation,original,candidate,original_decoder,candidate_decoder,cache,snapshots,arenas
      gc.collect()
    if hasattr(Buffer,'copyin'): raise AssertionError('A compatibility poison API alias exists')
    old_error=False
    try: Tensor.arange(2112,device='AMD')
    except TypeError: old_error=True
    if not old_error: raise AssertionError('Pinned obsolete arange API unexpectedly changed')
    np.testing.assert_array_equal(Tensor.arange(2112).to(first.device).numpy(),np.arange(2112))
    bad=Tensor([float('nan')],device='CPU').realize();reject(lambda:diagnostic.Mutations.exact(bad,bad))
  if len(writes)!=210: raise AssertionError('All30 cases require210 complete poison transfers')
  report.update(passed=True,production_mutation_cases=sum(len(r['cases']) for r in report['cpu_mutations']),
    complete_production_poison_writes=len(writes),poison_bytes_verified=sum(r['bytes'] for r in writes),
    poison_writes=writes,simulated_transport=False,Buffer_monkeypatch=False,devices=sorted(Device._opened_devices),
    fixture_scope='Actual production mutation/transfer/API/alias paths; CPU arithmetic and arena marker do not execute the AMD model',
    negative_controls=['changed_proof','changed_call_extent','changed_PROGRAM_ABI','changed_full_target','changed_ELF',
      'closed_owner_replacement','closed_weight_view_change','out_of_range_position','out_of_range_step',
      'missing_alias_staging','wrong_arena_extent','wrong_poison_pattern','nonfinite_output'],
    sources={name:contract.sha(EXP/name) for name in contract.SOURCES})
  args.output.write_text(json.dumps(report,indent=2)+'\n')
  print(json.dumps({k:v for k,v in report.items() if k not in ('sources','actual_members','cpu_mutations','poison_writes')},indent=2))

if __name__=='__main__': main()
