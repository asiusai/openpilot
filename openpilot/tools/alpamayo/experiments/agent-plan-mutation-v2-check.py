"""Execute the production mask/poison adapter through all six CPU setup cases."""
import argparse
import ast
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch

WORK=Path('/mnt/nvme-500/tmp/alpamayo-speed-20261116');EXP=Path(__file__).resolve().parent

def local(path):
  spec=importlib.util.spec_from_file_location(Path(path).stem.replace('-','_'),path)
  module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
  return module

def rejects(fn):
  try: fn()
  except (ValueError,AssertionError,KeyError): return
  raise AssertionError('Negative control was accepted')

def main():
  parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
  if args.output.exists() or not args.output.resolve().is_relative_to(WORK): raise ValueError('Use a new WORK output')
  for key in ('TMPDIR','CACHEDB'):
    if not Path(os.environ.get(key,'')).resolve().is_relative_to(WORK): raise ValueError('All caches and artifacts must remain under WORK')
  import numpy as np
  from tinygrad import Tensor,TinyJit,Device,dtypes
  from tinygrad.device import Buffer
  from tinygrad.uop.ops import UOp,Ops
  from tinygrad.engine.jit import _prepare_jit_inputs
  from tinygrad.helpers import Context
  adapter=local(EXP/'agent-plan-mutation-v2-adapter.py');original_source,modified,change=adapter.source()
  poison=local(EXP/'agent-plan-poison-copy.py')
  original_getitem=type(Device).__getitem__
  def cpu_only(self,key):
    if key is None or key.split(':')[0] not in ('CPU','NPY','PYTHON'): raise AssertionError('Hardware device forbidden: '+str(key))
    return original_getitem(self,key)
  report=dict(gpu_executed=False,gpu_qualified=False,scope='Actual adapter API/setup execution; CPU probe is not the Alpamayo model',adapter=change)
  with Context(DEV='CPU:LLVM'),patch.object(type(Device),'__getitem__',cpu_only):
    adapter.verify_base()
    if hasattr(Buffer,'copyin'): raise AssertionError('A compatibility alias exists unexpectedly')
    try: Tensor.arange(2112,device='AMD')
    except TypeError as error: report['old_arange_error_reproduced']=str(error)
    else: raise AssertionError('Frozen arange error did not reproduce')
    array=((np.arange(2112,dtype=np.int32)%33)-16).astype(np.float32).reshape(1,1,2112)
    first=Tensor(array,device='CPU').cast(dtypes.bfloat16).realize()
    values=eval(compile(ast.parse(adapter.MASK_NEW,mode='eval'),'<fixed-mask>','eval'),dict(Tensor=Tensor,first=first)).realize()
    np.testing.assert_array_equal(values.numpy(),np.arange(2112,dtype=np.int32))
    mask=(values==2111).reshape(1,1,2112).realize()
    expected_mask=(np.arange(2112)==2111).reshape(1,1,2112)
    np.testing.assert_array_equal(mask.numpy(),expected_mask)
    changed=mask.where(first.abs().max()+256,first).cast(dtypes.bfloat16).realize()
    expected=array.copy();expected[0,0,-1]=272
    np.testing.assert_array_equal(changed.bitcast(dtypes.uint16).numpy(),(expected.view(np.uint32)>>16).astype(np.uint16))
    rejects(lambda:np.testing.assert_array_equal((values==2110).reshape(1,1,2112).numpy(),expected_mask))
    rejects(lambda:adapter.derive((EXP/'agent-plan-reload-run.py').read_text().replace('pattern,alias=alias','pattern+1,alias=alias')))
    small=Buffer('CPU',4,initial_value=bytes(4))
    for pattern in (0,0xA4,0x5B,True): rejects(lambda pattern=pattern:poison.poison_copy(small,pattern))
    rejects(lambda:poison.poison_copy(None,0xA5))
    frozen=adapter.local(EXP/'agent-plan-reload-run.py');original_function=frozen.capture_one
    with adapter.adapted(frozen):
      if frozen.capture_one is original_function or frozen.plan_poison_copy.__code__.co_code!=poison.poison_copy.__code__.co_code:
        raise AssertionError('Production adapter installed a different transport')
    if frozen.capture_one is not original_function or 'plan_poison_copy' in vars(frozen): raise AssertionError('Adapter leaked')

    definitions={n.name:n for n in original_source.body if isinstance(n,ast.FunctionDef)}
    block=next(n for n in modified.body if isinstance(n,ast.Try));nested={n.name:n for n in block.body if isinstance(n,ast.FunctionDef)}
    selected=[copy.deepcopy(definitions[n]) for n in ('exact','reset','clone','same','kv_equal','variables')]
    selected+=[copy.deepcopy(nested[n]) for n in ('active_storage','poisoned_run')]
    count=0
    for node in ast.walk(selected[-2]):
      if isinstance(node,ast.Constant) and node.value=='AMD': node.value='CPU';count+=1
    if count!=1: raise AssertionError('Unreviewed test device translation')
    start=next(i for i,n in enumerate(block.body) if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name) and n.targets[0].id=='mask')
    # Run the original finally checks, too. Only the actual model computations
    # are supplied by a small CPU probe; all mutation/transfer/setup code is real.
    selected.append(ast.Try(body=copy.deepcopy(block.body[start:]),handlers=[],orelse=[],finalbody=copy.deepcopy(block.finalbody)))
    tail=compile(ast.fix_missing_locations(ast.Module(body=selected,type_ignores=[])),'<production-mutation-setup-CPU>','exec')
    zeros=np.zeros((2,1,8,2304,128),np.uint16)
    cache=[Tensor(zeros.copy(),device='CPU').bitcast(dtypes.bfloat16).realize() for _ in range(28)]
    model=SimpleNamespace(cache=cache,synchronize=lambda:Device['CPU'].synchronize())
    snapshots=[t.clone().realize() for t in cache]
    identity=lambda:tuple(id(t.uop.base.buffer.base) for t in model.cache)
    dc=Tensor(np.zeros((1,1,64,128),np.uint16),device='CPU').bitcast(dtypes.bfloat16).realize();ds=dc.clone().realize()
    prompt=2027;arenas=[];events=[]
    def make_jit():
      def run(x,c,s,p,step):
        return (x+1).cast(dtypes.bfloat16).realize(),x[:,:,-1:].cast(dtypes.float32).reshape(1,1).reshape(1).realize()
      jit=TinyJit(run)
      def invoke(x,c,s,p,step):
        ret=jit(x,c,s,UOp.variable('block_position',0,2303).bind(p),UOp.variable('block_step',0,63).bind(step))
        if jit.captured:
          for alias,(linear,staging) in list(jit.captured._linears.items()):
            if any(u.op is Ops.BUFFER and u.buffer.nbytes==10499328 for u in linear.toposort()): continue
            arena=Buffer('CPU',10499328,preallocate=True);arenas.append(arena)
            ref=UOp.from_buffer(arena,dtypes.int8)
            jit.captured._linears[alias]=(linear.replace(src=(linear.src[0].after(ref),*linear.src[1:])),staging)
        return ret
      for _ in range(3): Tensor.realize(*invoke(first,dc,ds,prompt,0))
      return jit,invoke
    original,original_invoke=make_jit();candidate,candidate_invoke=make_jit()
    class Loaded:
      def __init__(self): self.jit,self.ret=candidate,candidate.captured.ret
      def __call__(self,*a): return candidate_invoke(*a)
    decoder=SimpleNamespace(decode=lambda x,c,s,p,st,count:original_invoke(x,c,s,p,st),history=[],statistics=[],prefill_draft_records=[])
    owner=object();components=SimpleNamespace(inputs=owner)
    payload=Tensor(np.arange(7,dtype=np.int32).reshape(1,7),device='CPU').realize()
    model.padded_prefill=SimpleNamespace(last_targets=payload.clone().realize())
    def prefill():
      model.padded_prefill.last_targets.assign(payload).realize()
      return first.clone().realize()
    namespace=dict(Tensor=Tensor,dtypes=dtypes,UOp=UOp,Ops=Ops,_prepare_jit_inputs=_prepare_jit_inputs,
      model=model,decoder=decoder,original=original,original_ret=original.captured.ret,loaded=Loaded(),
      first=first,logits=first,dc=dc,ds=ds,prompt=prompt,snapshots=snapshots,identity=identity,original_identity=identity(),
      result={'cases':[]},stage=lambda name:events.append(name),plan_poison_copy=poison.poison_copy,
      warm=local(WORK/'startup-cache-parent-source/agent-startup-spec-warm.py'),prefill=prefill,payload=payload,c=components,owner=owner,
      history=copy.deepcopy({name:getattr(decoder,name) for name in ('history','statistics','prefill_draft_records')}))
    writes=[];previous_profile=sys.getprofile()
    def observe(frame,event,arg):
      if event!='return' or frame.f_code is not poison.poison_copy.__code__: return
      target,staging,pattern=(frame.f_locals[n] for n in ('target','staging','pattern'))
      actual=bytes(target.host[:target.nbytes])
      if actual!=bytes([pattern])*target.nbytes: raise AssertionError('Actual production transfer left unwritten bytes')
      if not staging.is_allocated() or staging.nbytes!=target.nbytes: raise AssertionError('Staging lifetime/extent differs')
      writes.append(dict(nbytes=target.nbytes,pattern=pattern,all_bytes_verified=True,staging_alive_through_synchronize=True,
        sha256=hashlib.sha256(actual).hexdigest()))
    try:
      sys.setprofile(observe);exec(tail,namespace)
    finally: sys.setprofile(previous_profile)
    if len(namespace['result']['cases'])!=6 or len(writes)!=42: raise AssertionError('Incomplete six-case poison coverage')
    if hasattr(Buffer,'copyin'): raise AssertionError('A compatibility alias leaked')
    # Negative controls act on real captured metadata and retain original guards.
    alias=next(key for key in original.captured._linears if key)
    saved=original.captured._linears[alias]
    try:
      original.captured._linears[alias]=(saved[0],{})
      rejects(lambda:namespace['poisoned_run']('original',True,first,prompt,0,False,0xA5))
    finally: original.captured._linears[alias]=saved
    linked,staging=original.captured._linears[()]
    arena_ref=next(u for u in linked.toposort() if u.op is Ops.BUFFER and u.buffer.nbytes==10499328)
    try:
      wrong=UOp.new_buffer('CPU',10499327,dtypes.int8)
      original.captured._linears[()]=(linked.substitute({arena_ref:wrong}),staging)
      rejects(lambda:namespace['active_storage'](original,first,prompt,0))
    finally: original.captured._linears[()]=(linked,staging)
    old_owner=components.inputs
    try:
      components.inputs=object()
      final=compile(ast.Module(body=copy.deepcopy(block.finalbody),type_ignores=[]),'<owner-negative>','exec')
      rejects(lambda:exec(final,namespace))
    finally: components.inputs=old_owner
    exec(compile(ast.Module(body=copy.deepcopy(block.finalbody),type_ignores=[]),'<final-restore>','exec'),namespace)
    report.update(mask_values_exact=2112,mask_true_indices=[2111],outlier_BF16_exact=272,untouched_logit_bits=2111,
      cpu_setup_cases=namespace['result']['cases'],cpu_poison_writes=writes,restoration=namespace['result'],
      cpu_original_alias_variants=len(original.captured._linears),cpu_candidate_alias_variants=len(candidate.captured._linears),
      actual_production_transport=True,simulated_transport=False,Buffer_monkeypatch=False,poison_bytes_verified=sum(r['nbytes'] for r in writes),
      negative_controls=['wrong_mask_index','changed_frozen_source','invalid_poison_pattern','invalid_buffer','missing_alias_staging',
                         'wrong_arena_extent','changed_current_owner'],CPU_setup_passed=True)
  report['opened_devices']=sorted(Device._opened_devices)
  if any(d.split(':')[0] not in ('CPU','NPY','PYTHON') for d in Device._opened_devices): raise AssertionError('A hardware device opened')
  report['source_sha256']={name:adapter.sha(EXP/name) for name in adapter.SOURCES if (EXP/name).exists()}
  args.output.write_text(json.dumps(report,indent=2)+'\n')
  print(json.dumps({k:v for k,v in report.items() if k not in ('adapter','cpu_setup_cases','cpu_poison_writes','restoration','source_sha256')},indent=2))

if __name__=='__main__': main()
