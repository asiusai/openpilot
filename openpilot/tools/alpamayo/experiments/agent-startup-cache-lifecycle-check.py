"""CPU checks of actual factory/prefill lifecycle and the narrow production overlay."""
import argparse
import ast
import copy
import importlib.util
import json
import os
from pathlib import Path
import socket
import sys
from types import SimpleNamespace
from unittest.mock import patch

EXP=Path(__file__).resolve().parent;WORK=Path('/mnt/nvme-500/tmp/alpamayo-speed-20261116')
TG=Path('/tmp/opencode/alpamayo-asius-20261114/tinygrad');ROOT=WORK/'agent-paced-dynamic-source-cpu02/alpamayo'
sys.dont_write_bytecode=True;sys.path[:0]=[str(TG),str(ROOT),str(ROOT/'experiments'),'/tmp/opencode/openpilot-alpamayo','/home/john/asius/openpilot/opendbc_repo']
spec=importlib.util.spec_from_file_location('lifecycle_check_contract',EXP/'agent-startup-cache-lifecycle-contract.py')
c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c);m=c.m

def reject(fn):
  try: fn()
  except (ValueError,AssertionError,TypeError,RuntimeError,KeyError): return
  raise AssertionError('Negative lifecycle control was accepted')

def reject_immutable(fn):
  try: fn()
  except AssertionError as error:
    if 'allocation or view changed' not in str(error): raise
    return
  raise AssertionError('Immutable owner changed before KV readiness')

def lifecycle(storage):
  from tinygrad import Tensor,Device,TinyJit,dtypes
  from tinygrad.helpers import prod
  from tinygrad.uop.ops import UOp
  fixture=c.local(EXP/'agent-plan-reload-check.py');overlay=c.local(EXP/'agent-startup-cache-lifecycle-overlay.py')
  previous=c.local(EXP/'agent-startup-persist-check.py')
  meta,inputs,options=fixture.metadata_model()
  def tensor(shape,dt=dtypes.bfloat16): return Tensor(UOp.new_buffer('AMD',prod(shape),dt)).reshape(shape)
  empty=storage/'empty-checkpoint';empty.mkdir()
  checkpoint=Path(options['checkpoint'])
  (empty/'config.json').write_bytes((checkpoint/'config.json').read_bytes())
  (empty/'model.safetensors.index.json').write_text('{"weight_map": {}}')
  original_model=c.local(ROOT/'model.py')
  # This is the actual empty-index initializer path used by benchmark-fast.
  with patch.object(type(Device),'__getitem__',lambda *a:SimpleNamespace(arch='gfx1200')):
    model=original_model.EdgeModel(str(empty),dtypes.bfloat16,'fp8')
  if model.cache is not None: raise AssertionError('Original factory cache lifecycle changed')
  model.weights=meta.weights;model.weight_scales=meta.weight_scales;model.output_token_mapping=meta.output_token_mapping
  model.weights['expert.lifecycle_probe']=tensor((2,2))
  oldowners=c.local(EXP/'agent-plan-family-reload-diagnostic.py').ClosedOwners
  try: oldowners(model)
  except TypeError: reproduced=True
  else: raise AssertionError('Original pre-cache failure was not reproduced')
  klass=previous.decoder_class(Tensor,TinyJit,UOp);decoder=klass();decoder.model=model;decoder.jits={}
  components=SimpleNamespace(model=model,decoder=decoder)
  args=SimpleNamespace(reduced_vocab=options['vocab'],checkpoint=options['checkpoint'])
  with overlay.imports(m) as state:
    installer=m.load(EXP/'agent-startup-persist-install.py')
    plans=installer.SerialPlans(m,components,args)
    if plans.owners.full is not None or plans.owners.bindings: raise AssertionError('Premature KV capture')
    reject(plans.owners.guard)
    reject(lambda:plans.decode(*inputs,2027,0,1))
    # Every immutable owner is checked before accepting the later KV allocation.
    for mapping,key in ((model.weights,next(iter(model.weights))),
                        (model.weight_scales,next(iter(model.weight_scales))),
                        (model.weights,'expert.lifecycle_probe')):
      old=mapping[key];mapping[key]=tensor(old.shape,old.dtype);reject_immutable(plans.owners.bind_ready);mapping[key]=old
    mapping=model.output_token_mapping;model.output_token_mapping=tensor(mapping.shape,mapping.dtype)
    reject_immutable(plans.owners.bind_ready);model.output_token_mapping=mapping
    policy=c.local(ROOT/'experiments/agent-speculation-policy.py')
    prefill_module=c.local(ROOT/'experiments/agent-speculation-prefill.py')
    prefill=prefill_module.install_prefill(model,policy.CHECKPOINT_DRAFT_IDS,policy.CHECKPOINT_DRAFT_IDS[-1])
    payload=tensor((12,),dtypes.int32);allocation_calls=[];reset_calls=[]
    class JitBoundary:
      def reset(self): reset_calls.append('reset')
      def __call__(self,x,cos,sin,length):
        if x.shape!=(1,2048,2048) or not length.is_bound_var or length.arg.val!=2027:
          raise AssertionError('Actual prefill arguments changed')
        return inputs[0],payload
    prefill.jit=JitBoundary()
    x=tensor((1,2048,2048));cos=tensor((1,1,2048,128));sin=tensor(cos.shape)
    def zeros(*shape,**kwargs):
      if kwargs!={'dtype':dtypes.bfloat16}: raise AssertionError('Original allocation API changed')
      allocation_calls.append(shape)
      return SimpleNamespace(realize=lambda:tensor(shape,kwargs['dtype']))
    # Execute the actual SpeculativePrefill control flow. Only allocation and arithmetic are metadata doubles.
    with patch.object(Tensor,'zeros',zeros):
      result=prefill(x,cos,sin,64,2027)
      plans.owners.bind_ready();plans.owners.guard()
      first_cache=tuple(t.uop.base for t in model.cache)
      prefill(x,cos,sin,64,2027);plans.owners.bind_ready()
    if result is not inputs[0] or prefill.last_targets is not payload or len(allocation_calls)!=28 or plans.owners.bindings!=1:
      raise AssertionError('Original prefill allocation/payload/lazy-bind lifecycle changed')
    if tuple(t.uop.base for t in model.cache)!=first_cache: raise AssertionError('Same-layout prefill replaced KV owners')
    cache=model.cache
    for bad in (None,cache[:-1],[cache[0]]*28,[tensor((2,1,8,2560,128)),*cache[1:]],
                [tensor(cache[0].shape,dtypes.float32),*cache[1:]],
                [cache[0].reshape(2,1,8,128,2304),*cache[1:]]):
      model.cache=bad;reject(plans.owners.bind_ready)
    model.cache=cache;plans.owners.guard()
    old=cache[0];cache[0]=tensor(old.shape,old.dtype);reject(plans.owners.guard);cache[0]=old
    # Already allocated initial owners bind immediately and reject later replacement.
    initialized=installer.SerialPlans(m,components,args)
    if initialized.owners.bindings!=1: raise AssertionError('Preallocated initial state was not guarded')
    initialized.owners.guard()
    # Actual decode passes through the repaired binding to each unchanged numerical loader.
    # Stop at its explicit runtime compilation boundary, before any device can open.
    reached=[]
    class Stop(Exception): pass
    def no_runtime(plan,*,parent_gpu_authorized=False):
      if parent_gpu_authorized is not True or plans.owners.bindings!=1: raise AssertionError('Wrong runtime admission')
      plan.guard()
      if [p.body.key.hex() for p in plan.linear.src]!=[r['program_key'] for r in plan.document['calls']]:
        raise AssertionError('Numerical PROGRAM sequence changed')
      reached.append(len(plan.document['calls']));raise Stop()
    # Each build imports its qualified class; intercept only the returned instance method in this CPU fixture.
    original_builder=plans.builder
    def builder(*a):
      value=original_builder(*a);value.compile_runtime=lambda **kw:no_runtime(value,**kw);return value
    plans.builder=builder
    for count in range(1,7):
      try: plans.decode(*inputs,2027,0,count)
      except Stop: pass
      else: raise AssertionError('Runtime boundary was not reached')
    if reached!=[764*n+1 for n in range(1,7)] or decoder.jits: raise AssertionError('Wrong plan family or premature JIT insertion')
    report=overlay.summary(state)
    # Execute the unchanged ordinary dispatcher with a warm-call boundary double.
    # It must never construct the candidate class or change any owner bindings.
    before=len(state['owners']);calls=[]
    def warm_boundary(*a,**kw): calls.append((a,kw));return {'ordinary_boundary':True}
    with patch.object(installer,'signatures',lambda *a:{}):
      ordinary,_=installer.initialize(m,SimpleNamespace(initialize=warm_boundary),components,args,{'current':True},'ordinary')
    if ordinary!={'ordinary_boundary':True} or len(calls)!=1 or len(state['owners'])!=before:
      raise AssertionError('Ordinary branch changed by candidate lifecycle overlay')
  return dict(original_none_cache_error_reproduced=reproduced,actual_empty_index_initializer=True,
    actual_speculative_prefill_control_flow=True,metadata_allocation_calls=len(allocation_calls),
    all28_shape=[2,1,8,2304,128],same_layout_owner_reuse=True,actual_numeric_loader_calls=reached,
    preallocated_initial_cache_guarded=True,ordinary_dispatch_unchanged_and_exercised=True,production_lifecycle=report,
    scope='Actual initializer and prefill control flow; metadata allocation/arithmetic doubles; no AMD execution')

def main():
  parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
  parser.add_argument('--protocol',type=Path,required=True);args=parser.parse_args()
  for p in (args.output,args.protocol,*[Path(os.environ.get(k,'')) for k in ('TMPDIR','CACHEDB','XDG_CACHE_HOME','TORCH_HOME')]): m.storage(p)
  if args.output.exists() or args.protocol.exists(): raise FileExistsError('Fresh evidence and protocol required')
  os.environ['ALPAMAYO_FP8_CACHE']=str(m.FP8)
  for name in c.SOURCES: ast.parse((EXP/name).read_text())
  from tinygrad import Device
  from tinygrad.helpers import Context
  get=type(Device).__getitem__
  def forbidden(*a,**kw): raise AssertionError('Hardware/network access forbidden')
  def cpu(owner,key):
    if key is None or key.split(':')[0] not in ('CPU','NPY'): raise AssertionError('Hardware access forbidden')
    return get(owner,key)
  storage=m.storage(args.output.with_suffix('.files'));storage.mkdir()
  with patch.object(socket.socket,'connect',forbidden),patch.object(socket.socket,'connect_ex',forbidden):
    with patch.object(type(Device),'__getitem__',forbidden):
      reuse=c.reuse();checked=lifecycle(storage)
    overlay=c.local(EXP/'agent-startup-cache-lifecycle-overlay.py')
    with patch.object(type(Device),'__getitem__',cpu),Context(DEV='CPU:LLVM'),overlay.imports(m):
      controls=c.local(EXP/'agent-startup-persist-check.py').cpu_paths()
  if any(k.split(':')[0] not in ('CPU','NPY') for k in Device._opened_devices): raise AssertionError('Unexpected device')
  report=dict(passed=True,gpu_executed=False,sources=c.sources(),devices=sorted(Device._opened_devices),
    lifecycle=checked,CPU_dispatch_and_alias_controls=controls,ordinary_reuse=reuse,
    negatives=['decode_before_prefill','premature_full_guard','weight_replacement','scale_replacement','mapping_replacement',
      'nonlanguage_owner_replacement','missing_KV','partial_KV','duplicate_KV','wrong_capacity','wrong_dtype','wrong_view','reallocated_KV'])
  m.save(args.output,report)
  protocol=dict(schema=1,base=dict(path=str(c.BASE),sha256=c.PINS[str(c.BASE)]),sources=c.sources(),pins=c.PINS,
    ordinary_reuse=reuse,cpu_evidence={str(args.output):m.sha(args.output)},role='persisted',child_seconds=450,outer_timeout_seconds=0)
  m.save(args.protocol,protocol);c.gate(protocol)
  print(json.dumps(dict(passed=True,output=str(args.output),protocol=str(args.protocol),protocol_sha256=m.sha(args.protocol),
    devices=report['devices'],counts=list(range(1,7)),original_failure_reproduced=True)),flush=True)

if __name__=='__main__': main()
