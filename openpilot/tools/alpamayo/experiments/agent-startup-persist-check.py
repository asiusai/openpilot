"""CPU production-path checks and immutable startup protocol preparation."""
import argparse
import ast
from contextlib import ExitStack
import copy
import importlib.util
import json
import os
from pathlib import Path
import socket
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

EXP=Path(__file__).resolve().parent
WORK=Path('/mnt/nvme-500/tmp/alpamayo-speed-20261116')
TG=Path('/tmp/opencode/alpamayo-asius-20261114/tinygrad')
ROOT=WORK/'agent-paced-dynamic-source-cpu02/alpamayo'
sys.dont_write_bytecode=True;sys.path[:0]=[str(TG),str(ROOT),str(ROOT/'experiments'),'/tmp/opencode/openpilot-alpamayo','/home/john/asius/openpilot/opendbc_repo']
spec=importlib.util.spec_from_file_location('startup_persist_check_common',EXP/'agent-startup-persist-common.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

def reject(fn):
  try: fn()
  except (AssertionError,ValueError,RuntimeError,KeyError): return
  raise AssertionError('Negative control was accepted')

def decoder_class(Tensor,TinyJit,UOp):
  tree=ast.parse((ROOT/'experiments/block-decode.py').read_text())
  cls=copy.deepcopy(next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='BlockDecoder'))
  cls.body=[n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='decode']
  namespace=dict(Tensor=Tensor,TinyJit=TinyJit,UOp=UOp)
  exec(compile(ast.Module(body=[cls],type_ignores=[]),'<actual-BlockDecoder.decode>','exec'),namespace)
  return namespace['BlockDecoder']

def metadata_checks():
  loader=m.load(EXP/'agent-plan-family-reload-loader.py');fixture=m.load(EXP/'agent-plan-reload-check.py')
  model,inputs,options=fixture.metadata_model();rows=[]
  for count in range(1,7):
    if count==1: prepared=loader.base.build(model,inputs,**options)
    else: prepared=loader.build(count,model,inputs,m.read(m.FAMILY),**options)
    if [p.body.key.hex() for p in prepared.linear.src]!=[r['program_key'] for r in prepared.document['calls']]:
      raise AssertionError('Actual numerical PROGRAMs changed')
    prepared.guard();old=model.cache[0];model.cache[0]=old.clone();reject(prepared.guard);model.cache[0]=old
    reject(lambda:prepared.compile_runtime())
    rows.append(dict(count=count,calls=len(prepared.document['calls']),scratch_bytes=prepared.proof['scratch_bytes'],
      initialized_bytes=prepared.proof['initialized_bytes'],input_contract=loader.base.contract_json(prepared.expected_info),
      owner_count=len(prepared.owner_bases),qualified_immutable_proof_loaded=True,numerical_PROGRAMs_exact=True))
    print(json.dumps({'CPU_original_member_rebound':count}),flush=True)
  return rows

def cpu_paths():
  import numpy as np
  from tinygrad import Tensor,Device,TinyJit,dtypes
  from tinygrad.uop.ops import UOp
  from tinygrad.engine.jit import _TinyJit,CapturedJit
  from tinygrad.helpers import Context
  installer=m.load(EXP/'agent-startup-persist-install.py');warm=m.load(m.STARTUP/'agent-startup-spec-warm.py')
  klass=decoder_class(Tensor,TinyJit,UOp);decoder=klass();decoder.jits={}
  first=Tensor(np.arange(16,dtype=np.float32).reshape(1,1,16),device='CPU').realize()
  cos=Tensor(np.arange(64,dtype=np.float32).reshape(1,1,64,1),device='CPU').realize();sin=cos.clone().realize()
  cache=[Tensor.zeros(2,1,1,2304,1,device='CPU').contiguous().realize() for _ in range(28)]
  snapshots=[v.clone().realize() for v in cache];identity=lambda:tuple(id(v.uop.base) for v in cache)
  saved_identity=identity();calls=[]
  def greedy(x,c,s,p,st):
    y=(x+c[:,:,st:st+1,:].reshape(1,1,1)+1).realize()
    values=y[:,:,:1].reshape(1,1,1,1,1).expand(2,1,1,1,1)
    for value in cache: Tensor(value.uop.after(value[:,:,:,p:p+1,:].uop.store(values.uop))).realize()
    return y,x.argmax(axis=-1).reshape(1,1).realize()
  model=SimpleNamespace(cache=cache,_greedy_decode=greedy,synchronize=lambda:Device['CPU'].synchronize())
  decoder.model=model;c=SimpleNamespace(model=model,decoder=decoder)
  def builder(count,inputs):
    original=klass();original.model=model;original.jits={}
    for _ in range(3): Tensor.realize(*original.decode(*inputs,2027,0,count))
    saved=original.jits[(inputs[1].shape,count,2304)].captured
    class Plan:
      document={'calls':[{}]*(764*count+1)};proof={'scratch_bytes':10499328+(count-1)*256}
      ret=saved.ret
      def guard(self):
        if identity()!=saved_identity: raise AssertionError('CPU closed allocation changed')
      def compile_runtime(self,*,parent_gpu_authorized):
        assert parent_gpu_authorized
        self.jit=_TinyJit(None,captured=CapturedJit(saved.ret,saved._linear,saved.expected_names,saved.expected_input_info))
    return Plan()
  serial=installer.SerialPlans(m,c,SimpleNamespace(),builder=builder)
  with serial.installing():
    for count in range(1,7):
      output,_=decoder.decode(first,cos,sin,2027,0,count)
      changed=(first+0.125).realize()
      Tensor.realize(*decoder.decode(changed,cos,sin,2030,3,count))
      Tensor.realize(*decoder.decode(output,cos,sin,2304-count,64-count,count))
      calls.append(dict(count=count,cnt=decoder.jits[(cos.shape,count,2304)].cnt,
        aliases=len(decoder.jits[(cos.shape,count,2304)].captured._linears)))
    reject(lambda:decoder.decode(first,cos,sin,2304,0,1))
    reject(lambda:decoder.decode(first,cos,sin,2027,64,1))
    reject(lambda:decoder.decode(first,cos,sin,2027,0,7))
  decoder.jits=warm.ClosedGraphs(decoder.jits);sealed=serial.seal(warm)
  if 'decode' in vars(decoder): raise AssertionError('Instance hook was not restored')
  Tensor.realize(*decoder.decode(first,cos,sin,2027,0,6))
  reject(lambda:decoder.decode(first,cos[:,:,:63,:].contiguous().realize(),sin[:,:,:63,:].contiguous().realize(),2027,0,1))
  key=(cos.shape,1,2304);old=cache[0];cache[0]=old.clone().realize();reject(lambda:decoder.jits[key]);cache[0]=old
  reject(lambda:decoder.jits.__setitem__(key,object()))
  count=[0]
  def actual_prefill():
    count[0]+=1
    for value,snapshot in zip(cache,snapshots,strict=True): value[:,:,:,:2033,:].assign(snapshot[:,:,:,:2033,:]).realize()
    return first
  exact=lambda a,b: np.testing.assert_array_equal(a.numpy().view(np.uint32),b.numpy().view(np.uint32))
  warm.restore_prefill(cache,snapshots,actual_prefill,identity=identity,equal=exact,copy_into=lambda a,b:a.assign(b).realize())
  if count!=[1] or identity()!=saved_identity: raise AssertionError('Actual restoration path or owner changed')
  # The production startup state and actual nominal hook controls cover acceptance/fallback and live owner lifecycle.
  previous=m.load(EXP/'agent-startup-spec-check.py')
  result=unittest.TextTestRunner(verbosity=1).run(unittest.defaultTestLoader.loadTestsFromTestCase(previous.Proofs))
  if not result.wasSuccessful(): raise AssertionError('Frozen startup controls failed')
  lifecycle=previous.live_tests()
  return dict(actual_original_decode=True,real_pinned_CPU_CapturedJit_replay=True,all_six= calls,
    full28_restored=True,real_prefill_called_once=True,guarded_raw_jit_map=True,accept_fallback_and_lifecycle=lifecycle,
    fixture='CPU arithmetic only; AMD numerical plans are separately reconstructed offline, never executed here')

def factory_preflight(environment):
  """Execute the actual factory through index selection, stopping before checkpoint tensor I/O."""
  from tinygrad import Device
  observed={};policy=m.load(ROOT/'experiments/agent-paced-dynamic-policy.py');policy.verify_snapshot()
  module=m.load(ROOT/'experiments/benchmark-fast.py')
  class Stop(Exception): pass
  realpath=module.Path
  class ReadPath(type(realpath())):
    def read_text(self,*args,**kwargs):
      value=super().read_text(*args,**kwargs)
      if self.name=='index.json':
        observed.update(index=str(self),index_sha256=m.sha(self),index_bytes=len(value))
      return value
  def stop(*args,**kwargs): raise Stop()
  try:
    with patch.object(module,'Path',ReadPath),patch.object(module,'safe_load',stop),\
         patch.object(type(Device),'__getitem__',lambda *args:SimpleNamespace(arch='gfx1200')):
      try: module.load(SimpleNamespace(),m.read(m.INVOCATION)['command'][m.read(m.INVOCATION)['command'].index('--checkpoint')+1],precision='fp8')
      except Stop: pass
      else: raise AssertionError('Factory did not stop before tensor load')
  finally: module.EdgeModel.__init__=module.original_init
  if observed.get('index')!=str(m.FP8/'index.json') or observed.get('index_sha256')!=m.PINS[str(m.FP8/'index.json')]:
    raise AssertionError('Actual factory read another FP8 index')
  return dict(**observed,actual_production_load_entered=True,stopped_before_checkpoint_tensor_load=True,device_double_only=True)

def main():
  parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
  parser.add_argument('--manifest',type=Path,required=True);args=parser.parse_args()
  for p in (args.output,args.manifest,*[Path(os.environ.get(k,'')) for k in ('TMPDIR','CACHEDB','XDG_CACHE_HOME','TORCH_HOME')]): m.storage(p)
  if args.output.exists() or args.manifest.exists(): raise FileExistsError('New immutable CPU evidence and protocol required')
  for p in EXP.glob('agent-startup-persist-*.py'): ast.parse(p.read_text())
  from tinygrad import Device
  original=type(Device).__getitem__
  def forbid(*a,**kw): raise AssertionError('CPU work attempted hardware access')
  def only_cpu(self,key):
    if key is None or key.split(':')[0] not in ('CPU','NPY'): raise AssertionError('CPU work attempted hardware access')
    return original(self,key)
  with patch.object(socket.socket,'connect',forbid),patch.object(socket.socket,'connect_ex',forbid):
    with patch.object(type(Device),'__getitem__',forbid):
      evidence=m.evidence();actual=metadata_checks()
      policy=m.load(ROOT/'experiments/agent-paced-dynamic-policy.py')
      replay=m.load(ROOT/'experiments/agent-live-replay.py');policy.adapt(replay)
      dataset=replay.Dataset(WORK/'quality-samples.json',replay.SOURCE/'manifest.json')
      pipeline=replay.local('benchmark-pipeline')
      _,_,original_protocol=replay.normalize(m.INVOCATION,dataset,pipeline,WORK/'agent-startup-persist-normalize-only.json')
    from tinygrad.helpers import Context
    with patch.object(type(Device),'__getitem__',only_cpu),Context(DEV='CPU:LLVM'):
      controls=cpu_paths()
    os.environ['ALPAMAYO_FP8_CACHE']=str(m.FP8)
    factory=factory_preflight(os.environ)
  blob=m.cache_check();selected=m.selected(dataset)
  files={str(p):m.sha(p) for p in sorted(ROOT.rglob('*')) if p.is_file()}
  for p in [*m.STARTUP.glob('agent-startup-spec-*.py'),EXP/'agent-startup-cache-capture.py',EXP/'agent-startup-spec-check.py',
            EXP/'agent-plan-reload-check.py',EXP/'agent-plan-mutation-v2-parent.py',Path('/tmp/opencode/run-alpamayo-speed-trial.py'),m.INVOCATION]:
    files[str(p)]=m.sha(p)
  manifest=dict(schema=1,sources=m.sources(),files=files,evidence=evidence,full_dataset=dataset.provenance,full_samples=dataset.samples,
    selected_dataset=selected.provenance,selected_samples=selected.samples,original_protocol=original_protocol,
    child_seconds=450,outer_timeout_seconds=0,compiler_cache_source=m.read(WORK/'agent-plan-family-capture1/report.json')['compiler_cache'])
  hooks=m.load(EXP/'agent-startup-persist-hooks.py')
  with hooks.Hooks(m,manifest,'persisted',{k:os.environ[k] for k in ('CACHEDB','TMPDIR','XDG_CACHE_HOME','TORCH_HOME')}) as h:
    m.load(ROOT/'experiments/agent-live-replay.py');manifest['transforms']=h.transforms
  bad=copy.deepcopy(manifest);bad['sources'][str(EXP/'agent-startup-persist-install.py')]='0'*64;reject(lambda:m.verify(bad))
  reject(lambda:m.equal(-0.0,0.0));reject(lambda:m.equal(float('nan'),float('nan')))
  prior_path=WORK/'agent-startup-spec-candidate/report.json';prior=m.read(prior_path)
  projected=m.projected_provenance(prior,prior_path)
  if 'model_load_ms' in projected['initializer']['factory']: raise AssertionError('Duration projection failed')
  bad=copy.deepcopy(prior);bad['consumer_provenance']['initializer']['factory']['model_load_ms']=float('nan')
  reject(lambda:m.projected_provenance(bad,prior_path))
  bad=copy.deepcopy(prior);bad['source_sha256'][next(iter(bad['source_sha256']))]='0'*64
  reject(lambda:m.projected_provenance(bad,prior_path))
  bad=copy.deepcopy(prior);bad['consumer_provenance']['initializer']['factory']['pad_prompt_to']=2027
  reject(lambda:m.equal(projected,m.projected_provenance(bad,prior_path)))
  report=dict(passed=True,gpu_executed=False,sources=m.sources(),devices=sorted(Device._opened_devices),actual_metadata=actual,
    CPU_production_paths=controls,factory_cache_preflight=factory,FP8_blob=blob,transforms=manifest['transforms'],
    negatives=['owner_replaced','unknown_shape','serial_count','position','step','jit_replacement','source_mutation','signed_zero','nonfinite',
      'nonfinite_metadata_duration','changed_snapshot_inventory','semantic_factory_metadata_retained'])
  if any(k.split(':')[0] not in ('CPU','NPY') for k in report['devices']): raise AssertionError('Unexpected hardware backend')
  m.save(args.output,report);manifest['files'][str(args.output)]=m.sha(args.output);m.save(args.manifest,manifest);m.verify(manifest)
  print(json.dumps(dict(passed=True,output=str(args.output),manifest=str(args.manifest),manifest_sha256=m.sha(args.manifest)),indent=2))

if __name__=='__main__': main()
