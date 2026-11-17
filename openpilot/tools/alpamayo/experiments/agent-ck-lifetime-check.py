"""Actual adapter, ABI scope, and complete frozen-runner control flow with GPU execution mocked."""
import argparse
import ast
from contextlib import ExitStack
import importlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
from types import MethodType,SimpleNamespace
from unittest.mock import patch

EXP=Path(__file__).parent
spec=importlib.util.spec_from_file_location('ck_lifetime_overlay',EXP/'agent-ck-lifetime-overlay.py')
O=importlib.util.module_from_spec(spec);spec.loader.exec_module(O)

def main(output):
  os.environ.update(DEV='USB+AMD:LLVM',ALLOW_DEVICE_USAGE='0',CACHEDB=str(O.ROOT/'cache.db'),TMPDIR=str(O.ROOT/'tmp'),
    OPENBLAS_NUM_THREADS='4',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',PROFILE='0',BEAM='0',JITBEAM='0',TC_OPT='2',LLVMOPT='1')
  tiny='/tmp/opencode/alpamayo-asius-20261114/tinygrad'
  base=O.WORK/'padding-source/alpamayo'
  sys.path[:0]=[tiny,str(base),str(base/'experiments')]
  import numpy as np
  from tinygrad import Tensor,dtypes,Device
  from tinygrad.device import Buffer,BufferSpec,HostAllocator
  from tinygrad.uop.ops import UOp
  from tinygrad.helpers import Target
  from tinygrad.renderer.llvmir import AMDLLVMRenderer
  from tinygrad.engine import realize
  from tinygrad.runtime import ops_amd
  from tinygrad.runtime.support import hcq2
  def audit(event,args):
    if event=='open' and isinstance(args[0],(str,bytes)):
      assert not os.fsdecode(args[0]).startswith(('/dev/bus/usb','/dev/dri','/dev/kfd'))
  sys.addaudithook(audit)
  renderer=AMDLLVMRenderer(Target('AMD','LLVM','gfx1200','USB'))
  dev=SimpleNamespace(device='AMD',host='CPU',arch='gfx1200',target=(12,0,0),is_usb=True,prof_enabled=False,
    renderer=renderer,iface=SimpleNamespace(props={'lds_size_in_kb':64}),synchronize=lambda:None)
  dev.allocator=HostAllocator(dev)
  dev._select_renderer=lambda *args:renderer
  def getitem(self,name): assert name=='AMD',('Forbidden device',name);return dev
  assert not Device._opened_devices;Device.__class__.__getitem__=getitem
  def tensor(shape,dt=dtypes.bfloat16,payload=None):
    b=Buffer('AMD',math.prod(shape)*dt.itemsize,options=BufferSpec(nolru=True)).ensure_allocated()
    if payload is not None: b.host.view(size=b.nbytes,fmt='B')[:]=payload
    return Tensor(UOp.from_buffer(b,dt)).reshape(shape)
  before_args=ops_amd.AMDComputeQueue.kernargs
  # Reproduce the real ownership defect, using the actual candidate_outputs call.
  old=O.previous.load_runner();old_common=old['run'].__globals__['C']
  h1=old_common.load('agent-ck-output-common');a1=h1.load('agent-ck-attention-v2-adapter')
  h2=old_common.load('agent-ck-output-common');a2=h2.load('agent-ck-attention-v2-adapter')
  assert h1 is not h2 and a1 is not a2
  xs=[tensor(shape) for shape in (h1.QSHAPE,h1.KVSHAPE,h1.KVSHAPE)]
  stores=[tensor((h1.OC+512,)),tensor((h1.LC+512,),dtypes.float32)]
  hook=old_common.load('agent-ck-model-hook')
  with a1.abi_scope():
    assert a1.ACTIVE and not a2.ACTIVE
    try: hook.candidate_outputs(old_common,*xs,*stores)
    except AssertionError as error: assert str(error)=='The parent must retain abi_scope through graph execution'
    else: raise AssertionError('Original helper duplication did not fail')
  assert not a1.ACTIVE and ops_amd.AMDComputeQueue.kernargs is before_args
  proof={'gpu_executed':False,'native_GPU_arithmetic_executed':False,'CPU_FP32_reference_executed':True,
    'original_duplicate_module_failure_reproduced':True,
    'mock_limit':'Recorded baseline and rounded FP32 reference substitute native output bytes. This is control-flow validation, not CK accuracy.',
    'runs':[],'lifetime_controls':[]}
  recorded=json.loads((O.WORK/'agent-ck-storage-parent/report.json').read_text())
  arrays={}
  for name,row in recorded['arrays'].items():
    assert O.sha(row['path'])==row['sha256'];arrays[name]=np.load(row['path'],allow_pickle=False)
  for failure in (False,True):
    namespace=O.load_runner();common=namespace['run'].__globals__['C'];normal_load=common.load
    helper=normal_load('agent-ck-output-common');adapter=helper.load('agent-ck-attention-v2-adapter')
    assert helper is normal_load('agent-ck-output-common')
    assert adapter is normal_load('agent-ck-output-common').load('agent-ck-attention-v2-adapter')
    assert adapter.attention.__globals__ is adapter.abi_scope.__wrapped__.__globals__
    # Other modules are intercepted only to expose host readback and reference results to the mock.
    modules={name:normal_load(name) for name in ('agent-ck-attention-v3-run','agent-ck-attention-v2-oracle')}
    common.load=lambda name,directory=EXP: modules[name] if name in modules and Path(directory)==EXP else normal_load(name,directory)
    old_io,oracle=modules.values();reference={};events=[];submitted=[]
    def fence():
      if events==['candidate_active_through_realize']:
        assert adapter.ACTIVE
        events.append('candidate_active_through_final_fence')
    dev.synchronize=fence
    actual_reference=oracle.reference
    def oracle_capture(inputs):
      result=actual_reference(inputs);reference['O'],reference['LSE']=result;return result
    oracle.reference=oracle_capture
    old_io.read=lambda t,device:bytes(t.uop.buffer.get_storage().host.view(size=t.uop.buffer.nbytes,fmt='B')[:])
    cpu_root=output.parent/(output.stem+('-CPU_MOCK_FAILURE' if failure else '-CPU_MOCK_COMPLETE'));cpu_root.mkdir(exist_ok=False)
    common.ROOT,common.PREPARED,common.PROTOCOL=cpu_root,O.PREPARED,O.OLD_PROTOCOL
    plan=O.plan();plan['launcher_command']=plan['launcher_command'][:]
    def put(buffer,payload):
      assert len(payload)==buffer.nbytes
      buffer.get_storage().host.view(size=buffer.nbytes,fmt='B')[:]=payload
    def mock_execute(linear,*args,**kwargs):
      compiled=realize.lower_and_compile(linear)
      batch=[(call.without_after,'AMD','COMPUTE:0') for call in compiled.src]
      if not batch: return
      # These are the actual source/AST/ELF, dependency, descriptor, and kernarg guards.
      hcq2._build_queues(hcq2.BatchCtx(batch,False))
      queue=SimpleNamespace(dev=dev,devs=('AMD',))
      for call,_,_ in batch:
        p=call.body;data,_=ops_amd.amd_build_program(dev,p,queue.devs)
        ops_amd.AMDComputeQueue.kernargs(queue,call,p,data)
        buffers=[u.buffer.ensure_allocated() for u in call.src[1:]]
        name=p.src[0].arg.name;submitted.append(name)
        if name=='alpamayo_ck_d128':
          assert adapter.ACTIVE and normal_load('agent-ck-output-common').load('agent-ck-attention-v2-adapter') is adapter
          events.append('candidate_active_through_realize')
          put(buffers[0],oracle.bf16_bits(reference['O']).tobytes());put(buffers[1],reference['LSE'].tobytes())
          if failure: raise RuntimeError('Deliberate CPU execution-seam failure')
        elif name=='agent_attention_rowsoftmax': put(buffers[0],arrays['baseline_O'].tobytes())
        else:
          assert len(buffers)==2 and buffers[0].nbytes==buffers[1].nbytes
          payload=bytes(buffers[1].get_storage().host.view(size=buffers[1].nbytes,fmt='B')[:])
          if buffers[0].nbytes==4194304:
            payload=np.frombuffer(payload,np.uint16).reshape(1,8,128,2,8,8,16).transpose(0,1,5,2,3,6,4).copy().tobytes()
          else: assert buffers[0].nbytes in (8388608,131072)
          put(buffers[0],payload)
    real_runpy=namespace['run'].__globals__['runpy'].run_path
    models=[]
    def mock_pipeline(path,run_name=None):
      assert Path(path)==base/'experiments/agent-speculation-run.py'
      # Startup and Q/K/V production are mocked with the captured real tensors.
      q,k,v=[tensor(shape,payload=arrays['live_'+name].tobytes()) for name,shape in zip(('q','k','v'),(helper.QSHAPE,helper.KVSHAPE,helper.KVSHAPE))]
      cache=tensor((2,1,8,2304,128),payload=arrays['live_cache'].tobytes())
      class Model:
        def __init__(self):
          self.config={'vlm_config':{'text_config':{'num_hidden_layers':28}}};self.cache=[cache]*28;self.output_token_mapping=object()
        def attention(self,*args,**kwargs): raise AssertionError('Unexpected fallback')
        def text_layer(self,x,cos,sin,prefix,cache=None,mask=None,cache_pos=None): return self.attention(q,k,v,mask,cache_pos==0)
        def linear(self,*args): raise AssertionError('Past first attention')
        def norm(self,*args): raise AssertionError('Past first attention')
      m=Model();models.append(m)
      selected=O.previous.load('agent-ck-model-common').load('agent-attention-rowsoftmax',base/'experiments')
      selected.install(m,'prefill','shuffle-fma',8)
      # Reuse the exact production quantize-packed text-layer wrapper, with its upstream computation mocked.
      path=base/'experiments/quantize-packed.py';tree=ast.parse(path.read_text())
      node=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='text_layer')
      ns={'state':{'prefill':False,'prefix':None,'index':0},'old_layer':m.text_layer}
      exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),'exec'),ns)
      m.text_layer=MethodType(ns['text_layer'],m)
      specmod=O.previous.load('agent-ck-model-common').load('agent-speculation-prefill',base/'experiments')
      specmod.install_prefill(m,[124549,11135,3174,4286,1561,135084],135084)
      # The original runner's import hook installs its real DiagnosticDecoder and Hook.
      spec=importlib.util.spec_from_file_location('cpu_lookup',base/'experiments/lookup-decode.py')
      module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);module.LookupDecoder(m)
      assert m._agent_ck_model_hook is not None
      x=tensor((1,2048,2048));cs=tensor((1,1,2048,128))
      m.text_layer(x,cs,cs,'vlm.model.language_model.layers.0',cache,None,cache_pos=0)
      raise AssertionError('Frozen runner did not stop after its first-layer comparison')
    output_path=cpu_root/'mock-control-flow-report.json'
    tm=importlib.import_module('tinygrad.tensor')
    def descriptor_only(device,program,devices): return ops_amd._amd_program_image(device,program.src[3].arg)
    with ExitStack() as stack:
      stack.enter_context(patch.object(tm,'run_linear',mock_execute))
      stack.enter_context(patch.object(ops_amd,'amd_build_program',descriptor_only))
      stack.enter_context(patch.object(namespace['run'].__globals__['runpy'],'run_path',mock_pipeline))
      try: namespace['run'](plan,output_path)
      except RuntimeError as error: assert failure and str(error)=='Deliberate CPU execution-seam failure'
      else: assert not failure
    assert not adapter.ACTIVE and ops_amd.AMDComputeQueue.kernargs is before_args
    assert all(not hasattr(m,'_agent_ck_model_hook') for m in models)
    result=json.loads(output_path.read_text())
    original_flags={k:result[k] for k in ('gpu_executed','completed','live_first_layer_micro_passed')}
    result.update(gpu_executed=False,native_GPU_arithmetic_executed=False,CPU_MOCK_ONLY=True,
      simulated_runner_flags=original_flags,numerical_acceptance=False,live_first_layer_micro_passed=False)
    output_path.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    if not failure:
      assert events==['candidate_active_through_realize','candidate_active_through_final_fence']
      assert original_flags['completed'] and result['io']=={'read_bytes':68423680,'poison_write_bytes':17045504}
      assert result['original_matches_guarded_baseline_bits'] and result['hooks_restored']
      assert len(result['actual_candidate_pointer_bindings'])==1
    proof['runs'].append({'mock_failure_injected':failure,'runner_reached_final_success':original_flags['completed'],
      'scope_restored':True,'hooks_restored':result['hooks_restored'],'events':events,'program_calls':submitted,'io':result['io']})
    # ACTIVE is still required before and after the genuine scope, and nested entry is rejected.
    hookmod=normal_load('agent-ck-model-hook')
    args=[tensor(shape) for shape in (helper.QSHAPE,helper.KVSHAPE,helper.KVSHAPE)]+[
      tensor((helper.OC+512,)),tensor((helper.LC+512,),dtypes.float32)]
    try: hookmod.candidate_outputs(common,*args)
    except AssertionError as error: assert 'retain abi_scope' in str(error)
    else: raise AssertionError('Inactive adapter was accepted')
    with adapter.abi_scope():
      try:
        with adapter.abi_scope(): raise AssertionError('Nested ABI scope accepted')
      except AssertionError: assert adapter.ACTIVE
    assert not adapter.ACTIVE and ops_amd.AMDComputeQueue.kernargs is before_args
    proof['lifetime_controls'].append('inactive invocation rejected; nested scope rejected; exception cleanup restored')
  assert not Device._opened_devices
  proof['source_sha256']={str(p):O.sha(p) for p in (Path(__file__),EXP/'agent-ck-lifetime-overlay.py')}
  with output.open('x') as stream: stream.write(json.dumps(proof,indent=2,allow_nan=False)+'\n')
  print(json.dumps({'gpu_executed':False,'complete_mock_run':True,'failure_cleanup':True,'proof':str(output)}))

if __name__=='__main__':
  parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args();main(args.output)
