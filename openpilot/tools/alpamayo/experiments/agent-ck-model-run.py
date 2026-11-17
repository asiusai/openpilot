"""Parent-only first-live-layer prerequisite; no route or full-model candidate run."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import runpy
import signal
import sys
import time

EXP = Path(__file__).parent
spec = importlib.util.spec_from_file_location('ck_model_common',EXP/'agent-ck-model-common.py')
C = importlib.util.module_from_spec(spec); spec.loader.exec_module(C)

class CaptureComplete(Exception): pass

def structural_graph(tensor):
  """Record movement operations and producer AST, without asserting a fictitious Q stride."""
  from tinygrad.uop.ops import Ops
  rows=[]
  for u in tensor.uop.toposort(enter_calls=True):
    if u.op in (Ops.RESHAPE,Ops.PERMUTE,Ops.SHRINK,Ops.AFTER,Ops.STORE,Ops.BUFFER,Ops.CALL):
      try: shape=list(u.shape)
      except Exception: shape=None
      rows.append({'op':u.op.name,'shape':shape,'arg':str(u.arg),'key':u.key.hex(),'sources':[x.key.hex() for x in u.src]})
  return {'shape':list(tensor.shape),'dtype':str(tensor.dtype),'uop_key':tensor.uop.key.hex(),'movement_and_dependencies':rows}

def run(plan, output):
  import numpy as np
  from tinygrad import Tensor,dtypes,Device
  from tinygrad.device import Compiled
  hookmod = C.load('agent-ck-model-hook')
  helper = C.load('agent-ck-output-common')
  old = C.load('agent-ck-attention-v3-run')
  adapter = helper.load('agent-ck-attention-v2-adapter')
  oracle = C.load('agent-ck-attention-v2-oracle')
  guard = C.load('agent-ck-model-guard')
  catalog = json.loads((C.PREPARED/'catalog.json').read_text())
  producer=C.load('agent-ck-model-producer')
  producer_catalog=json.loads((C.PREPARED/'producer-catalog.json').read_text())
  report={'schema':1,'completed':False,'gpu_executed':None,'gpu_requested':True,'live_layout_captured':False,'live_first_layer_micro_passed':False,
    'model_integrated':False,'route_screen_executed':False,'adoption_allowed':False,'timings_valid_for_throughput':False,
    'protocol_sha256':C.sha(C.PROTOCOL),'micro_evidence_sha256':C.sha(C.MICRO),'phase':'startup',
    'selected_layers':[],'io':{'read_bytes':0,'poison_write_bytes':0},'remaining_gates':plan['remaining_gates']}
  folder=C.ROOT/'parent-artifacts';folder.mkdir(exist_ok=False)
  started=time.monotonic();instances=[]
  def save(): output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
  def phase(name): report['phase']=name;report.setdefault('phase_started_seconds',{})[name]=time.monotonic()-started;save()
  def deadline(*_): raise TimeoutError('170-second first-live-layer prerequisite deadline')
  signal.signal(signal.SIGALRM,deadline);signal.alarm(170)
  Compiled.wait_timeout_ms=10000
  def read(tensor,device):
    data=old.read(tensor,device);report['io']['read_bytes']+=len(data);return data
  def array_file(name,array):
    path=folder/(name+'.npy');assert not path.exists();np.save(path,array,allow_pickle=False)
    report.setdefault('arrays',{})[name]={'path':str(path),'sha256':C.sha(path),'shape':list(array.shape),'dtype':str(array.dtype)}
  def snapshot(arm,stores,returned,device):
    bits=[];records=[]
    count=2 if arm=='candidate' else 1
    for index in range(count):
      dt,shape,poison,sentinel=(np.uint16,helper.QSHAPE,0x7fc1,0x5aa5) if index==0 else (np.uint32,helper.LSHAPE,0x7fc12345,0x5aa55aa5)
      whole=np.frombuffer(read(stores[index],device),dt)
      raw=np.frombuffer(read(returned[index],device),dt)
      interior=whole[256:-256].reshape(shape)
      assert raw.size==interior.size and np.array_equal(interior.reshape(-1),raw)
      assert np.all(whole[:256]==sentinel) and np.all(whole[-256:]==sentinel)
      values=oracle.fp32(interior) if index==0 else interior.view(np.float32)
      assert np.isfinite(values).all() and not np.any(interior==poison)
      records.append({'guard_mismatches':0,'returned_mismatches':0,'nonfinite':0,'poison':0,
        'guarded_storage':helper.storage_info(stores[index],True),'returned_storage':helper.storage_info(returned[index],True),
        'kernel_pointer':helper.storage_info(stores[index],True)['gpu_address']+256*np.dtype(dt).itemsize})
      bits.append(interior.copy());array_file(arm+('_O' if index==0 else '_LSE'),interior)
    report[arm+'_checks']=records;save();return bits
  def capture(handle,context,q,k,v,original):
    assert handle.calls==[0] and context['layer']==0 and not report['selected_layers']
    device=Device[q.device];assert device.device=='AMD' and device.is_usb and device.arch=='gfx1200' and not device.prof_enabled
    report['gpu_executed']=True;report['selected_layers']=[0]
    phase('live_producer_layout')
    report['logical_inputs']={n:structural_graph(t) for n,t in zip(('q','k','v'),(q,k,v))}
    for n,t in zip(('q','k','v'),(q,k,v)):
      (folder/(n+'-live-uop.txt')).write_text(str(t.uop)+'\n')
    cache=context['cache'];assert cache is handle.model.cache[0]
    report['cache_logical']=structural_graph(cache)
    # Force the real producer and cache STORE dependencies once. These are diagnostic fences.
    packed=tuple(t.contiguous() for t in (q,k,v))
    report['model_compiler_before']=type(device.renderer.compiler).__name__
    with producer.observe(producer_catalog,report,folder): packed[0].realize(*packed[1:])
    device.synchronize()
    report['packed_inputs']=[helper.storage_info(t,True) for t in packed]
    report['actual_cache_storage']=helper.storage_info(cache,True)
    C.storage_gate(report['packed_inputs'],[8388608,4194304,4194304])
    live={n:np.frombuffer(read(t,device),np.uint16).reshape(t.shape).copy() for n,t in zip(('q','k','v'),packed)}
    for name,bits in live.items(): assert np.isfinite(oracle.fp32(bits)).all();array_file('live_'+name,bits)
    cache_bits=np.frombuffer(read(cache,device),np.uint16).reshape(cache.shape)
    assert np.array_equal(cache_bits[0,:,:,:2048],live['k']) and np.array_equal(cache_bits[1,:,:,:2048],live['v'])
    assert np.all(cache_bits[:,:,:,2048:]==0), 'Unexpected first-prefill capacity tail'
    report['capacity_prefix_matches_packed']=True;report['capacity_tail_initial_zero']=True
    report['cache_physical_layout']={'shape':[2,1,8,2304,128],'element_strides':[2359296,2359296,294912,128,1],
      'K_byte_offset':0,'V_byte_offset':4718592,'prefix_rows':2048,'capacity_rows':2304}
    array_file('live_cache',cache_bits)
    report['live_layout_captured']=True;save()
    # The original method, not a recreated text layer, supplies the live comparator.
    phase('original_selected_attention')
    with guard.guard(C,catalog,report):
      direct=original(*packed,mask=None,causal=True).realize()
    device.synchronize();direct_bits=np.frombuffer(read(direct,device),np.uint16).reshape(helper.QSHAPE).copy()
    assert np.isfinite(oracle.fp32(direct_bits)).all();array_file('original_O',direct_bits)
    phase('independent_full_FP32')
    reference,lse_reference=oracle.reference(live);array_file('reference_O',reference);array_file('reference_LSE',lse_reference)
    results={}
    for arm in ('baseline','candidate'):
      phase('guarded_'+arm)
      stores=[old.allocate(device.device,(helper.OC+512,),dtypes.bfloat16),old.allocate(device.device,(helper.LC+512,),dtypes.float32)]
      for index,t in enumerate(stores):
        bits=np.full(t.numel(),0x7fc1 if index==0 else 0x7fc12345,np.uint16 if index==0 else np.uint32)
        bits[:256]=bits[-256:]=0x5aa5 if index==0 else 0x5aa55aa5
        old.write(t,bits.tobytes());report['io']['poison_write_bytes']+=bits.nbytes
      if arm=='candidate':
        report['expected_candidate_pointers']=[helper.storage_info(stores[0],True)['gpu_address']+512,
          helper.storage_info(stores[1],True)['gpu_address']+1024,*[r['gpu_address'] for r in report['packed_inputs']]]
      with adapter.abi_scope(),guard.guard(C,catalog,report):
        returned=(helper.execute('baseline',True,*packed,*stores) if arm=='baseline' else
                  hookmod.candidate_outputs(C,*packed,*stores))
        device.synchronize()
      results[arm]=snapshot(arm,stores,returned,device)
      report[arm+'_fp32_error']=old.metrics(oracle.fp32(results[arm][0]),reference)
    assert np.array_equal(direct_bits,results['baseline'][0]), 'Live selected baseline differs from frozen numerical kernel'
    old.error_gate(report['candidate_fp32_error'],report['baseline_fp32_error'],plan['limits'])
    error=results['candidate'][1].view(np.float32).astype(np.float64)-lse_reference
    report['candidate_lse_error']={'rmse':float(np.sqrt(np.mean(error*error))),'max_abs':float(np.max(np.abs(error)))}
    assert report['candidate_lse_error']['rmse']<=plan['limits']['lse_rmse_abs']
    assert report['candidate_lse_error']['max_abs']<=plan['limits']['lse_max_abs']
    report['original_matches_guarded_baseline_bits']=True;report['live_first_layer_micro_passed']=True
    assert len(report['actual_candidate_pointer_bindings'])==1
    assert report['io']=={'read_bytes':plan['budget']['read_bytes'],'poison_write_bytes':plan['budget']['poison_write_bytes']}
    assert type(device.renderer.compiler).__name__==report['model_compiler_before']
    report['model_renderer_unchanged']=True;report['standalone_micro_compiler']='Explicit CU renderer; exact frozen associations'
    phase('prerequisite_complete');raise CaptureComplete()
  old_spec,old_argv,old_path=importlib.util.spec_from_file_location,sys.argv,sys.path[:]
  def spec_hook(name,location,*args,**kwargs):
    value=old_spec(name,location,*args,**kwargs)
    if Path(location).resolve()!=C.BASE/'experiments/lookup-decode.py': return value
    loader=value.loader
    class Loader:
      def create_module(self,spec): return loader.create_module(spec)
      def exec_module(self,module):
        loader.exec_module(module);original=module.LookupDecoder
        class DiagnosticDecoder(original):
          def __init__(self,model):
            super().__init__(model);assert self.history==[]
            report['gpu_executed']=True
            report['initial_decoder_history']=[]
            report['installed_method_origins']={n:getattr(model,n).__func__.__code__.co_filename for n in ('text_layer','attention','linear','norm')}
            assert Path(report['installed_method_origins']['text_layer'])==C.BASE/'experiments/quantize-packed.py'
            assert Path(report['installed_method_origins']['attention'])==C.BASE/'experiments/agent-attention-rowsoftmax.py'
            assert type(model.padded_prefill).__name__=='SpeculativePrefill'
            instance=hookmod.Hook(model,capture);instances.append(instance)
        module.LookupDecoder=DiagnosticDecoder
    value.loader=Loader();return value
  save()
  try:
    importlib.util.spec_from_file_location=spec_hook
    sys.argv=plan['launcher_command']
    try: runpy.run_path(sys.argv[0],run_name='__main__')
    except CaptureComplete:
      assert report['live_first_layer_micro_passed'];report['completed']=True
    else: raise AssertionError('Live attention interception did not stop the pipeline')
  except BaseException as error:
    report['failure']={'type':type(error).__name__,'message':str(error),'phase':report['phase']};raise
  finally:
    importlib.util.spec_from_file_location=old_spec;sys.argv=old_argv;sys.path[:]=old_path
    for instance in reversed(instances): instance.close()
    report['hooks_restored']=all(x.closed for x in instances)
    signal.alarm(0);report['elapsed_seconds']=time.monotonic()-started;save()

def main():
  parser=argparse.ArgumentParser();parser.add_argument('--parent-run',action='store_true');args=parser.parse_args()
  plan=C.verify()
  assert not sys.flags.optimize
  if not args.parent_run:
    print(json.dumps({'verified':True,'gpu_executed':False,'protocol':str(C.PROTOCOL),'scope':plan['scope']}));return
  output=C.ROOT/'parent-result.json'
  assert not output.exists() and not (C.ROOT/'parent-artifacts').exists()
  invocation=json.loads((C.PREPARED/'invocation.json').read_text())
  for key,value in invocation['environment'].items():
    if value is None: os.environ.pop(key,None)
    else: os.environ[key]=value
  os.environ.update(ALLOW_DEVICE_USAGE='1',PYTHONDONTWRITEBYTECODE='1',TMPDIR=str(C.ROOT/'tmp'),OPENBLAS_NUM_THREADS='4',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4')
  sys.path[:0]=[str(C.TINY),str(C.BASE),str(C.BASE/'experiments')]
  run(plan,output)

if __name__=='__main__': main()
