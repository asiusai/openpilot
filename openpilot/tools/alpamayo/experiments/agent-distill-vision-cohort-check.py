"""CPU-only checks. Reuse completed old inputs; never inspect pending cohort outputs."""
import argparse
from copy import deepcopy
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
from types import MethodType, SimpleNamespace

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
BASE = Path('/tmp/opencode/alpamayo-asius-20261114')
WORK = Path('/mnt/nvme-500/tmp/alpamayo-speed-20261116')
ROOT = WORK/'agent-distill-vision-cohort-artifacts'


def load(name,path):
  spec=importlib.util.spec_from_file_location(name,path); value=importlib.util.module_from_spec(spec)
  sys.modules[name]=value; spec.loader.exec_module(value); return value


def main():
  parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('--output',type=Path,required=True)
  args=parser.parse_args(); root=args.output.resolve().parent
  if not root.is_relative_to(ROOT) or root.exists(): raise ValueError('Need new owned CPU-check output directory')
  root.mkdir(parents=True); (root/'tmp').mkdir()
  os.environ.update(DEV='CPU',TC='0',BEAM='0',JITBEAM='0',PARALLEL='2',OPENBLAS_NUM_THREADS='4',OMP_NUM_THREADS='4',
    HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',
    CUDA_VISIBLE_DEVICES='',HIP_VISIBLE_DEVICES='',ROCR_VISIBLE_DEVICES='',CACHEDB=str(root/'cpu-cache.db'),TMPDIR=str(root/'tmp'))
  def audit(event,values):
    if event!='open' or not isinstance(values[0],(str,bytes)): return
    path=os.fsdecode(values[0])
    if path.startswith(('/dev/kfd','/dev/dri/','/dev/nvidia','/dev/bus/usb/')): raise RuntimeError('CPU check attempted device access')
    if 'agent-distill-cohort-artifacts' in path: raise RuntimeError('CPU check attempted pending cohort input access')
    if any(x in path for x in ('0000008e--321b2ddc03','00000090--1cf3f7d926')): raise RuntimeError('Excluded recording access')
  sys.addaudithook(audit)
  sys.path.insert(0,str(BASE/'tinygrad'))
  import numpy as np
  from tinygrad import Tensor,TinyJit,Device,dtypes
  from tinygrad.helpers import Context
  C=load('vision_cohort_check_capture',HERE/'agent-distill-vision-cohort-capture.py'); plan=C.checked_plan()
  P=load('vision_cohort_check_parent',HERE/'agent-distill-joint-parent.py')
  R=load('vision_cohort_check_teacher',HERE/'agent-distill-joint-runtime.py')
  H=load('vision_cohort_check_reset',HERE/'agent-distill-vision-runtime-hook.py')
  old_input=load('vision_cohort_check_input',HERE/'agent-distill-command-capture.py')
  oldrow=P.selected_rows(P.INPUTS,['train-command-0'])[0]
  ready=old_input.prepared_inputs(plan,dict(samples=[oldrow]))[0]
  native=Path(plan['runtime_root']); sys.path[:0]=[str(native),str(native/'experiments')]
  Native=load('vision_cohort_check_native',native/'model.py').EdgeModel
  batched=load('vision_cohort_check_batched',native/'experiments/batched-vision.py')
  cache_module=load('vision_cohort_check_cache',native/'experiments/vision-cache.py')
  CONFIG=C.read(BASE/'model/config.json'); trace=[]; used={}; negatives=[]
  def reject(name,fn):
    try: fn()
    except (ValueError,AssertionError,KeyError): negatives.append(name)
    else: raise AssertionError('Accepted '+name)
  def fixture_model():
    model=Native.__new__(Native); model.config=deepcopy(CONFIG); model.dtype=dtypes.bfloat16
    model.weight_scales={}; model.fp8_wmma=model.cuda_kernels=False; model.cache=None
    model.prompt_length=model.greedy_shape=None; model.trajectory_jits={}; model.encode_jits={}
    model.weights={'vlm.model.language_model.embed_tokens.weight':Tensor.zeros(1,2048,dtype=dtypes.bfloat16).realize()}
    def linear(self,x,name,quantized_input=None,heads=None):
      if name=='vlm.model.visual.embeddings.patch_embedding': return x[:,:,:1].expand(*x.shape[:-1],1152)
      if name=='vlm.model.projector.linear_fc1': trace.append(name); return x
      if name=='vlm.model.projector.linear_fc2': trace.append(name); return x[:,:2048]
      raise AssertionError(name)
    model.linear=MethodType(linear,model)
    def norm(x,name,*unused): trace.append(name); return x
    model.norm=norm
    def layer(self,x,i): trace.append(('layer',x.shape[0],i)); return (x.float()+(i+1)/1024).cast(self.dtype).realize()
    model.vision_layer=MethodType(layer,model)
    encode=batched.install(model)
    # The selected packed projector bypasses model.linear; its outer hook saves each feature once.
    project_linear=model.linear
    def projector(value):
      outputs=[]
      for i in range(value.shape[0]):
        x=model.norm(value[i:i+1],'vlm.model.visual.post_layernorm')
        x=model.norm(x,'vlm.model.projector.norm').reshape(-1,4608)
        x=project_linear(x,'vlm.model.projector.linear_fc1').float().gelu(approximate='none').cast(model.dtype)
        outputs.append(project_linear(x,'vlm.model.projector.linear_fc2').realize())
      return tuple(outputs)
    model.batched_projector=projector
    cache=cache_module.VisionCache(model,capacity=144,grouped=True,views=False,batch_encoder=encode,group_limit=2)
    decoder=SimpleNamespace(history=[],previous=[],statistics=[])
    return model,cache,decoder
  def fixture_inputs(prepared,packed):
    # Full original shapes; only the expensive arithmetic and uploaded values are mocked.
    pixels=Tensor(np.repeat(np.arange(10,dtype=np.float32)/16,720).reshape(-1,1)).expand(7200,768).cast(dtypes.bfloat16).contiguous().realize()
    positions=Tensor.zeros(7200,1152,dtype=dtypes.bfloat16).realize()
    batches=tuple((i*720,(i+1)*720,1) for i in range(10))
    return SimpleNamespace(upload=lambda noise:(pixels,positions),batches=batches)
  def initialize(options,**unused):
    model,cache,decoder=fixture_model()
    padding=load('vision_cohort_check_padding',native/'experiments/padded-prefill.py')
    report=C.read(C.LEGACY['train-command-0'][0])['settings']; settings=SimpleNamespace(**report['settings'])
    settings.initial_noise=np.load(options.initial_noise,allow_pickle=False)
    # Warm stale graph/cache state must be removed before first boundary call.
    warm=TinyJit(lambda x:(x+1).realize()); x=Tensor.zeros(1).realize(); warm(x); warm(x)
    model.encode_jits['stale']=warm; cache.cache['stale']=x; cache.encoders['stale']=warm
    decoder.history.append([99])
    used.update(model=model,cache=cache,decoder=decoder,warm=warm)
    return SimpleNamespace(model=model,vision_cache=cache,decoder=decoder,pad_inputs=padding.pad_inputs,
      settings=settings,config=None,report=report,input_class=lambda settings,config,prepared,packed:fixture_inputs(prepared,packed))
  options=SimpleNamespace(output=root/'runner-report.json',legacy_control=None,inputs_sha256=None)
  result=C.capture_ready(options,plan,[ready],initialize=initialize,cpu_mock=True)
  captured=result['rows'][0]['targets']; mapping=result['rows'][0]['images']; all_trace=list(trace)
  assert len(captured['tensors'])==80 and captured['projector_calls']==10
  assert [t for t in trace if isinstance(t,tuple)]==[('layer',2,i) for _ in range(5) for i in range(27)]
  tail=['vlm.model.visual.post_layernorm','vlm.model.projector.norm','vlm.model.projector.linear_fc1','vlm.model.projector.linear_fc2']
  assert [t for t in trace if isinstance(t,str)]==tail*10
  assert [(r['batch'],r['member']) for r in mapping]==[(i//2,i%2) for i in range(10)]
  assert mapping[4]['camera']==1 and mapping[5]['camera']==6 and mapping[4]['batch']==mapping[5]['batch']==2
  assert used['warm'].cnt==0 and not used['cache'].cache and not used['cache'].encoders and not used['model'].encode_jits
  assert not used['decoder'].history and used['model'].cache is None

  def metadata(name,tensor):
    array=tensor.contiguous().bitcast(dtypes.uint16).numpy().copy(); stream=io.BytesIO(); np.save(stream,array,allow_pickle=False)
    return dict(shape=list(array.shape),dtype=str(array.dtype),bf16_bits=True,sha256=hashlib.sha256(stream.getvalue()).hexdigest())
  # Compare against the unchanged native VisionCache.encode callsite, not a second hand-written grouping function.
  model,cache,decoder=fixture_model(); cache.prepare(ready[1]); inputs=fixture_inputs(ready[1],ready[2]); pixels,positions=inputs.upload(None)
  with Context(JIT=0),R.teacher_targets(model,metadata,groups=R.VISION_GROUPS) as reference:
    output=cache.encode(pixels,positions,Tensor.zeros(1,1800,dtype=dtypes.int32).realize(),inputs.batches,((0,1800),))
    assert output.shape==(1,1800,2048)
  assert set(reference['tensors'])==set(captured['tensors'])
  assert all(reference['tensors'][k]['sha256']==v['sha256'] for k,v in captured['tensors'].items())
  H.reset_state(model,decoder,cache); cache.prepare(ready[1])
  with Context(JIT=0),R.teacher_targets(model,metadata) as original_onegroup:
    cache.encode(pixels,positions,Tensor.zeros(1,1800,dtype=dtypes.int32).realize(),inputs.batches,((0,1800),))
  group12=[k for k in original_onegroup['tensors'] if '.group12.' in k]
  assert len(group12)==10 and all(original_onegroup['tensors'][k]['sha256']==captured['tensors'][k]['sha256'] for k in group12)

  # Repeated images retain all ten logical positions, but execute once per unique key.
  H.reset_state(model,decoder,cache); cache.prepare(ready[1]); keys=list(cache.keys); keys[-1]=keys[0]; cache.keys=tuple(keys)
  with Context(JIT=0),C.vision_boundaries(model,metadata) as repeated:
    duplicates=C.vision_only(cache,pixels,positions,inputs.batches)
  C.validate_targets(repeated,duplicates)
  assert [r['images'] for r in repeated['vision_batches']]==[2,2,2,2,1]
  assert duplicates[9]['batch']==duplicates[0]['batch']==0 and duplicates[9]['member']==duplicates[0]['member']==0
  assert duplicates[9]['camera']==6 and duplicates[9]['history_index']==4 and repeated['projector_calls']==9
  original=model.text_layer
  try:
    with C.vision_boundaries(model,metadata):
      reject('language_execution',lambda:model.prefill(None)); raise RuntimeError('context test')
  except RuntimeError as error: assert str(error)=='context test'
  assert model.text_layer==original
  bad=deepcopy(captured); bad['vision_batches'][0]['layer_order'].pop(); reject('missing_layer',lambda:C.validate_targets(bad,mapping))
  bad=deepcopy(captured); bad['tensors']['vision0.group12.x']['shape']=[2,719,1152]; reject('cropped_boundary',lambda:C.validate_targets(bad,mapping))
  bad=deepcopy(captured); bad['tensors']['vision0.group12.x']['sha256']='0'*64; reject('broken_adjacent_boundary',lambda:C.validate_targets(bad,mapping))
  reject('nonfinite_target',lambda:C.save_bits(root,'nonfinite',Tensor([float('nan')]).cast(dtypes.bfloat16)))
  reject('wrong_target_dtype',lambda:C.save_bits(root,'float32',Tensor([1.])))
  reject('nonempty_cache',lambda:C.make_chunks(cache,inputs.batches))

  # Synthetic producer contract checks use closed metadata only. No pending prepared files are opened.
  anchors=C.read(C.COVERAGE)['anchors']; fake=dict(admissible_for_internal_layer_capture=True,
    admissible_for_driving_quality_validation=False,completed_native_frames=240,completed_conditioned_inputs=24,
    independent_drives=2,protected_and_sealed_drives_opened=False,protocol=dict(coverage=dict(path=str(C.COVERAGE),sha256=C.COVERAGE_SHA)),samples=[])
  for anchor in anchors:
    row={k:anchor[k] for k in ('id','split','route_id','drive_id','query_ns','conditioning')}
    row.update(command_index=0,actual_prompt_tokens=2027,complete_input=True,official_selected_tokens_and_image_normalization_exact=True,
      sample=str(WORK/'agent-distill-cohort-artifacts/inputs-v1/samples'/row['id']),input_file_sha256={n:'0'*64 for n in C.INPUT_NAMES})
    fake['samples'].append(row)
  assert [r['id'] for r in C.validate_manifest(fake,['validation-scene-00','train-scene-00'],verify_files=False)]==['train-scene-00','validation-scene-00']
  bad=deepcopy(fake); bad['completed_conditioned_inputs']=23; reject('unfinished_manifest',lambda:C.validate_manifest(bad,['train-scene-00'],verify_files=False))
  bad=deepcopy(fake); bad['samples'][0]['route_id']='0000008e--321b2ddc03'; reject('protected_route',lambda:C.validate_manifest(bad,['train-scene-00'],verify_files=False))
  bad=deepcopy(fake); bad['samples'][0]['route_id']='00000090--1cf3f7d926'; reject('sealed_route',lambda:C.validate_manifest(bad,['train-scene-00'],verify_files=False))
  reject('too_many_scenes',lambda:C.validate_manifest(fake,[r['id'] for r in fake['samples'][:3]],verify_files=False))
  reject('duplicate_scene',lambda:C.validate_manifest(fake,['train-scene-00']*2,verify_files=False))
  bad=deepcopy(fake); bad['samples'].reverse(); reject('reordered_cohort',lambda:C.validate_manifest(bad,['train-scene-00'],verify_files=False))

  # Completed pilot evidence: distinguish moment estimates from a newly measured gradient.
  trained,selection=H.load_selected(); seed,_=H.load_seed()
  checkpoint=WORK/'agent-distill-vision-artifacts/pilot-v1/checkpoint-0025.npz'
  assert C.sha(checkpoint)=='aef46994b09e4600c61f22d4ff12e734dbf231a4cf808ef5e6556b147101a21b'
  parameter=[]; sum_delta=sum_seed=0.; changed=total=0
  with np.load(checkpoint,allow_pickle=False) as recovery:
    for name,weight in trained.items():
      delta=weight.astype(np.float64)-seed[name]; dn=float(np.square(delta).sum()); sn=float(np.square(seed[name].astype(np.float64)).sum())
      sum_delta+=dn; sum_seed+=sn
      bits=Tensor(weight.copy()).cast(dtypes.bfloat16).bitcast(dtypes.uint16).numpy()
      seedbits=(seed[name].view(np.uint32)>>16).astype(np.uint16)
      count=int(np.count_nonzero(bits!=seedbits)); changed+=count; total+=weight.size
      moment1=recovery['moment1/'+name].astype(np.float64)/(1-.9**25)
      moment2=recovery['moment2/'+name].astype(np.float64)/(1-.999**25)
      parameter.append(dict(name=name,elements=weight.size,delta_l2=dn**.5,relative_delta_l2=(dn/sn)**.5 if sn else None,
        max_absolute_delta=float(np.abs(delta).max()),bf16_changed_elements=count,
        corrected_first_moment_rms=float(np.sqrt(np.square(moment1).mean())),corrected_second_moment_sqrt_mean=float(np.sqrt(moment2.mean()))))
  report=dict(complete=True,gpu_executed=False,training_updates=0,pending_cohort_outputs_read=False,
    actual_selected_launcher_parser=True,actual_original_batched_encoder_and_vision_cache=True,
    full720_mock_arithmetic=True,actual_completed_input_fingerprints=True,all_seven_groups=80,
    original_onegroup12_arrays_bitexact_in_cpu_mock=True,original_allgroup_cache_arrays_bitexact_in_cpu_mock=True,
    camera_history_batch_boundary_checked=True,duplicate_logical_images_preserved=True,exception_restoration=True,
    final_norm_and_projector_preserved=True,stale_jits_and_cache_cleared=True,negative_controls=negatives,
    pilot=dict(selection=selection,relative_parameter_delta_l2=(sum_delta/sum_seed)**.5,
      bf16_changed_elements=changed,elements=total,bf16_changed_fraction=changed/total,parameters=parameter),
    maximum_target_bytes_24_scenes=24*C.BYTES_PER_SCENE,quality_qualified=False,
    opened_devices=sorted(Device._opened_devices),sources={str(p):C.sha(p) for p in HERE.glob('agent-distill-vision-cohort-*.py')})
  P.no_devices(); assert all(d.startswith(('CPU','NPY','DISK','PYTHON')) for d in Device._opened_devices),Device._opened_devices
  args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
  print(json.dumps({k:v for k,v in report.items() if k not in ('pilot','sources')},indent=2))


if __name__=='__main__': main()
