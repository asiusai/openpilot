"""Complete real entrypoint/launcher tests with an explicitly mocked GPU factory. No GPU inference."""
import argparse
import ast
import importlib.util
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys
from types import MethodType,SimpleNamespace

sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('agent_distill_cohort_common_check',HERE/'agent-distill-cohort-common.py')
c=importlib.util.module_from_spec(spec);sys.modules[spec.name]=c;spec.loader.exec_module(c)

def mock_job(job,root,fail=False):
  """Execute the final capture CLI, real CPU preparation, real launcher/parser, and full serialization."""
  import numpy as np
  plan_path=c.ROOT/'capture-v1'/('agent-distill-cohort-job-'+str(job)+'.json');plan=c.read(plan_path)
  runtime=Path(plan['runtime_root']);exps=runtime/'experiments'
  runner=c.ROOT/'capture-v1/frozen/agent-distill-cohort-capture.py'
  output=root/'report.json';observations=[];models=[]
  old_spec,old_run,old_argv,old_env=importlib.util.spec_from_file_location,runpy.run_path,sys.argv[:],dict(os.environ)
  def factory(module,args,**kwargs):
    from tinygrad import Device,dtypes
    assert not Device._opened_devices
    assert args.samples==[next(row['sample'] for row in c.read(c.INPUTS)['samples'] if row['id']==identity) for identity in plan['scene_ids']]
    assert args.steps==3 and args.pad_prompt_to==2048 and args.vision_calibration is None
    assert args.warmup==0 and args.runs==12 and args.max_tokens==64
    assert args.native_manifest is args.native_calibrations is args.native_frame_cache is None
    assert args.pixel_scale==args.history_feature_scale==args.history_pixel_scale==1 and args.front_history_indices is None
    padded=module.local_module('padded-prefill');online=module.local_module('online-attention');lookup=module.local_module('lookup-decode')
    assert Path(padded.pad_inputs.__code__.co_filename)==exps/'agent-speculation-run.py'
    assert online.install.__name__=='install_attention' and lookup.LookupDecoder.__name__=='Decoder'
    observations.append(dict(real_pipeline_parser=True,real_canonical_import_hooks=True,exact_job_samples=args.samples,
      nominal_padding_function_from_original_launcher=True,factory_mocked=True))
    class Fake:
      def __init__(self,shape,code=0x3f80):self.shape=tuple(shape);self.dtype=dtypes.bfloat16;self.code=code
      def __getitem__(self,key):return Fake(np.broadcast_to(np.array(self.code,np.uint16),self.shape)[key].shape,self.code)
      def contiguous(self):return self
      def bitcast(self,dtype):assert dtype==dtypes.uint16;return self
      def numpy(self):return np.full(self.shape,self.code,np.uint16)
    class Scalar:
      def __init__(self,value):self.value=value
      def numpy(self):return np.float32(self.value)
    class MockPrefill:
      def __init__(self,model):self.model=model;self.shape=None;self.resets=0;self.jit=SimpleNamespace(reset=self.reset)
      def reset(self):self.resets+=1
      def __call__(self,x,cos,sin,max_tokens,length):
        assert max_tokens==64 and length==2027 and x.shape==(1,2048,2048)
        for index in range(28):x=self.model.text_layer(x,cos,sin,'vlm.model.language_model.layers.'+str(index),cache_pos=0)
        raise AssertionError('Capture must stop before layer16/reasoning')
    class Model:
      def __init__(self):
        self.dtype=dtypes.bfloat16;self.calls=[];self.encode_calls=0;self.cache=[None]*28
        self.cache[12]=Fake((2,1,8,2304,128),0x4000);self.padded_prefill=MockPrefill(self);self.non_target=object()
        fixed={key:Scalar(value) for key,value in c.read(args.prefill_calibration)['scales'].items()}
        # Deliberate mock of the installed closure interface. The actual finder still executes.
        namespace={};exec(compile('def make(fixed):\n def quantize_input(self,x):\n  return fixed[x]\n return quantize_input\n',str(exps/'quantize-packed.py'),'exec'),namespace)
        self.quantize=MethodType(namespace['make'](fixed),self)
      def encode(self,*values):self.encode_calls+=1;return Fake((1,2048,2048),0x3f80)
      def text_layer(self,x,cos,sin,prefix,cache_pos=None):
        index=int(prefix.rsplit('.',1)[1]);self.calls.append(index);assert cache_pos==0
        if fail and index==3:raise RuntimeError('explicit mock factory layer failure')
        return Fake(x.shape,0x3f80+index+1)
      def decode(self,*values):return 'original mock decode'
      def greedy_decode(self,*values):return 'original mock greedy'
      def trajectory(self,*values):return 'original mock trajectory'
    model=Model();model.before={key:getattr(model,key) for key in ('text_layer','decode','greedy_decode','trajectory')};models.append(model)
    class Cache:
      def __init__(self):self.hits=self.misses=0
      def prepare(self,prepared):self.misses+=10
    class Inputs:
      def __init__(self,settings,config,prepared,packed):
        self.batches=[];self.spans=[];self.packed=packed
        actual=prepared['actual_prompt_length'];assert actual==2027 and prepared['prefill_draft_eligible']
        assert packed[0].shape==(1,2048) and packed[1].shape==(3,1,2048)
        np.testing.assert_array_equal(packed[0][0,actual:actual+6],[124549,11135,3174,4286,1561,135084])
        unpadded_delta=int(packed[2])+(2048-actual)
        expected=np.broadcast_to(actual+unpadded_delta+np.arange(6),(3,1,6))
        np.testing.assert_array_equal(packed[1][:,:,actual:actual+6],expected)
        assert not packed[0][:,actual+6:].any() and not packed[1][:,:,actual+6:].any()
      def upload(self,noise):return Fake((1,)),self.packed[3],self.packed[0],Fake((1,1,2048,128)),Fake((1,1,2048,128),0)
    class Devices:
      DEFAULT='EXPLICIT_MOCK_ONLY'
      def __getitem__(self,name):assert name==self.DEFAULT;return SimpleNamespace(arch='gfx1200')
    return SimpleNamespace(model=model,settings=SimpleNamespace(diffusion_steps=3,max_tokens=64,initial_noise=None),
      Device=Devices(),config=c.read(c.BASE/'model/config.json'),vision_cache=Cache(),input_class=Inputs,pad_inputs=padded.pad_inputs,
      report=dict(camera_profile='nvidia',pad_prompt_to=2048,vision_calibration_sha256=None,
        prefill_calibration_sha256=c.sha(args.prefill_calibration),runtime_environment={'mock_factory':True}))
  def hooked_spec(name,location,*values,**options):
    spec=old_spec(name,location,*values,**options)
    if Path(location).resolve()!=exps/'benchmark-pipeline.py':return spec
    loader=spec.loader
    class Loader:
      def create_module(self,value):return loader.create_module(value)
      def exec_module(self,value):
        loader.exec_module(value);value.initialize=lambda args,**kwargs:factory(value,args,**kwargs)
    spec.loader=Loader();return spec
  importlib.util.spec_from_file_location=hooked_spec
  sys.argv=[str(runner),'--plan',str(plan_path),'--output',str(output),'--execute-gpu']
  caught=False
  try:
    try:old_run(str(runner),run_name='__main__')
    except RuntimeError as error:
      assert fail and str(error)=='explicit mock factory layer failure';caught=True
    assert importlib.util.spec_from_file_location is hooked_spec and runpy.run_path is old_run
    assert sys.argv==[str(runner),'--plan',str(plan_path),'--output',str(output),'--execute-gpu']
  finally:
    importlib.util.spec_from_file_location,runpy.run_path,sys.argv=old_spec,old_run,old_argv
    os.environ.clear();os.environ.update(old_env)
  assert len(models)==1 and len(observations)==1
  model=models[0];assert all(getattr(model,key)==value for key,value in model.before.items())
  report=c.read(output);assert report['launcher_hooks_restored'] and report['sealed_targets_opened'] is False
  assert (fail and caught and not report['complete']) or (not fail and report['complete'] and report['prefills']==12)
  if not fail:
    assert model.calls==list(range(16))*12 and model.encode_calls==12 and model.padded_prefill.resets==12
    assert [row['id'] for row in report['completed_examples']]==plan['scene_ids']
    for row in report['completed_examples']:
      assert row['finite_bf16_bits'] and row['layer_order']==list(range(16))
      path=Path(row['capture_path']);assert c.sha(path)==row['capture_sha256'] and path.stat().st_size==26215834
      with np.load(path,allow_pickle=False) as archive:
        assert set(archive.files)=={'x','y','k','v','cos','sin'}
        assert all(archive[key].dtype==np.uint16 and not np.any((archive[key]&0x7f80)==0x7f80) for key in archive.files)
    target_manifest=c.read(report['train_validation_manifest']);assert len(target_manifest['samples'])==12 and not target_manifest['sealed_test_targets_included']
    assert report['sealed_manifest'] is None
  from tinygrad import Device
  assert not Device._opened_devices;c.assert_no_devices()
  # Never let deliberately synthetic reports masquerade as hardware evidence.
  report.update(gpu_executed=False,mock_gpu_factory=True,teacher_inference_executed=False,synthetic_mock_payloads=True)
  output.write_text(json.dumps(report,indent=2)+'\n')
  result=dict(passed=True,job=job,complete_actual_runpy_entrypoint=True,real_CPU_preparation=True,real_canonical_launcher_parser_hooks=True,
    real_2048_padding_and_mrope=True,full_twelve_example_serialization=not fail,explicit_mock_factory_failure_control=fail,
    original_non_target_methods_restored=True,global_hooks_restored=True,actual_gpu_device_opens=0,teacher_inference_executed=False,
    mock_boundary='Only initialize/model/input-upload/device interfaces mocked. Real entrypoint, preflight, inputs, launcher, parser, padding, capture loop, and disk serialization execute.',
    observations=observations)
  c.write(root/'agent-distill-cohort-mock-check.json',result);print(json.dumps(result,indent=2))

def main():
  parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--child-job',type=int,choices=(0,1));parser.add_argument('--failure-control',action='store_true')
  parser.add_argument('--output-dir',type=Path,default=c.ROOT/'cpu-controls-v1');args=parser.parse_args()
  sys.addaudithook(c.cpu_guard);root=args.output_dir.resolve();assert root.is_relative_to(c.ROOT) and not root.exists();c.storage(root.parent,800*1024**2);root.mkdir()
  if args.child_job is not None:return mock_job(args.child_job,root,args.failure_control)
  manifest=c.validate_rows(c.read(c.INPUTS))
  adapter=c.module('agent_distill_cohort_exact_boundary_audit',c.ROOT/'capture-v1/frozen/agent-distill-cohort-capture.py')
  assert adapter.capture_boundary is adapter.original.capture_boundary and adapter.find_quantizer is adapter.original.find_quantizer
  def model_calls(text):
    return [ast.dump(node,include_attributes=False) for node in ast.walk(ast.parse(text)) if isinstance(node,ast.Call)
      and isinstance(node.func,ast.Attribute) and isinstance(node.func.value,ast.Name) and node.func.value.id=='model']
  import inspect
  assert model_calls(adapter.capture_source())==model_calls(inspect.getsource(adapter.original.execute))
  import numpy as np
  import copy
  negatives=[]
  for name,change in [
    ('sealed drive',lambda row:row.update(route_id=c.EXCLUDED[1])),
    ('protected route',lambda row:row.update(route_id=c.PROTECTED)),
    ('cross-drive split',lambda row:row.update(drive_id='wrong-physical-drive')),
    ('changed experimental command',lambda row:row['conditioning'].update(text='Turn left.')),
    ('changed input checksum',lambda row:row['input_file_sha256'].update(**{'image_frames.npy':'0'*64})),
    ('renamed input directory',lambda row:row.update(sample=str(c.ROOT/'inputs-v1/samples/renamed'))),
    ('invented recorded intent',lambda row:row['conditioning'].update(recorded_driver_intent=True))]:
    changed=copy.deepcopy(manifest);change(changed['samples'][0])
    try:c.validate_rows(changed)
    except (AssertionError,ValueError,KeyError):negatives.append(name)
    else:raise AssertionError('Negative control accepted: '+name)
  for path,digest in c.coverage()['source_sha256'].items():assert c.sha(path)==digest,path
  for path,digest in manifest['source_sha256'].items():assert c.sha(path)==digest,path
  for row in manifest['native_frames']:
    path=Path(row['file']);assert c.sha(path)==row['file_sha256']
    value=np.load(path,mmap_mode='r',allow_pickle=False);assert value.shape==(3,760,1344) and value.dtype==np.uint8
  for row in manifest['samples']:
    sample=Path(row['sample']);images=np.load(sample/'image_frames.npy',mmap_mode='r',allow_pickle=False)
    assert images.shape==(10,1,3,432,768) and images.dtype==np.uint8
    times=np.load(sample/'absolute_timestamps.npy',allow_pickle=False);assert np.all(times<=row['query_ns']//1000)
    assert all(np.all(np.diff(times[start:start+5])>0) for start in (0,5))
    for name,shape in [('ego_history_xyz',(1,20,3)),('ego_history_rot',(1,20,3,3))]:
      value=np.load(sample/(name+'.npy'),allow_pickle=False);assert value.shape==shape and np.isfinite(value).all()
  results=[]
  for job,fail in ((0,False),(1,False),(0,True)):
    child=root/('mock-job-'+str(job)+('-failure' if fail else ''))
    command=[str(c.BASE/'upstream/.venv/bin/python'),'-B',str(Path(__file__).resolve()),'--child-job',str(job),'--output-dir',str(child)]
    if fail:command.append('--failure-control')
    with (root/(child.name+'.log')).open('x') as stream:subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT,check=True,timeout=300)
    results.append(c.read(child/'agent-distill-cohort-mock-check.json'))
  result=dict(complete=True,all_passed=True,actual_inputs=24,native_RGB_frames=240,full_entrypoint_mock_jobs=results,
    gpu_executed=False,training_executed=False,teacher_inference_executed=False,sealed_drive_opened=False,protected_route_opened=False,
    original_coverage_source_hashes_rechecked=87,native_RGB_file_hashes_rechecked=240,negative_controls_rejected=negatives,
    qualified_boundary_and_quantizer_finder_reused_exactly=True,original_model_call_ASTs_unchanged=True,
    prepared_manifest_sha256=c.sha(c.INPUTS),helper_sha256=c.sha(Path(__file__)))
  c.assert_no_devices();c.write(root/'agent-distill-cohort-check.json',result);print(json.dumps(result,indent=2))

if __name__=='__main__':main()
