"""Explicit capture reuse for the failed residual/norm microcheck. Plan mode is CPU-only."""
import argparse
import ast
import copy
from contextlib import contextmanager
import hashlib
import importlib.util
import inspect
import json
import os
from pathlib import Path
import sys
import textwrap
import zipfile

HERE=Path(__file__).resolve().parent
FAILED=Path('/mnt/nvme-500/tmp/alpamayo-speed-20261116/agent-residual-norm-gpu')
TINY=Path('/tmp/opencode/alpamayo-asius-20261114/tinygrad')
PINS={'report.json':'22c9d54da9c71a5ca0a3132036a971952765b6d31c16422f130aa26d886aeef2',
 'report.agent-residual-norm-plan.json':'26ccf958fdf044491612fe135a92bb883f61890cf09fbb435bd4670e83a4ccce',
 'report.agent-residual-norm-inputs.npz':'063b140e70cf5e6e96eeefbd01e354d9789bc490912d2f1016b5f3cfa619ec87',
 'invocation.json':'7915dd8cae1b5cdc78a5dfee442e1c460bdfb3b0aa55d93bc9e92aba4ef24ab2',
 'wall.json':'8092b96813a6cc6e8c6bc611abd9ef300b903605096e4f1f2eb3c55808b4188d',
 'run.log':'42e9d694432e3b4fc2f1a572af73b99d46d78fe3feb3a3ce10929d62650ecccb'}
JIT_SHA='bdf5f3029e9e98faaddb0e83a51296189e7cf6ebf845d84fb46b8414f2a9424b'
RETRY_FILES=('agent-residual-norm-retry.py','agent-residual-norm-retry-buffers.py','agent-residual-norm-retry-check.py')


def sha(data):return hashlib.sha256(data).hexdigest()
def write(path,value):path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
def load(path):
  spec=importlib.util.spec_from_file_location(Path(path).stem.replace('-','_')+'_retry',path)
  m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


def verify_capture(trial=FAILED):
  import numpy as np
  for name,expected in PINS.items():
    if sha((trial/name).read_bytes())!=expected:raise ValueError('Bound failed evidence changed: '+name)
  report=json.loads((trial/'report.json').read_text());plan=json.loads((trial/'report.agent-residual-norm-plan.json').read_text())
  assert report['complete'] is False and report['gates']==[] and not report.get('timings')
  assert report['failure']==dict(type='JitError',message='JIT inputs must be real buffers; use .clone()')
  assert report['plan_sha256']==PINS['report.agent-residual-norm-plan.json']
  assert report['capture_context']==dict(vision_hits=6,vision_misses=4)
  assert plan['shape']==[1,2048,2048] and plan['layers']==[0,13,27] and plan['batch_calls']==16
  assert '--vision-calibration' not in plan['command'] and '--llvm-wgp' in plan['command']
  bundle=Path(plan['core_root']);manifest_path=bundle/'agent-residual-norm-manifest.json'
  assert sha(manifest_path.read_bytes())==plan['source_bundle_manifest_sha256']
  for path,expected in plan['original_core_source_sha256'].items():
    assert sha((bundle/Path(path).name).read_bytes())==expected
  for name,expected in plan['original_frozen_source_sha256'].items():
    assert sha((bundle/name if name=='model.py' else bundle/'experiments'/name).read_bytes())==expected
  capture_source_sha256={name:sha((trial/'report.capture-sources'/name).read_bytes()) for name in plan['original_frozen_source_sha256']}
  assert capture_source_sha256==plan['original_frozen_source_sha256']
  capture_invocation=json.loads((trial/'report.agent-residual-norm-invocation.json').read_text())
  assert capture_invocation==dict(command=plan['command'],environment=plan['environment'])
  for path,expected in plan['owned_source_sha256'].items():assert sha(Path(path).read_bytes())==expected
  assert sha((TINY/'tinygrad/engine/jit.py').read_bytes())==JIT_SHA
  for path,expected in plan['tinygrad']['source_sha256'].items():assert sha(Path(path).read_bytes())==expected
  # Retain every original input-artifact check, including capture-only provenance.
  for path,expected in plan['input_artifact_sha256'].items():assert sha(Path(path).read_bytes())==expected
  # Current BF16 checkpoint and the original language calibration remain independent capture anchors.
  calibration=next(Path(path) for path in plan['input_artifact_sha256'] if path.endswith('report.prefill-scales.json'))
  checkpoint=Path(plan['command'][plan['command'].index('--checkpoint')+1])
  for path in (calibration,checkpoint/'config.json',checkpoint/'model.safetensors.index.json'):
    assert sha(path.read_bytes())==plan['input_artifact_sha256'][str(path)]
  scales=json.loads(calibration.read_text())['scales']
  old=load(bundle/'agents/agent-residual-norm-run.py')
  kernel=load(bundle/'agents/agent-residual-norm-kernel.py')
  assert sha(kernel.source().encode())==report['compiled']['candidate_source_sha256']
  with zipfile.ZipFile(trial/'report.agent-residual-norm-inputs.npz') as archive:
    assert sum(info.file_size for info in archive.infolist())<128*1024*1024+16384
  samples={};actual_keys=set()
  with np.load(trial/'report.agent-residual-norm-inputs.npz',allow_pickle=False) as archive:
    for layer in (0,13,27):
      prefix='vlm.model.language_model.layers.'+str(layer);sample={}
      for key,expected in report['captured_sha256'][prefix].items():
        value=archive[prefix+'/'+key].copy();actual_keys.add(prefix+'/'+key)
        assert sha(value.tobytes())==expected
        shape=(2048,) if key=='weight' else () if key=='scale' else (2048,2048) if key=='packed' else (1,2048,2048)
        dtype=np.float32 if key=='scale' else np.uint8 if key=='packed' else np.uint16
        assert value.shape==shape and value.dtype==np.dtype(dtype)
        if key=='packed':assert np.all((value&127)!=127)
        elif key=='scale':assert np.isfinite(value).all() and float(value)>0
        else:assert np.all((value&0x7f80)!=0x7f80)
        sample[key]=value
      assert sample['weight'].tobytes()==old.checkpoint_norm_bytes(prefix+'.post_attention_layernorm.weight')
      assert sample['scale'].tobytes()==np.float32(scales[prefix+'/2']).tobytes()
      samples[prefix]=sample
    assert set(archive.files)==actual_keys and len(actual_keys)==21
  binding=dict(schema=1,reuse_only=True,recapture=False,model_load=False,model_warmups=0,
    original_failed_trial=str(trial),old_trial_status='failed before any micro gate; remains failed',
    failed_evidence_sha256=PINS,source_bundle=str(bundle),source_bundle_manifest_sha256=plan['source_bundle_manifest_sha256'],
    captured_sha256=report['captured_sha256'],capture_context=report['capture_context'],compiled=report['compiled'],
    target_query=plan['target_query'],numeric_kernel_file_sha256=sha((bundle/'agents/agent-residual-norm-kernel.py').read_bytes()),
    cpu_checkpoint_and_scale_bytes_reverified=True,array_count=21,array_bytes=sum(v.nbytes for s in samples.values() for v in s.values()),
    original_input_artifact_sha256=plan['input_artifact_sha256'],all_original_input_artifacts_reverified=True,
    actual_capture_source_sha256=capture_source_sha256,original_capture_invocation_reverified=True,
    original_batch_calls=16,parent_timeout_seconds=300)
  return binding,samples,plan,old


def adapted_micro(old,buffers,binding,records):
  """Three checked structural changes. Every original arithmetic/finite/queue gate remains."""
  source=textwrap.dedent(inspect.getsource(old.gpu_micro));tree=ast.parse(source)
  original=copy.deepcopy(tree);fn=tree.body[0]
  imported=next(n for n in fn.body if isinstance(n,ast.ImportFrom) and n.module=='tinygrad')
  alias=next(n for n in imported.names if n.name=='TinyJit');assert alias.asname is None
  alias.asname='_OriginalTinyJit'
  assignment=ast.parse('TinyJit = _retry_jit_factory(_OriginalTinyJit, result)').body[0]
  fn.body.insert(fn.body.index(imported)+1,assignment)
  constructor=next(n for n in fn.body if isinstance(n,ast.FunctionDef) and n.name=='tensors')
  original_constructor=copy.deepcopy(constructor.body)
  assert len(constructor.body)==1 and isinstance(constructor.body[0],ast.Return)
  constructor.body=ast.parse('return _retry_materialize(sample, dev.device)').body
  # Poison is diagnostic. Only the original clocked invocation bypasses poisoning.
  clock_loop=next(n for n in ast.walk(fn) if isinstance(n,ast.For) and isinstance(n.target,ast.Name) and n.target.id=='round_')
  timed=[n for n in ast.walk(clock_loop) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='jit']
  assert len(timed)==1
  timed[0].func=ast.Attribute(value=ast.Name(id='jit',ctx=ast.Load()),attr='run_without_poison',ctx=ast.Load())
  normalized=copy.deepcopy(tree);nf=normalized.body[0]
  ni=next(n for n in nf.body if isinstance(n,ast.ImportFrom) and n.module=='tinygrad')
  next(n for n in ni.names if n.name=='TinyJit').asname=None
  nf.body=[n for n in nf.body if not (isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name) and n.targets[0].id=='TinyJit')]
  next(n for n in nf.body if isinstance(n,ast.FunctionDef) and n.name=='tensors').body=original_constructor
  for n in ast.walk(normalized):
    if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='run_without_poison':n.func=ast.Name(id='jit',ctx=ast.Load())
  assert ast.dump(normalized,include_attributes=False)==ast.dump(original,include_attributes=False)
  assertions=lambda t:[ast.dump(n,include_attributes=False) for n in ast.walk(t) if isinstance(n,(ast.Assert,ast.Raise))]
  assert assertions(tree)==assertions(original)
  checked_compiled=binding['compiled'];original_load=old.load
  def checked_load(path):
    module=original_load(path)
    name=Path(path).name
    if name in ('agent-residual-norm-kernel.py','fused-norm.py'):
      compile_original=module.compiled
      prefix='candidate' if name.startswith('agent-') else 'frozen_norm'
      def compile_checked(*args,**kwargs):
        text,binary=compile_original(*args,**kwargs)
        assert sha(text.encode())==checked_compiled[prefix+'_source_sha256']
        assert sha(binary)==checked_compiled[prefix+'_binary_sha256']
        return text,binary
      module.compiled=compile_checked
    return module
  namespace=dict(old.__dict__,FROZEN=Path(binding['source_bundle'])/'experiments',
    HERE=Path(binding['source_bundle'])/'agents',load=checked_load,
    _retry_materialize=lambda sample,device:buffers.materialize(sample,device,records),_retry_jit_factory=buffers.jit_factory)
  ast.fix_missing_locations(tree)
  exec(compile(tree,str(HERE/'agent-residual-norm-retry.py')+'[adapted frozen micro]','exec'),namespace)
  proof=dict(original_micro_ast_sha256=sha(ast.dump(original,include_attributes=False).encode()),
    retry_micro_ast_sha256=sha(ast.dump(tree,include_attributes=False).encode()),
    reverse_transform_is_identical=True,all_original_assertions_and_raises_identical=True,
    modifications=['Materialize and verify four distinct physical input buffers.',
      'Wrap JITs with verified output poisoning outside timed calls.',
      'Delegate the single original clocked call through a poison-free method.'])
  return namespace['gpu_micro'],proof


@contextmanager
def reference_runtime(bundle):
  from tinygrad import Device
  from tinygrad.runtime import ops_amd
  device=Device[Device.DEFAULT];assert device.arch=='gfx1200'
  compiler,dispatch=device.renderer.compiler,ops_amd.dispatch_packet
  try:
    load(bundle/'experiments/wgp_compiler.py').install(device)
    load(bundle/'experiments/prefill-kernels.py').fix_dispatch_packet()
    yield
  finally:device.renderer.compiler=compiler;ops_amd.dispatch_packet=dispatch


def main():
  parser=argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--output',type=Path,required=True)
  parser.add_argument('--mode',choices=('plan','run'),default='plan')
  parser.add_argument('--failed-trial',type=Path,default=FAILED)
  parser.add_argument('--freeze-to',type=Path)
  args=parser.parse_args();output=args.output.resolve()
  if output.exists():raise FileExistsError(output)
  binding,samples,old_plan,old=verify_capture(args.failed_trial)
  buffers=load(HERE/'agent-residual-norm-retry-buffers.py');records=[]
  micro,adaptation=adapted_micro(old,buffers,binding,records)
  binding['ast_adaptation']=adaptation
  binding['retry_source_sha256']={name:sha((HERE/name).read_bytes()) for name in RETRY_FILES}
  manifest_path=HERE/'agent-residual-norm-retry-manifest.json'
  if manifest_path.exists():
    manifest=json.loads(manifest_path.read_text())
    assert manifest['source_sha256']==binding['retry_source_sha256'] and manifest['failed_evidence_sha256']==PINS
    assert manifest['numeric_kernel_file_sha256']==binding['numeric_kernel_file_sha256']
    binding['retry_manifest_sha256']=sha(manifest_path.read_bytes())
  elif args.mode=='run':raise ValueError('Run the frozen retry bundle; source manifest is required')
  if args.freeze_to:
    if args.mode!='plan':raise ValueError('Source freezing is CPU plan mode only')
    destination=args.freeze_to.resolve();destination.mkdir(parents=True,exist_ok=False)
    for name in RETRY_FILES:(destination/name).write_bytes((HERE/name).read_bytes())
    write(destination/'agent-residual-norm-retry-manifest.json',dict(schema=1,
      source_sha256=binding['retry_source_sha256'],failed_evidence_sha256=PINS,
      numeric_kernel_file_sha256=binding['numeric_kernel_file_sha256']))
    binding['created_frozen_retry_source']=str(destination)
  binding['original_environment']=old_plan['environment']
  output.parent.mkdir(parents=True,exist_ok=True)
  binding_path=output.with_suffix('.agent-residual-norm-reuse.json');write(binding_path,binding)
  if args.mode=='plan':print(json.dumps(dict(mode='plan',reuse_binding=str(binding_path),model_load=False,model_warmups=0)));return
  if any(name=='tinygrad' or name.startswith('tinygrad.') for name in sys.modules):raise RuntimeError('Fresh process required')
  old_env=dict(os.environ);old_path=sys.path[:]
  result=dict(schema=2,gpu_executed=False,complete=False,diagnostic_only=True,timings_valid_for_model_throughput=False,
    reuse_binding_sha256=sha(binding_path.read_bytes()),old_trial_status=binding['old_trial_status'],
    original_failed_report_sha256=PINS['report.json'],materialization_checks=records,
    model_loaded=False,model_warmups=0)
  write(output,result)
  try:
    for key,value in old_plan['environment'].items():
      if value is None:os.environ.pop(key,None)
      else:os.environ[key]=value
    os.environ.update(PROFILE='0',CACHEDB=str(output.with_suffix('.agent-residual-norm-retry-cache.db')))
    bundle=Path(binding['source_bundle'])
    sys.path[:0]=[str(TINY),str(bundle),str(bundle/'experiments')]
    with reference_runtime(bundle):
      result['gpu_executed']=True
      micro(samples,result,output)
    assert len(result['gates'])==18 and len(result['timings'])==3 and len(result['poison_checks'])==12
    assert all(s['poisoned_replays']>0 for s in result['poison_checks'])
    result['complete']=True;write(output,result);print(json.dumps(result,indent=2))
  except Exception as error:
    result['failure']=dict(type=type(error).__name__,message=str(error));write(output,result);raise
  finally:sys.path[:]=old_path;os.environ.clear();os.environ.update(old_env)


if __name__=='__main__':main()
