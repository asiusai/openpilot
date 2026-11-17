"""Reuse the frozen residual microcheck with a strict resolved-alias HCQ proof. Plan mode is CPU-only."""
import argparse
import ast
import builtins
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
EXP=Path('/tmp/opencode/openpilot-alpamayo/openpilot/tools/alpamayo/experiments')
OLD=EXP/'agent-residual-norm-retry-source-final'
TRIAL=Path('/mnt/nvme-500/tmp/alpamayo-speed-20261116/agent-residual-norm-retry-gpu')
OLD_MANIFEST_SHA='8eadda73f9e81c74338f4508155be173d7a4a692a5cf463133925793d9839c81'
PRIOR={'report.json':'86f8d7fbb38f1f5971b82f6c4e7edd25ccd04849cb2330c54e0f2886999d5245',
 'run.log':'c4def76bfef96a2e8a49875d768ba5e7fbcb15bd60a21eec38eebbc7ffe191c1',
 'invocation.json':'682791dcb5c85f5d9955d0f3f0accc313e07a977c35286e8d275cc530acf83a2',
 'wall.json':'1bd099a04b25bb43cd0b2c701667561810a427ae6c68c99b87f897a8385de630',
 'report.agent-residual-norm-reuse.json':'82df784f42b4b7ed5b4ec9dbabd1c7000ff88bfc087a78306c1a0030db9af6ac'}
FILES=('agent-residual-norm-queue-run.py','agent-residual-norm-queue-proof.py',
       'agent-residual-norm-queue-capture.py','agent-residual-norm-queue-check.py')


def sha(data):return hashlib.sha256(data).hexdigest()
def write(path,value):path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
def load(path):
  spec=importlib.util.spec_from_file_location(Path(path).stem.replace('-','_')+'_launch',path)
  module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def dependencies():
  manifest=OLD/'agent-residual-norm-retry-manifest.json'
  assert sha(manifest.read_bytes())==OLD_MANIFEST_SHA
  for name,expected in json.loads(manifest.read_text())['source_sha256'].items():assert sha((OLD/name).read_bytes())==expected
  for name,expected in PRIOR.items():assert sha((TRIAL/name).read_bytes())==expected
  report=json.loads((TRIAL/'report.json').read_text())
  assert report['complete'] is False and report['failure']==dict(type='AssertionError',message='') and not report.get('timings')
  assert [gate['case'] for gate in report['gates']]==['current','a','b','weight','scale','zeros-ties-outliers']
  assert all(gate['prefix']=='vlm.model.language_model.layers.0' and gate['all_bf16_residual_bits'] and gate['all_fp8_packed_bytes']
             for gate in report['gates'])
  return load(OLD/'agent-residual-norm-retry.py'),load(OLD/'agent-residual-norm-retry-buffers.py')


def build_micro(retry,buffers,old,binding,records,queue,result):
  """Add actual inputs to the observer call only; retain the frozen retry's other transformations."""
  delta=[]
  assert 'compile' not in retry.__dict__
  def checked_compile(tree,filename,mode,*args,**kwargs):
    before=copy.deepcopy(tree)
    calls=[n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='observe']
    assert len(calls)==1 and len(calls[0].args)==2 and not calls[0].keywords
    assert [arg.id for arg in calls[0].args]==['proofs','label']
    calls[0].args.append(ast.Name(id='inputs',ctx=ast.Load()))
    ast.fix_missing_locations(tree)
    restored=copy.deepcopy(tree)
    next(n for n in ast.walk(restored) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='observe').args.pop()
    assert ast.dump(restored,include_attributes=False)==ast.dump(before,include_attributes=False)
    asserts=lambda t:[ast.dump(n,include_attributes=False) for n in ast.walk(t) if isinstance(n,(ast.Assert,ast.Raise))]
    assert asserts(tree)==asserts(before)
    delta.append(dict(before_ast_sha256=sha(ast.dump(before,include_attributes=False).encode()),
      after_ast_sha256=sha(ast.dump(tree,include_attributes=False).encode()),only_observer_inputs_argument_changed=True,
      reverse_transform_is_identical=True,all_original_assertions_and_raises_identical=True))
    assert delta[0]['before_ast_sha256']=='358fc2a2aecc3921ccf373ed4698400a1dec27baedf2c75c4305b5233d69a239'
    return builtins.compile(tree,filename,mode,*args,**kwargs)
  retry.compile=checked_compile
  try:micro,adaptation=retry.adapted_micro(old,buffers,binding,records)
  finally:del retry.compile
  assert len(delta)==1
  micro.__globals__['observe']=lambda proofs,label,inputs:queue.observe(proofs,label,inputs,result.setdefault('queue_proofs',[]))
  return micro,dict(frozen_retry_adaptation=adaptation,queue_observer_delta=delta[0],
    changed_guard='Resolve PARAM inputs, retained BUFFERs and ALLOC arena views to exact byte spans; require all RAW/WAR/WAW edges.')


def main():
  parser=argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--output',type=Path,required=True)
  parser.add_argument('--mode',choices=('plan','run'),default='plan')
  parser.add_argument('--freeze-to',type=Path)
  args=parser.parse_args();output=args.output.resolve()
  if output.exists():raise FileExistsError(output)
  retry,buffers=dependencies()
  binding,samples,plan,old=retry.verify_capture()
  queue=load(HERE/'agent-residual-norm-queue-proof.py');records=[];result={}
  micro,adaptation=build_micro(retry,buffers,old,binding,records,queue,result)
  sources={name:sha((HERE/name).read_bytes()) for name in FILES}
  manifest_path=HERE/'agent-residual-norm-queue-manifest.json'
  audit_path=HERE/'agent-residual-norm-queue-cpu-proof.json' if manifest_path.exists() else EXP/'agent-residual-norm-queue-cpu/agent-residual-norm-queue-check.json'
  audit=json.loads(audit_path.read_text())
  assert audit['source_sha256']==sources
  assert audit['gpu_executed'] is False and audit['numeric_execution'] is False
  assert audit['actual_tinyjit_capture'] and audit['actual_hcq_queue_builder'] and audit['strict_observe_hook_tested']
  assert len(audit['negative_controls'])==71 and all(row['rejected'] for row in audit['negative_controls'])
  assert [r['kernel_calls'] for r in audit['records']]==[48,16] and audit['hook_restored_after_failure']
  assert audit['compiled']==binding['compiled']
  for name,expected in audit['tinygrad_source_sha256'].items():assert sha((retry.TINY/name).read_bytes())==expected
  assert audit['launcher_adapter']==adaptation
  manifest=dict(schema=1,source_sha256=sources,cpu_audit_sha256=sha(audit_path.read_bytes()),
    prior_failed_retry_sha256=PRIOR,original_retry_manifest_sha256=OLD_MANIFEST_SHA,
    numeric_kernel_file_sha256=binding['numeric_kernel_file_sha256'])
  if manifest_path.exists():assert json.loads(manifest_path.read_text())==manifest
  elif args.mode=='run':raise ValueError('Use the frozen queue retry bundle with its actual graph audit')
  if args.freeze_to:
    assert args.mode=='plan'
    destination=args.freeze_to.resolve();destination.mkdir(parents=True,exist_ok=False)
    for name in FILES:(destination/name).write_bytes((HERE/name).read_bytes())
    (destination/'agent-residual-norm-queue-cpu-proof.json').write_bytes(audit_path.read_bytes())
    write(destination/'agent-residual-norm-queue-manifest.json',manifest)
    binding['created_frozen_queue_source']=str(destination)
  binding.update(queue_adapter=adaptation,queue_source_sha256=sources,queue_cpu_audit_sha256=manifest['cpu_audit_sha256'],
    prior_failed_retry_sha256=PRIOR,prior_retry_status='failed HCQ guard after first-layer six exact cases; no timing',
    original_retry_manifest_sha256=OLD_MANIFEST_SHA,model_load=False,model_warmups=0)
  output.parent.mkdir(parents=True,exist_ok=True)
  binding_path=output.with_suffix('.agent-residual-norm-queue-reuse.json');write(binding_path,binding)
  if args.mode=='plan':print(json.dumps(dict(mode='plan',gpu_executed=False,reuse_binding=str(binding_path))));return
  if any(name=='tinygrad' or name.startswith('tinygrad.') for name in sys.modules):raise RuntimeError('Fresh process required')
  environment=dict(os.environ);path=sys.path[:]
  result.update(schema=3,gpu_executed=False,complete=False,diagnostic_only=True,timings_valid_for_model_throughput=False,
    reuse_binding_sha256=sha(binding_path.read_bytes()),prior_retry_status=binding['prior_retry_status'],
    model_loaded=False,model_warmups=0,materialization_checks=records)
  write(output,result)
  try:
    for key,value in plan['environment'].items():
      if value is None:os.environ.pop(key,None)
      else:os.environ[key]=value
    os.environ.update(PROFILE='0',CACHEDB=str(output.with_suffix('.agent-residual-norm-queue-cache.db')))
    bundle=Path(binding['source_bundle']);sys.path[:0]=[str(retry.TINY),str(bundle),str(bundle/'experiments')]
    with retry.reference_runtime(bundle):
      result['gpu_executed']=True;micro(samples,result,output)
    assert len(result['gates'])==18 and len(result['timings'])==3 and len(result['poison_checks'])==12 and len(result['queue_proofs'])==6
    assert all(row['poisoned_replays']>0 for row in result['poison_checks'])
    result['complete']=True;write(output,result);print(json.dumps(result,indent=2))
  except Exception as error:
    result['failure']=dict(type=type(error).__name__,message=str(error));write(output,result);raise
  finally:sys.path[:]=path;os.environ.clear();os.environ.update(environment)


if __name__=='__main__':main()
