"""Exact comparison projection and independent admission of a completed failed audit."""
import copy
import hashlib
import importlib.util
import inspect
import json
import math
from pathlib import Path
from unittest.mock import patch

WORK=Path('/mnt/nvme-500/tmp/alpamayo-speed-20261116')
FROZEN=WORK/'agent-expert-live-storage-source'
BASE=WORK/'agent-expert-live-storage-protocol.json'
BASE_SHA='3702770f1b1f11139f96ec4d1aac4be93373dddd7251e9ccbed5cc6e539422cc'
CURRENT=WORK/'agent-expert-live-storage-current/report.json'
CURRENT_SHA='40a8210c500ee8b6b82f3b108bb61071c0e263806f1032ab8b619a880e34d4f2'
ROOT=WORK/'agent-expert-live-compare-source-v2'
ARTIFACTS=WORK/'agent-expert-live-compare-artifacts'
PROTOCOL=WORK/'agent-expert-live-compare-protocol-v2.json'
ADMISSION=WORK/'agent-expert-live-compare-current-admission-v2.json'
CANDIDATE=WORK/'agent-expert-live-compare-candidate/report.json'


def load(path,name):
  spec=importlib.util.spec_from_file_location(name,path)
  module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


S=load(FROZEN/'agent-expert-live-storage-common.py','compare_original_storage_common')
N=S.N
sha,read,write=N.sha,N.read,N.write
assert sha(BASE)==BASE_SHA
assert sha(FROZEN/'agent-expert-live-storage-common.py')==read(BASE)['files'][str(FROZEN/'agent-expert-live-storage-common.py')]


def configure():
  N.ARTIFACTS=ARTIFACTS
  proof=N.configure()
  target=ARTIFACTS/'hooks/agent-exact-combined-install.py'
  assert sha(target)==read(N.MANIFEST)['files'][str(N.EXP/target.name)]
  N.paths();return proof


def snapshot(report,report_path):
  report_path=Path(report_path)
  recorded=report['consumer_provenance']['initializer']['factory']['source_snapshot']
  assert type(recorded) is str
  folder=Path(recorded)
  expected=report_path.parent/(report_path.stem+'.benchmark-sources')
  assert folder.is_absolute() and folder==expected and folder==folder.resolve() and folder.is_relative_to(WORK)
  assert folder.is_dir() and not folder.is_symlink()
  sources=report['source_sha256']
  assert sources==report['initialized']['source_sha256'] and len(sources)==253
  entries=list(folder.iterdir())
  assert len(entries)==253 and {p.name for p in entries}==set(sources),'Snapshot must contain exactly all 253 recorded files'
  for path in entries:
    assert path.is_file() and not path.is_symlink() and path.resolve().parent==folder
    assert type(sources[path.name]) is str and sha(path)==sources[path.name],str(path)
  identity={'complete_file_sha256':dict(sorted(sources.items()))}
  return identity


def project(report,report_path):
  identity=snapshot(report,report_path)
  result=copy.deepcopy(report['consumer_provenance']);factory=result['initializer']['factory']
  elapsed=factory['model_load_ms']
  assert type(elapsed) in (int,float) and math.isfinite(elapsed) and elapsed>=0,'Invalid model_load_ms'
  del factory['model_load_ms']
  factory['source_snapshot']=identity
  return result


def projected_qualify(module,qualifier,report,data,digest,arm,report_path):
  m,_=module.N.live();old=read(module.N.CONSUMER)
  a,b=project(report,report_path),project(old,module.N.CONSUMER)
  equal=m.equal;seen=[]
  def compare(left,right,path=''):
    if path!='consumer_owner_and_code':return equal(left,right,path)
    equal(left,report['consumer_provenance'],'projection_original_left')
    equal(right,old['consumer_provenance'],'projection_original_right')
    seen.append(path);return equal(a,b,path)
  with patch.object(m,'equal',compare):result=qualifier(report,data,digest,arm)
  assert seen==['consumer_owner_and_code']
  return result


def coverage(status):
  c=status['coverage'];s=c['language_stats']
  required={f'vlm.model.language_model.layers.{i}' for i in range(28)}
  assert set(s['layer_hits'])==required and len(set(s['layer_hits'].values()))==1 and min(s['layer_hits'].values())>0
  assert s['python_trace_fc1']==s['python_trace_fc2']==s['python_trace_fused_quantize']==s['layer_completions']>0
  assert s['failed_text_layers']==0 and set(s['shape_hits'])=={'2048'} and s['methods_restored']
  assert any(x['fused_fc1']==28 for x in c['language_graphs'])
  traces=c['vision_hook']['trace_calls']
  for batch in (1,2):
    names={f'vlm.model.visual.encoder.layers.{i}.mlp.fc{fc}/batch{batch}' for i in range(2,27) for fc in (1,2)}
    assert names<={k for k,v in traces.items() if v>0}
  assert len(c['vision_graphs'])==2 and all(x['fc1']==x['fc2']==25 for x in c['vision_graphs'])
  assert status['quantizer_idle'] and status['constructor_restored'] and status['hooks_restored']
  assert status['composition_restorations'] and all(x['restored'] for x in status['composition_restorations'])
  return dict(mlp_layers=28,vision_projections_per_batch=50,vision_batches=[1,2],quantizer_idle=True,cleanup_exact=True)


def recover_current():
  assert sha(CURRENT)==CURRENT_SHA
  original=read(CURRENT);status=original['agent_expert_live_storage']
  assert status==read(CURRENT.with_suffix('.agent-expert-live-storage.json'))
  expected={'type':'ValueError','message':'Float bits differ: consumer_owner_and_code.initializer.factory.model_load_ms'}
  assert status['failure']==expected and not status['passed'] and status['gpu_executed']
  wall=read(CURRENT.parent/'wall.json')
  assert wall['returncode']==1 and not wall['timed_out'] and wall['timeout_seconds']==1200
  log=(CURRENT.parent/'run.log').read_text()
  assert log.count('Traceback (most recent call last):')==1 and log.rstrip().endswith('ValueError: '+expected['message'])
  assert 'status[\'comparison\']=S.qualify(actual,data,args.protocol_sha,args.arm)' in log
  data=read(BASE);assert status['protocol_sha256']==BASE_SHA
  # Keep storage validation bound to the actual completed arm's recorded paths.
  recovery=load(FROZEN/'agent-expert-live-storage-common.py','compare_current_storage_validator')
  view=copy.deepcopy(original);view['agent_expert_live_storage']['passed']=True
  try:recovery.qualify(view,data,BASE_SHA,'current')
  except ValueError as error:assert str(error)==expected['message']
  else:raise AssertionError('Original sole postrun failure was not reproduced')
  result=projected_qualify(recovery,recovery.qualify,view,data,BASE_SHA,'current',CURRENT)
  counted=coverage(status)
  assert sha(CURRENT)==CURRENT_SHA and original['agent_expert_live_storage']['passed'] is False
  return dict(admission_succeeded=True,gpu_executed_during_admission=False,current_report_sha256=CURRENT_SHA,
    original_passed=False,original_failure=expected,comparison=result,coverage=counted,
    source_snapshot_identity=snapshot(original,CURRENT),saved_consumer_snapshot_identity=snapshot(read(N.CONSUMER),N.CONSUMER),
    temporary_role_view={'field':'agent_expert_live_storage.passed','before':False,'after':True,
      'scope':'In-memory only for unchanged qualifier; original failure and report remain unchanged',
      'justification':'All recorded stages independently checked; sole original failure reproduced; exact two-field metadata projection passes'},
    projection={'removed':['consumer_provenance.initializer.factory.model_load_ms'],
      'replaced':['consumer_provenance.initializer.factory.source_snapshot'],
      'replacement':'Verified complete 253-file name/SHA256 identity','all_other_fields':'Exact, including unknown keys'})


def runtime_module():
  R=load(FROZEN/'agent-expert-live-storage-run.py','compare_frozen_runtime')
  runtime=R.S;runtime.ARTIFACTS=ARTIFACTS;runtime.N.ARTIFACTS=ARTIFACTS
  original_qualify=runtime.qualify
  runtime.verify=verify
  runtime.qualify=lambda report,data,digest,arm:projected_qualify(runtime,original_qualify,report,data,digest,arm,CANDIDATE)
  runtime.paired=lambda current,candidate,data,digest:paired(candidate,data,digest)
  replacements=[
    ("S.WORK/f'agent-expert-live-storage-{args.arm}/report.json'","S.WORK/f'agent-expert-live-compare-{args.arm}/report.json'"),
    ("S.WORK/'agent-expert-live-storage-candidate/report.json'","S.WORK/'agent-expert-live-compare-candidate/report.json'"),
    ("HERE/'agent-expert-live-storage-check.py'","HERE/'agent-expert-live-compare-check.py'"),
    ("HERE/'agent-expert-live-storage-install.py'","FROZEN/'agent-expert-live-storage-install.py'")]
  source=inspect.getsource(R.run)
  for before,after in replacements:
    assert source.count(before)==1;source=source.replace(before,after,1)
  namespace=dict(R.__dict__,HERE=ROOT,FROZEN=FROZEN)
  exec(compile(source,str(ROOT/'agent-expert-live-compare-run.py')+':frozen_runtime_overlay','exec'),namespace)
  return namespace['run'],runtime,dict(frozen_runner_sha256=sha(FROZEN/'agent-expert-live-storage-run.py'),
    transformed_function_sha256=hashlib.sha256(source.encode()).hexdigest(),literal_path_replacements=[list(x) for x in replacements],
    runtime_comparison_overlay='projected_qualify: exact two metadata leaves only',
    frozen_installer_sha256=sha(FROZEN/'agent-expert-live-storage-install.py'))


def paired(candidate,data,digest):
  assert recover_current()==read(ADMISSION)
  run,runtime,_=runtime_module()
  result=runtime.qualify(candidate,data,digest,'candidate');coverage(candidate['agent_expert_live_storage'])
  current=read(CURRENT);m,_=N.live()
  m.equal(project(current,CURRENT),project(candidate,CANDIDATE),'fresh_pair_projected_consumer_provenance')
  reference=dict(current,mode='reference');comparison=m.compare(reference,candidate)
  for i,(a,b) in enumerate(zip(current['rows'],candidate['rows'],strict=True)):
    m.equal(a['consumer_protocol'],b['consumer_protocol'],f'fresh_pair_owner.{i}')
  return dict(**comparison,admitted_current_sha256=CURRENT_SHA,candidate_vs_original_reference=result,
    current_gpu_rerun=False,current_admission_sha256=sha(ADMISSION))


def verify(path,digest):
  assert sha(path)==digest
  data=read(path);assert data['storage_protocol_sha256']==BASE_SHA and data['current_report_sha256']==CURRENT_SHA
  for p,h in data['files'].items():assert sha(p)==h,p
  original=load(FROZEN/'agent-expert-live-storage-common.py','compare_original_protocol_validator')
  original.verify(BASE,BASE_SHA)
  assert recover_current()==read(ADMISSION)
  assert runtime_module()[2]==data['runtime_overlay']
  assert data['gpu_arms']==['candidate'] and data['automatic_paced'] is False
  return data
