"""Independent repair protocol; every earlier live artifact remains immutable."""
import copy
import importlib.util
import os
from pathlib import Path
import sys

WORK=Path('/mnt/nvme-500/tmp/alpamayo-speed-20261116')
OLD=WORK/'agent-expert-live-source'
BASE=WORK/'agent-expert-live-protocol.json'
BASE_SHA='e4b52c12b56695c518394ceced344eda5aeec6bfa46b014cd44eb4030249f5ae'
ROOT=WORK/'agent-expert-live-storage-source'
PROTOCOL=WORK/'agent-expert-live-storage-protocol.json'
ARTIFACTS=WORK/'agent-expert-live-storage-artifacts'
FAILED=WORK/'agent-expert-live-consumer'
FAILED_SHA='451cde6d713a0776cf75ac652685e3ee3595b2669ac628e94d3bf4d5eec8d977'
spec=importlib.util.spec_from_file_location('storage_frozen_live_common',OLD/'agent-expert-live-common.py')
N=importlib.util.module_from_spec(spec);spec.loader.exec_module(N)
assert N.sha(BASE)==BASE_SHA
assert N.sha(OLD/'agent-expert-live-common.py')==N.read(BASE)['files'][str(OLD/'agent-expert-live-common.py')]
# Only this private common-module instance gets the new storage directory.
N.ARTIFACTS=ARTIFACTS
sha,read,write,load=N.sha,N.read,N.write,N.load


def configure():
  result=N.configure()
  target=ARTIFACTS/'hooks/agent-exact-combined-install.py'
  assert target.is_file() and sha(target)==read(N.MANIFEST)['files'][str(N.EXP/target.name)]
  return result


def failed_evidence():
  assert sha(FAILED/'report.json')==FAILED_SHA
  report=read(FAILED/'report.json');status=report['agent_expert_live']
  assert report['gpu_executed'] and report['predictions_qualified']
  assert not status['passed'] and not status['gpu_executed'] and status['hooks_restored']
  m,_=N.live();comparison=m.compare(read(N.REFERENCE),report)
  assert comparison==report['comparison'];N.validate_startup(report)
  _,binding=N.pv();N.validate_pv(status['pv'],True,5,binding)
  actual=read(FAILED/'report.normalized-invocation.json')['environment']
  assert actual['ALPAMAYO_FP8_CACHE']=='/dev/shm/alpamayo-9060-fp8'
  assert actual['CACHEDB']=='/tmp/opencode/alpamayo-asius-20261114/cache.db'
  assert 'coverage' not in status and 'storage_bindings' not in status
  return dict(report_sha256=FAILED_SHA,original_status_passed=False,comparison=comparison,
    full_startup_saved=True,pv_capture_complete=True,actual_normalized_cache=actual['ALPAMAYO_FP8_CACHE'],
    numerical_outputs_reused_as_new_qualification=False,
    missing=['actual intended 500GB weight-cache binding','serialized complete MLP/vision coverage and quantizer state'],
    decision='Fresh matched current/candidate two-query Consumers; keep failed run as exact-output diagnostic evidence')


def check_storage(events,*,paced=False):
  normal=[x for x in events if x['event']=='normalize']
  enter=[x for x in events if x['event']=='factory_enter']
  done=[x for x in events if x['event']=='factory_complete']
  assert len(normal)==len(enter)==len(done)==1,'Require actual normalizer and completed canonical factory exactly once'
  assert events.index(normal[0])<events.index(enter[0])<events.index(done[0])
  for event in (normal[0],enter[0],done[0]):assert event['actual_environment']==N.storage()
  assert enter[0]['cachedb_constant']==N.storage()['CACHEDB']
  files=[x for x in events if x['event']=='cache_open']
  assert [x['path'] for x in files]==[str(N.WEIGHTS/'index.json'),str(N.WEIGHTS/'weights.bin'),str(N.WEIGHTS/'weights.bin')]
  assert all(x['mode'] in ('r','rb') for x in files)
  temporary=[x for x in events if x['event']=='factory_temporary_directory']
  assert len(temporary)==1 and Path(temporary[0]['path']).parent==Path(N.storage()['TMPDIR'])
  prep=[x for x in events if x['event']=='paced_preparer']
  assert len(prep)==int(paced)
  if paced:assert prep[0]['actual_environment']==N.storage() and prep[0]['sampler_artifacts']==str(ARTIFACTS/'hooks/agent-exact-live-artifacts')
  return dict(complete=True,canonical_factory_completed=True,actual_cache_files=[x['path'] for x in files],
    actual_environment=enter[0]['actual_environment'],cachedb_constant=enter[0]['cachedb_constant'],
    factory_temporary_directory=temporary[0]['path'],paced_preparer_observed=paced)


def qualify(report,data,digest,arm):
  assert arm in ('current','candidate')
  m,manifest=N.live();old=read(N.CONSUMER)
  comparison=m.compare(read(N.REFERENCE),report)
  assert report['comparison']==comparison and report['gpu_executed'] and report['predictions_qualified']
  status=report['agent_expert_live_storage']
  assert status['arm']==arm and status['passed'] and status['gpu_executed'] and status['hooks_restored']
  assert status['protocol_sha256']==digest and status['base_protocol_sha256']==BASE_SHA
  assert status['storage_proof']==check_storage(status['storage_events'])
  assert status['constructor_restored'] and all(x['restored'] for x in status['composition_restorations'])
  assert status['quantizer_idle'] and status['coverage']['consumer_bindings']==1 and status['coverage']['live_sampler_calls']==0
  assert status['transforms']=={k:v for k,v in manifest['expected_transforms'].items() if k!='paced'}
  assert [r['sample'] for r in report['rows']]==manifest['selected_samples']
  assert [r['speculation']['accepted'] for r in report['rows']]==[True,False]
  assert report['rows'][0]['history_before']=={'history':[],'prefill_draft_old_history':[]}
  N.validate_startup(report)
  m.equal(report['source_sha256'],old['source_sha256'],'all_model_sources')
  m.equal(report['consumer_provenance'],old['consumer_provenance'],'consumer_owner_and_code')
  for i,(a,b) in enumerate(zip(old['rows'],report['rows'],strict=True)):
    m.equal(a['consumer_protocol'],b['consumer_protocol'],f'owner.{i}')
    assert len(b['defined_kv_prefix_bits'])==28
  N.validate_pv(status['pv'],arm=='candidate',5,data['evidence']['pv'])
  return comparison


def paired(current,candidate,data,digest):
  qualify(current,data,digest,'current');qualify(candidate,data,digest,'candidate')
  m,_=N.live()
  # Comparator role view only; original reports remain unchanged.
  reference=dict(current,mode='reference');result=m.compare(reference,candidate)
  for i,(a,b) in enumerate(zip(current['rows'],candidate['rows'],strict=True)):
    m.equal(a['consumer_protocol'],b['consumer_protocol'],f'fresh_pair.owner.{i}')
  return dict(**result,current_sha256=sha(WORK/'agent-expert-live-storage-current/report.json'),
    candidate_sha256=sha(WORK/'agent-expert-live-storage-candidate/report.json'),fresh_matched_pair=True)


def verify(path,digest):
  assert sha(path)==digest
  data=read(path);assert data['base_protocol_sha256']==BASE_SHA
  for p,h in data['files'].items():assert sha(p)==h,p
  N.verify(BASE,BASE_SHA)
  m,_=N.live();m.equal(N.evidence(),data['evidence'],'independent_qualified_evidence')
  m.equal(failed_evidence(),data['failed_evidence'],'immutable_failed_run')
  assert data['arms']==['current','candidate'] and data['automatic_paced'] is False
  return data
