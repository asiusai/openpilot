"""Small integration contract; reuse qualified numerical installers without copying their trees."""
import copy
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys

WORK=Path('/mnt/nvme-500/tmp/alpamayo-speed-20261116')
EXP=Path('/tmp/opencode/openpilot-alpamayo/openpilot/tools/alpamayo/experiments')
ROOT=WORK/'agent-live-cpu-integrated-source'
ARTIFACTS=WORK/'agent-live-cpu-integrated-artifacts'
PROTOCOL=WORK/'agent-live-cpu-integrated-protocol.json'
REFERENCE=WORK/'agent-expert-live-compare-candidate/report.json'
REFERENCE_SHA='751549150a637df24df3fed6c1782a33bf89b77871f358c2b87b9e06a3269367'
PACED=WORK/'agent-expert-paced-stem-run/report.json'
PACED_SHA='7d719c9ad7db5b714f23ae98a415e2b2f0f568dcc71a7e0e24a0e599a371aa6b'
CPU_REPORT=WORK/'agent-live-cpu-fast-report-v2.json'
CPU_SHA='aefcdaf57f1098bce0e3faf4fef903f8408041521982c79a46d13b98cfddfb00'
HOOK=EXP/'agent-live-cpu-fast-hook.py'
HOOK_SHA='d213b0210ccce941b4d6d1134f9eb7aac37074a417f0ada1d3c53af4f0530c4b'
OLD=WORK/'agent-expert-live-storage-source'
COMPARE=WORK/'agent-expert-live-compare-source-v2'


def sha(path):
  with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
def read(path):return json.loads(Path(path).read_text())
def write(path,value,*,new=False):
  with Path(path).open('x' if new else 'w') as stream:json.dump(value,stream,indent=2,allow_nan=False);stream.write('\n')
def load(path,name):
  spec=importlib.util.spec_from_file_location(name,path);value=importlib.util.module_from_spec(spec)
  sys.modules[name]=value;spec.loader.exec_module(value);return value


# Preserve all existing numerical, source, storage, and qualification checks.
S=load(OLD/'agent-expert-live-storage-common.py','cpu_integrated_storage')
N=S.N;S.ARTIFACTS=ARTIFACTS;N.ARTIFACTS=ARTIFACTS
V=load(COMPARE/'agent-expert-live-compare-common.py','cpu_integrated_comparator')


def configure():
  N.configure();target=ARTIFACTS/'hooks/agent-exact-combined-install.py'
  target.parent.mkdir(exist_ok=True)
  original=N.EXP/target.name
  expected=read(N.MANIFEST)['files'][str(original)]
  assert sha(original)==expected
  if not target.exists():target.write_bytes(original.read_bytes());target.chmod(0o444)
  assert sha(target)==expected
  N.paths()


def output(arm):
  assert arm in ('consumer','paced')
  return WORK/f'agent-live-cpu-integrated-{arm}/report.json'


def parity(reference,candidate,path):
  m,manifest=N.live()
  result=m.compare(dict(reference,mode='reference'),candidate)
  m.equal(V.project(reference,REFERENCE),V.project(candidate,path),'identical_consumer_owner_and_source')
  for i,(a,b) in enumerate(zip(reference['rows'],candidate['rows'],strict=True)):
    m.equal(a['consumer_protocol'],b['consumer_protocol'],f'owner.{i}')
    assert len(b['defined_kv_prefix_bits'])==28
  assert [r['sample'] for r in candidate['rows']]==manifest['selected_samples']
  assert [r['speculation']['accepted'] for r in candidate['rows']]==[True,False]
  assert candidate['rows'][0]['history_before']=={'history':[],'prefill_draft_old_history':[]}
  return dict(**result,reference_sha256=REFERENCE_SHA,reference_reused=True,reference_gpu_rerun=False)


def foundations():
  assert sha(REFERENCE)==REFERENCE_SHA and sha(PACED)==PACED_SHA
  assert sha(CPU_REPORT)==CPU_SHA and sha(HOOK)==HOOK_SHA
  r=read(REFERENCE);p=read(PACED);cpu=read(CPU_REPORT)
  assert cpu['complete'] and not cpu['gpu_executed'] and cpu['hooks_restored']
  assert r['predictions_qualified'] and r['gpu_executed'] and r['agent_expert_live_compare']['passed']
  assert p['agent_expert_paced_stem']['passed'] and p['agent_expert_live_storage']['passed']
  m,manifest=N.live();_,pv=N.pv()
  actual={q.name:sha(q) for q in [*(N.LIVE/'experiments').glob('*.py'),*(N.LIVE/'experiments').glob('*.c'),N.LIVE/'model.py']}
  assert actual==r['source_sha256']==r['initialized']['source_sha256'] and len(actual)==253
  V.snapshot(r,REFERENCE)
  for report,paced,count in ((r,False,5),(p,True,3+len(p['rows']))):
    status=report['agent_expert_live_storage'];V.coverage(status)
    N.validate_startup(report,paced=paced);N.validate_pv(status['pv'],True,count,pv)
  assert [row['speculation']['accepted'] for row in r['rows']]==[True,False]
  assert r['rows'][0]['history_before']=={'history':[],'prefill_draft_old_history':[]}
  return dict(reference_sha256=REFERENCE_SHA,paced_sha256=PACED_SHA,cpu_report_sha256=CPU_SHA,hook_sha256=HOOK_SHA,
    live_manifest_sha256=sha(N.MANIFEST),invocation_sha256=sha(m.INVOCATION),selected_samples=manifest['selected_samples'],
    initial_history='empty',acceptance=[True,False],actual_source_files=253,numerical_source_unchanged=True)


def files():
  # Pin helpers and existing contracts, without another recursive source copy/list.
  paths=[REFERENCE,PACED,CPU_REPORT,HOOK,N.MANIFEST,m_path()/'pipeline-invocation.json',
    OLD/'agent-expert-live-storage-common.py',OLD/'agent-expert-live-storage-install.py',
    WORK/'agent-expert-live-source/agent-expert-live-common.py',COMPARE/'agent-expert-live-compare-common.py',
    N.EXP/'agent-exact-live-install.py',N.EXP/'agent-exact-live-common.py']
  paths += list(Path(__file__).parent.glob('agent-live-cpu-integrated-*.py'))
  return {str(p):sha(p) for p in paths}

def m_path():return N.LIVE.parent


def verify(path,digest):
  assert sha(path)==digest
  data=read(path)
  for name,expected in data['files'].items():assert sha(name)==expected,name
  assert foundations()==data['foundations']
  assert data['arms']==['consumer','paced'] and data['automatic_paced'] is False
  if data.get('cpu_only'):assert os.environ.get('DEV')=='CPU' and os.environ.get('ALLOW_DEVICE_USAGE')=='0'
  return data


def validate_status(report,status,arm,path):
  assert status['passed'] and status['gpu_executed'] and status['hooks_restored'] and status['constructor_restored']
  assert status['quantizer_idle'] and all(x['restored'] for x in status['composition_restorations'])
  S.check_storage(status['storage_events'],paced=arm=='paced')
  V.coverage(status);N.validate_startup(report,paced=arm=='paced')
  _,pv=N.pv();N.validate_pv(status['pv'],True,3+len(report['rows']),pv)
  cpu=status['cpu_hook'];calls=cpu['calls']
  assert cpu['consumer_bindings']==1 and cpu['hook_source_sha256']==HOOK_SHA
  assert cpu['provenance']['overlay_source_sha256']==HOOK_SHA
  assert cpu['cleanup_exact'] and cpu['shadow_after_cleanup']==0
  assert len(calls)==len(report['rows']) and calls[0]['shadow_before']==0
  assert [c['query_ns'] for c in calls]==[r['query_ns'] for r in report['rows']]
  assert all(c['inside_consumer_predict'] and c['shadow_after']==33177600 and c['completed'] for c in calls)
  assert calls[0]['misses_delta']==1 and calls[0]['hits_delta']==0
  assert all(c['misses_delta']==0 and c['hits_delta']==1 for c in calls[1:])
  assert all(math.isfinite(c['elapsed_ms']) and c['elapsed_ms']>=0 for c in calls)
  if arm=='consumer':
    assert report['gpu_executed'] and report['predictions_qualified']
    return parity(read(REFERENCE),report,path)
  assert report['requested_hz']==10 and report['source_window_s']==8 and report['inference_budget_seconds']==180
  assert report['maximum_active_inference']==1 and report['queued_prediction_snapshots']==0
  assert report['rows'][0]['vision_hits']==0 and report['rows'][0]['vision_misses']==10
  return dict(continuous_only=True,comparison_boundary='same original full startup; 35..43 seconds; received 20Hz FLOOR; max10Hz; one active; zero queue',baseline_sha256=PACED_SHA)
