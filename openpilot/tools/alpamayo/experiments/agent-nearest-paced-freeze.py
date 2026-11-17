"""Freeze one parent protocol only after source-bound CPU checks pass."""
import importlib.util,json
from pathlib import Path
HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('nearest_paced_freeze_admit',HERE/'agent-nearest-paced-admit.py')
a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a)

def main():
  a.setup_cpu();c,plan,policy,startup,dynamic,order,reference,saved,admission=a.verify()
  checks=a.read(HERE/'agent-nearest-paced-checks.json')
  if not checks['passed'] or checks['gpu_executed'] or checks['source_sha256']!=a.sources() or checks['admission']!=admission:
    raise ValueError('Current passing CPU admission/adapter checks required')
  files=[*a.PINS,HERE/'agent-nearest-paced-checks.json',c.SAVED,c.LIVE_PLAN,
    c.FROZEN/'agent-nearest-live-policy.py',Path(plan['startup_runner']),Path(plan['startup_runner']).with_name('agent-startup-spec-warm.py')]
  protocol=dict(schema=1,source_sha256=a.sources(),artifact_sha256={str(p):a.sha(p) for p in files},admission=admission,
    parent_only=True,requested_hz_ceiling=10,source_start_seconds=35,source_duration_seconds=8,raw_rgb_frames=450,
    maximum_active_inference=1,queued_prediction_snapshots=0,active_budget_seconds=180,outer_timeout_seconds=900,
    global_lock='/tmp/opencode/alpamayo-chestnut-speed.lock',startup='Unchanged full qualified current-input-only helper before source clock',
    compiler_cache='Only CACHEDB pathname changes into the new owned run directory; no numerical runtime flags change',
    replay='Original qualified startup.main(task=paced) under the immutable received-only nearest policy; no scheduler/model source substitution',
    output=str(HERE/'agent-nearest-paced-parent-run/report.json'),automatic_parity_rerun=False,automatic_adoption=False,
    goal_qualified=False,old64_failure_preserved=True,
    known_baseline=dict(policy='FLOOR',completion_hz=2.4294274756233003,actual_mean_vision_misses=8.4,
      nearest_metadata_mean_misses_at_same_times=8.6,interpretation='No guaranteed speed improvement; nearest actual timing remains unmeasured'),
    measurement_limit='PC recorded replay with retrospective navigation, integrity-warmed OS cache, and no assertion of live SmallModel/device health')
  a.save(a.PROTOCOL,protocol);print(json.dumps(protocol,indent=2))

if __name__=='__main__':main()
