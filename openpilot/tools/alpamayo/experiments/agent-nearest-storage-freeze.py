"""Freeze one500GB-only parent retry after the full pre-model call path passes onCPU."""
import importlib.util,json
from pathlib import Path
HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('nearest_storage_freeze_run',HERE/'agent-nearest-storage-run.py')
run=importlib.util.module_from_spec(spec);spec.loader.exec_module(run)

def main():
  old=run.inherited();run.setup_cpu(old);old.a.verify();weight_proof=run.weights()
  checks=run.read(run.CHECKS);_,transformation=run.transform(old,old.verify_frozen)
  if not checks['passed'] or checks['gpu_executed'] or checks['source_sha256']!=run.sources() or checks['transformation']!=transformation:
    raise ValueError('Current full-call-path CPU proof required')
  if len(checks['factory_boundaries'])!=1 or not checks['actual_output_constructors']:raise ValueError('Actual canonical Output preflight missing')
  files=[*run.PINS,run.CHECKS,run.WORK/'agent-nearest-storage-weights.json',Path('/tmp/opencode/run-alpamayo-speed-trial.py')]
  protocol=dict(schema=1,source_sha256=run.sources(),artifact_sha256={str(p):run.sha(p) for p in files},
    admission=old.verify_frozen()['admission'],transformation=transformation,weights=weight_proof,cache_environment=run.cache_environment(run.DEFAULT.parent),
    output=str(run.DEFAULT),parent_output=str(run.WORK/'agent-nearest-storage-parent/report.json'),
    parent_program=str(HERE/'agent-nearest-storage-parent.py'),original_storage_guard_unchanged=True,
    original_no_overwrite_guard_unchanged=True,output_filesystem='/mnt/nvme-500',root_output_forbidden=True,
    global_lock='/tmp/opencode/alpamayo-chestnut-speed.lock',timeout_environment={'AGENT_NEAREST_PACED_PARENT_TIMEOUT':'1'},
    outer_timeout_seconds=900,active_budget_seconds=180,source_start_seconds=35,source_duration_seconds=8,raw_rgb_frames=450,
    requested_hz_ceiling=10,maximum_active_inference=1,queued_prediction_snapshots=0,original_full_startup_unchanged=True,
    source_clock_changes=False,numerical_changes=False,qualification_changes=False,extra_parity_runs=0,
    automatic_adoption=False,goal_qualified=False,gpu_executed=False)
  run.save(run.PROTOCOL,protocol);print(json.dumps(protocol,indent=2))

if __name__=='__main__':main()
