"""Freeze the bounded optional parent protocol after current CPU checks."""
import importlib.util,json
from pathlib import Path
HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('nearest_order_freeze_common',HERE/'agent-nearest-order-common.py')
c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c)

def main():
  c.setup_cpu();plan,policy,startup,dynamic=c.verify_assets();saved=c.validate_saved(plan,startup)
  check=c.read(HERE/'agent-nearest-order-checks.json')
  if not check['passed'] or check['gpu_executed'] or check['source_sha256']!=c.own_sources():raise ValueError('Current passing CPU controls required')
  files=[*map(Path,c.PINS),HERE/'agent-nearest-order-checks.json',HERE/'agent-nearest-order-diagnosis.json',
    FROZEN_RUN:=c.FROZEN/'agent-nearest-live-run.py',c.FROZEN/'agent-nearest-live-policy.py',
    Path(plan['saved_nearest_report']),Path(plan['startup_runner']),Path(plan['startup_runner']).with_name('agent-startup-spec-warm.py')]
  record=dict(schema=1,queries=4,actual_reference_order=[x['sample'] for x in plan['cases']],
    display_times_s=[x['display_time_s'] for x in plan['cases']],saved_consumer=str(c.SAVED),saved_consumer_sha256=c.sha(c.SAVED),
    source_sha256=c.own_sources(),artifact_sha256={str(p):c.sha(p) for p in files},
    canonical_main_transform=check['canonical_main_transform'],bounded_recipe_changes=check['bounded_recipe_changes'],
    source_snapshot=str(c.BASE),old_nearest_plan=str(c.LIVE_PLAN),inherited_frozen_asset_count=len(plan['frozen_sha256']),
    startup='Same qualified helper, same first nearest input, empty decoder history, cold logical vision cache, same linked graph counts and retained owner',
    reference='Original canonical benchmark.main; only local input owner initialization changes from None to components.inputs',
    saved_outputs_used_by_GPU_reference=False,comparison='Exact all audit fields, prediction bits, histories, target verification, ownership and startup restoration',
    old_failure_remains_failed=True,numerical_differences_vs64_remain_diagnostic=True,
    paired_new_consumer_required_now=False,paired_new_consumer_condition='Only if same-order reference fails or saved source/state gates no longer hold',
    parent_only=True,global_lock='/tmp/opencode/alpamayo-chestnut-speed.lock',timeout_seconds=900,
    output=str(HERE/'agent-nearest-order-parent-reference/report.json'),automatic_pace=False,automatic_adoption=False)
  c.save(c.PROTOCOL,record);print(json.dumps(record,indent=2))

if __name__=='__main__':main()
