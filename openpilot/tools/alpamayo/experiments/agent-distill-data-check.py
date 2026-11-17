"""Independent metadata/causality checks. Never imports Torch or opens numerical test targets."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
EXP = Path(__file__).resolve().parent
WORK = Path('/mnt/nvme-500/tmp/alpamayo-speed-20261116')
ARTIFACTS = WORK/'agent-distill-data-artifacts'
PROTECTED = '0000008e--321b2ddc03'


def sha(path):
  with Path(path).open('rb') as stream: return hashlib.file_digest(stream, 'sha256').hexdigest()


def load(name, path):
  spec = importlib.util.spec_from_file_location(name, path)
  module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module); return module


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--output', type=Path, required=True)
  args = parser.parse_args(); output = args.output.resolve()
  assert output.parent.is_relative_to(WORK) and output.name.startswith('agent-distill-data-')
  manifest_path = ARTIFACTS/'agent-distill-data-cohort.json'
  manifest = json.loads(manifest_path.read_text()); counts = {}
  assert manifest['admissible'] is False and manifest['parent_gpu_capture_command'] is None
  assert manifest['fit_records'] == manifest['validation_records'] == manifest['sealed_teacher_records'] == 0
  assert not manifest['pixel_decode_count'] and not manifest['training_executed'] and not manifest['gpu_initialized']
  for path, digest in manifest['source_sha256'].items(): assert sha(path) == digest
  for key in ('audit_evidence', 'navigation_evidence', 'teacher_protocol'):
    pin = manifest[key]; assert sha(pin['path']) == pin['sha256']
  assert len(manifest['records']) == 3
  assert {r['provisional_split'] for r in manifest['records']} == {'train', 'validation', 'sealed_test'}
  dates, boots = [], []
  for record in manifest['records']:
    assert PROTECTED not in json.dumps(record) and not record['admissible_training_sample']
    dates.append(record['evidenced_date_utc']); boots.append(record['inferred_boot_epoch_ns'])
    assert abs(record['gps_minus_init_clock_ns']) < 1_000_000_000
    assert record['source_rlog']['sha256'] == sha(record['source_rlog']['path'])
    for path, digest in record['source_qlog_sha256'].items(): assert sha(path) == digest
    for pin in record['source_videos']+record['supplementary_state_logs']:
      assert sha(pin['path']) == pin['sha256'] and PROTECTED not in pin['path']
    h = record['history_metadata']; q = record['anchor_query_ns']
    assert h['query_ns'] == q and h['motion_input_max_timestamp_ns'] <= q
    assert h['max_motion_history_gap_ns'] <= 150_000_000 and q-h['last_motion_ns'] <= 100_000_000
    assert [r['view'] for r in h['frames']] == [1]*5+[6]*5
    for view in (1, 6):
      rows = [r for r in h['frames'] if r['view'] == view]
      assert len({r['frame_id'] for r in rows}) == len({r['capture_ns'] for r in rows}) == 5
      assert [r['requested_ns']-q for r in rows] == [-2_000_000_000, -1_000_000_000, -500_000_000, -100_000_000, 0]
      for row in rows:
        assert row['capture_ns'] <= row['requested_ns'] <= q
        assert row['requested_ns']-row['capture_ns'] <= 150_000_000
        assert 0 <= row['capture_ns']-row['prior_extrinsics_ns'] <= 1_000_000_000
        assert row['capture_ns'] == row['encoding']['capture_ns'] and row['eof_ns'] == row['encoding']['eof_ns']
    s = record['selected_history_checks']
    assert s['max_car_state_age_ns'] <= 150_000_000 and s['max_vehicle_parameters_age_ns'] <= 1_000_000_000
    navigation = record['original_rlog_navigation_metadata']
    assert all(not r['valid'] and not r['nonempty_instruction'] for r in navigation if r['kind'] == 'navInstruction')
    assert sum(r['kind'] == 'navRoute' for r in navigation) == 6
    assert all(r['coordinate_count'] == 0 for r in navigation if r['kind'] == 'navRoute')
    if record['provisional_split'] == 'sealed_test':
      assert s['sealed_numeric_state_values_extracted'] is False
      assert 'causal_pose_check' not in h and not h['motion_values_retained']
      assert not any(k in s for k in ('xyz_sha256', 'rot_sha256', 'original_vehicle_history_executed'))
    else: assert s['original_vehicle_history_executed'] and not s['exported_pose_arrays']
  assert len(set(dates)) == 3 and min(b-a for a, b in zip(sorted(boots), sorted(boots)[1:])) > 12*3600*1_000_000_000
  counts['three_independent_date_session_records'] = True
  counts['thirty_causal_distinct_frames_with_matching_encode_indices'] = True
  counts['selected_vehicle_history_bounds_and_original_constructor'] = True
  counts['sealed_test_values_not_loaded_for_fit_or_stopping'] = True
  trainer = load('agent_distill_data_frozen_trainer', EXP/'agent-distill-feasibility-train.py')
  try: trainer.verify_records(manifest)
  except (AssertionError, KeyError, ValueError) as error:
    counts['original_trainer_rejects_incomplete_metadata_cohort'] = dict(type=type(error).__name__, message=str(error))
  else: raise AssertionError('Incomplete metadata was accepted as a training capture split')
  audit = load('agent_distill_data_frozen_audit', EXP/'agent-distill-data-audit.py')
  try: audit.load_events(audit.ARCHIVE/'realdata'/(PROTECTED+'--0')/'rlog.zst', 1024)
  except AssertionError: counts['protected_original_path_rejected_before_open'] = True
  else: raise AssertionError('Protected route path was accepted')
  assert not any(name == 'torch' or name.startswith('tinygrad.runtime.ops_') for name in sys.modules)
  result = dict(schema=1, passed=True, checks=counts, manifest_sha256=sha(manifest_path),
    helper_sha256=sha(Path(__file__)), gpu_initialized=False, training_executed=False,
    boundary='Metadata and original CPU input-contract checks only. No complete new sample, teacher capture, fit, quality or speed claim.')
  with output.open('x') as stream: stream.write(json.dumps(result, indent=2)+'\n')
  print(json.dumps(result, indent=2))


if __name__ == '__main__': main()
