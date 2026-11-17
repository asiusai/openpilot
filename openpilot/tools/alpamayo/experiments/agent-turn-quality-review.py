"""Evaluate completed predictions against recorded motion; never run a model."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import zstandard

WORK = Path('/mnt/nvme-500/tmp/alpamayo-speed-20261116')
REPLAY = WORK/'agent-paced-dynamic-source-cpu02/alpamayo/experiments/replay-route.py'
SOURCE = Path('/mnt/nvme-256/footage/f9ad9a96/exports/0000008e--321b2ddc03/alpamayo-4hz-calibrated-two-six')
LOGS = Path('/mnt/nvme-256/footage/f9ad9a96/realdata').resolve()
COMPLETED = WORK/'agent-turn-paced-111-119/report.jsonl'
OUTPUT = WORK/'agent-turn-paced-111-119/parent-motion-quality-review.json'


def sha(path):
  return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
  from openpilot.cereal import log
  assert not OUTPUT.exists(), 'Preserve existing review'
  manifest = json.loads((SOURCE/'manifest.json').read_text())
  completed = [json.loads(line) for line in COMPLETED.read_text().splitlines()]
  completed = [row for row in completed if row.get('kind') == 'model_completion']
  assert len(completed) == 18 and manifest['route'] == '0000008e--321b2ddc03'
  motions = {}
  verified = {}
  for name, expected in manifest['rlog_sha256'].items():
    path = (LOGS/name).resolve()
    assert path.is_relative_to(LOGS) and path.parent.name.startswith('0000008e--321b2ddc03--')
    assert sha(path) == expected
    with path.open('rb') as stream, zstandard.ZstdDecompressor().stream_reader(stream) as decoder:
      raw = decoder.read(256*1024**2+1)
      assert len(raw) <= 256*1024**2, 'Unexpected decompressed log size'
    for event in log.Event.read_multiple_bytes(raw):
      if not event.valid or event.which() != 'deviceMotion':
        continue
      message = event.deviceMotion
      if message.timestamp > 0 and message.inputsOK and message.orientationNED.valid and message.velocityDevice.valid:
        motions[int(message.timestamp)] = ([getattr(message.orientationNED, axis) for axis in 'xyz'],
                                           [getattr(message.velocityDevice, axis) for axis in 'xyz'])
    verified[name] = expected
  spec = importlib.util.spec_from_file_location('parent_completed_turn_motion', REPLAY)
  replay = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(replay)
  motion = replay.Motion(motions)
  low = min(row['input']['query_ns'] for row in completed)
  high = max(row['input']['query_ns'] for row in completed)
  references = [row for row in manifest['schedule'] if low <= row['query_ns'] <= high
                and row.get('recorded_motion_future_xyz') is not None]
  assert references
  differences = []
  for row in references:
    actual = motion.sample(row['query_ns'], row['calibration_rpy'])[2]
    assert actual is not None
    differences.append(float(np.max(np.abs(np.asarray(actual)-row['recorded_motion_future_xyz']))))
  assert max(differences) <= 1e-7, ('Existing reference mismatch', max(differences))
  rows = []
  for row in completed:
    inp, prediction = row['input'], row['prediction']
    actual = np.asarray(motion.sample(inp['query_ns'], inp['calibration_rpy'])[2], dtype=np.float64)
    predicted = replay.predicted_xyz(row).astype(np.float64)
    assert actual.shape == predicted.shape == (64, 3) and np.isfinite(actual).all()
    error = np.linalg.norm(predicted[:, :2]-actual[:, :2], axis=1)
    derived = {}
    for key, path in (('predicted', predicted), ('recorded', actual)):
      velocity = np.diff(np.vstack((np.zeros((1, 3)), path)), axis=0)/.1
      speed = np.linalg.norm(velocity[:, :2], axis=1)
      heading = np.arctan2(velocity[:, 1], velocity[:, 0])
      derived[key] = dict(speed_start_m_s=float(speed[0]), speed_min_m_s=float(speed.min()),
        speed_end_m_s=float(speed[-1]), heading_at_1_3_6_4_s_rad=heading[[9, 29, 63]].tolist(),
        speed_at_1_3_6_4_s_m_s=speed[[9, 29, 63]].tolist())
    rows.append(dict(query_ns=inp['query_ns'], offset_s=(inp['query_ns']-manifest['capture_start_ns'])/1e9,
      navigation=inp['navigation_text'], reasoning=prediction['reasoning'], ade_m=float(error.mean()),
      fde_m=float(error[-1]), error_at_1_3_6_4_s_m=error[[9, 29, 63]].tolist(), **derived))
  result = dict(complete=True, gpu_executed=False, evaluation_only=True, route=manifest['route'],
    model_inputs_modified=False, reference='Original Motion.sample integrated valid deviceMotion with each actual input calibration',
    limitations=['Recorded motion is an estimated path, not independent surveyed ground truth.',
                 'Future measurements are used only after inference for this evaluation.',
                 'Eighteen overlapping predictions from one turn are not broad driving validation.'],
    completed_sha256=sha(COMPLETED), manifest_sha256=sha(SOURCE/'manifest.json'), motion_source_sha256=sha(REPLAY),
    script_sha256=sha(Path(__file__)), rlog_sha256=verified, motion_records=len(motions),
    existing_reference_checks=len(references), existing_reference_max_abs_difference=max(differences),
    predictions=len(rows), ade_m=float(np.mean([row['ade_m'] for row in rows])),
    fde_m=float(np.mean([row['fde_m'] for row in rows])), rows=rows)
  OUTPUT.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
  print(json.dumps({key:result[key] for key in ('complete','predictions','ade_m','fde_m','existing_reference_checks','existing_reference_max_abs_difference')}))
  for row in rows:
    print(json.dumps({key:row[key] for key in ('offset_s','ade_m','fde_m','predicted','recorded')}))


if __name__ == '__main__':
  main()
