"""Compare predicted speed, slowing and path shape against a recorded drive."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from calibration_trials import vehicle_records
from camera_profiles import replay_module

TIMES = np.arange(1, 65, dtype=np.float64)*.1
MIDPOINTS = TIMES-.05


def kinematics(path):
  """Use identical position-derived interval speeds for every prediction."""
  points = np.vstack([np.zeros(2), np.asarray(path, dtype=np.float64)[:, :2]])
  if points.shape != (65, 2) or not np.isfinite(points).all():
    raise ValueError('Expected a finite 64-point trajectory')
  return np.linalg.norm(np.diff(points, axis=0), axis=1)/.1


def slowing(speed):
  # Five-sample smoothing and a sustained 0.5 m/s² threshold avoid a one-point trigger.
  smooth = np.convolve(speed, np.ones(5)/5, mode='valid')
  times = MIDPOINTS[2:-2]
  acceleration = np.gradient(smooth, .1)
  sustained = np.convolve((acceleration < -.5).astype(int), np.ones(5, dtype=int), mode='valid')
  indices = np.flatnonzero(sustained == 5)
  return {'onset_s': float(times[indices[0]]) if len(indices) else None,
          'minimum_smoothed_acceleration_m_s2': float(acceleration.min()),
          'speed_drop_m_s': float(max(0., smooth[0]-smooth.min()))}


def path_shape(predicted, recorded):
  """Compare shared travelled distances separately from the time-aligned error."""
  def curve(path):
    points = np.vstack([np.zeros(2), np.asarray(path)[:, :2]])
    distance = np.r_[0., np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
    keep = np.r_[True, np.diff(distance) > 1e-6]
    return distance[keep], points[keep]
  ps, pp = curve(predicted)
  rs, rp = curve(recorded)
  limit = min(ps[-1], rs[-1])
  if limit < 2:
    return {'shared_distance_m': float(limit), 'mean_error_m': None, 'mean_absolute_lateral_error_m': None}
  distance = np.linspace(.5, limit, 64)
  p = np.stack([np.interp(distance, ps, pp[:, i]) for i in range(2)], axis=-1)
  r = np.stack([np.interp(distance, rs, rp[:, i]) for i in range(2)], axis=-1)
  tangent = np.gradient(r, axis=0)
  norm = np.linalg.norm(tangent, axis=1)
  valid = norm > 1e-6
  cross = ((p-r)[:, 0]*tangent[:, 1]-(p-r)[:, 1]*tangent[:, 0])[valid]/norm[valid]
  return {'shared_distance_m': float(limit), 'mean_error_m': float(np.linalg.norm(p-r, axis=1).mean()),
          'mean_absolute_lateral_error_m': float(np.abs(cross).mean()) if len(cross) else None}


def compare_one(path, row, records):
  ns, states = records['carState']
  query = row['query_ns']
  start = max(0, int(np.searchsorted(ns, query))-1)
  end = min(len(ns), int(np.searchsorted(ns, query+6_400_000_000, side='right'))+1)
  if ns[start] > query or ns[end-1] < query+6_400_000_000 or np.max(np.diff(ns[start:end])) > 150_000_000:
    raise ValueError('Missing contiguous future CAN speed for comparison')
  seconds = (ns[start:end]-query)/1e9
  actual = np.interp(MIDPOINTS, seconds, [v['vEgo'] for v in states[start:end]])
  predicted = kinematics(path)
  reference = np.asarray(row['recorded_motion_future_xyz'])
  model_braking, actual_braking = slowing(predicted), slowing(actual)
  pressed = [t for t, v in zip(seconds, states[start:end], strict=True) if 0 <= t <= 6.4 and v['brakePressed']]
  return {'speed_bias_km_h': float((predicted-actual).mean()*3.6),
          'speed_mae_km_h': float(np.abs(predicted-actual).mean()*3.6),
          'predicted_speed_km_h': [float(np.interp(t, MIDPOINTS, predicted)*3.6) for t in (1., 3., 6.)],
          'recorded_speed_km_h': [float(np.interp(t, MIDPOINTS, actual)*3.6) for t in (1., 3., 6.)],
          'predicted_slowing': model_braking, 'recorded_slowing': actual_braking,
          'slowing_onset_difference_s': (model_braking['onset_s']-actual_braking['onset_s']
             if model_braking['onset_s'] is not None and actual_braking['onset_s'] is not None else None),
          'first_recorded_brake_pressed_s': float(pressed[0]) if pressed else None,
          'timed_path_error_m': float(np.linalg.norm(np.asarray(path)[:, :2]-reference[:, :2], axis=1).mean()),
          'distance_aligned': path_shape(path, reference)}


def summarize(rows):
  if not rows:
    return {'samples': 0}
  result = {'samples': len(rows)}
  for field in ('speed_bias_km_h', 'speed_mae_km_h', 'timed_path_error_m', 'slowing_onset_difference_s'):
    values = [r[field] for r in rows if r[field] is not None]
    result[field] = {'mean': float(np.mean(values)), 'median': float(np.median(values)), 'samples': len(values)} if values else None
  for field in ('mean_error_m', 'mean_absolute_lateral_error_m', 'shared_distance_m'):
    values = [r['distance_aligned'][field] for r in rows if r['distance_aligned'][field] is not None]
    result['distance_aligned_'+field] = float(np.mean(values)) if values else None
  result['slowing_cases'] = {
    'both': sum(r['predicted_slowing']['onset_s'] is not None and r['recorded_slowing']['onset_s'] is not None for r in rows),
    'predicted_only': sum(r['predicted_slowing']['onset_s'] is not None and r['recorded_slowing']['onset_s'] is None for r in rows),
    'recorded_only': sum(r['predicted_slowing']['onset_s'] is None and r['recorded_slowing']['onset_s'] is not None for r in rows)}
  result['horizons_s'] = [1., 3., 6.]
  for field in ('predicted_speed_km_h', 'recorded_speed_km_h'):
    result[field] = np.mean([r[field] for r in rows], axis=0).tolist()
  for source in ('predicted_slowing', 'recorded_slowing'):
    result[source] = {field: float(np.mean([r[source][field] for r in rows]))
                     for field in ('minimum_smoothed_acceleration_m_s2', 'speed_drop_m_s')}
  result['recorded_brake_pressed_windows'] = sum(r['first_recorded_brake_pressed_s'] is not None for r in rows)
  return result


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--source', type=Path, required=True)
  parser.add_argument('--logs', type=Path, required=True)
  parser.add_argument('--model', nargs=2, action='append', default=[], metavar=('NAME', 'DIRECTORY'))
  parser.add_argument('--big-model', type=Path)
  parser.add_argument('--trials', type=Path)
  parser.add_argument('--output', type=Path, required=True)
  parser.add_argument('--recording-uuid')
  args = parser.parse_args()
  replay = replay_module()
  replay.check_mount(args.output, args.recording_uuid)
  args.output.parent.mkdir(parents=True, exist_ok=True)
  manifest = replay.read_json(args.source/'manifest.json')
  records, hashes = vehicle_records(args.logs)
  if hashes != manifest['rlog_sha256']:
    raise ValueError('Log provenance differs')
  paths, provenance = {}, {}
  for name, directory in args.model:
    file = Path(directory)/'predictions.jsonl'
    predictions = [json.loads(line) for line in file.read_text().splitlines()]
    paths[name] = {r['query_ns']: replay.predicted_xyz(r) for r in predictions}
    if len(paths[name]) != len(predictions):
      raise ValueError('Duplicate prediction timestamp')
    provenance[name] = {'source': str(file), 'sha256': hashlib.sha256(file.read_bytes()).hexdigest()}
  big = np.load(args.big_model/'render-openpilot-big.npz') if args.big_model else None
  if big is not None:
    file = args.big_model/'render-openpilot-big.npz'
    provenance['openpilot_big'] = {'source': str(file), 'sha256': hashlib.sha256(file.read_bytes()).hexdigest()}
  big_indices = {int(t): i for i, t in enumerate(big['ns'])} if big is not None else {}
  rows = []
  for row in manifest['schedule']:
    if row['skip_reasons']:
      continue
    position = row['small_model']['model']['position']
    small = np.stack([np.interp(TIMES, position['t'], position[k]) for k in ('x', 'y', 'z')], axis=-1)*[1,-1,-1]
    current = {'openpilot_small': small, **{name: p[row['query_ns']] for name, p in paths.items()}}
    if big is not None:
      from scipy.spatial.transform import Rotation
      i = big_indices[row['query_ns']]
      axes = np.diag([1,-1,-1])
      current['openpilot_big'] = big['xyz'][i] @ axes @ Rotation.from_euler('xyz', big['rpy'][i]).as_matrix().T @ Rotation.from_euler('xyz', row['calibration_rpy']).as_matrix() @ axes
    for name, path in current.items():
      rows.append({'model': name, 'tick': row['tick'], 'time_s': row['display_time_s'], **compare_one(path, row, records)})
  if args.trials:
    trials = replay.read_json(args.trials/'manifest.json')
    predictions = {p['sample']: p for p in (json.loads(line) for line in (args.trials/'predictions.jsonl').read_text().splitlines())}
    for row in trials['schedule']:
      path = replay.predicted_xyz(predictions[row['sample']])
      rows.append({'model': 'trial:'+row['camera_profile'], 'split': row['split'], 'tick': row['tick'],
                   'time_s': row['display_time_s'], **compare_one(path, row, records)})
  names = sorted({r['model'] for r in rows})
  result = {'summary': {name: summarize([r for r in rows if r['model'] == name]) for name in names},
    'trial_splits': {name: {split: summarize([r for r in rows if r['model'] == name and r.get('split') == split])
                          for split in ('tune', 'holdout')} for name in names if name.startswith('trial:')},
    'rows': rows, 'provenance': provenance, 'log_sha256': hashes,
    'definition': {'speed': 'XY trajectory arc-length increments / 0.1s; all models use the same estimator',
      'reference_speed': 'Recorded CAN vEgo; future CAN is comparison-only, never model input',
      'slowing': '0.5s smoothed speed, acceleration below -0.5 m/s² for at least 0.5s; not a brake command',
      'shape': 'Positions at equal travelled arc distance over shared coverage; reported separately from timed ADE',
      'sampling': 'Overlapping 6.4s prediction windows; slowing counts are windows, not independent braking events',
      'openpilot_speed': 'Derived from interpolated plan positions, not the native velocity head or executed controller commands',
      'limits': 'A slower human drive can increase timed error for a faster model. Faster alone does not establish safe or correct behavior.'}}
  replay.write_json(args.output, result)
  print(json.dumps(result['summary'], indent=2), flush=True)


if __name__ == '__main__':
  main()
