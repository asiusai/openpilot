"""Compare camera configurations at identical recorded times."""
import argparse
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np

from camera_profiles import replay_module


def load_predictions(directory, schedule):
  entries = [json.loads(line) for line in (directory/'predictions.jsonl').read_text().splitlines()]
  result = {p['tick']: p for p in entries}
  expected = {r['tick']: r['query_ns'] for r in schedule if 'sample' in r}
  if len(entries) != len(result) or {tick: p['query_ns'] for tick, p in result.items()} != expected:
    raise ValueError('Predictions must cover each requested timestamp exactly once')
  return result


def summary(rows, names, pairs):
  result = {'samples': len(rows)}
  if not rows:
    return result
  result['models'] = {name: {metric: float(np.mean([r['metrics'][name][metric] for r in rows]))
                            for metric in ('ade_m', 'fde_m')} for name in rows[0]['metrics']}
  result['pairs'] = {}
  for a, b in pairs:
    key = f'{a}_vs_{b}'
    delta = np.array([r['pairs'][key]['ade_delta_m'] for r in rows])
    result['pairs'][key] = {
      'ade_delta_m': float(delta.mean()), 'median_ade_delta_m': float(np.median(delta)),
      'first_better': int(np.count_nonzero(delta < -1e-6)), 'second_better': int(np.count_nonzero(delta > 1e-6)),
      'equal': int(np.count_nonzero(np.abs(delta) <= 1e-6)),
      'reasoning_changed': sum(r['pairs'][key]['reasoning_changed'] for r in rows),
    }
  return result


def compare(output, reference, others=(), big_output=None):
  replay = replay_module()
  baseline = replay.read_json(reference/'manifest.json')
  baseline_hash = hashlib.sha256((reference/'manifest.json').read_bytes()).hexdigest()
  runs = {}
  for directory in [reference, output, *others]:
    manifest = replay.read_json(directory/'manifest.json')
    cameras = manifest.get('camera_subset', [0, 1, 2, 3, 5, 6])
    name = f'{len(cameras)}cam'
    if name in runs:
      raise ValueError('Use one run per camera configuration')
    if directory != reference and manifest.get('camera_subset_source_sha256') != baseline_hash:
      raise ValueError('Reference differs from the camera subset source')
    for key in ('route', 'sample_hz', 'capture_start_ns', 'capture_end_ns', 'future_times_s', 'rlog_sha256'):
      if manifest[key] != baseline[key]:
        raise ValueError(f'Experiment changes more than cameras: {key}')
    runs[name] = {'directory': directory, 'cameras': cameras, 'manifest': manifest,
                  'schedule': {r['tick']: r for r in manifest['schedule']},
                  'predictions': load_predictions(directory, manifest['schedule'])}
  names = sorted(runs)
  pairs = list(itertools.combinations(names, 2))
  big = None
  if big_output is not None:
    from scipy.spatial.transform import Rotation
    big_manifest = replay.read_json(big_output/'manifest.json')
    for key in ('route', 'capture_start_ns', 'rlog_sha256'):
      if big_manifest[key] != baseline[key]:
        raise ValueError(f'Big-model recording differs: {key}')
    big = np.load(big_output/'render-openpilot-big.npz', allow_pickle=False)
    if np.any(np.diff(big['ns']) <= 0):
      raise ValueError('Big-model timestamps must strictly increase')
    big_indices = {int(ns): i for i, ns in enumerate(big['ns'])}
  rows = []
  for old in baseline['schedule']:
    if 'sample' not in old:
      continue
    tick = old['tick']
    paths, reasons, views, token_ids = {}, {}, {}, {}
    for name, run in runs.items():
      row = run['schedule'][tick]
      for key in ('query_ns', 'navigation_text', 'ego_history_xyz', 'ego_history_rot', 'calibration_rpy',
                  'recorded_motion_future_xyz', 'small_model'):
        if row[key] != old[key]:
          raise ValueError(f'Non-camera input differs at tick {tick}: {key}')
      prediction = run['predictions'][tick]
      paths[name] = replay.predicted_xyz(prediction)
      reasons[name] = prediction['prediction']['reasoning']
      token_ids[name] = prediction['prediction']['generated_token_ids']
      views[name] = row['available_views']
    pos = old['small_model']['model']['position']
    paths['openpilot'] = np.stack([np.interp(baseline['future_times_s'], pos['t'], pos[a]) for a in 'xyz'], axis=-1)
    paths['openpilot'] *= np.array([1., -1., -1.])
    if big is not None:
      index = big_indices.get(old['query_ns'])
      if index is None:
        raise ValueError(f'No exact big-model frame at tick {tick}')
      axes = np.diag([1., -1., -1.])
      from_rotation = Rotation.from_euler('xyz', big['rpy'][index]).as_matrix()
      to_rotation = Rotation.from_euler('xyz', old['calibration_rpy']).as_matrix()
      paths['openpilot_big'] = big['xyz'][index] @ axes @ from_rotation.T @ to_rotation @ axes
      if paths['openpilot_big'].shape != (64, 3) or not np.isfinite(paths['openpilot_big']).all():
        raise ValueError('Invalid big-model trajectory')
    actual = np.asarray(old['recorded_motion_future_xyz'])
    metrics, differences = {}, {}
    for name, path in paths.items():
      error = np.linalg.norm(path[:, :2]-actual[:, :2], axis=-1)
      metrics[name] = {'ade_m': float(error.mean()), 'fde_m': float(error[-1])}
    for a, b in pairs:
      difference = paths[a]-paths[b]
      differences[f'{a}_vs_{b}'] = {
        'ade_delta_m': metrics[a]['ade_m']-metrics[b]['ade_m'], 'identical_camera_inputs': views[a] == views[b],
        'mean_path_difference_xy_m': float(np.linalg.norm(difference[:, :2], axis=-1).mean()),
        'max_coordinate_difference_m': float(np.abs(difference).max()), 'bit_exact_path': np.array_equal(paths[a], paths[b]),
        'reasoning_changed': reasons[a] != reasons[b], 'token_ids_equal': token_ids[a] == token_ids[b],
      }
    rows.append({'tick': tick, 'time_s': old['display_time_s'], 'query_ns': old['query_ns'],
                 'reference_views': old['available_views'], 'views': views, 'navigation': old['navigation_text'],
                 'metrics': metrics, 'pairs': differences, 'reasoning': reasons})
  turns = baseline['navigation_annotations']['turns']
  is_turn = lambda r: any(t['start_s']-6 <= r['time_s'] <= t['end_s'] for t in turns)
  groups = {'all': rows, 'side_views_available': [r for r in rows if len(r['reference_views']) > 2],
            'all_six_available': [r for r in rows if len(r['reference_views']) == 6],
            'turn_windows': [r for r in rows if is_turn(r)], 'outside_turn_windows': [r for r in rows if not is_turn(r)]}
  controls, examples = {}, {}
  for a, b in pairs:
    key = f'{a}_vs_{b}'
    selected = [r['pairs'][key] for r in rows if r['pairs'][key]['identical_camera_inputs']]
    controls[key] = {
      'samples': len(selected), 'bit_exact_paths': sum(r['bit_exact_path'] for r in selected),
      'equal_token_ids': sum(r['token_ids_equal'] for r in selected),
      'mean_path_difference_xy_m': float(np.mean([r['mean_path_difference_xy_m'] for r in selected])) if selected else None,
      'max_coordinate_difference_m': max((r['max_coordinate_difference_m'] for r in selected), default=None),
      'changed_reasoning': sum(r['reasoning_changed'] for r in selected),
    }
    ordered = sorted(rows, key=lambda r: r['pairs'][key]['ade_delta_m'])
    examples[key] = {'first_largest_improvements': ordered[:10], 'first_largest_regressions': ordered[-10:][::-1]}
  return {
    'outputs': {name: str(run['directory']) for name, run in runs.items()}, 'reference_output': str(reference),
    'sample_hz': baseline['sample_hz'], 'horizon_s': baseline['future_times_s'][-1],
    'reference_limit': baseline['actual_path_limit'],
    'comparison_limit': 'Paired offline predictions from one correlated drive with retrospective navigation; not closed-loop driving validation',
    'delta_definition': 'First configuration error minus second; negative favors the first',
    'controlled_inputs': 'Same retained camera pixels, timestamps, motion history, navigation, noise, checkpoint, precision, diffusion steps',
    'big_model': None if big is None else {'output': str(big_output), 'pre_roll_s': big_manifest['pre_roll_s'],
      'artifacts': big_manifest['artifacts'], 'image_policy': big_manifest['image_policy'],
      'comparison_limit': 'Big openpilot uses its normal inputs and continuous recurrent history; camera-only controls apply to Alpamayo runs.',
      'predictions_sha256': hashlib.sha256((big_output/'predictions.jsonl').read_bytes()).hexdigest()},
    'summary': {name: summary(selected, names, pairs) for name, selected in groups.items()},
    'unchanged_input_variability': controls,
    'turns': [{**turn, **summary([r for r in rows if turn['start_s']-6 <= r['time_s'] <= turn['end_s']], names, pairs)} for turn in turns],
    'examples': examples, 'samples': rows,
    'predictions_sha256': {name: hashlib.sha256((run['directory']/'predictions.jsonl').read_bytes()).hexdigest() for name, run in runs.items()},
  }


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--output', type=Path, required=True)
  parser.add_argument('--reference', type=Path, required=True)
  parser.add_argument('--other', type=Path, nargs='*', default=[])
  parser.add_argument('--big-model-output', type=Path)
  parser.add_argument('--recording-uuid')
  args = parser.parse_args()
  replay = replay_module()
  replay.check_mount(args.output, args.recording_uuid)
  result = compare(args.output, args.reference, args.other, args.big_model_output)
  replay.write_json(args.output/'camera-comparison.json', result)
  print(json.dumps({'summary': result['summary'], 'unchanged_input_variability': result['unchanged_input_variability']}, indent=2), flush=True)


if __name__ == '__main__':
  main()
