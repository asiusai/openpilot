"""CPU regression against saved outputs. Agreement does not establish driving accuracy."""
import json
from pathlib import Path

import numpy as np

DEFAULT_THRESHOLDS = {'ade_m': .25, 'fde_m': .5, 'speed_mae_m_s': .25, 'acceleration_mae_m_s2': .5, 'max_heading_deg': 2.}
FIELDS = ('image_frames', 'camera_indices', 'absolute_timestamps', 'ego_history_xyz', 'ego_history_rot', 'nav_text')
IDENTITY_FIELDS = ('initial_noise_sha256', 'checkpoint_config_sha256', 'checkpoint_weight_index_sha256', 'checkpoint_weights_sha256')


def trajectory_config(model_config):
  action = model_config['expert_config']['action_space_cfg']
  dt, steps = float(action.get('dt', .1)), action.get('n_waypoints', 64)
  if not np.isfinite(dt) or dt <= 0 or steps != 64:
    raise ValueError('Require positive finite trajectory dt and 64 waypoints')
  return {'dt_s': dt, 'steps': steps}


def _prediction(value):
  arrays = []
  for key, shape in (('xyz', (64, 3)), ('rot', (64, 3, 3))):
    array = np.asarray(value[key], dtype=np.float64)
    if array.ndim < len(shape) or array.shape[-len(shape):] != shape or any(n != 1 for n in array.shape[:-len(shape)]):
      raise ValueError(f'{key}: expected {shape} with optional singleton leading axes')
    if not np.isfinite(array).all():
      raise ValueError(f'{key}: nonfinite values')
    arrays.append(array.reshape(shape))
  if not isinstance(value.get('reasoning'), str):
    raise ValueError('Missing reasoning string')
  ids = value.get('generated_token_ids')
  if not isinstance(ids, list) or not ids or any(type(x) is not int or x < 0 for x in ids):
    raise ValueError('Missing or invalid generated_token_ids')
  return arrays


def compare_predictions(actual, expected, thresholds=None, *, dt=.1):
  limits = dict(DEFAULT_THRESHOLDS)
  if thresholds is not None:
    if set(thresholds)-set(limits)-{'max_point_distance_m'}:
      raise ValueError('Unknown regression threshold')
    limits.update(thresholds)
  if not np.isfinite(dt) or dt <= 0 or any(not np.isfinite(v) or v < 0 for v in limits.values()):
    raise ValueError('Require finite positive dt and finite nonnegative thresholds')
  result = {'result': 'FAIL', 'valid': False, 'thresholds': limits, 'errors': []}
  try:
    (a, ar), (b, br) = _prediction(actual), _prediction(expected)
    with np.errstate(over='raise', invalid='raise', divide='raise'):
      distance = np.linalg.norm(a[:, :2]-b[:, :2], axis=1)
      yaw = np.arctan2(ar[:, 1, 0], ar[:, 0, 0])-np.arctan2(br[:, 1, 0], br[:, 0, 0])
      heading = np.abs(np.degrees(np.arctan2(np.sin(yaw), np.cos(yaw))))
      speeds = [np.linalg.norm(np.diff(x[:, :2], axis=0), axis=1)/dt for x in (a, b)]
      accels = [np.diff(x)/dt for x in speeds]
      metrics = {'ade_m': float(distance.mean()), 'fde_m': float(distance[-1]), 'max_point_distance_m': float(distance.max()),
                 'max_heading_deg': float(heading.max()), 'speed_mae_m_s': float(np.abs(speeds[0]-speeds[1]).mean()),
                 'acceleration_mae_m_s2': float(np.abs(accels[0]-accels[1]).mean())}
    if not all(np.isfinite(x) for x in metrics.values()):
      raise ValueError('Nonfinite derived metrics')
    exceeded = [key for key, limit in limits.items() if metrics[key] > limit]
    result.update(valid=True, result='FAIL' if exceeded else 'PASS', metrics=metrics, exceeded=exceeded,
                  dt_s=dt, horizon_s=64*dt, reasoning_mismatch=actual['reasoning'] != expected['reasoning'],
                  token_mismatch=actual['generated_token_ids'] != expected['generated_token_ids'],
                  curves={'speed_time_s': ((np.arange(63)+1.5)*dt).tolist(), 'acceleration_time_s': ((np.arange(62)+2)*dt).tolist(),
                          'actual_speed_m_s': speeds[0].tolist(), 'expected_speed_m_s': speeds[1].tolist(),
                          'actual_acceleration_m_s2': accels[0].tolist(), 'expected_acceleration_m_s2': accels[1].tolist()})
  except (KeyError, TypeError, ValueError, OverflowError, FloatingPointError) as error:
    result['errors'].append(str(error))
  return result


def _digest(value):
  if not isinstance(value, str) or len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
    raise ValueError('Require SHA256 identities')
  return value


def _sequence(config, rows):
  if type(config.get('warmup')) is not int or type(config.get('runs')) is not int or config['warmup'] < 0 or config['runs'] < 1:
    raise ValueError('Require explicit warmup and measured counts')
  if len(rows) != config['warmup']+config['runs']:
    raise ValueError('Missing or extra predictions, including warmup sequence')
  identities = config['identities']
  model = tuple(_digest(identities[k]) for k in IDENTITY_FIELDS)
  trajectory = config['trajectory']
  if trajectory['steps'] != 64 or not np.isfinite(trajectory['dt_s']) or trajectory['dt_s'] <= 0:
    raise ValueError('Invalid trajectory configuration')
  keys = []
  for index, row in enumerate(rows):
    if type(row.get('index')) is not int or row['index'] != index or row.get('warmup') is not (index < config['warmup']):
      raise ValueError('Duplicate, reordered, or incompatible prediction sequence')
    if row.get('kind') != 'prediction' or 'prediction' not in row or row.get('error') or row.get('result') in ('FAIL', 'ERROR'):
      raise ValueError(f'Failed prediction at index {index}')
    sample = identities['samples'][row['sample']]
    hashes = {key: _digest(sample['sha256'][key]) for key in FIELDS}
    refs = sample['frame_references']
    if type(refs.get('query_ns')) is not int or refs['query_ns'] <= 0 or not isinstance(refs.get('frames'), list) or not refs['frames']:
      raise ValueError('Missing camera references/query_ns')
    keys.append(json.dumps([hashes, refs], sort_keys=True, allow_nan=False))
  if len(set(keys)) != len(keys):
    raise ValueError('Duplicate reference sample identities')
  return model, trajectory, keys


def load_reference(path):
  path = Path(path)
  if path.is_dir():
    path /= 'predictions.jsonl'
  events = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
  if not events or events[0].get('kind') != 'configuration' or sum(e.get('kind') == 'configuration' for e in events) != 1:
    raise ValueError('Require one initial reference configuration')
  if any(e.get('kind') not in ('configuration', 'initialized', 'prediction', 'summary') for e in events):
    raise ValueError('Unexpected or failed reference event')
  if events[-1].get('kind') != 'summary' or sum(e.get('kind') == 'summary' for e in events) != 1:
    raise ValueError('Reference must have one completed final summary')
  if events[-1].get('gpu_executed') is not True:
    raise ValueError('Reference must contain actual GPU predictions')
  config, rows = events[0], [e for e in events if e.get('kind') == 'prediction']
  _sequence(config, rows)
  for row in rows:
    _prediction(row['prediction'])
  return config, rows


def validate_reference(config):
  recipe = config.get('runtime', config.get('recipe', {}))
  variant = config.get('variant', recipe.get('variant'))
  required = {'precision': 'bf16', 'diffusion_steps': 10, 'max_tokens': 256, 'greedy': True}
  if variant != 'default' or any(recipe.get(key) != value for key, value in required.items()):
    raise ValueError('Use the saved default BF16, ten-step, full-reasoning reference')


def compare_replays(actual_config, actual_rows, expected_config, expected_rows, thresholds=None):
  actual, expected = _sequence(actual_config, actual_rows), _sequence(expected_config, expected_rows)
  if actual != expected or actual_config['warmup'] != expected_config['warmup']:
    raise ValueError('Incompatible input sequence, noise, checkpoint identity, dt, or warmup count')
  samples = [dict(index=a['index'], warmup=a['warmup'], sample=a['sample'],
                  **compare_predictions(a['prediction'], b['prediction'], thresholds, dt=actual[1]['dt_s']))
             for a, b in zip(actual_rows, expected_rows, strict=True)]
  measured = [s for s in samples if not s['warmup']]
  return {'result': 'PASS' if all(s['valid'] for s in samples) and all(s['result'] == 'PASS' for s in measured) else 'FAIL',
          'compared': len(samples), 'measured': len(measured), 'samples': samples,
          'measured_reasoning_mismatches': sum(s.get('reasoning_mismatch', False) for s in measured),
          'measured_token_mismatches': sum(s.get('token_mismatch', False) for s in measured),
          'scope': 'Saved default-model regression; agreement does not establish driving accuracy or safety'}


def performance_gate(rows, elapsed_s, period_ms=100):
  if period_ms not in (100, 250, 500):
    raise ValueError('Use a 100, 250, or 500 ms period')
  measured = [r for r in rows if not r['warmup']]
  times = [r['full_pipeline_ms'] for r in measured]
  valid = bool(times) and np.isfinite(elapsed_s) and elapsed_s > 0 and all(np.isfinite(t) and t > 0 for t in times)
  hz = len(times)/elapsed_s if valid else 0.
  def stats(values):
    if not values or not all(np.isfinite(v) and v >= 0 for v in values):
      return None
    return {'mean': float(np.mean(values)), 'p95': float(np.percentile(values, 95)), 'max': float(max(values))}
  stages = [r.get('prediction', {}).get('timings_ms', {}) for r in measured]
  common = set.intersection(*(set(s) for s in stages)) if stages else set()
  return {'result': 'PASS' if valid and max(times) <= period_ms and hz >= 1000/period_ms else 'FAIL', 'elapsed_hz': hz,
          'maximum_full_pipeline_ms': max(times) if valid else None, 'limit_ms': period_ms, 'minimum_hz': 1000/period_ms,
          'full_pipeline_ms': stats(times), 'model_execution_ms': stats([r['inference_ms'] for r in measured]),
          'stage_ms': {key: stats([s[key] for s in stages]) for key in sorted(common)}}


def qualification(performance, regression):
  accuracy = regression['result'] if regression is not None else 'REFERENCE SAVED'
  return {'result': 'PASS' if performance['result'] == accuracy == 'PASS' else 'FAIL',
          'speed_result': performance['result'], 'accuracy_result': accuracy}
