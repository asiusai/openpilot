from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from openpilot.selfdrive.alpamayo import compare as C


def prediction(speed=2., yaw=0.):
  xyz = np.zeros((64, 3))
  xyz[:, 0] = np.arange(1, 65)*.1*speed
  angle = np.radians(yaw)
  rot = np.array([[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
  return {'xyz': xyz.tolist(), 'rot': np.tile(rot, (64, 1, 1)).tolist(), 'reasoning': '<NOMINAL>', 'generated_token_ids': [1, 2]}


def replay(root='first'):
  config = {'kind': 'configuration', 'warmup': 1, 'runs': 1, 'trajectory': {'dt_s': .1, 'steps': 64},
            'identities': dict.fromkeys(C.IDENTITY_FIELDS, 'a'*64)}
  config['identities']['samples'] = {}
  rows = []
  for i in range(2):
    name = f'{root}/sample-{i}'
    config['identities']['samples'][name] = {'sha256': dict.fromkeys(C.FIELDS, 'b'*64),
      'frame_references': {'query_ns': (i+1)*100_000_000, 'frames': [{'view': 1, 'capture_ns': i+1}]}}
    rows.append({'kind': 'prediction', 'index': i, 'warmup': i == 0, 'sample': name, 'prediction': prediction()})
  return config, rows


class ComparisonTests(unittest.TestCase):
  def test_identical_and_shapes(self):
    p = prediction()
    result = C.compare_predictions(p, p)
    self.assertEqual(result['result'], 'PASS')
    self.assertEqual(set(result['metrics'].values()), {0.})
    self.assertAlmostEqual(result['horizon_s'], 6.4)
    self.assertEqual(len(result['curves']['actual_speed_m_s']), 63)
    self.assertEqual(len(result['curves']['actual_acceleration_m_s2']), 62)
    wrapped = dict(p, xyz=np.array(p['xyz'])[None, None].tolist(), rot=np.array(p['rot'])[None].tolist())
    self.assertEqual(C.compare_predictions(wrapped, p)['result'], 'PASS')
    for key, bad in [('xyz', np.zeros((32, 6))), ('xyz', np.zeros((2, 64, 3))),
                     ('rot', np.zeros((64, 9))), ('rot', np.full((64, 3, 3), np.nan)), ('xyz', np.full((64, 3), np.inf))]:
      with self.subTest(key=key, shape=bad.shape):
        self.assertFalse(C.compare_predictions(dict(p, **{key: bad}), p)['valid'])

  def test_position_and_thresholds(self):
    p, q = prediction(), prediction()
    for xyz in p['xyz']:
      xyz[1], xyz[2] = .3, 99
    result = C.compare_predictions(p, q)
    self.assertEqual(result['exceeded'], ['ade_m'])
    for key in ('ade_m', 'fde_m', 'max_point_distance_m'):
      self.assertAlmostEqual(result['metrics'][key], .3)
    self.assertEqual(C.compare_predictions(p, q, {'ade_m': .3})['result'], 'PASS')
    for limits in ({'typo': 1}, {'ade_m': float('nan')}, {'fde_m': -1}):
      with self.assertRaises(ValueError):
        C.compare_predictions(p, q, limits)

  def test_speed_acceleration_and_dt(self):
    p, q = prediction(3.), prediction(2.)
    self.assertAlmostEqual(C.compare_predictions(p, q)['metrics']['speed_mae_m_s'], 1.)
    self.assertAlmostEqual(C.compare_predictions(p, q, dt=.2)['metrics']['speed_mae_m_s'], .5)
    p = prediction(0)
    for i, xyz in enumerate(p['xyz']):
      xyz[0] = .5*((i+1)*.1)**2
    result = C.compare_predictions(p, prediction(0))
    np.testing.assert_allclose(result['curves']['actual_acceleration_m_s2'], 1., atol=1e-11)
    self.assertAlmostEqual(result['metrics']['acceleration_mae_m_s2'], 1.)
    model = {'expert_config': {'action_space_cfg': {'n_waypoints': 64}}}
    self.assertEqual(C.trajectory_config(model), {'dt_s': .1, 'steps': 64})
    model['expert_config']['action_space_cfg']['dt'] = .2
    self.assertEqual(C.trajectory_config(model)['dt_s'], .2)

  def test_heading_wrap_and_text_separation(self):
    p, q = prediction(yaw=179.5), prediction(yaw=-179.5)
    p.update(reasoning='changed', generated_token_ids=[3])
    result = C.compare_predictions(p, q)
    self.assertEqual(result['result'], 'PASS')
    self.assertAlmostEqual(result['metrics']['max_heading_deg'], 1.)
    self.assertTrue(result['reasoning_mismatch'])
    self.assertTrue(result['token_mismatch'])
    self.assertIn('max_heading_deg', C.compare_predictions(prediction(yaw=3), prediction())['exceeded'])

  def test_reference_roundtrip_and_relocation(self):
    config, rows = replay()
    with tempfile.TemporaryDirectory() as folder:
      path = Path(folder)/'predictions.jsonl'
      completed = {'kind': 'summary', 'gpu_executed': True}
      path.write_text('\n'.join(json.dumps(e) for e in [config, *rows, completed]))
      loaded = C.load_reference(folder)
      self.assertEqual(loaded, (config, rows))
      self.assertEqual(C.compare_replays(*replay('relocated'), *loaded)['result'], 'PASS')
      for events in ([config, *rows, rows[1], completed], [config, rows[0], completed], [config, config, *rows, completed],
                     [config, *rows, {'kind': 'failure'}], [config, *rows], [config, *rows, {'kind': 'summary', 'gpu_executed': False}]):
        path.write_text('\n'.join(json.dumps(e) for e in events))
        with self.assertRaises(ValueError):
          C.load_reference(path)

  def test_identity_rejections(self):
    expected = replay()
    for field in C.IDENTITY_FIELDS:
      actual = deepcopy(expected)
      actual[0]['identities'][field] = 'c'*64
      with self.assertRaises(ValueError):
        C.compare_replays(*actual, *expected)
    for kind in ('hash', 'query', 'camera', 'order', 'warmup', 'missing', 'duplicate', 'dt', 'failure'):
      config, rows = deepcopy(expected)
      sample = config['identities']['samples'][rows[1]['sample']]
      if kind == 'hash':
        sample['sha256']['nav_text'] = 'd'*64
      if kind == 'query':
        sample['frame_references']['query_ns'] += 1
      if kind == 'camera':
        sample['frame_references']['frames'][0]['view'] = 6
      if kind == 'order':
        rows.reverse()
      if kind == 'warmup':
        rows[0]['warmup'] = False
      if kind == 'missing':
        rows.pop()
      if kind == 'duplicate':
        rows[1]['sample'] = rows[0]['sample']
      if kind == 'dt':
        config['trajectory']['dt_s'] = .2
      if kind == 'failure':
        rows[1]['error'] = 'inference failed'
      with self.subTest(kind=kind), self.assertRaises(ValueError):
        C.compare_replays(config, rows, *expected)

  def test_warmup_and_failed_sample_handling(self):
    config, rows = replay()
    rows[0]['prediction'] = prediction(100)
    result = C.compare_replays(config, rows, *replay())
    self.assertEqual(result['result'], 'PASS')
    self.assertEqual(result['samples'][0]['result'], 'FAIL')
    rows[0]['prediction']['xyz'][0][0] = float('nan')
    self.assertEqual(C.compare_replays(config, rows, *replay())['result'], 'FAIL')
    rows[0]['prediction'] = prediction()
    rows[1]['prediction'] = prediction(100)
    self.assertEqual(C.compare_replays(config, rows, *replay())['result'], 'FAIL')

  def test_every_prediction_and_elapsed_throughput_must_pass(self):
    rows = [{'warmup': True, 'full_pipeline_ms': 10000.}, {'warmup': False, 'full_pipeline_ms': 100.}]
    self.assertEqual(C.performance_gate(rows, .1)['result'], 'PASS')
    self.assertEqual(C.performance_gate(rows, .10001)['result'], 'FAIL')
    rows.append({'warmup': False, 'full_pipeline_ms': 101.})
    self.assertEqual(C.performance_gate(rows, .1)['result'], 'FAIL')
    rows[-1]['full_pipeline_ms'] = float('nan')
    self.assertEqual(C.performance_gate(rows, .1)['result'], 'FAIL')

  def test_timing_statistics_exclude_warmup(self):
    rows = [{'warmup': warmup, 'full_pipeline_ms': total,
             'prediction': {'timings_ms': {'vision': total/4, 'prefill': total/4, 'reasoning': total/8, 'diffusion': total/8}}}
            for warmup, total in ((True, 10000.), (False, 60.), (False, 80.))]
    result = C.performance_gate(rows, .18)
    self.assertEqual(result['result'], 'PASS')
    self.assertEqual(result['full_pipeline_ms'], {'mean': 70., 'p95': 79., 'max': 80.})
    self.assertEqual(result['model_execution_ms']['mean'], 52.5)
    self.assertEqual(result['stage_ms']['vision']['mean'], 17.5)

  def test_half_second_deadline_does_not_claim_ten_hz(self):
    rows = [{'warmup': True, 'full_pipeline_ms': 10000.}, {'warmup': False, 'full_pipeline_ms': 392.}]
    result = C.performance_gate(rows, .4, 500)
    self.assertEqual((result['result'], result['limit_ms'], result['minimum_hz']), ('PASS', 500, 2.))
    self.assertEqual(C.performance_gate(rows, .4, 100)['result'], 'FAIL')
    self.assertEqual(C.performance_gate(rows, .4, 250)['result'], 'FAIL')
    self.assertEqual(C.performance_gate(rows, .501, 500)['result'], 'FAIL')
    rows[-1]['full_pipeline_ms'] = 501.
    self.assertEqual(C.performance_gate(rows, .4, 500)['result'], 'FAIL')
    with self.assertRaises(ValueError):
      C.performance_gate(rows, .4, 0)

  def test_default_reference_and_qualification_are_required(self):
    recipe = {'variant': 'default', 'precision': 'bf16', 'diffusion_steps': 10, 'max_tokens': 256, 'greedy': True}
    C.validate_reference({'runtime': recipe})
    C.validate_reference({'variant': 'default', 'recipe': recipe})
    for field, value in (('variant', 'selected'), ('precision', 'fp8'), ('diffusion_steps', 3), ('max_tokens', 64), ('greedy', False)):
      with self.subTest(field=field), self.assertRaises(ValueError):
        C.validate_reference({'runtime': dict(recipe, **{field: value})})
    self.assertEqual(C.qualification({'result': 'PASS'}, None)['result'], 'FAIL')
    self.assertEqual(C.qualification({'result': 'FAIL'}, {'result': 'PASS'})['result'], 'FAIL')
    self.assertEqual(C.qualification({'result': 'PASS'}, {'result': 'FAIL'})['result'], 'FAIL')
    self.assertEqual(C.qualification({'result': 'PASS'}, {'result': 'PASS'})['result'], 'PASS')


if __name__ == '__main__':
  unittest.main()
