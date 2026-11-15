import ast
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

HERE = Path(__file__).parent
spec = importlib.util.spec_from_file_location('shadow_worker', HERE / 'shadow-worker.py')
worker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(worker)


class ShadowTest(unittest.TestCase):
  def fixture(self, path, *, yaw=0., times=None, lateral=0.):
    times = np.arange(100, 103.05, .05) if times is None else times
    query = 102.5
    metadata = {'query_ns': round(query*1e9), 'calibration_rpy': [0., 0., 0.],
                'camera_timestamps_ns': [round(query*1e9)]*10, 'navigation': 'Follow the road.'}
    np.savez(path, metadata=np.array(json.dumps(metadata)),
             image_frames=np.zeros((10, 1, 3, 4, 4), dtype=np.uint8),
             motion_times=np.round(times*1e9).astype(np.int64),
             motion_rpy=np.tile([0., 0., yaw], (len(times), 1)),
             motion_velocity=np.tile([10., lateral, 0.], (len(times), 1)))

  def test_history_uses_vehicle_axes_and_current_origin(self):
    with tempfile.TemporaryDirectory() as directory:
      path = Path(directory)/'input.npz'
      for yaw in (0., .7, -2.4):
        self.fixture(path, yaw=yaw, lateral=2.)
        data, _ = worker.sample(path)
        expected = np.stack((np.arange(-19, 1), -.2*np.arange(-19, 1), np.zeros(20)), axis=-1)
        np.testing.assert_allclose(data['ego_history_xyz'][0].numpy(), expected, atol=1e-5)
        np.testing.assert_allclose(data['ego_history_rot'][0].numpy(), np.tile(np.eye(3), (20, 1, 1)), atol=1e-6)

  def test_bad_measurement_history_rejected(self):
    with tempfile.TemporaryDirectory() as directory:
      path = Path(directory)/'input.npz'
      for times in (np.zeros(61), np.arange(102, 103, .05), np.r_[np.arange(100, 101, .05), np.arange(102, 103, .05)]):
        self.fixture(path, times=times)
        with self.assertRaises(ValueError):
          worker.sample(path)

  def test_disabled_model_never_probes_chestnut(self):
    source = HERE.parents[2] / 'selfdrive/modeld/modeld.py'
    tree = ast.parse(source.read_text())
    main = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'main')
    assignment = next(node for node in main.body if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'CHESTNUT' for t in node.targets))
    expr = compile(ast.Expression(assignment.value), str(source), 'eval')
    def forbidden():
      self.fail('Disabled model attempted a Chestnut probe')
    with patch.dict(os.environ, {'DISABLE_CHESTNUT': '1'}):
      self.assertFalse(eval(expr, {'os': os, 'chestnut_compiled': forbidden}))
    with patch.dict(os.environ, {'DISABLE_CHESTNUT': '0'}):
      self.assertTrue(eval(expr, {'os': os, 'chestnut_compiled': lambda: True,
                                  'ASIUS_HARDWARE': True, 'chestnut_present': lambda **kwargs: True}))


if __name__ == '__main__':
  unittest.main()
