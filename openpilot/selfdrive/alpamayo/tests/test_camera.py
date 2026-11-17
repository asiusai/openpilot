import tempfile
import unittest
from functools import partial

import numpy as np

from openpilot.selfdrive.alpamayo.frontend import Preparer
from openpilot.selfdrive.alpamayo.geometry import calibrated_image
from openpilot.selfdrive.alpamayo.selected.camera import calibrated_image as native_image
from openpilot.selfdrive.alpamayo.tests.test_frontend import BASE, MS, geometry, populated


class CameraTests(unittest.TestCase):
  def test_native_sampler_preserves_pixels_and_calibration(self):
    pixels = np.random.default_rng(42).integers(0, 256, (3, 96, 160), dtype=np.uint8)
    with tempfile.TemporaryDirectory() as cache:
      for camera in (1, 6):
        for rotation in ([0., 0., 0.], [.01, -.02, .03]):
          calibration = {'rpyCalib': rotation, 'wideFromDeviceEuler': [.003, -.004, .002]}
          expected = calibrated_image(pixels, camera, geometry(), calibration)
          actual = native_image(pixels, camera, geometry(), calibration, cache_dir=cache)
          np.testing.assert_array_equal(actual, expected)

  def test_preparer_uses_sampler_without_changing_history(self):
    selected = populated().select(BASE+3015*MS)
    expected, metadata = Preparer(geometry()).prepare(selected)
    with tempfile.TemporaryDirectory() as cache:
      preparer = Preparer(geometry(), sampler=partial(native_image, cache_dir=cache))
      actual, native_metadata = preparer.prepare(selected)
      for key in expected:
        np.testing.assert_array_equal(actual[key], expected[key])
      self.assertEqual(native_metadata, metadata)
      self.assertEqual(preparer.prepare(selected)[1]['image_cache_hits'], 10)


if __name__ == '__main__':
  unittest.main()
