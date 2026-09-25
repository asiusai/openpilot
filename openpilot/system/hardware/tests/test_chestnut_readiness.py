from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from openpilot.common.parameterized import parameterized
from openpilot.selfdrive.modeld import helpers


class TestChestnutReadiness(unittest.TestCase):
  def setUp(self):
    root = Path(self.enterContext(TemporaryDirectory()))
    self.enterContext(patch.object(helpers, "USB_DEVICES_PATH", root))
    self.device = root / "2-1"
    self.device.mkdir()
    for name, value in {"idVendor": "3801", "idProduct": "0001", "product": helpers.CHESTNUT_USB_PRODUCT, "speed": "480"}.items():
      (self.device / name).write_text(value)
    self.now = 0.
    self.waits = 0
    self.enterContext(patch.object(helpers, "time", SimpleNamespace(monotonic=lambda: self.now, sleep=self.sleep)))

  def sleep(self, seconds):
    self.now += seconds
    self.waits += 1

  @parameterized.expand([("480", False), ("5000", True), ("10000", True), ("12", False)])
  def test_readiness_requires_superspeed(self, speed, ready):
    (self.device / "speed").write_text(speed)
    self.assertEqual(helpers.chestnut_present(), ready)

  def test_wait_survives_firmware_disconnect(self):
    def sleep(seconds):
      self.sleep(seconds)
      if self.waits == 1:
        (self.device / "idVendor").unlink()
      else:
        (self.device / "idVendor").write_text("3801")
        (self.device / "speed").write_text("5000")

    with patch.object(helpers.time, "sleep", sleep):
      self.assertTrue(helpers.chestnut_present(timeout=45.))
    self.assertEqual(self.waits, 2)

  def test_usb2_wait_is_bounded(self):
    self.assertFalse(helpers.chestnut_present(timeout=1.))
    self.assertGreaterEqual(self.now, 1.)
    self.assertLess(self.now, 1.2)

  @parameterized.expand([None, "custom ed4e39b7-CLEAN"])
  def test_absent_or_other_firmware_does_not_delay_startup(self, product):
    if product is None:
      (self.device / "product").unlink()
    else:
      (self.device / "product").write_text(product)
    self.assertFalse(helpers.chestnut_present(timeout=45.))
    self.assertEqual(self.waits, 0)
