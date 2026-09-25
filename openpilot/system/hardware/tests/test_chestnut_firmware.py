import unittest
from unittest.mock import patch

from openpilot.common.hardware.usb import CHESTNUT_FW_VERSION, CHESTNUT_USB_PRODUCT
from openpilot.common.parameterized import parameterized
from openpilot.system.hardware.chestnut import flash


class TestChestnutFirmware(unittest.TestCase):
  def test_bundled_firmware_matches_device_selection(self):
    image = flash.FIRMWARE_PATH.read_bytes()
    flash.validate_image(image)
    self.assertEqual(flash.image_product(image), CHESTNUT_USB_PRODUCT)

  @parameterized.expand([
    b"custom ed4e39b7-quiet1",
    b"custom ed4e39b7-quiet1\0custom ed4e39b7-CLEAN\0",
    b"custom not-a-version\0",
  ])
  def test_reject_ambiguous_or_incomplete_product(self, image):
    with self.assertRaisesRegex(ValueError, "expected one product"):
      flash.image_product(image)

  def test_corrupt_image_rejected(self):
    image = bytearray(flash.FIRMWARE_PATH.read_bytes())
    image[100] ^= 1
    with self.assertRaisesRegex(ValueError, "checksum"):
      flash.validate_image(image)

  def test_same_commit_different_build_rejected_before_usb_access(self):
    with patch.object(flash, "find_chestnut") as find:
      with self.assertRaisesRegex(RuntimeError, "expected version"):
        flash.flash_chestnut(expected_version="ed4e39b7-CLEAN")
      find.assert_not_called()

  def test_matching_device_does_not_reflash(self):
    with patch.object(flash, "find_chestnut", return_value=("unused", ("3801", "0001"), CHESTNUT_USB_PRODUCT)), \
         patch.object(flash, "Flash") as writer:
      flash.flash_chestnut(expected_version=CHESTNUT_FW_VERSION)
      writer.assert_not_called()
