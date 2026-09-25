import pytest

from openpilot.common.hardware.usb import CHESTNUT_FW_VERSION, CHESTNUT_USB_PRODUCT
from openpilot.system.hardware.chestnut import flash


def test_bundled_firmware_matches_device_selection():
  image = flash.FIRMWARE_PATH.read_bytes()
  flash.validate_image(image)
  assert flash.image_product(image) == CHESTNUT_USB_PRODUCT


@pytest.mark.parametrize("image", [
  b"custom ed4e39b7-quiet1",
  b"custom ed4e39b7-quiet1\0custom ed4e39b7-CLEAN\0",
  b"custom not-a-version\0",
])
def test_reject_ambiguous_or_incomplete_product(image):
  with pytest.raises(ValueError, match="expected one product"):
    flash.image_product(image)


def test_corrupt_image_rejected():
  image = bytearray(flash.FIRMWARE_PATH.read_bytes())
  image[100] ^= 1
  with pytest.raises(ValueError, match="checksum"):
    flash.validate_image(image)


def test_same_commit_different_build_rejected_before_usb_access(monkeypatch):
  def unexpected_usb_access():
    pytest.fail("version mismatch must be rejected before accessing USB")

  monkeypatch.setattr(flash, "find_chestnut", unexpected_usb_access)
  with pytest.raises(RuntimeError, match="expected version"):
    flash.flash_chestnut(expected_version="ed4e39b7-CLEAN")


def test_matching_device_does_not_reflash(monkeypatch, capsys):
  monkeypatch.setattr(flash, "find_chestnut", lambda: ("unused", ("3801", "0001"), CHESTNUT_USB_PRODUCT))

  def unexpected_flash():
    pytest.fail("matching device must not be reflashed")

  monkeypatch.setattr(flash, "Flash", unexpected_flash)
  flash.flash_chestnut(expected_version=CHESTNUT_FW_VERSION)
  assert "firmware is up to date" in capsys.readouterr().out
