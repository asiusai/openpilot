from types import SimpleNamespace

import pytest

from openpilot.selfdrive.modeld import helpers


@pytest.fixture
def usb_device(tmp_path, monkeypatch):
  monkeypatch.setattr(helpers, "USB_DEVICES_PATH", tmp_path)
  device = tmp_path / "2-1"
  device.mkdir()
  for name, value in {"idVendor": "3801", "idProduct": "0001", "product": helpers.CHESTNUT_USB_PRODUCT, "speed": "480"}.items():
    (device / name).write_text(value)
  return device


@pytest.mark.parametrize("speed,ready", [("480", False), ("5000", True), ("10000", True), ("12", False)])
def test_readiness_requires_superspeed(usb_device, speed, ready):
  (usb_device / "speed").write_text(speed)
  assert helpers.chestnut_present() == ready


def test_wait_survives_firmware_disconnect(usb_device, monkeypatch):
  now = 0.
  waits = 0

  def sleep(seconds):
    nonlocal now, waits
    now += seconds
    waits += 1
    if waits == 1:
      (usb_device / "idVendor").unlink()
    else:
      (usb_device / "idVendor").write_text("3801")
      (usb_device / "speed").write_text("5000")

  monkeypatch.setattr(helpers, "time", SimpleNamespace(monotonic=lambda: now, sleep=sleep))
  assert helpers.chestnut_present(timeout=45.)
  assert waits == 2


def test_usb2_wait_is_bounded(usb_device, monkeypatch):
  now = 0.

  def sleep(seconds):
    nonlocal now
    now += seconds

  monkeypatch.setattr(helpers, "time", SimpleNamespace(monotonic=lambda: now, sleep=sleep))
  assert not helpers.chestnut_present(timeout=1.)
  assert 1. <= now < 1.2


@pytest.mark.parametrize("product", [None, "custom ed4e39b7-CLEAN"])
def test_absent_or_other_firmware_does_not_delay_startup(usb_device, monkeypatch, product):
  if product is None:
    (usb_device / "product").unlink()
  else:
    (usb_device / "product").write_text(product)

  def unexpected_sleep(_):
    pytest.fail("only a detected matching Chestnut may delay startup")

  monkeypatch.setattr(helpers, "time", SimpleNamespace(monotonic=lambda: 0., sleep=unexpected_sleep))
  assert not helpers.chestnut_present(timeout=45.)
