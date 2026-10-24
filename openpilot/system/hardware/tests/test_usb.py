import pytest

from openpilot.cereal import messaging
from openpilot.common.hardware import usb


@pytest.fixture
def topology(tmp_path, monkeypatch):
  devices = tmp_path / 'usb'
  devices.mkdir()
  monkeypatch.setattr(usb, 'USB_DEVICES_PATH', devices)
  monkeypatch.setattr(usb, 'TYPEC_CC_ORIENTATION_PATH', tmp_path / 'cc')
  monkeypatch.setattr(usb, 'ASIUS_HARDWARE', True)

  def add(name, controller='a800000.usb', vid=0x1D6B, pid=0x0002):
    path = tmp_path / controller / 'host' / name
    path.mkdir(parents=True)
    for field, value in {'idVendor': f'{vid:04x}', 'idProduct': f'{pid:04x}', 'busnum': '1', 'devnum': '2', 'speed': '480'}.items():
      (path / field).write_text(value)
    (devices / name).symlink_to(path, target_is_directory=True)
    return path

  return add


def state():
  ds = messaging.new_message('deviceState').deviceState
  usb.set_usb_state(ds, usb.get_usb_state())
  return ds


def test_onboard_hubs_and_radio_do_not_indicate_an_external_device(topology):
  topology('usb1')
  topology('usb2', controller='a600000.usb')
  topology('usb3', controller='a600000.usb')
  topology('1-1', vid=0x1A40, pid=0x0101)
  topology('1-1.4', vid=0xA69C, pid=0x8D81)
  topology('1-1.4.1')
  assert not state().usbState.connected
  assert not state().chestnutPresent


@pytest.mark.parametrize('ctrl', ['a600000.usb', 'a600000.ssusb'])
@pytest.mark.parametrize('name', ['2-1', '3-1', '3-1.2'])
def test_primary_usb2_and_usb3_devices_and_external_hubs_count(topology, ctrl, name):
  path = topology(name, controller=ctrl, vid=0x3801, pid=0x0001)
  assert usb.controller(path).name == ctrl
  assert state().usbState.connected
  assert state().chestnutPresent
  (path / 'idVendor').unlink()
  assert not state().usbState.connected


def test_second_dragon_port_counts_peripherals_but_not_just_its_hub(topology):
  topology('1-1', vid=0x1A40, pid=0x0101)
  assert not state().usbState.connected
  topology('1-1.2', vid=0x04E8, pid=0x4001)
  assert state().usbState.connected
  assert not state().chestnutPresent


def test_comma_cable_detection_does_not_require_enumeration(topology, monkeypatch):
  monkeypatch.setattr(usb, 'ASIUS_HARDWARE', False)
  usb.TYPEC_CC_ORIENTATION_PATH.write_text('1')
  assert state().usbState.connected
  usb.TYPEC_CC_ORIENTATION_PATH.write_text('0')
  assert not state().usbState.connected
