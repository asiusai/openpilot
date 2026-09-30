import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from openpilot.common.params import Params, ParamKeyFlag
from openpilot.system.asius import methods
from openpilot.system.asius.bluetoothd import Advertisement, keep_advertising


def test_device_name_is_persistent_and_shared(tmp_path, monkeypatch):
  monkeypatch.setattr(methods, 'Params', lambda: Params(str(tmp_path)))
  assert methods.getDeviceName() == 'Asius v0'
  assert methods.setDeviceName('  My car  ') == {'name': 'My car'}
  params = Params(str(tmp_path))
  for flag in (ParamKeyFlag.CLEAR_ON_MANAGER_START, ParamKeyFlag.CLEAR_ON_ONROAD_TRANSITION,
               ParamKeyFlag.CLEAR_ON_OFFROAD_TRANSITION):
    params.clear_all(flag)
  assert params.get('DeviceName') == 'My car'
  assert methods.getDeviceName() == Advertisement().LocalName == 'My car'
  for name in ('', '   ', 'a' * 41, 'Car\nName'):
    with pytest.raises(ValueError):
      methods.setDeviceName(name)
  assert methods.getDeviceName() == 'My car'


def test_rename_refreshes_bluetooth_alias_and_advertisement():
  stop = MagicMock()
  stop.is_set.side_effect = [False, False, True]
  stop.wait = AsyncMock()
  bus = MagicMock()
  with patch('openpilot.system.asius.bluetoothd.connected_device_count', new=AsyncMock(return_value=0)), \
       patch('openpilot.system.asius.bluetoothd.pairing_mode_active', return_value=True), \
       patch.object(methods, 'getDeviceName', side_effect=['Old name', 'New name']), \
       patch('openpilot.system.asius.bluetoothd.set_adapter_property', new=AsyncMock()) as set_property, \
       patch('openpilot.system.asius.bluetoothd.refresh_advertisement', new=AsyncMock()) as refresh:
    asyncio.run(keep_advertising(bus, '/adapter', stop))
  assert [call.args[3].value for call in set_property.call_args_list if call.args[2] == 'Alias'] == ['Old name', 'New name']
  assert refresh.await_count == 2


@pytest.mark.parametrize('name,expected', [
  ('Asius v0 2', 'asius-v0-2'),
  ('  My Car!! ', 'my-car'),
  ('$(touch /tmp/nope); CAR', 'touch-tmp-nope-car'),
  ('My\nCar', 'my-car'),
  ('🚗', 'asius-v0'),
  ('', 'asius-v0'),
  ('a' * 62 + ' - tail', 'a' * 62),
])
def test_hostname_normalization(name, expected):
  from openpilot.system.asius.device_name import device_hostname
  assert device_hostname(name) == expected


def test_hostname_updates_only_when_changed():
  from openpilot.system.asius import device_name
  with patch.object(device_name.socket, 'gethostname', return_value='my-car'), \
       patch.object(device_name.subprocess, 'run') as run:
    device_name.sync_device_hostname('My Car')
    run.assert_not_called()
    device_name.sync_device_hostname('Other Car')
    run.assert_called_once_with(['sudo', '-n', 'hostname', 'other-car'], check=True, timeout=5)


@pytest.mark.parametrize('initial', [
  {'connected': True},
  {'connected': False, 'authUrl': 'https://example.com/existing-login'},
  {'connected': False},
])
def test_tailscale_name_is_set_only_when_starting_login(tmp_path, monkeypatch, initial):
  socket = tmp_path / 'tailscaled.sock'
  socket.touch()
  monkeypatch.setattr(methods, 'TAILSCALE_SOCKET', socket)
  monkeypatch.setattr(methods, 'TAILSCALE_ENABLED', tmp_path / 'enabled')
  monkeypatch.setattr(methods, 'getDeviceName', lambda: 'Asius v0 2')
  with patch.object(methods, 'getTailscaleState', side_effect=[initial, {'authUrl': 'https://example.com/new-login'}]), \
       patch.object(methods.subprocess, 'run') as run:
    methods.configureTailscale()
  logins = [call.args[0] for call in run.call_args_list if 'login' in call.args[0]]
  if initial.get('connected') or initial.get('authUrl'):
    assert logins == []
  else:
    assert len(logins) == 1
    assert '--hostname=asius-v0-2' in logins[0]
  assert not any('set' in call.args[0] for call in run.call_args_list)
