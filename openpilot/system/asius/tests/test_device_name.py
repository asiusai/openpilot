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
