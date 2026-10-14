import asyncio
import runpy
from pathlib import Path
from unittest.mock import AsyncMock, Mock

import pytest

from openpilot.common import hardware
from openpilot.common.params import Params
from openpilot.system.asius import access_policy, bluetoothd, bluetooth_ui


@pytest.mark.parametrize(('asius', 'comma', 'kernel', 'expected'), [
  (True, False, False, True),
  (False, True, False, False),
  (False, True, True, True),
  (False, False, True, False),
])
def test_bluetooth_kernel_gate(monkeypatch, asius, comma, kernel, expected):
  isfile, isdir = hardware.os.path.isfile, hardware.os.path.isdir
  monkeypatch.setattr(hardware.os.path, 'isfile', lambda p: {'/ASIUS': asius, '/AGNOS': comma}.get(p, isfile(p)))
  monkeypatch.setattr(hardware.os.path, 'isdir', lambda p: kernel if p == '/sys/class/bluetooth' else isdir(p))
  settings = runpy.run_path(str(Path(hardware.__file__)))
  assert settings['BLUETOOTH_AVAILABLE'] == expected


@pytest.mark.parametrize('ignition', [False, True, None])
def test_screen_button_requires_ignition_off(monkeypatch, tmp_path, ignition):
  params = Params(str(tmp_path))
  monkeypatch.setattr(bluetoothd, 'Params', lambda: params)
  monkeypatch.setattr(access_policy, 'ignition_state', lambda: ignition)
  monkeypatch.setattr(bluetoothd.time, 'monotonic', lambda: 1000.)
  result = bluetooth_ui.start_pairing()
  until = params.get('BluetoothPairingUntil')
  if ignition is False:
    assert 'two minutes' in result
    assert until == 1120
  else:
    assert 'Turn the vehicle off' in result
    assert until is None


def test_comma_bluetooth_keeps_builtin_gps(monkeypatch, tmp_path):
  monkeypatch.setattr(bluetoothd, 'ASIUS_HARDWARE', False)
  monkeypatch.setattr(bluetoothd, 'Params', lambda: Params(str(tmp_path)))
  monkeypatch.setattr(bluetoothd.messaging, 'SubMaster', Mock())
  pubmaster = Mock()
  monkeypatch.setattr(bluetoothd.messaging, 'PubMaster', pubmaster)
  engine = bluetoothd.BlePeerEngine(Mock())
  pubmaster.assert_called_once_with([])
  engine.send_body = AsyncMock()
  for method in ('startPhoneGps', 'updatePhoneGps', 'stopPhoneGps'):
    asyncio.run(engine.handle_rpc('app', {'jsonrpc': '2.0', 'id': 1, 'method': method}))
    assert engine.send_body.call_args.args[1]['error']['code'] == -32601
  assert engine.phone_gps.owner is None
  pubmaster.return_value.send.assert_not_called()
