import copy
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, Mock

import pytest

from openpilot.system.app import methods


@pytest.fixture
def device(monkeypatch):
  values = {
    "DongleId": "device", "ExperimentalMode": True,
    "UpdaterCurrentReleaseNotes": b"Release notes " * 12000,
    "UpdaterNewReleaseNotes": b"More release notes " * 12000,
    "UpdaterAvailableBranches": [f"branch-{i}" for i in range(2000)],
    "UpdaterState": "downloading", "UpdaterProgress": 25,
    "Offroad_TemperatureTooHigh": {"text": "Cooling down: %1", "extra": "85°C", "severity": 0},
  }
  params = Mock()
  params.get.side_effect = lambda key, **_: values.get(key)
  services = {
    "deviceState": {"started": True, "cpuTempC": [85], "networkType": "none", "chestnutPresent": False},
    "gpsLocation": {"hasFix": True, "latitude": 59.4, "longitude": 24.7, "horizontalAccuracy": 3, "source": "android"},
    "extrinsicsCalibration": {"calStatus": "uncalibrated", "calPerc": 42, "rpyCalib": [0.1, 0.2, 0.3]},
    "selfdriveState": {"alertText1": "Take control", "alertText2": "Camera error", "alertStatus": "critical", "alertSize": "full"},
    "managerState": {"processes": [
      *[{"name": f"worker{i}", "running": True, "shouldBeRunning": True, "cmdline": ["arg"] * 1000} for i in range(50)],
      {"name": "camerad", "running": False, "shouldBeRunning": True, "exitCode": 1},
    ]},
  }
  sm = MagicMock(alive=dict.fromkeys(methods.LIVE_STATE_SERVICES, True), valid=dict.fromkeys(methods.LIVE_STATE_SERVICES, True),
                 recv_frame={service: int(service in services) for service in methods.LIVE_STATE_SERVICES})
  sm.__getitem__.side_effect = lambda service: SimpleNamespace(to_dict=lambda: copy.deepcopy(services[service]))
  build = SimpleNamespace(channel='master', openpilot=SimpleNamespace(version='test', git_normalized_origin='', git_commit=''))
  monkeypatch.setattr(methods, 'get_build_metadata', lambda: build)
  monkeypatch.setattr(methods, 'get_device_name', lambda: 'Asius v0')
  monkeypatch.setattr(methods, 'load_authorized_peers', lambda: {'app': {}})
  monkeypatch.setattr(methods, '_read_vamos_update_state', lambda: None)
  return sm, params, values


def test_bluetooth_snapshot_stays_small_without_losing_status(device):
  sm, params, _ = device
  snapshot = methods._live_state_snapshot(sm, params, compact=True)
  assert len(json.dumps(snapshot).encode()) < 5000
  assert snapshot['params']['ExperimentalMode'] is True
  assert snapshot['software']['UpdaterProgress'] == 25
  assert snapshot['services']['extrinsicsCalibration']['calPerc'] == 42
  assert snapshot['services']['selfdriveState']['alertText1'] == 'Take control'
  assert snapshot['services']['gpsLocation']['horizontalAccuracy'] == 3
  assert snapshot['services']['managerState']['processes'] == [{'name': 'camerad', 'running': False, 'shouldBeRunning': True, 'exitCode': 1}]
  assert 'UpdaterCurrentReleaseNotes' not in snapshot['software']
  assert 'UpdaterAvailableBranches' not in snapshot['params']
  # On-demand Software details still include the complete release notes.
  assert len(methods._software_update_state(params)['UpdaterCurrentReleaseNotes']) > 100000


@pytest.mark.parametrize('compact', [False, True])
def test_live_alerts_match_raylib_and_clear_with_device_params(device, compact):
  sm, params, values = device
  assert methods._live_state_snapshot(sm, params, compact=compact)['offroadAlerts'] == [
    {'key': 'Offroad_TemperatureTooHigh', 'text': 'Cooling down: 85°C', 'severity': 0},
  ]
  del values['Offroad_TemperatureTooHigh']
  assert methods._live_state_snapshot(sm, params, compact=compact)['offroadAlerts'] == []


def test_bluetooth_queues_uploads_but_rejects_media_signaling(monkeypatch, tmp_path):
  import asyncio
  from openpilot.system.app.bluetoothd import BlePeerEngine
  from openpilot.system.loggerd import data_upload_queue
  from openpilot.common.hardware.hw import Paths

  engine = BlePeerEngine.__new__(BlePeerEngine)
  engine.send_body = AsyncMock()
  params = Mock()
  params.get_bool.return_value = True
  monkeypatch.setattr(methods, 'Params', lambda: params)
  monkeypatch.setattr(Paths, 'log_root', lambda: str(tmp_path))
  queued = Mock(return_value={'queued': ['route--0/qlog.zst'], 'uploaded': []})
  monkeypatch.setattr(data_upload_queue, 'request_uploads', queued)
  asyncio.run(engine.handle_rpc('app', {'jsonrpc': '2.0', 'id': 1, 'method': 'requestRouteUpload', 'params': {'paths': ['route--0/qlog.zst']}}))
  queued.assert_called_once_with(str(tmp_path), ['route--0/qlog.zst'])
  assert engine.send_body.call_args.args[1]['result']['queued'] == ['route--0/qlog.zst']
  for method in ('startStream', 'startRouteStream'):
    asyncio.run(engine.handle_rpc('app', {'jsonrpc': '2.0', 'id': 2, 'method': method, 'params': {'sdp': 'test'}}))
    assert 'network connection' in engine.send_body.call_args.args[1]['error']['message']
