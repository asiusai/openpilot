import copy
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, Mock

import pytest

from openpilot.system.asius import methods
from openpilot.system.asius.live_state import StateEncoder


def test_batched_telemetry_reduces_repeated_metadata_and_recovers_baselines(device):
  sm, params, _ = device
  snapshot = methods._live_state_snapshot(sm, params, compact=True)
  encoder = StateEncoder()
  full = encoder.encode(snapshot, 0)
  changed = copy.deepcopy(snapshot)
  changed["services"]["carState"]["vEgo"] = 21.
  changed["sampledAt"] += 0.2
  changed["services"].pop("driverMonitoringState")
  delta = encoder.encode(changed, .2)
  assert len(json.dumps(delta)) < len(json.dumps(full)) / 4
  assert delta["base"] == full["sequence"]
  assert delta["changes"]["services"]["driverMonitoringState"] is None
  assert "params" not in delta["changes"]
  assert encoder.encode(changed, 3.1)["full"]


@pytest.fixture
def device(monkeypatch):
  values = {
    "DongleId": "device", "IsMetric": True, "AppTerminalEnabled": True, "VoiceAlertsEnabled": True,
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
    "carState": {"vEgo": 20.0, "vEgoCluster": 20.5, "vCruiseCluster": 90.0, "gearShifter": "drive", "canValid": True, "extra": "unused" * 1000},
    "gpsLocation": {"hasFix": True, "latitude": 59.4, "longitude": 24.7, "horizontalAccuracy": 3, "source": "android", "unixTimestampMillis": 1800000000000},
    "extrinsicsCalibration": {"calStatus": "uncalibrated", "calPerc": 42, "rpyCalib": [0.1, 0.2, 0.3]},
    "selfdriveState": {"alertText1": "Take control", "alertText2": "Camera error", "alertStatus": "critical",
                       "alertSize": "full", "alertType": "cameraMalfunction/softDisable"},
    "driverMonitoringState": {"activePolicy": "vision", "isRHD": False, "alertLevel": "three",
                              "lockout": True, "lockoutMinutesRemaining": 30, "lockoutCount": 4,
                              "visionPolicyState": {"faceDetected": True, "awarenessPercent": 100, "isDistracted": False,
                                                    "pose": {"pitch": 0.2, "yaw": 0.1, "uncertainty": 0.01}}},
    "managerState": {"processes": [
      *[{"name": f"worker{i}", "running": True, "shouldBeRunning": True, "cmdline": ["arg"] * 1000} for i in range(50)],
      {"name": "camerad", "running": False, "shouldBeRunning": True, "exitCode": 1},
    ]},
  }
  sm = MagicMock(alive=dict.fromkeys(methods.LIVE_STATE_SERVICES, True), valid=dict.fromkeys(methods.LIVE_STATE_SERVICES, True),
                 recv_frame={service: int(service in services) for service in methods.LIVE_STATE_SERVICES})
  sm.__getitem__.side_effect = lambda service: SimpleNamespace(to_dict=lambda: copy.deepcopy(services[service]))
  build = SimpleNamespace(channel='master', openpilot=SimpleNamespace(version='test', git_normalized_origin='', git_commit=''))
  monkeypatch.setattr(methods, 'ignition_state', lambda: True)
  monkeypatch.setattr(methods, 'get_build_metadata', lambda: build)
  monkeypatch.setattr(methods, 'getDeviceName', lambda: 'Asius v0')
  monkeypatch.setattr(methods, 'load_authorized_peers', lambda: {'app': {}})
  monkeypatch.setattr(methods, '_read_vamos_update_state', lambda: None)
  return sm, params, values


def test_bluetooth_snapshot_stays_small_without_losing_status(device):
  sm, params, _ = device
  snapshot = methods._live_state_snapshot(sm, params, compact=True)
  assert len(json.dumps(snapshot).encode()) < 5000
  assert snapshot['ignition'] is True
  assert snapshot['params']['AppTerminalEnabled'] is True
  assert snapshot['params']['IsMetric'] is True
  assert snapshot['params']['VoiceAlertsEnabled'] is True
  assert snapshot['software']['UpdaterProgress'] == 25
  assert snapshot['services']['extrinsicsCalibration']['calPerc'] == 42
  assert snapshot['services']['selfdriveState']['alertText1'] == 'Take control'
  assert snapshot['services']['selfdriveState']['alertType'] == 'cameraMalfunction/softDisable'
  assert snapshot['services']['gpsLocation']['horizontalAccuracy'] == 3
  assert snapshot['services']['gpsLocation']['unixTimestampMillis'] == 1800000000000
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


@pytest.mark.parametrize('compact', [False, True])
def test_home_vehicle_readings_require_live_valid_car_data(device, compact):
  sm, params, _ = device
  assert methods._live_state_snapshot(sm, params, compact=compact)['services']['carState'] == {
    'vEgo': 20.0, 'vEgoCluster': 20.5, 'vCruiseCluster': 90.0, 'gearShifter': 'drive', 'canValid': True,
  }
  for alive, valid in [(False, True), (True, False)]:
    sm.alive['carState'], sm.valid['carState'] = alive, valid
    assert 'carState' not in methods._live_state_snapshot(sm, params, compact=compact)['services']


@pytest.mark.parametrize('compact', [False, True])
@pytest.mark.parametrize('service', ['selfdriveState', 'onroadEvents', 'driverMonitoringState'])
def test_driving_prompts_require_live_valid_services(device, compact, service):
  sm, params, _ = device
  original = sm.__getitem__.side_effect
  sm.recv_frame['onroadEvents'] = 1
  sm.__getitem__.side_effect = lambda name: [SimpleNamespace(to_dict=lambda: {'name': 'preLaneChangeLeft'})] if name == 'onroadEvents' else original(name)
  assert service in methods._live_state_snapshot(sm, params, compact=compact)['services']
  for alive, valid in [(False, True), (True, False)]:
    sm.alive[service], sm.valid[service] = alive, valid
    assert service not in methods._live_state_snapshot(sm, params, compact=compact)['services']


@pytest.mark.parametrize('compact', [False, True])
def test_monitoring_snapshot_keeps_home_fields_without_model_details(device, compact):
  sm, params, _ = device
  assert methods._live_state_snapshot(sm, params, compact=compact)['services']['driverMonitoringState'] == {
    'activePolicy': 'vision', 'isRHD': False, 'alertLevel': 'three', 'lockout': True, 'lockoutMinutesRemaining': 30,
    'visionPolicyState': {'faceDetected': True, 'awarenessPercent': 100, 'isDistracted': False, 'pose': {'pitch': 0.2, 'yaw': 0.1}},
  }


@pytest.mark.parametrize('compact', [False, True])
def test_monitoring_lockout_clear_is_sent_explicitly(device, compact):
  sm, params, _ = device
  original = sm.__getitem__.side_effect
  sm.__getitem__.side_effect = lambda name: SimpleNamespace(to_dict=lambda: {
    'activePolicy': 'vision', 'alertLevel': 'none', 'lockout': False, 'lockoutMinutesRemaining': 0,
  }) if name == 'driverMonitoringState' else original(name)
  state = methods._live_state_snapshot(sm, params, compact=compact)['services']['driverMonitoringState']
  assert state['lockout'] is False
  assert state['lockoutMinutesRemaining'] == 0


def test_bluetooth_queues_uploads_but_rejects_media_signaling(monkeypatch, tmp_path):
  import asyncio
  from openpilot.system.asius.bluetoothd import BlePeerEngine
  from openpilot.system.loggerd import data_upload_queue
  from openpilot.common.hardware.hw import Paths

  engine = BlePeerEngine.__new__(BlePeerEngine)
  engine.rpc_active = 0
  engine.send_body = AsyncMock()
  engine.peer_devices = {'app': '/test'}
  engine.sessions = {'/test': Mock(session_id=lambda _peer: 'test-session')}
  params = Mock()
  params.get_bool.return_value = True
  monkeypatch.setattr(methods, 'Params', lambda: params)
  monkeypatch.setattr(Paths, 'log_root', lambda: str(tmp_path))
  queued = Mock(return_value={'queued': ['route--0/qlog.zst'], 'uploaded': []})
  monkeypatch.setattr(data_upload_queue, 'request_uploads', queued)
  asyncio.run(engine.handle_rpc('app', {'jsonrpc': '2.0', 'id': 1, 'method': 'requestRouteUpload', 'params': {'paths': ['route--0/qlog.zst']}}))
  queued.assert_called_once_with(str(tmp_path), ['route--0/qlog.zst'])
  assert engine.send_body.call_args.args[1]['result']['queued'] == ['route--0/qlog.zst']
  asyncio.run(engine.handle_rpc('app', {'jsonrpc': '2.0', 'id': 2, 'method': 'startStream', 'params': {'sdp': 'test'}}))
  assert 'network connection' in engine.send_body.call_args.args[1]['error']['message']


def test_live_snapshot_marks_stale_gps_unavailable(monkeypatch):
  params = Mock()
  params.get.return_value = None
  sm = Mock()
  sm.alive = dict.fromkeys(methods.LIVE_STATE_SERVICES, False)
  sm.valid = dict.fromkeys(methods.LIVE_STATE_SERVICES, True)
  sm.recv_frame = dict.fromkeys(methods.LIVE_STATE_SERVICES, 0)
  build = SimpleNamespace(channel='master', openpilot=SimpleNamespace(version='test', git_normalized_origin='', git_commit=''))
  monkeypatch.setattr(methods, 'ignition_state', lambda: True)
  monkeypatch.setattr(methods, 'get_build_metadata', lambda: build)
  monkeypatch.setattr(methods, 'getDeviceName', lambda: 'Asius v0')
  monkeypatch.setattr(methods, 'load_authorized_peers', dict)
  monkeypatch.setattr(methods, '_software_update_state', lambda _: {})
  assert methods._live_state_snapshot(sm, params)['services']['gpsLocation'] == {'hasFix': False}
