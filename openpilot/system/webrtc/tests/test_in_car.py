import asyncio
import json
import time
from unittest.mock import AsyncMock, Mock, patch

from openpilot.cereal import messaging
from openpilot.system.webrtc.in_car import InCarTelemetry


def make_session():
  return InCarTelemetry()


def test_live_video_camera_demand_no_longer_uses_image_relay_flag():
  from itertools import product
  from types import SimpleNamespace
  from openpilot.system.manager.process_config import managed_processes
  for started, driver_view, livestream in product((False, True), repeat=3):
    params = Mock()
    params.get_bool.side_effect = {'IsDriverViewEnabled': driver_view, 'IsLiveStreaming': livestream}.__getitem__
    cp = SimpleNamespace(notCar=False)
    assert managed_processes['camerad'].should_run(started, params, cp) == (started or driver_view or livestream)
    assert managed_processes['stream_encoderd'].should_run(started, params, cp) == livestream


def publish(session, name, data, age=0):
  msg = messaging.new_message(name)
  msg.valid = True
  msg.logMonoTime = time.monotonic_ns() - int(age * 1e9)
  setattr(msg, name, data)
  session.sm.update_msgs(time.monotonic(), [msg])


def test_snapshot_removes_stale_speed_engagement_monitoring_and_path():
  session = make_session()
  publish(session, 'deviceState', {'started': True, 'chestnutPresent': False})
  publish(session, 'carState', {'vEgo': 20, 'vEgoCluster': 21})
  publish(session, 'selfdriveState', {'enabled': True, 'state': 'enabled'})
  snapshot = session.snapshot()
  assert snapshot['services']['carState']['vEgo'] == 20
  assert snapshot['services']['selfdriveState']['enabled']
  publish(session, 'carState', {'vEgo': 30}, age=1)
  publish(session, 'selfdriveState', {'enabled': True}, age=1)
  publish(session, 'driverMonitoringState', {'activePolicy': 'vision'}, age=1)
  assert not {'carState', 'selfdriveState', 'driverMonitoringState'} & session.snapshot()['services'].keys()


def test_path_and_lanes_come_from_model_v2_and_expire_when_stale():
  session = make_session()
  path = {'x': [0, 10, 30], 'y': [0, 0, 0], 'z': [0, 0, 0]}
  model = {'position': path, 'laneLines': [path] * 4, 'laneLineProbs': [0.1, 0.9, 0.9, 0.1],
           'roadEdges': [path] * 2, 'roadEdgeStds': [0.2, 0.2]}
  publish(session, 'modelV2', model)
  received = session.snapshot()['services']['modelV2']
  assert received['position']['x'] == path['x']
  assert len(received['laneLines']) == 4
  assert len(received['roadEdges']) == 2
  publish(session, 'modelV2', model, age=1)
  assert 'modelV2' not in session.snapshot()['services']


def test_gpu_states_and_alerts_use_real_device_values():
  session = make_session()
  session.params = Mock()
  session.params.get_bool.return_value = False
  session.params.get.side_effect = lambda key: {'text': 'GPU temperature %1', 'extra': '85 C'} if key == 'Offroad_ChestnutOverheated' else None
  with patch('openpilot.system.webrtc.in_car.chestnut_compiled', return_value=True):
    publish(session, 'deviceState', {'started': False, 'chestnutPresent': True})
    snapshot = session.snapshot()
    assert snapshot['gpu'] == 'ready'
    assert snapshot['offroadAlerts'][0]['text'] == 'GPU temperature 85 C'
    publish(session, 'deviceState', {'started': True, 'chestnutPresent': True})
    assert session.snapshot()['gpu'] == 'loading'
    publish(session, 'modelV2', {'big': True})
    assert session.snapshot()['gpu'] == 'active'
    publish(session, 'modelV2', {'big': False})
    assert session.snapshot()['gpu'] == 'failed'
    publish(session, 'deviceState', {'started': False, 'chestnutPresent': True})
    assert session.snapshot()['gpu'] == 'ready'


def test_selfdrive_timeout_uses_stock_alert_without_replaying_enabled_state():
  session = make_session()
  publish(session, 'deviceState', {'started': True})
  session.was_started = True
  session.started_at = time.monotonic() - 30
  publish(session, 'selfdriveState', {'enabled': True}, age=6)
  snapshot = session.snapshot()
  assert 'selfdriveState' not in snapshot['services']
  assert snapshot['alert']['alertText1'] == 'TAKE CONTROL IMMEDIATELY'
  assert snapshot['alert']['alertStatus'] == 'critical'
  publish(session, 'selfdriveState', {'enabled': True}, age=16)
  assert session.snapshot()['alert']['alertText2'] == 'Reboot Device'


def test_in_car_video_is_read_only_even_on_not_car_devices():
  from opendbc.car.structs import car
  from openpilot.system.athena.athenad import startStream
  cp = car.CarParams.new_message(notCar=True)
  with patch('openpilot.system.athena.athenad.Params') as params, \
       patch('openpilot.system.webrtc.helpers.wait_for_webrtcd'), \
       patch('openpilot.system.webrtc.helpers.post_stream_request') as post:
    params.return_value.get.return_value = cp.to_bytes()
    startStream('sdp', True, inCar=True)
    request = post.call_args.args[0]
    assert request.in_car and request.bridge_services_in == [] and request.bridge_services_out == []
    startStream('sdp', True)
    assert post.call_args.args[0].bridge_services_in == ['testJoystick']


def test_in_car_video_waits_for_disconnection_without_five_minute_limit():
  from openpilot.system.webrtc.webrtcd import StreamSession, SESSION_TIMEOUT_SECONDS
  from openpilot.system.webrtc.helpers import StreamRequestBody
  async def run():
    with patch('teleoprtc.builder.WebRTCAnswerBuilder'), patch('openpilot.system.webrtc.webrtcd._default_route_ip'):
      display = StreamSession(StreamRequestBody('sdp', [], True, in_car=True))
      ordinary = StreamSession(StreamRequestBody('sdp', [], True))
    assert display.session_timeout is None and ordinary.session_timeout == SESSION_TIMEOUT_SECONDS
    display.stream.wait_for_disconnection = AsyncMock()
    with patch('openpilot.system.webrtc.webrtcd.asyncio.wait_for', new_callable=AsyncMock) as wait:
      await display.run_normal_session()
      pending = wait.call_args.args[0]
      assert wait.call_args.kwargs['timeout'] is None
      await pending
  asyncio.run(run())


def test_webrtc_hud_is_read_only_sequenced_and_only_available_in_car_mode():
  from openpilot.system.webrtc.webrtcd import StreamSession
  from openpilot.system.webrtc.helpers import StreamRequestBody
  with patch('teleoprtc.builder.WebRTCAnswerBuilder'), patch('openpilot.system.webrtc.webrtcd._default_route_ip'):
    display = StreamSession(StreamRequestBody('sdp', [], True, in_car=True))
    ordinary = StreamSession(StreamRequestBody('sdp', [], True))
    display.in_car_state.snapshot = Mock(return_value={'test': True})
    for session in [ordinary, display]:
      session.stream.get_messaging_channel().send.reset_mock()
      session.message_handler(json.dumps({'type': 'inCarState', 'id': 1}))
      if session is ordinary:
        session.stream.get_messaging_channel().send.assert_not_called()
      else:
        session.stream.get_messaging_channel().send.assert_called_once()
        session.stream.get_messaging_channel().send.reset_mock()
        session.message_handler(json.dumps({'type': 'inCarState', 'id': 1}))
        session.message_handler(json.dumps({'type': 'inCarState', 'id': True}))
        session.message_handler(json.dumps({'type': 'inCarState', 'id': 2}))  # rate limited
        session.stream.get_messaging_channel().send.assert_not_called()
