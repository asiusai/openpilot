import asyncio
import json
import time
import uuid
from itertools import product
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import numpy as np

from openpilot.cereal import messaging
from openpilot.common.params import ParamKeyFlag, Params
from openpilot.system.webrtc.in_car import CameraSource, InCarSession, JpegEncoder, pack_nv12
from openpilot.system.webrtc.webrtcd import ServerState, handle_in_car_frame, schedule_teardown


def make_session():
  return InCarSession(str(uuid.uuid4()), lambda: True)


def publish(session, name, data, age=0):
  msg = messaging.new_message(name)
  msg.valid = True
  msg.logMonoTime = time.monotonic_ns() - int(age * 1e9)
  setattr(msg, name, data)
  session.sm.update_msgs(time.monotonic(), [msg])


def test_nv12_camera_padding_and_downsampling():
  width, height, stride, uv_offset = 1344, 760, 1408, 1408 * 768
  raw = np.zeros(uv_offset + stride * 384, dtype=np.uint8)
  y = raw[:uv_offset].reshape(-1, stride)
  y[:height, :width] = 123
  uv = raw[uv_offset:].reshape(-1, stride)
  uv[:height // 2, :width:2] = 80
  uv[:height // 2, 1:width:2] = 180
  pixels, w, h = pack_nv12(SimpleNamespace(data=memoryview(raw), width=width, height=height, stride=stride, uv_offset=uv_offset))
  assert (w, h) == (672, 380)
  assert pixels[:w * h] == bytes([123]) * w * h
  assert pixels[w * h:] == bytes([80, 180]) * (w * h // 4)


def test_jpeg_encoder_produces_independent_frames_and_restarts_on_size_change():
  async def run():
    encoder = JpegEncoder()
    try:
      for width, height in [(320, 240), (320, 240), (640, 480)]:
        frame = await encoder.encode(bytes([100]) * width * height + bytes([128]) * (width * height // 2), width, height)
        assert frame.startswith(b'\xff\xd8') and frame.endswith(b'\xff\xd9')
      process = encoder.process
    finally:
      await encoder.close()
    assert process.returncode is not None
  asyncio.run(run())


def test_ignition_does_not_clear_display_or_livestream(tmp_path):
  params = Params(str(tmp_path))
  for name in ('IsLiveStreaming', 'IsInCarDisplay'):
    params.put_bool(name, True, block=True)
  params.clear_all(ParamKeyFlag.CLEAR_ON_IGNITION_ON)
  assert params.get_bool('IsLiveStreaming') and params.get_bool('IsInCarDisplay')
  params.clear_all(ParamKeyFlag.CLEAR_ON_MANAGER_START)
  assert not params.get_bool('IsLiveStreaming') and not params.get_bool('IsInCarDisplay')


def test_camera_process_runs_for_display_without_starting_stream_encoder():
  from openpilot.system.manager.process_config import managed_processes
  cp = SimpleNamespace(notCar=False)
  for started, driver_view, livestream, display in product((False, True), repeat=4):
    flags = {'IsDriverViewEnabled': driver_view, 'IsLiveStreaming': livestream, 'IsInCarDisplay': display}
    params = Mock()
    params.get_bool.side_effect = flags.__getitem__
    assert managed_processes['camerad'].should_run(started, params, cp) == (started or driver_view or livestream or display)
    assert managed_processes['stream_encoderd'].should_run(started, params, cp) == livestream


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


def test_camera_reconnects_after_producer_restart():
  source = CameraSource()
  source.camera = source.requested_camera = 'road'
  source.client = Mock()
  source.last_frame = time.monotonic() - 2
  with patch('msgq.visionipc.VisionIpcClient') as client:
    from openpilot.cereal.visionipc import VisionStreamType
    client.available_streams.return_value = [VisionStreamType.VISION_STREAM_NARROW_ROAD]
    client.return_value.connect.return_value = True
    client.return_value.recv.return_value = None
    assert source.read('road') is None
    client.assert_called_once()


def test_frame_requests_require_authorized_peer_and_valid_identifiers():
  async def run():
    with patch('openpilot.system.app.websocketd.load_authorized_peers', return_value=['allowed']):
      state = ServerState()
      for peer, session, identifier, code in [('other', str(uuid.uuid4()), 1, 403), ('allowed', '', 1, 400),
                                              ('allowed', str(uuid.uuid4()), True, 400), ('allowed', str(uuid.uuid4()), 0, 400)]:
        response = await handle_in_car_frame(state, json.dumps({'peer': peer, 'session': session, 'id': identifier}).encode())
        assert response[0] == code
      assert not state.in_car
  asyncio.run(run())


def test_display_pull_is_serial_expires_and_does_not_create_webrtc():
  async def run():
    state = ServerState()
    identifier = str(uuid.uuid4())
    body = {'peer': 'allowed', 'session': identifier, 'id': 1}
    async def request(**changes):
      return await handle_in_car_frame(state, json.dumps({**body, **changes}).encode())
    with patch('openpilot.system.app.websocketd.load_authorized_peers', return_value=['allowed']), \
         patch('openpilot.system.webrtc.in_car.InCarSession.frame', new_callable=AsyncMock, return_value={'id': 1}) as frame, \
         patch('teleoprtc.builder.WebRTCAnswerBuilder') as rtc, patch('openpilot.system.webrtc.webrtcd.Params'):
      assert (await request())[0] == 200
      session = state.in_car['allowed']
      assert (await request())[0] == 409  # a duplicate never captures another image
      session.busy = True
      assert (await request(id=2))[0] == 409
      session.busy = False
      assert (await request(id=2))[0] == 200
      frame.assert_awaited()
      rtc.assert_not_called()
      # Closing an obsolete tab must not close its replacement's display.
      assert (await request(session=str(uuid.uuid4()), close=True))[0] == 200
      assert state.in_car['allowed'] is session
      expiry = session.expiry
      callback = expiry._callback
      expiry.cancel()
      callback()
      await asyncio.sleep(0)
      assert not state.in_car and session.closed
      state.teardown.cancel()
  asyncio.run(run())


def test_revocation_during_capture_never_returns_an_image():
  async def run():
    session = make_session()
    def revoke(_):
      session.authorized = lambda: False
      return None
    session.camera.read = revoke
    try:
      await session.frame(1)
      raise AssertionError('revoked peer received a frame')
    except PermissionError:
      pass
    finally:
      await session.stop()
  asyncio.run(run())


def test_display_frame_encodes_camera_and_refreshes_telemetry():
  async def run():
    import base64
    session = make_session()
    session.camera.read = Mock(return_value=(b'pixels', 64, 32, 'road', time.monotonic_ns()))
    session.encoder.encode = AsyncMock(return_value=b'jpeg')
    result = await session.frame(7)
    assert result['id'] == 7 and result['width'] == 64 and result['height'] == 32
    assert base64.b64decode(result['jpeg']) == b'jpeg'
    assert 'state' in result
    await session.stop()
  asyncio.run(run())


def test_teardown_keeps_independent_display_and_video_lifetimes():
  async def run():
    state = ServerState()
    state.in_car['peer'] = object()
    with patch('openpilot.system.webrtc.webrtcd.Params') as params:
      schedule_teardown(state)
      callback = state.teardown._callback
      state.teardown.cancel()
      callback()
      params.return_value.put_bool.assert_called_once_with('IsLiveStreaming', False)
      params.return_value.put_bool.reset_mock()
      state.streams['video'] = object()
      state.in_car.clear()
      callback()
      params.return_value.put_bool.assert_called_once_with('IsInCarDisplay', False)
  asyncio.run(run())


def test_real_visionipc_frames_are_fresh_and_survive_camera_restart():
  from msgq.visionipc import VisionIpcServer
  from openpilot.cereal.visionipc import VisionStreamType
  stream = VisionStreamType.VISION_STREAM_NARROW_ROAD
  name = 'display-test-' + uuid.uuid4().hex[:8]
  source = CameraSource(name)
  for _ in range(2):
    server = VisionIpcServer(name)
    server.create_buffers(stream, 2, 128, 96)
    server.start_listener()
    source.last_frame = 0  # model the reconnect timeout without sleeping a second
    deadline = time.monotonic() + 2
    while source.client is None or source.last_frame == 0:
      assert source.read('road') is None  # connects, no camera frame yet
      assert time.monotonic() < deadline
      time.sleep(0.01)
    pixels = bytes([128]) * source.client.buffer_len
    server.send(stream, pixels, frame_id=1, timestamp_sof=time.monotonic_ns() - 1_000_000_000)
    assert source.read('road') is None  # never replay an old frame
    server.send(stream, pixels, frame_id=2, timestamp_sof=time.monotonic_ns())
    frame = source.read('road')
    assert frame is not None
    assert frame[1:4] == (128, 96, 'road')
    assert len(frame[0]) == 128 * 96 * 3 // 2
    del server


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
