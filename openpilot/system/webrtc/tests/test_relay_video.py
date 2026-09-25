import asyncio
import json
import time
from unittest.mock import Mock, patch

from openpilot.system.webrtc.relay_video import RelayVideoSession, RelayWindow, h264_codec
from openpilot.system.webrtc.webrtcd import ServerState, handle_relay_stream


def test_annex_b_codec_uses_encoder_sps():
  assert h264_codec(b'\x00\x00\x00\x01\x67\x64\x00\x28\xaa') == 'avc1.640028'
  assert h264_codec(b'\x00\x00\x01\x09\x10\x00\x00\x01\x67\x42\xc0\x1e') == 'avc1.42c01e'
  assert h264_codec(b'\x00\x00\x01\x65\xff') is None


def test_window_bounds_memory_rejects_future_ack_and_detects_stall():
  window = RelayWindow()
  for _ in range(6):
    assert window.available(100)
    window.sent(10, 100)
  assert not window.available(1)
  window.acknowledge(7)
  assert len(window.pending) == 6
  window.acknowledge(True)
  assert len(window.pending) == 6
  window.acknowledge(3)
  assert list(window.pending) == [4, 5, 6]
  assert not window.available(768 * 1024)
  assert not window.stalled(14)
  assert window.stalled(16)
  window.acknowledge(6)
  assert not window.stalled(16)


def test_controls_are_session_scoped_ordered_and_read_only():
  with patch('openpilot.system.webrtc.relay_video.Params'):
    session = RelayVideoSession('peer', 'session', 'road')
    original = session.control_at
    session.control({'session': 'other', 'id': 99, 'camera': 'driver'})
    session.control({'session': 'session', 'id': True, 'camera': 'driver'})
    assert session.camera == 'road' and session.control_at == original
    session.control({'session': 'session', 'id': 2, 'camera': 'driver', 'keyframe': 1, 'ack': 0})
    assert session.camera == 'driver' and session.keyframe_id == 1
    session.control({'session': 'session', 'id': 1, 'camera': 'road', 'stop': True})
    assert not session.closed and session.camera == 'driver'
    session.control({'session': 'session', 'id': 3, 'camera': 'driver', 'stop': True})
    assert session.closed


def test_stream_endpoint_enforces_access_and_owner_scoped_stop():
  async def run():
    state = ServerState()
    identifier = '2dc36cd6-af3e-4ae0-b37a-1cc0ea67808b'
    with patch('openpilot.system.app.websocketd.load_authorized_peers', return_value={'owner': {}, 'other': {}}), \
         patch('openpilot.system.webrtc.webrtcd.Params'), \
         patch('openpilot.system.webrtc.relay_video.RelayVideoSession') as factory:
      denied = await handle_relay_stream(state, json.dumps({'peer': 'stranger', 'session': identifier, 'camera': 'road'}).encode())
      assert denied[0] == 403 and not factory.called
      session = factory.return_value
      session.peer, session.identifier = 'owner', identifier
      session.run_task = asyncio.create_task(asyncio.sleep(100))
      async def stop():
        session.run_task.cancel()
      session.stop = Mock(side_effect=stop)
      result = await handle_relay_stream(state, json.dumps({'peer': 'owner', 'session': identifier, 'camera': 'road'}).encode())
      assert result[0] == 200 and state.relay is session
      await handle_relay_stream(state, json.dumps({'peer': 'other', 'session': identifier, 'stop': True}).encode())
      assert not session.stop.called
      await handle_relay_stream(state, json.dumps({'peer': 'owner', 'session': identifier, 'stop': True}).encode())
      assert session.stop.called and state.relay is None
      if state.teardown:
        state.teardown.cancel()
  asyncio.run(run())


def test_missing_ack_releases_stream():
  async def run():
    with patch('openpilot.system.webrtc.relay_video.Params'), patch('openpilot.system.webrtc.relay_video.InCarTelemetry'):
      session = RelayVideoSession('peer', 'session', 'road')
      session.control_at = time.monotonic() - 9
      try:
        await session.send('dongle')
        raise AssertionError('missing acknowledgements must end the stream')
      except TimeoutError:
        pass
  asyncio.run(run())
