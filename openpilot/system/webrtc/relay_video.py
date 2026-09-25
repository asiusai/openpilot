"""Bounded, encrypted H.264 delivery for browsers whose WebRTC takes audio focus."""
from __future__ import annotations

import asyncio
import base64
import contextlib
import json
import re
import ssl
import time

from websocket import create_connection, WebSocketTimeoutException

from openpilot.cereal import messaging
from openpilot.common.params import Params
from openpilot.common.swaglog import cloudlog
from openpilot.system.app.identity import get_device_private_key
from openpilot.system.app.websocketd import base64url_encode, load_authorized_peers, pack_peer_message, unpack_peer_message
from openpilot.system.webrtc.in_car import InCarTelemetry

CAMERAS = {'road': 'livestreamNarrowRoadEncodeData', 'wideRoad': 'livestreamWideRoadEncodeData', 'driver': 'livestreamCabinEncodeData'}
MAX_FRAME_BYTES = 512 * 1024


def h264_codec(data: bytes) -> str | None:
  for nal in re.split(b'\x00\x00\x01', data):
    if len(nal) >= 4 and nal[0] & 0x1f == 7:
      return 'avc1.' + nal[1:4].hex()
  return None


class RelayWindow:
  """Cumulative acknowledgements bound both bytes and time in the network."""
  def __init__(self):
    self.sequence = 0
    self.pending: dict[int, tuple[float, int]] = {}

  def available(self, size: int = MAX_FRAME_BYTES) -> bool:
    return len(self.pending) < 6 and sum(item[1] for item in self.pending.values()) + size <= 768 * 1024

  def sent(self, now: float, size: int) -> int:
    self.sequence += 1
    self.pending[self.sequence] = (now, size)
    return self.sequence

  def acknowledge(self, sequence: int):
    if type(sequence) is int and 0 <= sequence <= self.sequence:
      self.pending = {key: value for key, value in self.pending.items() if key > sequence}

  def stalled(self, now: float) -> bool:
    return any(now - item[0] > 5 for item in self.pending.values())


class RelayVideoSession:
  def __init__(self, peer: str, identifier: str, camera: str):
    self.peer, self.identifier, self.camera = peer, identifier, camera
    self.params = Params()
    self.closed = False
    self.ws = None
    self.run_task: asyncio.Task | None = None
    self.control_id = 0
    self.control_at = time.monotonic()
    self.keyframe_id = 0
    self.need_keyframe = True
    self.window = RelayWindow()

  def start(self):
    self.run_task = asyncio.create_task(self.run())

  def _connect(self):
    dongle = self.params.get('DongleId')
    tls = ssl.create_default_context()
    tls.verify_flags |= 0x200000  # Match relay auth: trust/hostname checks without requiring a correct RTC.
    ws = create_connection(self.params.get('WebsocketHost', return_default=True) + '/ws/v2/' + dongle,
                           enable_multithread=True, timeout=5, sslopt={'context': tls})
    self.ws = ws
    try:
      if self.closed:
        raise RuntimeError('stream cancelled')
      challenge = json.loads(ws.recv())
      if challenge.get('type') != 'challenge' or not isinstance(challenge.get('challenge'), str):
        raise ValueError('invalid relay challenge')
      proof = f"asius-relay-auth-v1\n{dongle}\n{challenge['challenge']}".encode()
      ws.send(json.dumps({'type': 'authenticate', 'signature': base64url_encode(get_device_private_key().sign(proof))}))
      if json.loads(ws.recv()).get('type') != 'ready' or self.closed:
        raise ValueError('relay authentication failed or stream cancelled')
      ws.settimeout(1)
      return dongle
    except Exception:
      ws.shutdown()
      raise

  def control(self, payload: dict):
    identifier = payload.get('id')
    if payload.get('session') != self.identifier or type(identifier) is not int or not self.control_id < identifier < 2**53:
      return
    if payload.get('camera') not in CAMERAS:
      return
    self.control_id, self.control_at = identifier, time.monotonic()
    self.window.acknowledge(payload.get('ack'))
    if payload.get('stop') is True:
      self.closed = True
    self.camera = payload['camera']
    keyframe = payload.get('keyframe')
    if type(keyframe) is int and keyframe > self.keyframe_id:
      self.keyframe_id = keyframe
      self.need_keyframe = True

  async def receive(self, dongle: str):
    sequences: dict[str, int] = {}
    while not self.closed:
      try:
        raw = await asyncio.to_thread(self.ws.recv)
      except WebSocketTimeoutException:
        continue
      if not raw:
        raise ConnectionError('relay closed')
      envelope = json.loads(raw)
      if envelope.get('type') != 'peer' or envelope.get('from') != self.peer:
        continue
      session, sequence = envelope.get('relaySession'), envelope.get('sequence')
      if not isinstance(session, str) or type(sequence) is not int or sequence <= sequences.get(session, 0):
        continue
      if len(sequences) >= 16 and session not in sequences:
        sequences.pop(next(iter(sequences)))
      sequences[session] = sequence
      message = await asyncio.to_thread(unpack_peer_message, raw, dongle, False)
      if message and message[0] == self.peer and isinstance(message[1], dict):
        body = message[1]
        if body.get('type') == 'event' and body.get('name') == 'relayVideoControl' and isinstance(body.get('payload'), dict):
          self.control(body['payload'])

  def _send(self, dongle: str, payload: dict, sequence: int):
    body = {'type': 'event', 'name': 'relayVideo', 'payload': payload}
    envelope = json.loads(pack_peer_message(dongle, self.peer, body))
    envelope['sequence'] = sequence
    self.ws.send(json.dumps(envelope, separators=(',', ':')))

  async def send(self, dongle: str):
    telemetry = InCarTelemetry()
    camera, sock, codec, previous_frame = None, None, None, None
    last_state, last_keyframe, last_access = 0.0, 0.0, 0.0
    # The same existing encoder feeds both transports; no capture or software encode.
    self.params.put('LivestreamEncoderBitrate', 1_500_000)
    while not self.closed:
      now = time.monotonic()
      if now - self.control_at > 8 or self.window.stalled(now):
        raise TimeoutError('live viewer stopped acknowledging')
      if now - last_access > 1:
        if self.peer not in load_authorized_peers():
          raise PermissionError('device access removed')
        last_access = now
      if camera != self.camera:
        camera = self.camera
        sock = messaging.sub_sock(CAMERAS[camera], conflate=True)
        codec, previous_frame = None, None
        self.need_keyframe = True
      if self.need_keyframe and now - last_keyframe > 0.5:
        self.params.put_bool('LivestreamRequestKeyframe', True)
        last_keyframe = now
      if not self.window.available(0):
        self.need_keyframe = True
        await asyncio.sleep(0.01)
        continue
      payload = {'session': self.identifier, 'camera': camera, 'stateId': self.control_id}
      msg = messaging.recv_one_or_none(sock)
      if msg is not None:
        encoded = getattr(msg, msg.which())
        index = encoded.idx
        keyframe = bool(index.flags & 8)
        if previous_frame is not None and index.encodeId != previous_frame + 1:
          self.need_keyframe = True
        previous_frame = index.encodeId
        data = encoded.header + encoded.data
        fresh = 0 <= time.monotonic() - msg.logMonoTime / 1e9 < 0.25
        if not fresh or len(data) > MAX_FRAME_BYTES or not self.window.available(len(data)):
          self.need_keyframe = True
        elif keyframe or not self.need_keyframe:
          if keyframe:
            codec = h264_codec(data)
            self.need_keyframe = not bool(codec)
            self.params.put_bool('LivestreamRequestKeyframe', False)
          if codec and not self.need_keyframe:
            payload.update(data=base64.b64encode(data).decode(), keyframe=keyframe, codec=codec, timestamp=index.timestampSof // 1000)
      if self.control_id and now - last_state >= 0.1:
        payload['state'] = telemetry.snapshot()
        last_state = now
      if 'data' in payload or 'state' in payload:
        size = len(json.dumps(payload))
        if self.window.available(size):
          sequence = self.window.sent(now, size)
          payload['sequence'] = sequence
          await asyncio.to_thread(self._send, dongle, payload, sequence)
        else:
          self.need_keyframe = True
      await asyncio.sleep(0.005)

  async def run(self):
    tasks = []
    try:
      dongle = await asyncio.to_thread(self._connect)
      self.control_at = time.monotonic()
      tasks = [asyncio.create_task(self.receive(dongle)), asyncio.create_task(self.send(dongle))]
      done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
      for task in done:
        task.result()
    except asyncio.CancelledError:
      raise
    except Exception:
      cloudlog.exception('relay_video.session_failed')
    finally:
      self.closed = True
      if self.ws is not None:
        self.ws.shutdown()
      for task in tasks:
        task.cancel()
      await asyncio.gather(*tasks, return_exceptions=True)

  async def stop(self):
    self.closed = True
    if self.ws is not None:
      self.ws.shutdown()
    if self.run_task is not None:
      self.run_task.cancel()
      with contextlib.suppress(asyncio.CancelledError):
        await self.run_task
