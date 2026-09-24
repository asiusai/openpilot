"""Silent, read-only driving display over an authenticated WebRTC data channel.

The browser pulls at most one JPEG at a time. There are no RTP media tracks,
media playback APIs, or queues of old frames, including across ignition changes.
"""
from __future__ import annotations

import asyncio
import contextlib
import json
import math
import struct
import time
import uuid
from collections.abc import Callable

import numpy as np

from openpilot.cereal import messaging
from openpilot.common.params import Params
from openpilot.selfdrive.modeld.helpers import chestnut_compiled
from openpilot.selfdrive.selfdrived.alertmanager import OFFROAD_ALERTS

CHUNK_BYTES = 16 * 1024
MAX_FRAME_BYTES = 512 * 1024
SERVICES = ['deviceState', 'carState', 'selfdriveState', 'driverMonitoringState', 'drivingModelData', 'extrinsicsCalibration',
            'modelV2', 'carOutput', 'narrowRoadCameraState', 'wideRoadCameraState']
FIELDS = {
  'deviceState': ['deviceType', 'started', 'chestnutPresent', 'networkType', 'thermalStatus', 'gpuTempC', 'cpuTempC', 'freeSpacePercent'],
  'carState': ['vEgo', 'vEgoCluster', 'vCruiseCluster', 'steeringAngleDeg', 'leftBlinker', 'rightBlinker'],
  'selfdriveState': ['enabled', 'state', 'experimentalMode', 'alertText1', 'alertText2', 'alertSize', 'alertStatus', 'alertType', 'alertHudVisual'],
  'driverMonitoringState': ['activePolicy', 'isRHD', 'alertLevel', 'visionPolicyState'],
  'extrinsicsCalibration': ['rpyCalib', 'wideFromDeviceEuler', 'calStatus', 'height'],
  'drivingModelData': ['position', 'laneLines', 'laneLineProbs', 'roadEdges', 'roadEdgeStds'],
  'narrowRoadCameraState': ['sensor'],
  'wideRoadCameraState': ['sensor'],
}


def pack_nv12(buf) -> tuple[bytes, int, int]:
  """Remove camera stride/UV padding and downsample before feeding the encoder."""
  data = np.frombuffer(buf.data, dtype=np.uint8)
  step = max(1, math.ceil(buf.width / 1024))
  width = (buf.width // step) & ~1
  height = (buf.height // step) & ~1
  y = data[:buf.uv_offset].reshape(-1, buf.stride)[:height * step:step, :width * step:step]
  uv = data[buf.uv_offset:buf.uv_offset + (buf.height // 2) * buf.stride].reshape(-1, buf.stride // 2, 2)
  uv = uv[:height // 2 * step:step, :width // 2 * step:step].reshape(height // 2, width)
  return y.tobytes() + uv.tobytes(), width, height


class CameraSource:
  def __init__(self, name: str = 'camerad'):
    self.name = name
    self.client = None
    self.camera = ''
    self.requested_camera = ''
    self.last_frame = 0.0

  def read(self, camera: str):
    from msgq.visionipc import VisionIpcClient
    from openpilot.cereal.visionipc import VisionStreamType
    streams = {'road': VisionStreamType.VISION_STREAM_NARROW_ROAD, 'wideRoad': VisionStreamType.VISION_STREAM_WIDE_ROAD}
    now = time.monotonic()
    if camera != self.requested_camera or now - self.last_frame > 1.0:
      self.client = None
    if self.client is None:
      self.requested_camera = camera
      available = VisionIpcClient.available_streams(self.name, False)
      if streams[camera] not in available:
        camera = 'wideRoad' if camera == 'road' else 'road'
      if streams[camera] not in available:
        return None
      self.client = VisionIpcClient(self.name, streams[camera], True)
      self.camera = camera
      self.last_frame = now
      if not self.client.connect(False):
        self.client = None
        return None
    buf = self.client.recv(100)
    if buf is None or time.monotonic_ns() - self.client.timestamp_sof > 500_000_000:
      return None
    self.last_frame = time.monotonic()
    pixels, width, height = pack_nv12(buf)
    return pixels, width, height, self.camera, self.client.timestamp_sof


class JpegEncoder:
  def __init__(self):
    self.process = None
    self.size = None

  async def encode(self, pixels: bytes, width: int, height: int) -> bytes:
    if self.size != (width, height) or self.process is None or self.process.returncode is not None:
      await self.close()
      self.process = await asyncio.create_subprocess_exec(
        'ffmpeg', '-hide_banner', '-loglevel', 'error', '-f', 'rawvideo', '-pixel_format', 'nv12',
        '-video_size', f'{width}x{height}', '-framerate', '10', '-i', 'pipe:0', '-an',
        # The device FFmpeg build has no image2pipe muxer; rawvideo writes the encoded JPEG packets unchanged.
        '-threads', '1', '-c:v', 'mjpeg', '-q:v', '6', '-f', 'rawvideo', '-flush_packets', '1', 'pipe:1',
        stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL, limit=MAX_FRAME_BYTES)
      self.size = (width, height)
    try:
      async with asyncio.timeout(2):
        self.process.stdin.write(pixels)
        await self.process.stdin.drain()
        jpeg = await self.process.stdout.readuntil(b'\xff\xd9')
        if not jpeg.startswith(b'\xff\xd8') or len(jpeg) > MAX_FRAME_BYTES:
          raise ValueError('invalid JPEG frame')
        return jpeg
    except BaseException:
      await self.close()
      raise

  async def close(self):
    if self.process is not None:
      if self.process.returncode is None:
        with contextlib.suppress(ProcessLookupError):
          self.process.kill()
      await self.process.wait()
      self.process = None
      self.size = None


class InCarSession:
  def __init__(self, sdp: str, authorized: Callable[[], bool]):
    from teleoprtc.builder import WebRTCAnswerBuilder
    self.identifier = str(uuid.uuid4())
    self.stream = WebRTCAnswerBuilder(sdp).stream()
    self.authorized = authorized
    self.params = Params()
    self.sm = messaging.SubMaster(SERVICES)
    self.camera = CameraSource()
    self.encoder = JpegEncoder()
    self.run_task: asyncio.Task | None = None
    self.closed = False
    self.request = asyncio.Event()
    self.request_id = 0
    self.drained = asyncio.Event()
    self.channel_ready = False
    self.compiled = False
    self.gpu = 'disconnected'
    self.was_started = False
    self.present_at_start = False
    self.started_at = 0.0
    self.wide = False
    self.alerts = []
    self.alerts_updated = 0.0
    self.stream.set_message_handler(self.message_handler)

  async def get_answer(self):
    return await self.stream.start()

  def start(self):
    self.run_task = asyncio.create_task(self.run())

  def message_handler(self, message: bytes | str):
    try:
      if len(message) > 256:
        return
      body = json.loads(message)
      identifier = body.get('id')
      if body.get('op') == 'next' and type(identifier) is int and 0 < identifier < 2**32 and not self.request.is_set():
        self.request_id = identifier
        self.request.set()
    except (ValueError, TypeError, AttributeError):
      pass

  def snapshot(self) -> dict:
    self.sm.update(0)
    now = time.monotonic()
    services = {}
    for name, fields in FIELDS.items():
      if not self.sm.seen[name]:
        continue
      age = max(0, now - self.sm.logMonoTime[name] / 1e9)
      max_age = 5 if name == 'deviceState' else 0.5 if name in ('carState', 'selfdriveState', 'driverMonitoringState', 'drivingModelData') else 2
      if age > max_age or not self.sm.valid[name]:
        continue
      values = self.sm[name].to_dict()
      if name == 'drivingModelData' and not all(key in values for key in FIELDS[name]):
        continue
      if name == 'extrinsicsCalibration' and not all(key in values for key in ('rpyCalib', 'wideFromDeviceEuler', 'calStatus')):
        continue
      services[name] = {key: values[key] for key in fields if key in values}
    ds = services.get('deviceState', {})
    started = ds.get('started', False)
    detected = ds.get('chestnutPresent', False)
    self.compiled = self.compiled or chestnut_compiled()
    if started and not self.was_started:
      self.started_at = now
      self.present_at_start = detected
    model_seen = self.sm.seen['modelV2'] and self.sm.logMonoTime['modelV2'] / 1e9 > self.started_at
    model_fresh = model_seen and self.sm.valid['modelV2'] and now - self.sm.logMonoTime['modelV2'] / 1e9 < 1
    if not started:
      self.present_at_start = detected
      self.gpu = 'ready' if detected and self.compiled else 'uncompiled' if detected else 'disconnected'
    elif not self.present_at_start:
      self.gpu = 'disconnected'
    elif not self.compiled:
      self.gpu = 'uncompiled'
    elif self.gpu == 'failed' or not detected or (model_seen and (not model_fresh or not self.sm['modelV2'].big)):
      self.gpu = 'failed'
    elif self.params.get_bool('ChestnutLoading') or not model_seen:
      self.gpu = 'loading'
    else:
      self.gpu = 'failed' if self.params.get('ChestnutActive') is False else 'active'
    self.was_started = started
    alert = services.get('selfdriveState')
    if started and alert is None:
      ss_time = self.sm.logMonoTime['selfdriveState'] / 1e9
      if ss_time < self.started_at and now - self.started_at > 10:
        alert = {'alertText1': 'openpilot Unavailable', 'alertText2': 'Waiting to start', 'alertSize': 'mid', 'alertStatus': 'normal'}
      elif ss_time >= self.started_at and now - ss_time > 5:
        take_control = self.sm['selfdriveState'].enabled and now - ss_time < 15
        alert = {'alertText1': 'TAKE CONTROL IMMEDIATELY' if take_control else 'System Unresponsive',
                 'alertText2': 'System Unresponsive' if take_control else 'Reboot Device', 'alertSize': 'full', 'alertStatus': 'critical',
                 'alertHudVisual': 'steerRequired'}
    if alert:
      alert = {key: value for key, value in alert.items() if key.startswith('alert')}
    confidence = None
    if model_fresh:
      predictions = self.sm['modelV2'].meta.disengagePredictions
      confidence = (1 - max(predictions.brakeDisengageProbs or [1])) * (1 - max(predictions.steerOverrideProbs or [1]))
    torque = None
    if self.sm.seen['carOutput'] and self.sm.valid['carOutput'] and now - self.sm.logMonoTime['carOutput'] / 1e9 < 1:
      torque = -self.sm['carOutput'].actuatorsOutput.torque
    if now - self.alerts_updated > 1:
      self.alerts = []
      if not started:
        for key, definition in OFFROAD_ALERTS.items():
          value = self.params.get(key)
          if isinstance(value, dict) and isinstance(value.get('text'), str):
            self.alerts.append({'key': key, 'text': value['text'].replace('%1', str(value.get('extra', '')))[:1024],
                                'severity': definition.get('severity', 0)})
        self.alerts.sort(key=lambda alert: -alert['severity'])
      self.alerts_updated = now
    car = services.get('carState', {})
    if car.get('vEgo', 0) < 5:
      self.wide = True
    elif car.get('vEgo', 0) > 10:
      self.wide = False
    camera = 'wideRoad' if self.wide and services.get('selfdriveState', {}).get('experimentalMode') else 'road'
    return {'services': services, 'gpu': self.gpu if ds else 'unknown', 'confidence': confidence, 'torque': torque,
            'alert': alert, 'offroadAlerts': self.alerts if not started else [], 'camera': camera,
            'isMetric': self.params.get_bool('IsMetric'), 'alwaysOnDM': self.params.get_bool('AlwaysOnDM')}

  async def send(self, message: str | bytes):
    channel = self.stream.get_messaging_channel()
    if not self.channel_ready:
      loop = asyncio.get_running_loop()
      channel.set_buffered_amount_low_threshold(0)
      channel.on_buffered_amount_low(lambda: loop.call_soon_threadsafe(self.drained.set))
      self.channel_ready = True
    if self.closed or not channel.is_open():
      raise ConnectionError('in-car connection closed')
    self.drained.clear()
    # buffered_amount() in the deployed libdatachannel binding crashes. Wait for
    # the native drain notification instead, keeping at most one chunk queued.
    if not channel.send(message):
      await asyncio.wait_for(self.drained.wait(), timeout=2)

  async def run(self):
    try:
      await asyncio.wait_for(self.stream.wait_for_connection(), timeout=30)
      while not self.closed and self.authorized():
        await asyncio.wait_for(self.request.wait(), timeout=10)
        if not self.authorized():
          break
        identifier = self.request_id
        started = time.monotonic()
        state = self.snapshot()
        frame = await asyncio.to_thread(self.camera.read, state['camera'])
        jpeg = b''
        meta = {}
        if frame is not None:
          pixels, width, height, camera, captured = frame
          try:
            jpeg = await self.encoder.encode(pixels, width, height)
            # Do not deliver a frame held up by encoder startup/stalls.
            if time.monotonic_ns() - captured > 500_000_000:
              jpeg = b''
            else:
              meta = {'width': width, 'height': height, 'camera': camera}
          except (TimeoutError, ValueError, OSError, asyncio.IncompleteReadError, asyncio.LimitOverrunError):
            await self.encoder.close()
        # Refresh HUD after camera work so blocked capture never replays engagement.
        state = self.snapshot()
        if not self.authorized():
          break
        await self.send(json.dumps({'type': 'frame', 'id': identifier, 'length': len(jpeg), 'state': state, **meta}, separators=(',', ':')))
        for offset in range(0, len(jpeg), CHUNK_BYTES):
          await self.send(struct.pack('>I', identifier) + jpeg[offset:offset + CHUNK_BYTES])
        self.request.clear()
        await self.send(json.dumps({'type': 'done', 'id': identifier}))
        await asyncio.sleep(max(0, 0.1 - (time.monotonic() - started)))
    except (TimeoutError, ConnectionError):
      pass
    finally:
      await self.stop()

  async def stop(self):
    if self.closed:
      return
    self.closed = True
    if self.run_task and self.run_task is not asyncio.current_task():
      self.run_task.cancel()
      with contextlib.suppress(asyncio.CancelledError):
        await self.run_task
    await self.encoder.close()
    await self.stream.stop()
