"""Read-only live input capture. Never publishes model or vehicle commands."""
import json
import os
from pathlib import Path
import threading
import time
from collections import deque

import cv2
import numpy as np
from openpilot.cereal import messaging
from openpilot.cereal.visionipc import VisionStreamType
from msgq.visionipc import VisionIpcClient
from openpilot.common.params import Params


def main():
  root = Path(os.environ['ALPAMAYO_SHADOW_DIR'])
  root.mkdir(parents=True, exist_ok=True)
  cv2.setNumThreads(1)
  lock = threading.Lock()
  frames = {'wide': deque(maxlen=65), 'narrow': deque(maxlen=65)}
  errors = []

  def camera(name, stream):
    try:
      client = VisionIpcClient('camerad', stream, True)
      while not client.connect(False):
        time.sleep(.2)
      last = 0
      while True:
        buf = client.recv(100)
        # camera_v0.cc leaves the optional VIPC valid member unset, as does the
        # common camerad sender. Match modeld: require a buffer and measured time.
        if buf is None or int(client.timestamp_eof) <= 0:
          continue
        timestamp = client.timestamp_eof
        if timestamp <= last:
          continue
        last = timestamp
        raw = np.frombuffer(buf.data, dtype=np.uint8)
        h, w, stride = buf.height, buf.width, buf.stride
        y = raw[:h*stride].reshape(h, stride)[:, :w]
        uv = raw[buf.uv_offset:buf.uv_offset+h//2*stride].reshape(h//2, stride)[:, :w]
        rgb = cv2.cvtColor(np.concatenate((y, uv)), cv2.COLOR_YUV2RGB_NV12)
        rgb = cv2.resize(rgb, (1344, 760), interpolation=cv2.INTER_AREA)
        record = (timestamp, client.frame_id, rgb.transpose(2, 0, 1).copy(), [w, h])
        with lock:
          frames[name].append(record)
    except Exception as exc:
      errors.append(repr(exc))

  for name, stream in [('wide', VisionStreamType.VISION_STREAM_WIDE_ROAD), ('narrow', VisionStreamType.VISION_STREAM_NARROW_ROAD)]:
    threading.Thread(target=camera, args=(name, stream), daemon=True).start()
  sm = messaging.SubMaster(['deviceMotion', 'extrinsicsCalibration', 'modelV2', 'deviceState'])
  motions, models, calibrations = deque(maxlen=120), deque(maxlen=120), deque(maxlen=20)
  params = Params()
  last_written = 0
  while True:
    if errors:
      raise RuntimeError(errors)
    sm.update(20)
    if params.get_bool('ChestnutActive') or params.get_bool('ChestnutLoading'):
      raise RuntimeError('openpilot attempted to use Chestnut; stopping shadow capture')
    if sm.updated['deviceMotion'] and sm.valid['deviceMotion']:
      m = sm['deviceMotion']
      if m.timestamp > 0 and m.orientationNED.valid and m.velocityDevice.valid and m.inputsOK:
        motions.append([m.timestamp, [getattr(m.orientationNED, a) for a in 'xyz'], [getattr(m.velocityDevice, a) for a in 'xyz']])
    if sm.updated['modelV2'] and sm.valid['modelV2']:
      models.append({'log_mono_time': int(sm.logMonoTime['modelV2']), 'model': sm['modelV2'].to_dict()})
    if sm.updated['extrinsicsCalibration'] and sm.valid['extrinsicsCalibration'] and str(sm['extrinsicsCalibration'].calStatus) == 'calibrated':
      calibrations.append((int(sm.logMonoTime['extrinsicsCalibration']), list(sm['extrinsicsCalibration'].rpyCalib)))
    now = time.monotonic_ns()
    health = {'time_ns': now, 'model_alive': bool(sm.alive['modelV2'] and sm.valid['modelV2']),
              'model_age_s': (now-int(sm.logMonoTime['modelV2']))/1e9,
              'thermal_status': str(sm['deviceState'].thermalStatus), 'motion_samples': len(motions),
              'camera_samples': {k: len(v) for k, v in frames.items()},
              'camera_latest': {k: [v[-1][0], v[-1][1]] if v else None for k, v in frames.items()},
              'model_frame': int(sm['modelV2'].frameId), 'calibrations': len(calibrations)}
    tmp = root / 'health.tmp'
    tmp.write_text(json.dumps(health))
    tmp.replace(root / 'health.json')
    if (root / 'pending.npz').exists() or now-last_written < 1_000_000_000 or len(motions) < 40 or not models:
      continue
    if not health['model_alive'] or health['model_age_s'] > .25:
      continue
    if not calibrations:
      continue
    with lock:
      candidates = {k: list(v) for k, v in frames.items()}
    if any(len(v) < 40 for v in candidates.values()):
      continue
    query = min(candidates['wide'][-1][0], candidates['narrow'][-1][0], motions[-1][0], now-150_000_000)
    if now-query > 500_000_000:
      continue
    chosen = [min(candidates[name], key=lambda f: abs(f[0]-(query+offset)))
              for name in ('wide', 'narrow') for offset in (-2_000_000_000, -1_000_000_000, -500_000_000, -100_000_000, 0)]
    offsets = (-2_000_000_000, -1_000_000_000, -500_000_000, -100_000_000, 0)*2
    if any(abs(f[0]-(query+offset)) >= 60_000_000 for f, offset in zip(chosen, offsets)):
      continue
    query = chosen[-1][0]
    available_calibrations = [c for c in calibrations if c[0] <= query]
    if not available_calibrations:
      continue
    calibration = available_calibrations[-1]
    small = min(models, key=lambda item: abs(item['model']['frameId']-chosen[-1][1]))
    if small['model']['frameId'] != chosen[-1][1]:
      continue
    route = params.get('CurrentRoute')
    if not route:
      continue
    metadata = {'query_ns': query, 'route': route.decode() if isinstance(route, bytes) else route,
                'boot_id': Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
                'camera_order': ['wide']*5+['narrow']*5, 'frame_ids': [f[1] for f in chosen],
                'camera_timestamps_ns': [f[0] for f in chosen], 'source_dimensions': [f[3] for f in chosen],
                'image_transform': 'full frame resized to 1344x760; no lens crop',
                'motion_timestamp_source': 'deviceMotion.timestamp', 'small_model': small,
                'calibration_rpy': calibration[1],
                'calibration_log_mono_time': calibration[0],
                'navigation': 'Follow the road.', 'navigation_source': 'fixed test instruction'}
    with (root / 'pending.tmp').open('wb') as output:
      np.savez(output, image_frames=np.stack([f[2] for f in chosen])[:, None],
               motion_times=np.array([m[0] for m in motions], dtype=np.int64),
               motion_rpy=np.array([m[1] for m in motions]), motion_velocity=np.array([m[2] for m in motions]),
               metadata=np.array(json.dumps(metadata)))
    (root / 'pending.tmp').replace(root / 'pending.npz')
    last_written = now


if __name__ == '__main__':
  main()
