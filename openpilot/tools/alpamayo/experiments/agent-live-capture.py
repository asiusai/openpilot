"""Optional native VisionIPC CPU probe. Importing this file opens no connections.

The CLI requires explicit local device identifiers. It never starts openpilot,
loads a model, changes parameters, or publishes messages. Hardware use is deferred.
"""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
import threading
import time

import numpy as np


def local(name):
  spec = importlib.util.spec_from_file_location(name.replace('-', '_'), Path(__file__).with_name(name+'.py'))
  module = importlib.util.module_from_spec(spec)
  sys.modules[spec.name] = module
  spec.loader.exec_module(module)
  return module


bridge = local('agent-live-bridge')


def copy_vision_buffer(camera, client, buf, clock=time.monotonic_ns):
  received = clock()
  width, height, stride, uv_offset = int(buf.width), int(buf.height), int(buf.stride), int(buf.uv_offset)
  if (width, height) != (1344, 760) or stride < width or uv_offset < height*stride:
    raise ValueError('Native geometry differs from the recorded 1344x760 front source; qualify it explicitly')
  raw = np.frombuffer(buf.data, dtype=np.uint8)
  if len(raw) < uv_offset+(height//2-1)*stride+width:
    raise ValueError('Truncated VisionIPC NV12 buffer')
  # Copy before the next recv can release/reuse the VisionIPC slot. No resizing.
  y = np.ndarray((height, width), np.uint8, buffer=raw, strides=(stride, 1))
  uv = np.ndarray((height//2, width), np.uint8, buffer=raw, offset=uv_offset, strides=(stride, 1))
  packed = np.concatenate((y, uv))
  packed.flags.writeable = False
  ready = clock()
  return bridge.Frame(camera, int(client.frame_id), int(client.timestamp_sof), int(client.timestamp_eof),
                      received, ready, packed, 'nv12', ready-received)


def state_value(kind, msg):
  """Use current service names; liveCalibration/liveParameters are historical names."""
  if kind == 'carState':
    return {k: float(getattr(msg, k)) for k in ('vEgo', 'steeringAngleDeg', 'aEgo')}, True
  if kind == 'carParams':
    from calibration_trials import CP_FIELDS
    return {k: float(getattr(msg, k)) for k in CP_FIELDS}, True
  if kind == 'vehicleParameters':
    return {k: float(getattr(msg, k)) for k in ('stiffnessFactor', 'steerRatio', 'angleOffsetDeg', 'roll')}, bool(msg.valid)
  if kind == 'extrinsicsCalibration':
    return {k: list(getattr(msg, k)) for k in ('rpyCalib', 'wideFromDeviceEuler')}, str(msg.calStatus) == 'calibrated'
  if kind == 'deviceMotion':
    value = [[float(getattr(msg.orientationNED, a)) for a in 'xyz'],
             [float(getattr(msg.velocityDevice, a)) for a in 'xyz']]
    return value, bool(msg.inputsOK and msg.orientationNED.valid and msg.velocityDevice.valid and msg.timestamp > 0)
  if kind == 'modelV2':
    return msg.to_dict(), True
  raise ValueError(kind)


class NativeProducer:
  """Two conflated camera readers plus one state reader. No inference queue."""
  def __init__(self, store):
    self.store = store
    self.stop = threading.Event()
    self.threads = []
    self.error = None
    self.health = None

  def guarded(self, work, *args):
    try:
      work(*args)
    except Exception as exc:
      self.error = exc
      self.stop.set()

  def camera(self, camera, stream):
    from msgq.visionipc import VisionIpcClient
    client = VisionIpcClient('camerad', stream, True)
    connect_deadline = time.monotonic()+5
    while not self.stop.is_set() and not client.connect(False):
      if time.monotonic() > connect_deadline:
        raise RuntimeError('Existing camerad did not provide the requested native stream')
      self.stop.wait(.05)
    while not self.stop.is_set():
      buf = client.recv(100)
      if buf is not None:
        self.store.add_frame(copy_vision_buffer(camera, client, buf))

  def states(self):
    from openpilot.cereal import messaging
    from openpilot.common.params import Params
    kinds = (*bridge.STATE_LIMITS, 'modelV2')
    sm = messaging.SubMaster([*kinds, 'deviceState'])
    params = Params()
    while not self.stop.is_set():
      # Parsing/deserialization happens inside update. Record its CPU boundary
      # independently; avoid claiming that upstream CAN decoding is measured.
      begin = time.thread_time_ns()
      sm.update(0)
      received = time.monotonic_ns()
      parse_ns = time.thread_time_ns()-begin
      updated = [k for k in kinds if sm.updated[k]]
      for index, kind in enumerate(updated):
        work = time.thread_time_ns()
        msg = sm[kind]
        value, valid = state_value(kind, msg)
        stamp = int(msg.timestamp) if kind == 'deviceMotion' else int(sm.logMonoTime[kind])
        # Reject NaN/Infinity before retaining state. Invalid message updates
        # remain timestamped tombstones rather than silently holding old values.
        try:
          json.dumps(value, allow_nan=False)
        except ValueError:
          valid = False
        self.store.add_state(kind, stamp, received, value, bool(sm.valid[kind] and valid),
                             time.thread_time_ns()-work+(parse_ns if index == 0 else 0))
      if params.get_bool('ChestnutActive') or params.get_bool('ChestnutLoading'):
        raise RuntimeError('Normal openpilot attempted Chestnut ownership')
      now = time.monotonic_ns()
      self.health = dict(time_ns=now, model_alive=bool(sm.alive['modelV2'] and sm.valid['modelV2']),
                         model_age_ns=now-int(sm.logMonoTime['modelV2']),
                         device_valid=bool(sm.alive['deviceState'] and sm.valid['deviceState']),
                         thermal_status=str(sm['deviceState'].thermalStatus))
      self.stop.wait(.005)

  def start(self):
    from openpilot.cereal.visionipc import VisionStreamType
    jobs = [(self.camera, (1, VisionStreamType.VISION_STREAM_WIDE_ROAD)),
            (self.camera, (6, VisionStreamType.VISION_STREAM_NARROW_ROAD)), (self.states, ())]
    for work, args in jobs:
      thread = threading.Thread(target=self.guarded, args=(work, *args), daemon=True)
      self.threads.append(thread)
      thread.start()

  def close(self):
    self.stop.set()
    for thread in self.threads:
      thread.join(timeout=2)
    if any(thread.is_alive() for thread in self.threads):
      raise RuntimeError('A native reader did not stop within its bounded receive timeout')


def verify_small_model(model_pid, manager_pid, prior=None):
  model, manager = Path('/proc')/str(model_pid), Path('/proc')/str(manager_pid)
  status = dict(line.split(':', 1) for line in (model/'status').read_text().splitlines())
  if int(status['PPid']) != manager_pid or b'manager.py' not in (manager/'cmdline').read_bytes():
    raise RuntimeError('Model/manager process relation changed')
  command = (model/'cmdline').read_bytes()
  if (model/'comm').read_text().strip() != 'modeld' and b'modeld' not in command:
    raise RuntimeError('The selected PID is not modeld')
  # Read only the needed setting; never display process environments.
  if b'DISABLE_CHESTNUT=1' not in (manager/'environ').read_bytes().split(b'\0'):
    raise RuntimeError('The running model supervisor must already select the small model')
  identity = tuple((p/'stat').read_text().rsplit(')', 1)[1].split()[19] for p in (model, manager))
  if prior is not None and identity != prior:
    raise RuntimeError('Model or manager restarted; stop the probe')
  return identity


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--live-cpu-only', action='store_true', required=True)
  parser.add_argument('--serial', required=True)
  parser.add_argument('--model-pid', type=int, required=True)
  parser.add_argument('--manager-pid', type=int, required=True)
  parser.add_argument('--geometry-manifest', type=Path, required=True)
  parser.add_argument('--output', type=Path, required=True, help='New JSONL file; existing parent directory')
  parser.add_argument('--seconds', type=float, default=60)
  parser.add_argument('--hz', type=float, choices=(5., 10.), default=5.)
  parser.add_argument('--warp', choices=('native', 'scipy'), default='native')
  args = parser.parse_args()
  if os.getuid() == 0 or not 0 < args.seconds <= 600:
    parser.error('Run as comma for at most 600 seconds')
  if Path('/sys/devices/soc0/serial_number').read_text().strip() != args.serial:
    raise RuntimeError('Wrong SoC serial')
  identity = verify_small_model(args.model_pid, args.manager_pid)
  reserve = 64*1024**2
  usage = shutil.disk_usage(args.output.parent)
  if not bridge.storage_ok(usage.total, usage.free, reserve):
    raise RuntimeError('Probe would cross the 10% or 5 GiB storage reserve')
  manifest = json.loads(args.geometry_manifest.read_text())
  if manifest.get('camera_input_profile') != 'calibrated_vehicle_history' or manifest.get('camera_subset') != [1, 6]:
    raise ValueError('Use the calibrated two-front-camera geometry manifest')
  warp = bridge.calibrated_image
  if args.warp == 'native':
    from native_warp import calibrated_image as warp, library
    library()  # Explicit initialization, before the measured observation window.
  store = bridge.Store()
  prep = bridge.Preparer(manifest['source_camera_geometry'], manifest['target_camera_geometry'], warp=warp)
  runner = bridge.Bridge(store, prep, hz=args.hz, require_small_model=True)
  producer = NativeProducer(store)
  os.umask(0o077)
  with args.output.open('x') as output:
    written = 0
    def save(record):
      nonlocal written
      raw = json.dumps(record, allow_nan=False, separators=(',', ':'))+'\n'
      size = len(raw.encode())
      usage = shutil.disk_usage(args.output.parent)
      if written+size > reserve or not bridge.storage_ok(usage.total, usage.free, size):
        raise RuntimeError('Bounded output or storage reserve reached; no files are deleted')
      output.write(raw)
      output.flush()
      written += size
    save(dict(kind='manifest', model_predictions=False, warp=args.warp, source_sof='v0 EOF minus sensor readout estimate',
              services=list(bridge.STATE_LIMITS), camera_hz='native; no temporal decimation',
              normal_model_pid=args.model_pid, geometry_manifest=str(args.geometry_manifest),
              scope='native receive/copy, RGB conversion, calibrated projection, causal state and learned history'))
    started = time.monotonic()
    next_guard = started
    producer.start()
    try:
      while time.monotonic()-started < args.seconds:
        if producer.error is not None:
          raise RuntimeError('Native producer failed') from producer.error
        if time.monotonic() >= next_guard:
          verify_small_model(args.model_pid, args.manager_pid, identity)
          health = producer.health
          if time.monotonic()-started > 5 and (health is None or not health['model_alive'] or
              not health['device_valid'] or not 0 <= health['model_age_ns'] < 250_000_000 or
              time.monotonic_ns()-health['time_ns'] > 500_000_000 or health['thermal_status'] in ('red', 'danger')):
            raise RuntimeError('Small-model or device health guard failed')
          mem = dict(line.split(':', 1) for line in Path('/proc/meminfo').read_text().splitlines())
          if int(mem['MemAvailable'].split()[0]) < 768*1024:
            raise RuntimeError('Host memory reserve reached')
          usage = shutil.disk_usage(args.output.parent)
          if not bridge.storage_ok(usage.total, usage.free):
            raise RuntimeError('Storage reserve reached')
          next_guard = time.monotonic()+.25
        row = runner.run_once()
        if row is not None:
          save(row)
        producer.stop.wait(.002)
    finally:
      producer.close()
      save(dict(kind='summary', **runner.summary(), limits=store.bounds()))


if __name__ == '__main__':
  main()
