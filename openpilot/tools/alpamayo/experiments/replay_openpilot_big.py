"""Replay the installed openpilot big model offline, with continuous recurrent state.

Run with the openpilot checkout's Python environment and pinned tinygrad package.
This module does not start messaging publishers or vehicle control processes.
"""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import time
from types import SimpleNamespace

import numpy as np

AXES = np.diag([1., -1., -1.])
FUTURE = np.arange(1, 65)*.1
CAR_FIELDS = ('vEgo', 'leftBlinker', 'rightBlinker', 'steeringPressed', 'steeringTorque', 'leftBlindspot', 'rightBlindspot')


def read_json(path):
  return json.loads(path.read_text())


def write_json(path, value):
  temporary = path.with_suffix(path.suffix+'.partial')
  temporary.write_text(json.dumps(value, indent=2)+'\n')
  temporary.replace(path)


def sha(path):
  with path.open('rb') as source:
    return hashlib.file_digest(source, 'sha256').hexdigest()


def check_mount(path, uuid):
  parent = path
  while not parent.exists():
    parent = parent.parent
  actual = subprocess.check_output(['findmnt', '-n', '-o', 'UUID', '--target', str(parent)], text=True).strip()
  if actual != uuid:
    raise ValueError(f'Recording filesystem differs: {actual}')


def source_hashes(checkout):
  names = ('openpilot/selfdrive/modeld/modeld.py', 'openpilot/selfdrive/modeld/parse_model_outputs.py',
           'openpilot/selfdrive/modeld/constants.py', 'openpilot/selfdrive/controls/lib/desire_helper.py',
           'openpilot/common/transformations/model.py', 'openpilot/common/transformations/camera.py',
           'openpilot/system/camerad/cameras/nv12_info.py', 'openpilot/selfdrive/modeld/helpers.py',
           'tinygrad_repo/examples/openpilot/helpers.py')
  return {name: sha(checkout/name) for name in names}


def prepare(args):
  from openpilot.cereal import log
  import zstandard
  check_mount(args.output, args.recording_uuid)
  if (args.output/'manifest.json').exists():
    raise FileExistsError('Prepared replay inputs are immutable; use a new output directory')
  manifest = read_json(args.source/'manifest.json')
  export = Path(manifest['export'])
  records = read_json(export/'clips/native-timestamps.json')['views']
  selected = ('carState', 'carControl', 'driverMonitoringState', 'extrinsicsCalibration', 'lateralDelay', 'carParams')
  events = defaultdict(list)
  log_hashes = {}
  devices, sensors = set(), set()
  for path in sorted(args.logs.glob('*/rlog.zst')):
    log_hashes[str(path.relative_to(args.logs))] = sha(path)
    with path.open('rb') as source:
      raw = zstandard.ZstdDecompressor().stream_reader(source).read()
    for event in log.Event.read_multiple_bytes(raw):
      kind = event.which()
      if kind == 'deviceState':
        devices.add(str(event.deviceState.deviceType))
      if kind == 'narrowRoadCameraState':
        sensors.add(str(event.narrowRoadCameraState.sensor))
      if kind not in selected:
        continue
      body = getattr(event, kind)
      if kind == 'carState':
        value = {key: getattr(body, key) for key in CAR_FIELDS}
      elif kind == 'carControl':
        value = {'latActive': bool(body.latActive)}
      elif kind == 'driverMonitoringState':
        value = {'isRHD': bool(body.isRHD)}
      elif kind == 'extrinsicsCalibration':
        if str(body.calStatus) != 'calibrated' or len(body.rpyCalib) != 3:
          continue
        value = {'rpyCalib': list(body.rpyCalib)}
      elif kind == 'lateralDelay':
        value = {'lateralDelay': float(body.lateralDelay)}
      else:
        value = {'longitudinalActuatorDelay': float(body.longitudinalActuatorDelay)}
      events[kind].append({'ns': int(event.logMonoTime), 'valid': bool(event.valid), 'value': value})
  if len(devices) != 1 or len(sensors) != 1:
    raise ValueError(f'Ambiguous camera hardware: {devices}, {sensors}')
  times = {}
  for kind in selected:
    events[kind] = sorted({row['ns']: row for row in events[kind]}.values(), key=lambda row: row['ns'])
    if not events[kind]:
      raise ValueError(f'Missing recorded input: {kind}')
    times[kind] = np.array([row['ns'] for row in events[kind]], dtype=np.int64)
  wide_by_frame = {row['frame_id']: (index, row) for index, row in enumerate(records['1'])}
  rows, invalid = [], Counter()
  end_ns = manifest['capture_start_ns']+round(np.ceil(manifest['duration_s']*20)/20*1e9)
  for narrow_index, frame in enumerate(records['6']):
    ns = frame['capture_ns']
    if ns > end_ns:
      break
    state = {}
    for kind in selected:
      index = int(np.searchsorted(times[kind], ns, side='right'))-1
      if index < 0:
        break
      state[kind] = events[kind][index]
    if len(state) != len(selected):
      continue  # Start once recorded inputs exist, well before the displayed interval.
    wide_index, wide = wide_by_frame[frame['frame_id']]
    if abs(wide['capture_ns']-ns) > 10_000_000:
      raise ValueError('Native front cameras are not synchronized')
    for kind, entry in state.items():
      limit = 120_000_000_000 if kind == 'carParams' else 1_000_000_000
      if not 0 <= ns-entry['ns'] <= limit:
        raise ValueError(f'Stale {kind} at native frame {frame["frame_id"]}')
      invalid[kind] += not entry['valid']
    rows.append({'query_ns': ns, 'frame_id': frame['frame_id'], 'eof_ns': frame['eof_ns'],
                 'narrow_index': narrow_index, 'wide_index': wide_index, 'wide_capture_ns': wide['capture_ns'], 'state': state})
  if not rows or manifest['capture_start_ns']-rows[0]['query_ns'] < 20_000_000_000:
    raise ValueError('Insufficient continuous recurrent pre-roll')
  if np.any(np.diff([row['narrow_index'] for row in rows]) != 1) or np.any(np.diff([row['query_ns'] for row in rows]) > 100_000_000):
    raise ValueError('Recurrent replay has a camera gap')
  model_dir = args.checkout/'openpilot/selfdrive/modeld/models'
  artifacts = {name: {'bytes': (model_dir/name).stat().st_size, 'sha256': sha(model_dir/name)}
               for name in ('big_driving_tinygrad.pkl', 'big_driving_warp_1344x760_tinygrad.pkl')}
  if artifacts['big_driving_tinygrad.pkl']['bytes'] < 100_000_000:
    raise ValueError('Big model is absent or is a Git LFS pointer')
  args.output.mkdir(parents=True, exist_ok=True)
  with (args.output/'inputs.jsonl').open('x') as output:
    for row in rows:
      output.write(json.dumps(row)+'\n')
  result = {'schema': 1, 'route': manifest['route'], 'source': str(args.source), 'export': str(export),
            'checkout': str(args.checkout), 'device_type': next(iter(devices)), 'sensor': next(iter(sensors)),
            'capture_start_ns': manifest['capture_start_ns'], 'capture_end_ns': end_ns,
            'first_inference_ns': rows[0]['query_ns'], 'continuous_frames': len(rows),
            'pre_roll_s': (manifest['capture_start_ns']-rows[0]['query_ns'])/1e9,
            'source_manifest_sha256': sha(args.source/'manifest.json'), 'rlog_sha256': log_hashes,
            'inputs_sha256': sha(args.output/'inputs.jsonl'), 'artifacts': artifacts,
            'source_sha256': source_hashes(args.checkout),
            'tinygrad_revision': subprocess.check_output(['git', '-C', str(args.checkout/'tinygrad_repo'), 'rev-parse', 'HEAD'], text=True).strip(),
            'invalid_message_counts': dict(invalid),
            'input_policy': 'Latest messages published at or before camera capture. As modeld does, consume recorded values even when event.valid is false.',
            'image_policy': 'Original front MP4 frames decoded directly to NV12; normal openpilot calibration warps; no Alpamayo crops.',
            'recurrent_policy': 'Zero initial state, then every recorded front-camera frame continuously; independent DesireHelper driven by big model outputs.',
            'limit': 'Offline replay of compressed camera recordings; no vehicle messages are published. Recurrent pre-roll precedes all displayed predictions.'}
  write_json(args.output/'manifest.json', result)
  print(json.dumps({key: result[key] for key in ('continuous_frames', 'pre_roll_s', 'artifacts', 'invalid_message_counts')}, indent=2), flush=True)


class NV12Reader:
  def __init__(self, path, width=1344, height=760):
    from openpilot.system.camerad.cameras.nv12_info import get_nv12_info
    self.width, self.height = width, height
    self.stride, self.y_height, self.uv_height, _ = get_nv12_info(width, height)
    self.frame = np.zeros((self.y_height+self.uv_height, self.stride), dtype=np.uint8)
    self.index = -1
    self.process = subprocess.Popen(['ffmpeg', '-v', 'error', '-threads', '2', '-noautorotate', '-i', str(path),
                                     '-map', '0:v:0', '-vsync', '0', '-pix_fmt', 'nv12', '-f', 'rawvideo', '-'], stdout=subprocess.PIPE)

  def at(self, index):
    if index < self.index:
      raise ValueError('Recurrent replay cannot seek backward')
    while self.index < index:
      size = self.width*self.height*3//2
      raw = self.process.stdout.read(size)
      if len(raw) != size:
        raise EOFError(f'Incomplete native frame {self.index+1}')
      self.index += 1
      if self.index == index:
        image = np.frombuffer(raw, dtype=np.uint8).reshape(self.height*3//2, self.width)
        self.frame[:self.height, :self.width] = image[:self.height]
        self.frame[self.y_height:self.y_height+self.height//2, :self.width] = image[self.height:]
    return self.frame.reshape(-1)

  def close(self):
    if self.process.poll() is None:
      self.process.terminate()
    self.process.stdout.close()
    self.process.wait()


def cpu_warp(frame, matrix):
  """Independent nearest-neighbor reference for the shipped NV12 warp."""
  packed = frame.reshape(1152, 1408)
  def sample(image, transform, width, height):
    yy, xx = np.mgrid[:height, :width].astype(np.float32)
    denom = transform[2, 0]*xx+transform[2, 1]*yy+transform[2, 2]
    x = np.clip(np.rint((transform[0, 0]*xx+transform[0, 1]*yy+transform[0, 2])/denom), 0, image.shape[1]-1).astype(int)
    y = np.clip(np.rint((transform[1, 0]*xx+transform[1, 1]*yy+transform[1, 2])/denom), 0, image.shape[0]-1).astype(int)
    return image[y, x]
  y = sample(packed[:760, :1344], matrix, 512, 256)
  uv_matrix = matrix*np.array([[1, 1, .5], [1, 1, .5], [2, 2, 1]], dtype=np.float32)
  u = sample(packed[768:1148, :1344:2], uv_matrix, 256, 128)
  v = sample(packed[768:1148, 1:1344:2], uv_matrix, 256, 128)
  return np.stack((y[::2, ::2], y[1::2, ::2], y[::2, 1::2], y[1::2, 1::2], u, v))


def infer(args):
  import inspect
  import tinygrad
  from openpilot.cereal import log
  from openpilot.selfdrive.modeld.modeld import ModelState
  from openpilot.selfdrive.modeld.constants import ModelConstants, Plan
  from openpilot.selfdrive.controls.lib.desire_helper import DesireHelper
  from openpilot.common.transformations.camera import DEVICE_CAMERAS
  from openpilot.common.transformations.model import get_warp_matrix
  from tinygrad.runtime.ops_amd import AMDDevice
  check_mount(args.output, args.recording_uuid)
  manifest = read_json(args.output/'manifest.json')
  checkout = Path(manifest['checkout'])
  if Path(inspect.getfile(ModelState)).resolve() != (checkout/'openpilot/selfdrive/modeld/modeld.py').resolve():
    raise ValueError('Imported ModelState is from another checkout')
  if Path(tinygrad.__file__).resolve().parent != (checkout/'tinygrad_repo/tinygrad').resolve():
    raise ValueError('Imported tinygrad is from another checkout')
  revision = subprocess.check_output(['git', '-C', str(checkout/'tinygrad_repo'), 'rev-parse', 'HEAD'], text=True).strip()
  if revision != manifest['tinygrad_revision']:
    raise ValueError('Pinned tinygrad revision changed')
  if source_hashes(checkout) != manifest['source_sha256'] or sha(args.output/'inputs.jsonl') != manifest['inputs_sha256']:
    raise ValueError('Replay sources changed after preparation')
  for name, info in manifest['artifacts'].items():
    if sha(checkout/'openpilot/selfdrive/modeld/models'/name) != info['sha256']:
      raise ValueError(f'Compiled artifact changed: {name}')
  if (args.output/'predictions.jsonl').exists() or (args.output/'predictions.partial.jsonl').exists():
    raise FileExistsError('Continuous recurrent replay must start in a new output directory')
  AMDDevice.wait_timeout_ms = 10000
  start_utc, started = datetime.now(timezone.utc).isoformat(), time.perf_counter()
  model = ModelState(1344, 760, chestnut=True)
  model.warmup()
  load_s = time.perf_counter()-started
  camera = DEVICE_CAMERAS[(manifest['device_type'], manifest['sensor'])]
  readers = {'img': NV12Reader(Path(manifest['export'])/'clips/front-telephoto.mp4'),
             'big_img': NV12Reader(Path(manifest['export'])/'clips/front-wide.mp4')}
  desire = DesireHelper()
  rows = [json.loads(line) for line in (args.output/'inputs.jsonl').read_text().splitlines()]
  timings, rendered, warp_checks = [], [], []
  partial = args.output/'predictions.partial.jsonl'
  try:
    with partial.open('x') as output:
      for index, row in enumerate(rows):
        state = {key: entry['value'] for key, entry in row['state'].items()}
        rpy = state['extrinsicsCalibration']['rpyCalib']
        transforms = {'img': get_warp_matrix(np.array(rpy), camera.narrow_road.intrinsics, False).astype(np.float32),
                      'big_img': get_warp_matrix(np.array(rpy), camera.wide_road.intrinsics, True).astype(np.float32)}
        frames = {'img': readers['img'].at(row['narrow_index']), 'big_img': readers['big_img'].at(row['wide_index'])}
        pulse = np.eye(ModelConstants.DESIRE_LEN, dtype=np.float32)[int(desire.desire)].copy()
        traffic = np.eye(2, dtype=np.float32)[int(state['driverMonitoringState']['isRHD'])]
        action_t = np.array([state['lateralDelay']['lateralDelay']+.075,
                             state['carParams']['longitudinalActuatorDelay']+.375], dtype=np.float32)
        before = time.perf_counter()
        result = model.run(frames, transforms, {'desire_pulse': pulse, 'traffic_convention': traffic, 'action_t': action_t})
        elapsed_ms = (time.perf_counter()-before)*1000
        timings.append(elapsed_ms)
        if index == 0:
          actual = model.input_queues['new_img'].numpy()
          expected = np.stack([cpu_warp(frames[key], transforms[key]) for key in ('img', 'big_img')])
          difference = actual.astype(float)-expected.astype(float)
          mismatch = float(np.mean(difference != 0))
          rms = float(np.sqrt(np.mean(difference**2)))
          warp_checks.append({'frame_id': row['frame_id'], 'different_pixel_fraction': mismatch, 'rms_byte_error': rms})
          write_json(args.output/'warp-validation.json', warp_checks)
          if actual.shape != (2, 6, 128, 256) or mismatch > .001 or rms > 2:
            raise ValueError(f'Compiled NV12 warp differs from independent CPU reference: {warp_checks}')
        path = result['plan'][0, :, Plan.POSITION]
        if path.shape != (33, 3) or not np.isfinite(path).all():
          raise ValueError('Invalid big model plan')
        canonical = np.stack([np.interp(FUTURE, ModelConstants.T_IDXS, path[:, a]) for a in range(3)], axis=-1) @ AXES
        probabilities = result['desire_state'][0]
        desire_input = model.npy['desire'].copy().tolist()
        desire.update(SimpleNamespace(**state['carState']), state['carControl']['latActive'],
                      float(probabilities[log.Desire.laneChangeLeft]+probabilities[log.Desire.laneChangeRight]))
        saved = {'query_ns': row['query_ns'], 'frame_id': row['frame_id'], 'eof_ns': row['eof_ns'],
                 'calibration_rpy': rpy, 'position': {'t': np.asarray(ModelConstants.T_IDXS).tolist(), **{a: path[:, i].tolist() for i, a in enumerate('xyz')}},
                 'xyz': canonical.tolist(), 'desire_state': probabilities.tolist(), 'desire_input': desire_input,
                 'lane_change_state': int(desire.lane_change_state), 'lane_change_direction': int(desire.lane_change_direction),
                 'action_t': action_t.tolist(), 'run_ms': elapsed_ms, 'big': True}
        output.write(json.dumps(saved)+'\n')
        output.flush()
        rendered.append((row['query_ns'], row['eof_ns'], row['frame_id'], rpy, canonical))
        if index % 200 == 0:
          print(f'Big model: {index+1}/{len(rows)} continuous frames; inference {elapsed_ms:.2f} ms', flush=True)
    if len(rendered) != manifest['continuous_frames']:
      raise ValueError('Incomplete recurrent replay')
    partial.replace(args.output/'predictions.jsonl')
    ns, eof, frame_ids, rpy, xyz = zip(*rendered, strict=True)
    np.savez_compressed(args.output/'render-openpilot-big.npz', ns=np.array(ns, dtype=np.int64),
                        published_ns=np.array(eof, dtype=np.int64), frame_id=np.array(frame_ids), rpy=np.array(rpy), xyz=np.array(xyz))
    total = time.perf_counter()-started
    write_json(args.output/'inference-summary.json', {'started_utc': start_utc, 'finished_utc': datetime.now(timezone.utc).isoformat(),
               'frames': len(rows), 'wall_seconds': total, 'including_startup_hz': len(rows)/total, 'load_and_warmup_seconds': load_s,
               'model_ms': {'median': float(np.median(timings)), 'p90': float(np.percentile(timings, 90)), 'max': max(timings)},
               'pre_roll_s': manifest['pre_roll_s'], 'warp_validation': warp_checks, 'model': manifest['artifacts'],
               'predictions_sha256': sha(args.output/'predictions.jsonl'), 'native_decode': 'FFmpeg NV12 without RGB conversion or rescaling',
               'limit': manifest['limit'], 'input_policy': manifest['input_policy']})
    print('Big model replay complete:', args.output, flush=True)
  finally:
    for reader in readers.values():
      reader.close()


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('action', choices=('prepare', 'infer'))
  parser.add_argument('--source', type=Path)
  parser.add_argument('--logs', type=Path)
  parser.add_argument('--checkout', type=Path)
  parser.add_argument('--output', type=Path, required=True)
  parser.add_argument('--recording-uuid', required=True)
  args = parser.parse_args()
  globals()[args.action](args)


if __name__ == '__main__':
  main()
