"""Short, controlled lens, camera-calibration and measured-history experiments."""
import argparse
from collections import OrderedDict
import copy
import hashlib
import json
import math
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from scipy.ndimage import map_coordinates
from scipy.spatial.transform import Rotation

from camera_profiles import NAMES, Transform, replay_module

VIEW = np.array([[0., 1., 0.], [0., 0., 1.], [1., 0., 0.]])
PROFILES = {
  'baseline': {}, 'wide110': {1: 110}, 'wide130': {1: 130},
  'tele25': {6: 25}, 'tele35': {6: 35}, 'resample_control': {},
  'calibrated_fronts': {}, 'calibrated_nvidia_fronts': {},
  'learned_vehicle_history': {}, 'calibrated_vehicle_history': {},
}
LENS_COMBINATIONS = {
  'baseline': {}, 'wide110': {1: 110},
  'wide110_tele25': {1: 110, 6: 25},
  'wide100_tele25': {1: 100, 6: 25},
  'wide110_tele20': {1: 110, 6: 20},
}
FULL_CALIBRATION = {name: {} for name in (
  'calibrated_fronts', 'calibrated_nvidia_fronts', 'learned_vehicle_history', 'calibrated_vehicle_history')}
CP_FIELDS = ('rotationalInertiaPerMass', 'wheelbase', 'centerToFront',
             'tireStiffnessFrontPerMass', 'tireStiffnessRearPerMass', 'steerRatio')


def asof(records, ns, max_age):
  times, values = records
  i = int(np.searchsorted(times, ns, side='right'))-1
  if i < 0 or ns-int(times[i]) > max_age:
    raise ValueError(f'Missing causal state at {ns}')
  return values[i], int(times[i])


def vehicle_records(logs):
  from openpilot.cereal import log
  import zstandard
  records = {k: {} for k in ('carState', 'vehicleParameters', 'carParams')}
  checksums = {}
  for path in sorted(logs.glob('*/rlog.zst')):
    checksums[str(path.relative_to(logs))] = hashlib.sha256(path.read_bytes()).hexdigest()
    raw = zstandard.ZstdDecompressor().stream_reader(path.open('rb'))
    with raw:
      for event in log.Event.read_multiple_bytes(raw.read()):
        kind = event.which()
        if kind not in records or not event.valid:
          continue
        msg = getattr(event, kind)
        if kind == 'carState':
          value = {k: float(getattr(msg, k)) for k in ('vEgo', 'steeringAngleDeg', 'aEgo')}
          value.update(brakePressed=bool(msg.brakePressed), gasPressed=bool(msg.gasPressed))
        elif kind == 'vehicleParameters':
          if not msg.valid:
            continue
          value = {k: float(getattr(msg, k)) for k in ('stiffnessFactor', 'steerRatio', 'angleOffsetDeg', 'roll')}
        else:
          value = {k: float(getattr(msg, k)) for k in CP_FIELDS}
        records[kind][int(event.logMonoTime)] = value
  if any(not rows for rows in records.values()):
    raise ValueError('Missing valid vehicle state or learned parameters')
  return {kind: (np.array(sorted(rows), dtype=np.int64), [rows[k] for k in sorted(rows)])
          for kind, rows in records.items()}, checksums


def vehicle_history(row, records):
  """Causal planar bicycle-model odometry; keep measured vertical motion and tilt."""
  from opendbc.car.vehicle_model import VehicleModel
  ns = int(row['query_ns'])
  cp, _ = asof(records['carParams'], ns, 120_000_000_000)
  vm = VehicleModel(SimpleNamespace(**cp))
  times = ns + np.arange(-38, 1, dtype=np.int64)*50_000_000
  speed, yaw_rate, ages = [], [], []
  for stamp in times:
    cs, ct = asof(records['carState'], int(stamp), 150_000_000)
    lp, lt = asof(records['vehicleParameters'], int(stamp), 1_000_000_000)
    vm.update_params(max(lp['stiffnessFactor'], .1), max(lp['steerRatio'], .1))
    v = max(cs['vEgo'], 0.)
    # VehicleModel positive steering curvature is left in Alpamayo FLU axes.
    curvature = vm.calc_curvature(math.radians(cs['steeringAngleDeg']-lp['angleOffsetDeg']), v, lp['roll'])
    speed.append(v)
    yaw_rate.append(v*curvature)
    ages.append([int(stamp)-ct, int(stamp)-lt])
  speed, yaw_rate = np.array(speed), np.array(yaw_rate)
  yaw = np.r_[0., np.cumsum((yaw_rate[:-1]+yaw_rate[1:])*.025)]
  velocity = speed[:, None]*np.stack([np.cos(yaw), np.sin(yaw)], axis=-1)
  xy = np.vstack([np.zeros(2), np.cumsum((velocity[:-1]+velocity[1:])*.025, axis=0)])
  current = Rotation.from_euler('z', -yaw[-1]).as_matrix()[:2, :2]
  xy = (xy-xy[-1]) @ current.T
  xyz = np.asarray(row['ego_history_xyz'], dtype=np.float32).copy()
  rotations = np.asarray(row['ego_history_rot'], dtype=np.float32).copy()
  xyz[0, :, :2] = xy[::2]
  tilt = Rotation.from_matrix(rotations[0]).as_euler('xyz')
  tilt[:, 2] = yaw[::2]-yaw[-1]
  rotations[0] = Rotation.from_euler('xyz', tilt).as_matrix()
  if not np.isfinite(xyz).all() or not np.isfinite(rotations).all():
    raise ValueError('Nonfinite learned-parameter history')
  return xyz, rotations, {'max_car_state_age_ms': max(a[0] for a in ages)/1e6,
                          'max_learned_parameters_age_ms': max(a[1] for a in ages)/1e6,
                          'speed_m_s': float(speed[-1]), 'left_yaw_rate_rad_s': float(yaw_rate[-1])}


def calibrated_image(array, camera, source, target, calibration, target_nvidia=False, identity=False):
  """Rotate virtual calibrated rays into each measured front camera's source rays."""
  src = source['cameras'][str(camera)]
  w, h = src['prepared_size']
  left, top, right, bottom = src['content_box']
  fx, fy = src['focal_pixels']
  cx, cy = src['principal_point']
  yy, xx = np.mgrid[:h, :w]
  if target_nvidia:
    dst = target['cameras'][NAMES[camera]]
    tx = (xx+.5)*dst['width']/w-.5-dst['principal_point'][0]
    ty = (yy+.5)*dst['height']/h-.5-dst['principal_point'][1]
    radius = np.hypot(tx, ty)
    theta = np.polynomial.polynomial.polyval(radius, dst['r2th'])
  else:
    fov = 120 if camera == 1 else 30
    fraction = (fx*math.radians(fov)/w if src['projection'] == 'equidistant'
                else 2*fx*math.tan(math.radians(fov/2))/w)
    tx, ty = (xx-(w-1)/2)*fraction/fx, (yy-(h-1)/2)*fraction/fy
    radius = np.hypot(tx, ty)
    theta = radius if src['projection'] == 'equidistant' else np.arctan(radius)
  scale = np.divide(np.sin(theta), radius, out=np.ones_like(radius), where=radius > 0)
  rays = np.stack([tx*scale, ty*scale, np.cos(theta)], axis=-1)
  rotation = np.eye(3)
  if not identity:
    rotation = Rotation.from_euler('xyz', calibration['rpyCalib']).as_matrix()
    if camera == 1:
      rotation = Rotation.from_euler('xyz', calibration['wideFromDeviceEuler']).as_matrix() @ rotation
    rotation = VIEW @ rotation @ VIEW.T
  rays = rays @ rotation.T
  radial = np.hypot(rays[..., 0], rays[..., 1])
  theta = np.arctan2(radial, rays[..., 2])
  radius = theta if src['projection'] == 'equidistant' else np.tan(theta)
  factor = np.divide(radius, radial, out=np.ones_like(radius), where=radial > 1e-12)
  sx, sy = cx+fx*rays[..., 0]*factor, cy+fy*rays[..., 1]*factor
  if (rays[..., 2] <= 0).any() or sx.min() < left or sx.max() > right-1 or sy.min() < top or sy.max() > bottom-1:
    raise ValueError(f'Camera {camera}: calibrated rays exceed recorded image coverage')
  source_h, source_w = array.shape[-2:]
  sx = (sx-left+.5)*source_w/(right-left)-.5
  sy = (sy-top+.5)*source_h/(bottom-top)-.5
  return np.stack([map_coordinates(channel, (sy, sx), order=1, mode='nearest', prefilter=False)
                   for channel in array]).astype(np.uint8)


def prepare(args):
  replay = replay_module()
  if (args.output/'manifest.json').exists():
    raise FileExistsError('Use a new trial directory')
  baseline = replay.read_json(args.source/'manifest.json')
  source, target = replay.read_json(args.source_geometry), replay.read_json(args.target_geometry)
  calibrations = replay.read_json(args.source/'render-calibrations.json')
  calibration_records = (np.array([r['ns'] for r in calibrations]), calibrations)
  records, hashes = vehicle_records(args.logs)
  if hashes != baseline['rlog_sha256']:
    raise ValueError('Vehicle logs differ from the baseline recording')
  study = getattr(args, 'study', 'calibration')
  profiles = FULL_CALIBRATION if study == 'full-calibration' else PROFILES if study == 'calibration' else LENS_COMBINATIONS
  windows = ([('tune', 112., 120.), ('holdout', 164., 172.)] if study == 'calibration'
             else [('tune', 112., 120.), ('tune', 164., 172.), ('holdout', 228., 236.)])
  selected = ([('full', row) for row in baseline['schedule'] if not row['skip_reasons']] if study == 'full-calibration' else
              [(split, row) for split, start, end in windows for row in baseline['schedule']
               if start <= row['display_time_s'] < end and row['tick'] % 2 == 0])
  if study == 'full-calibration':
    windows = [('full', 0., baseline['duration_s'])]
    if len(selected) != baseline['valid_samples']:
      raise ValueError('Full calibration study must retain every valid baseline query')
  elif len(selected) != 16*len(windows) or any(r['available_views'] != [0, 1, 2, 3, 5, 6] for _, r in selected):
    raise ValueError('Require 16 complete six-camera snapshots in each predeclared window')
  args.output.mkdir(parents=True, exist_ok=True)
  selection = {'windows': windows, 'ticks': {s: [r['tick'] for split, r in selected if split == s] for s, _, _ in windows},
               'profiles': profiles, 'study': study,
               'source_manifest_sha256': hashlib.sha256((args.source/'manifest.json').read_bytes()).hexdigest(),
               'limit': 'One previously inspected drive; reserved temporal window is not independent driving validation',
               'reference': 'Future recorded motion is comparison-only; no future measurements enter model inputs'}
  replay.write_json(args.output/'selection.json', selection)
  rows, audit = [], []
  for profile, operations in profiles.items():
    transform = Transform(profile, source, target, operations)
    image_cache = OrderedDict()
    for split, original in selected:
      row = copy.deepcopy(original)
      old = Path(row['sample'])
      new = args.output/'samples'/profile/old.name
      new.mkdir(parents=True, exist_ok=False)
      image_change = bool(operations) or profile in ('resample_control', 'calibrated_fronts', 'calibrated_nvidia_fronts', 'calibrated_vehicle_history')
      history_change = profile in ('learned_vehicle_history', 'calibrated_vehicle_history')
      for path in old.glob('*.npy'):
        if (path.name == 'image_frames.npy' and image_change) or (path.stem in ('ego_history_xyz', 'ego_history_rot') and history_change):
          continue
        os.link(path, new/path.name)
      shared_images = args.output/'samples'/'calibrated_fronts'/old.name/'image_frames.npy'
      if profile == 'calibrated_vehicle_history' and shared_images.exists():
        os.link(shared_images, new/'image_frames.npy')
        for frame in row['frames']:
          if frame['view'] in (1, 6):
            _, frame['image_calibration_ns'] = asof(calibration_records, frame['capture_ns'], 1_000_000_000)
      elif image_change:
        old_images = np.load(old/'image_frames.npy', mmap_mode='r')
        images = []
        for image, frame in zip(old_images, row['frames'], strict=True):
          camera = frame['view']
          if camera in operations or (not operations and camera in (1, 6)):
            raw = np.load(args.raw_frames/str(camera)/f'{frame["video_index"]}.npy')
            if operations:
              image = transform.image(raw, camera)[None]
            else:
              calibration, stamp = asof(calibration_records, frame['capture_ns'], 1_000_000_000)
              key = (camera, frame['video_index'], stamp)
              if key not in image_cache:
                image_cache[key] = calibrated_image(raw, camera, source, target, calibration,
                                                    target_nvidia=profile == 'calibrated_nvidia_fronts',
                                                    identity=profile == 'resample_control')[None]
              image = image_cache[key]
              image_cache.move_to_end(key)
              if len(image_cache) > 96:
                image_cache.popitem(last=False)
              frame['image_calibration_ns'] = stamp
          images.append(image)
        np.save(new/'image_frames.npy', np.stack(images), allow_pickle=False)
      if history_change:
        xyz, rotations, detail = vehicle_history(row, records)
        np.save(new/'ego_history_xyz.npy', xyz, allow_pickle=False)
        np.save(new/'ego_history_rot.npy', rotations, allow_pickle=False)
        row['ego_history_xyz'], row['ego_history_rot'] = xyz.tolist(), rotations.tolist()
        row['vehicle_history_audit'] = detail
        audit.append({'profile': profile, 'tick': row['tick'], **detail})
      row.update(sample=str(new.resolve()), camera_profile=profile, split=split)
      rows.append(row)
    print(f'Prepared {profile}: {len(selected)} inputs', flush=True)
  manifest = copy.deepcopy(baseline)
  manifest.update(schedule=rows, requested_samples=len(rows), valid_samples=len(rows), calibration_trials=selection,
                  input_policy='Historical frames retain official causal floor selection; calibration is causal per frame; CAN history is an explicit planar experiment')
  replay.write_json(args.output/'manifest.json', manifest)
  replay.write_json(args.output/'vehicle-history-audit.json', audit)


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--source', type=Path, required=True)
  parser.add_argument('--output', type=Path, required=True)
  parser.add_argument('--logs', type=Path, required=True)
  parser.add_argument('--raw-frames', type=Path, required=True)
  parser.add_argument('--source-geometry', type=Path, required=True)
  parser.add_argument('--target-geometry', type=Path, required=True)
  parser.add_argument('--study', choices=('calibration', 'lens-combinations', 'full-calibration'), default='calibration')
  prepare(parser.parse_args())


if __name__ == '__main__':
  main()
