"""Controlled camera-input crops and nominal reprojection for recorded replay."""
import argparse
from collections import OrderedDict
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil

import numpy as np
from PIL import Image
from scipy.ndimage import map_coordinates

NAMES = {0: 'camera_cross_left_120fov', 1: 'camera_front_wide_120fov', 2: 'camera_cross_right_120fov',
         3: 'camera_rear_left_70fov', 5: 'camera_rear_right_70fov', 6: 'camera_front_tele_30fov'}
PROFILES = {'baseline': {}, 'tele45': {6: 45}, 'tele35': {6: 35}, 'tele30': {6: 30},
            'fronts_crop': {1: 120, 6: 30},
            'all_crop': {0: 120, 1: 120, 2: 120, 3: 70, 5: 70, 6: 30},
            'all_map': dict.fromkeys(NAMES, 'nvidia')}


def replay_module():
  spec = importlib.util.spec_from_file_location('replay_route', Path(__file__).with_name('replay-route.py'))
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module


class Transform:
  def __init__(self, profile, source, target, operations=None):
    self.profile, self.source, self.target = profile, source, target
    self.operations = PROFILES[profile] if operations is None else operations
    self.maps, self.cache = {}, OrderedDict()

  def image(self, array, camera):
    operation = self.operations.get(camera)
    if operation is None:
      return array
    key = (camera, array.shape, hashlib.sha256(array.tobytes()).digest())
    if key in self.cache:
      self.cache.move_to_end(key)
      return self.cache[key]
    source_h, source_w = array.shape[-2:]
    src = self.source['cameras'][str(camera)]
    w, h = src['prepared_size']
    native = (source_w, source_h) != (w, h)
    if native and (source_w, source_h) != ((1344, 760) if camera in (1, 6) else (416, 224)):
      raise ValueError('Unexpected source video dimensions')
    left, top, right, bottom = src['content_box']
    def source_coordinates(x, y):
      if not native:
        return x, y
      return (x-left+.5)*source_w/(right-left)-.5, (y-top+.5)*source_h/(bottom-top)-.5
    fx, fy = src['focal_pixels']
    cx, cy = src['principal_point']
    if operation == 'nvidia':
      map_key = (camera, native)
      if map_key not in self.maps:
        target = self.target['cameras'][NAMES[camera]]
        yy, xx = np.mgrid[:h, :w]
        # Map output pixels through NVIDIA's measured F-theta polynomial and
        # an explicitly nominal source lens. No camera extrinsics are guessed.
        tx = (xx+.5)*target['width']/w-.5-target['principal_point'][0]
        ty = (yy+.5)*target['height']/h-.5-target['principal_point'][1]
        radius = np.hypot(tx, ty)
        theta = np.polynomial.polynomial.polyval(radius, target['r2th'])
        source_radius = theta if src['projection'] == 'equidistant' else np.tan(theta)
        factor = np.divide(source_radius, radius, out=np.zeros_like(radius), where=radius > 0)
        sx, sy = cx+fx*tx*factor, cy+fy*ty*factor
        if sx.min() < left or sx.max() > right-1 or sy.min() < top or sy.max() > bottom-1:
          raise ValueError(f'Camera {camera}: target rays exceed source coverage; do not fabricate image borders')
        sx, sy = source_coordinates(sx, sy)
        self.maps[map_key] = (sy, sx)
      result = np.stack([map_coordinates(channel, self.maps[map_key], order=1, mode='nearest', prefilter=False)
                         for channel in array]).astype(np.uint8)
    else:
      fraction = (fx*math.radians(operation)/w if src['projection'] == 'equidistant'
                  else 2*fx*math.tan(math.radians(operation/2))/w)
      if not 0 < fraction <= 1:
        raise ValueError(f'Camera {camera}: requested crop would need wider source coverage')
      box = (cx-w*fraction/2, cy-h*fraction/2, cx+w*fraction/2, cy+h*fraction/2)
      if native:
        box = (*source_coordinates(box[0], box[1]), *source_coordinates(box[2], box[3]))
      image = Image.fromarray(array.transpose(1, 2, 0))
      result = np.asarray(image.transform((w, h), Image.Transform.EXTENT, box, Image.Resampling.BICUBIC)).transpose(2, 0, 1).copy()
    self.cache[key] = result
    while len(self.cache) > 96:
      self.cache.popitem(last=False)
    return result


def extract(args):
  """Decode referenced frames once, before any resize, using corrected camera IDs."""
  import av
  replay = replay_module()
  replay.check_mount(args.output, args.recording_uuid)
  manifest = replay.read_json(args.source/'manifest.json')
  if (args.output/'manifest.json').exists():
    raise FileExistsError('Use a new raw-frame directory')
  rows = [r for r in manifest['schedule'] if args.ticks is None or r['tick'] in args.ticks]
  args.output.mkdir(parents=True, exist_ok=True)
  inventory = {}
  for camera in manifest['camera_manifest']['cameras']:
    view = camera['view']
    needed = {f['video_index'] for row in rows for f in row.get('frames', []) if f['view'] == view}
    if not needed:
      continue
    directory = args.output/str(view)
    directory.mkdir(exist_ok=False)
    video = Path(manifest['export'])/camera['video']
    with av.open(str(video)) as container:
      container.streams.video[0].thread_type = 'AUTO'
      for index, frame in enumerate(container.decode(video=0)):
        if index not in needed:
          continue
        image = frame.to_image()
        if not camera['video_pixels_rotated']:
          rotation = camera['video_display_rotation_degrees']
          if rotation not in (0, 180):
            raise ValueError('Unsupported source orientation')
          if rotation:
            image = image.transpose(Image.Transpose.ROTATE_180)
        np.save(directory/f'{index}.npy', np.asarray(image).transpose(2, 0, 1), allow_pickle=False)
        needed.remove(index)
        if not needed:
          break
    if needed:
      raise ValueError(f'Incomplete video for camera {view}')
    inventory[str(view)] = {'video': str(video), 'sha256': hashlib.sha256(video.read_bytes()).hexdigest(),
                            'frames': len(list(directory.glob('*.npy')))}
    print(f'Decoded camera {view}: {inventory[str(view)]["frames"]} native frames', flush=True)
  replay.write_json(args.output/'manifest.json', {'source_manifest': str(args.source/'manifest.json'), 'cameras': inventory,
    'source_manifest_sha256': hashlib.sha256((args.source/'manifest.json').read_bytes()).hexdigest(),
    'orientation': 'Upright pixels; remote display rotations applied exactly once'})


def prepare(args):
  replay = replay_module()
  replay.check_mount(args.output, args.recording_uuid)
  if (args.output/'manifest.json').exists():
    raise FileExistsError('Use a new output directory')
  manifest = replay.read_json(args.source/'manifest.json')
  source, target = replay.read_json(args.source_calibration), replay.read_json(args.target_calibration)
  if getattr(args, 'profile_json', None) is not None:
    definition = replay.read_json(args.profile_json)
    profile = definition['name']
    operations = {int(k): v for k, v in definition['operations'].items()}
    if not isinstance(profile, str) or not profile or not set(operations).issubset({0, 1, 2, 3, 5, 6}):
      raise ValueError('Invalid camera profile name or camera IDs')
    for operation in operations.values():
      if operation != 'nvidia' and (isinstance(operation, bool) or not isinstance(operation, (int, float)) or
                                     not np.isfinite(operation) or not 0 < operation < 180):
        raise ValueError('Camera operations require an FOV in (0, 180) or nvidia')
  else:
    profile, operations = args.profile, PROFILES[args.profile]
  transform = Transform(profile, source, target, operations=operations)
  if args.ticks is not None:
    wanted = set(args.ticks)
    manifest['schedule'] = [r for r in manifest['schedule'] if r['tick'] in wanted]
    if {r['tick'] for r in manifest['schedule']} != wanted:
      raise ValueError('Requested ticks are not in the prepared route')
  args.output.mkdir(parents=True, exist_ok=True)
  args.samples.mkdir(parents=True, exist_ok=True)
  for row in manifest['schedule']:
    if 'sample' not in row:
      continue
    old = Path(row['sample'])
    new = args.samples/old.name
    new.mkdir(exist_ok=False)
    for path in old.glob('*.npy'):
      if path.name != 'image_frames.npy' or not operations:
        os.link(path, new/path.name)
    if operations:
      images = np.load(old/'image_frames.npy', mmap_mode='r', allow_pickle=False)
      cameras = np.load(old/'camera_indices.npy', allow_pickle=False)
      frames = row['frames']
      updated = []
      for a, c, frame in zip(images, cameras, frames, strict=True):
        assert int(c) == frame['view']
        if args.raw_frames is not None and int(c) in operations:
          a = np.load(args.raw_frames/str(int(c))/f'{frame["video_index"]}.npy', allow_pickle=False)[None]
        updated.append(transform.image(a[0], int(c))[None])
      updated = np.stack(updated)
      np.save(new/'image_frames.npy', updated, allow_pickle=False)
    row.update(sample=str(new.resolve()), camera_profile=profile)
  for name in ('render-motion.npz', 'render-calibrations.json', 'render-openpilot.npz', 'navigation.json'):
    if (args.source/name).exists():
      shutil.copyfile(args.source/name, args.output/name)
  manifest.update(camera_input_profile=profile, camera_input_operations=operations,
                  source_camera_geometry=source, target_camera_geometry=target,
                  baseline_manifest=str((args.source/'manifest.json').resolve()),
                  baseline_manifest_sha256=hashlib.sha256((args.source/'manifest.json').read_bytes()).hexdigest(),
                  valid_samples=sum('sample' in r for r in manifest['schedule']), requested_samples=len(manifest['schedule']),
                  image_transform=f'{profile}: model-input transformation only; display footage remains original',
                  camera_crop_source='native video before resizing' if args.raw_frames is not None else 'prepared 768x432 images',
                  camera_calibration_limit='Source lens models are nominal; side extrinsics and measured source distortion remain unavailable')
  replay.write_json(args.output/'manifest.json', manifest)
  print(f'Prepared {manifest["valid_samples"]} {profile} inputs', flush=True)


def subset(args):
  """Remove camera inputs without resampling retained images or changing other inputs."""
  replay = replay_module()
  replay.check_mount(args.output, args.recording_uuid)
  if (args.output/'manifest.json').exists():
    raise FileExistsError('Use a new output directory')
  views = sorted(set(args.views))
  if not {1, 6}.issubset(views) or not set(views).issubset({0, 1, 2, 3, 5, 6}):
    raise ValueError('Keep both front cameras and use known camera IDs')
  source_path = args.source/'manifest.json'
  manifest = replay.read_json(source_path)
  args.output.mkdir(parents=True, exist_ok=True)
  args.samples.mkdir(parents=True, exist_ok=True)
  counts = {0: 3, 1: 5, 2: 3, 3: 3, 5: 3, 6: 5}
  for row in manifest['schedule']:
    if 'sample' not in row:
      continue
    old = Path(row['sample'])
    new = args.samples/old.name
    new.mkdir(exist_ok=False)
    cameras = np.load(old/'camera_indices.npy', allow_pickle=False)
    keep = np.isin(cameras, views)
    if [f['view'] for f in row['frames']] != cameras.tolist():
      raise ValueError('Camera metadata and image order disagree')
    retained = cameras[keep]
    if any(np.count_nonzero(retained == v) != counts[v] for v in views if v in cameras):
      raise ValueError('Incomplete camera history')
    if not {1, 6}.issubset(retained.tolist()):
      raise ValueError('Missing required front camera')
    for path in old.glob('*.npy'):
      if path.name in ('image_frames.npy', 'camera_indices.npy', 'absolute_timestamps.npy') and not keep.all():
        array = np.load(path, mmap_mode='r', allow_pickle=False)
        if len(array) != len(cameras):
          raise ValueError(f'Unexpected camera axis: {path.name}')
        np.save(new/path.name, np.ascontiguousarray(array[keep]), allow_pickle=False)
      else:
        os.link(path, new/path.name)
    row.update(sample=str(new.resolve()), reference_available_views=row['available_views'],
               excluded_views=[v for v in row['available_views'] if v not in views],
               available_views=sorted(set(retained.tolist())),
               frames=[f for f, selected in zip(row['frames'], keep, strict=True) if selected],
               omitted_views={k: v for k, v in row['omitted_views'].items() if int(k) in views})
  for name in ('render-motion.npz', 'render-calibrations.json', 'render-openpilot.npz', 'navigation.json',
               'gps-route.json', 'map-background.png', 'map-layout.json'):
    if (args.source/name).exists():
      shutil.copyfile(args.source/name, args.output/name)
  manifest.update(camera_subset=views, display_views=views,
                  camera_subset_source_manifest=str(source_path.resolve()),
                  camera_subset_source_sha256=hashlib.sha256(source_path.read_bytes()).hexdigest(),
                  camera_subset_limit='Selected cameras are removed from the model prompt; retained inputs are unchanged',
                  valid_samples=sum('sample' in r for r in manifest['schedule']))
  replay.write_json(args.output/'manifest.json', manifest)
  print(f'Prepared {manifest["valid_samples"]} inputs with cameras {views}', flush=True)


def evaluate(args):
  replay = replay_module()
  manifest = replay.read_json(args.source/'manifest.json')
  predictions = {p['sample']: p for line in (args.source/'predictions.jsonl').read_text().splitlines() if (p := json.loads(line))}
  errors = {}
  for row in manifest['schedule']:
    prediction = predictions[row['sample']]
    xyz = replay.predicted_xyz(prediction)
    actual = np.asarray(row['recorded_motion_future_xyz'])
    error = np.linalg.norm(xyz[:, :2]-actual[:, :2], axis=-1)
    profile = row['camera_profile']
    errors.setdefault(profile, []).append({'tick': row['tick'], 'split': row['split'],
      'ade_m': float(error.mean()), 'fde_m': float(error[-1]),
      'reasoning': prediction['prediction']['reasoning'], 'endpoint_xyz': xyz[-1].tolist()})
  summary = {}
  for profile, rows in errors.items():
    summary[profile] = {}
    for split in ('tune', 'holdout', 'all'):
      selected = [r for r in rows if split == 'all' or r['split'] == split]
      summary[profile][split] = {'samples': len(selected), **{
        metric: float(np.mean([r[metric] for r in selected])) for metric in ('ade_m', 'fde_m')}}
  winner = min(summary, key=lambda p: summary[p]['tune']['ade_m'])
  result = {'selection': winner, 'selection_rule': 'Minimum mean 6.4-second XY deviation on predeclared tuning ticks',
            'reference_limit': manifest['actual_path_limit'], 'summary': summary, 'samples': errors,
            'validation_limit': 'Same recorded drive; holdout timestamps are correlated, not independent driving validation'}
  replay.write_json(args.output, result)
  print(json.dumps({k: v for k, v in result.items() if k != 'samples'}, indent=2))


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('action', choices=('prepare', 'evaluate', 'extract', 'subset'))
  parser.add_argument('--source', type=Path, required=True)
  parser.add_argument('--output', type=Path, required=True)
  parser.add_argument('--samples', type=Path)
  profiles = parser.add_mutually_exclusive_group()
  profiles.add_argument('--profile', choices=PROFILES)
  profiles.add_argument('--profile-json', type=Path, help='JSON with name and per-camera operations')
  parser.add_argument('--source-calibration', type=Path)
  parser.add_argument('--target-calibration', type=Path)
  parser.add_argument('--raw-frames', type=Path, help='Native frames from extract, for crops before image resizing')
  parser.add_argument('--ticks', nargs='+', type=int)
  parser.add_argument('--views', nargs='+', type=int)
  parser.add_argument('--recording-uuid')
  args = parser.parse_args()
  if args.action == 'prepare':
    for name in ('samples', 'source_calibration', 'target_calibration'):
      if getattr(args, name) is None:
        parser.error(f'--{name.replace("_", "-")} is required')
    if args.profile is None and args.profile_json is None:
      parser.error('prepare requires --profile or --profile-json')
  if args.action == 'subset' and (args.samples is None or not args.views):
    parser.error('subset requires --samples and --views')
  globals()[args.action](args)


if __name__ == '__main__':
  main()
