"""Prepare six-camera recordings, replay Alpamayo, and render synchronized comparisons."""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
from PIL import Image, ImageOps
from scipy.spatial.transform import Rotation, Slerp

CAMERAS = (0, 1, 2, 3, 5, 6)
AXES = np.diag([1., -1., -1.])
FUTURE = np.arange(1, 65)*.1
SIZE = (768, 432)


def read_json(path):
  return json.loads(Path(path).read_text())


def write_json(path, value):
  path = Path(path)
  temporary = path.with_suffix(path.suffix+'.tmp')
  temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
  temporary.replace(path)


def predicted_xyz(entry):
  xyz = np.asarray(entry['prediction']['xyz'])
  if xyz.shape == (1, 64, 3):
    xyz = xyz[0]
  if xyz.shape != (64, 3) or not np.isfinite(xyz).all():
    raise ValueError('Invalid predicted trajectory')
  return xyz


def check_mount(path, expected):
  if expected:
    parent = Path(path).resolve()
    while not parent.exists():
      parent = parent.parent
    actual = subprocess.check_output(['findmnt', '-n', '-o', 'UUID', '-T', str(parent)], text=True).strip()
    if actual != expected:
      raise RuntimeError(f'Recording filesystem UUID mismatch: {actual!r}')


def read_logs(root):
  # Read local files directly; no remote log-source fallback or credentials.
  from openpilot.cereal import log
  import zstandard
  motions, calibrations, models = {}, {}, {}
  counts, checksums = Counter(), {}
  paths = sorted(Path(root).glob('*/rlog.zst'), key=lambda p: int(p.parent.name.rsplit('--', 1)[1]))
  if not paths:
    raise ValueError('No native rlogs')
  for path in paths:
    checksums[str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()
    with path.open('rb') as stream, zstandard.ZstdDecompressor().stream_reader(stream) as decoder:
      raw = decoder.read()
    for event in log.Event.read_multiple_bytes(raw):
      kind = event.which()
      if kind not in ('deviceMotion', 'extrinsicsCalibration', 'modelV2', 'navInstruction'):
        continue
      counts[kind] += 1
      if not event.valid:
        continue
      message = getattr(event, kind)
      if kind == 'deviceMotion':
        if message.timestamp > 0 and message.inputsOK and message.orientationNED.valid and message.velocityDevice.valid:
          motions[int(message.timestamp)] = ([getattr(message.orientationNED, a) for a in 'xyz'],
                                            [getattr(message.velocityDevice, a) for a in 'xyz'])
      elif kind == 'extrinsicsCalibration' and str(message.calStatus) == 'calibrated':
        calibrations[int(event.logMonoTime)] = list(message.rpyCalib)
      elif kind == 'modelV2':
        models[int(message.frameId)] = {'log_mono_time': int(event.logMonoTime), 'model': message.to_dict()}
      elif kind == 'navInstruction':
        counts['valid_navigation'] += 1
    print(f'Read {path.parent.name}', flush=True)
  return motions, calibrations, models, dict(counts), checksums


def save_render_models(output, export, models, calibrations):
  """Keep every logged model frame for video, independently of Alpamayo cadence."""
  times = np.array(sorted(calibrations), dtype=np.int64)
  rows = []
  for frame in read_json(export/'clips/native-timestamps.json')['views']['6']:
    entry = models.get(frame['frame_id'])
    if entry is None or entry['model']['timestampEof'] != frame['eof_ns']:
      continue
    ns = frame['capture_ns']
    index = int(np.searchsorted(times, ns, side='right'))-1
    if index < 0 or ns-int(times[index]) > 1_000_000_000:
      continue
    pos = entry['model']['position']
    if not pos['t'] or pos['t'][-1] < FUTURE[-1]:
      continue
    path = np.stack([np.interp(FUTURE, pos['t'], pos[a]) for a in 'xyz'], axis=-1) @ AXES
    if not np.isfinite(path).all():
      continue
    rows.append((ns, entry['log_mono_time'], frame['frame_id'], calibrations[int(times[index])], path))
  if not rows:
    raise ValueError('No frame-matched logged openpilot predictions')
  ns, published, frames, rpy, xyz = zip(*rows, strict=True)
  np.savez_compressed(output/'render-openpilot.npz', ns=np.array(ns, dtype=np.int64),
                      published_ns=np.array(published, dtype=np.int64), frame_id=np.array(frames),
                      rpy=np.array(rpy), xyz=np.array(xyz))
  return len(rows)


class Motion:
  def __init__(self, records):
    keys = sorted(records)
    self.ns = np.array(keys, dtype=np.int64)
    self.origin = keys[0]
    self.t = (self.ns-self.origin)/1e9
    values = [records[k] for k in keys]
    self.r = Rotation.from_euler('xyz', [v[0] for v in values]).as_matrix()
    self.v = np.einsum('nij,nj->ni', self.r, np.array([v[1] for v in values]))
    self.p = np.concatenate((np.zeros((1, 3)), np.cumsum(np.diff(self.t)[:, None]*(self.v[1:]+self.v[:-1])/2, axis=0)))

  def sample(self, query_ns, rpy):
    q = (query_ns-self.origin)/1e9
    last = int(np.searchsorted(self.ns, query_ns, side='right'))-1
    first = max(0, int(np.searchsorted(self.t, q-1.9))-1)
    if last <= first or self.t[first] > q-1.9 or q-self.t[last] > .1:
      raise ValueError('Missing measured ego history')
    if np.max(np.diff(self.t[first:last+1])) > .15:
      raise ValueError('Gap in measured ego history')
    t, p, r = self.t[first:last+1], self.p[first:last+1], self.r[first:last+1]
    # Only measurements at or before the query enter inference. Propagate the
    # final measured velocity for at most 100 ms; hold its orientation.
    if q > t[-1]:
      p = np.concatenate((p, [p[-1]+self.v[last]*(q-t[-1])]))
      r = np.concatenate((r, r[-1:]))
      t = np.append(t, q)
    wanted = q+np.arange(-19, 1)*.1
    history = np.stack([np.interp(wanted, t, p[:, a]) for a in range(3)], axis=-1)
    calibration = Rotation.from_euler('xyz', rpy).as_matrix()
    world_vehicle = Slerp(t, Rotation.from_matrix(r))(wanted).as_matrix() @ calibration
    current_world = AXES @ world_vehicle[-1].T
    xyz = ((history-history[-1]) @ current_world.T)[None].astype(np.float32)
    rot = (current_world @ world_vehicle @ AXES)[None].astype(np.float32)
    actual = None
    end = int(np.searchsorted(self.t, q+6.4))+1
    if end <= len(self.t) and self.t[-1] >= q+6.4 and np.max(np.diff(self.t[last:end])) <= .15:
      future = np.stack([np.interp(q+FUTURE, self.t, self.p[:, a]) for a in range(3)], axis=-1)
      actual = ((future-history[-1]) @ current_world.T).tolist()
    return xyz, rot, actual, (query_ns-int(self.ns[last]))/1e6


def camera_records(export):
  cameras = {c['view']: c for c in read_json(export/'cameras.json')['cameras']}
  native = read_json(export/'clips/native-timestamps.json')['views']
  records = {}
  for camera in CAMERAS:
    source = cameras[camera]
    if camera in (1, 6):
      rows = native[str(camera)]
    else:
      rows = [read_json_line for line in (export/source['frame_log']).read_text().splitlines()
              if (read_json_line := json.loads(line)).get('exported', True)]
    timestamps = [row['capture_ns'] for row in rows]
    if any(b <= a for a, b in zip(timestamps, timestamps[1:])):
      raise ValueError(f'Unordered camera {camera} timestamps')
    records[camera] = rows
  return cameras, records


def prepare(args):
  import av
  check_mount(args.output, args.recording_uuid)
  if (args.output/'manifest.json').exists():
    raise FileExistsError('Use a new output directory; prepared runs are immutable')
  args.output.mkdir(parents=True, exist_ok=True)
  args.samples.mkdir(parents=True, exist_ok=True)
  cameras, records = camera_records(args.export)
  motions, calibrations, models, counts, hashes = read_logs(args.logs)
  motion = Motion(motions)
  calibration_times = np.array(sorted(calibrations), dtype=np.int64)
  grid = read_json(args.export/'grid-summary.json')
  start_ns, end_ns = grid['capture_start_ns'], grid['capture_end_ns']
  original_duration = (end_ns-start_ns)/1e9
  if not 0 <= args.trim_end_seconds < original_duration:
    raise ValueError('Trim must leave a positive recording duration')
  end_ns -= round(args.trim_end_seconds*1e9)
  times = {c: np.array([r['capture_ns'] for r in records[c]], dtype=np.int64) for c in CAMERAS}
  schedule, needed, arrays = [], defaultdict(lambda: defaultdict(list)), {}
  duration = (end_ns-start_ns)/1e9
  for tick in range(math.ceil(duration*args.hz)):
    display_s = tick/args.hz
    requested = start_ns+round(display_s*1e9)
    anchor = int(np.searchsorted(times[6], requested, side='right'))-1
    if anchor < 0:
      continue
    query = int(times[6][anchor])
    row = {'tick': tick, 'display_time_s': display_s, 'query_ns': query, 'skip_reasons': [], 'omitted_views': {}}
    selected, camera_ids, timestamps, frame_refs = [], [], [], []
    for c in CAMERAS:
      offsets = (-2., -1., -.5, -.1, 0.) if c in (1, 6) else (-1., -.1, 0.)
      chosen, missing = [], []
      for offset in offsets:
        desired = query+round(offset*1e9)
        index = int(np.searchsorted(times[c], desired, side='right'))-1
        if index < 0 or desired-int(times[c][index]) > args.max_frame_age_ms*1e6:
          missing.append(f'missing frame at {offset:g}s')
          continue
        chosen.append((desired, index))
      if missing:
        row['omitted_views'][str(c)] = {'position': cameras[c]['camera_position'], 'reasons': missing}
        if c in (1, 6):
          row['skip_reasons'].append(f'{cameras[c]["camera_position"]}: missing required front history')
        continue
      for desired, index in chosen:
        selected.append((c, index))
        camera_ids.append(c)
        timestamps.append(int(times[c][index])//1000)
        frame_refs.append({'view': c, 'video_index': index, 'capture_ns': int(times[c][index]),
                           'requested_ns': desired, 'frame_id': records[c][index].get('frame_id', records[c][index].get('frame'))})
    row['frames'] = frame_refs
    row['available_views'] = sorted(set(camera_ids))
    frame_id = records[6][anchor]['frame_id']
    small = models.get(frame_id)
    row['small_model'] = small
    if small is None:
      row['skip_reasons'].append('No openpilot model with the matching front frame ID')
    else:
      delta = int(small['model']['timestampEof'])-int(records[6][anchor]['eof_ns'])
      row['small_model_eof_difference_ns'] = delta
      if delta != 0:
        row['skip_reasons'].append('Openpilot frame ID matches but EOF differs')
    cal_index = int(np.searchsorted(calibration_times, query, side='right'))-1
    if cal_index < 0:
      row['skip_reasons'].append('No prior calibrated extrinsics')
    else:
      cal_time = int(calibration_times[cal_index])
      row['calibration_rpy'] = calibrations[cal_time]
      row['calibration_log_mono_time'] = cal_time
      try:
        xyz, rot, actual, age = motion.sample(query, row['calibration_rpy'])
        row.update(ego_history_xyz=xyz.tolist(), ego_history_rot=rot.tolist(),
                   recorded_motion_future_xyz=actual, motion_extrapolation_ms=age,
                   speed_m_s=float(np.linalg.norm(xyz[0, -1]-xyz[0, -2])/.1))
      except ValueError as exc:
        row['skip_reasons'].append(str(exc))
    if not row['skip_reasons']:
      path = args.samples/f'sample-{query//1000}'
      path.mkdir(exist_ok=False)
      row['sample'] = str(path.resolve())
      for name, value in {'camera_indices': np.array(camera_ids, dtype=np.int64),
                          'absolute_timestamps': np.array(timestamps, dtype=np.int64),
                          'ego_history_xyz': xyz, 'ego_history_rot': rot, 'nav_text': np.array(args.navigation)}.items():
        np.save(path/(name+'.npy'), value, allow_pickle=False)
      arrays[tick] = np.lib.format.open_memmap(path/'image_frames.npy', mode='w+', dtype=np.uint8, shape=(len(selected), 1, 3, SIZE[1], SIZE[0]))
      for position, (c, index) in enumerate(selected):
        needed[c][index].append((tick, position))
    schedule.append(row)
  print(f'{len(arrays)}/{len(schedule)} valid {args.hz:g} Hz snapshots', flush=True)
  if not arrays:
    write_json(args.output/'rejected-schedule.json', schedule)
    raise ValueError('No complete snapshots')
  for c in CAMERAS:
    source = cameras[c]
    count, used = 0, 0
    with av.open(str(args.export/source['video'])) as container:
      stream = container.streams.video[0]
      stream.thread_type = 'AUTO'
      for index, frame in enumerate(container.decode(stream)):
        count += 1
        if index not in needed[c]:
          continue
        # PyAV returns coded pixels without applying the MP4 display matrix.
        image = frame.to_image()
        rotation = source['video_display_rotation_degrees'] if not source.get('video_pixels_rotated') else 0
        if rotation:
          image = image.rotate(rotation, expand=True)
        image = ImageOps.pad(image, SIZE, method=Image.Resampling.LANCZOS, color=(0, 0, 0))
        rgb = np.asarray(image).transpose(2, 0, 1)
        for tick, position in needed[c][index]:
          arrays[tick][position, 0] = rgb
        used += 1
    if count != len(records[c]) or used != len(needed[c]):
      raise ValueError(f'Camera {c}: video/log frame mismatch {count}/{len(records[c])}; selected {used}/{len(needed[c])}')
    for array in arrays.values():
      array.flush()
    print(f'{source["camera_position"]}: verified {count} frames; selected {used}', flush=True)
  manifest = {'schema': 1, 'export': str(args.export.resolve()), 'route': read_json(args.export/'cameras.json')['route'],
              'sample_hz': args.hz, 'capture_start_ns': start_ns, 'capture_end_ns': end_ns,
              'duration_s': duration, 'original_duration_s': original_duration, 'max_frame_age_ms': args.max_frame_age_ms,
              'navigation': args.navigation, 'navigation_source': 'explicit fixed test instruction; no logged destination navigation',
              'image_transform': f'Correct display orientation once; aspect-preserving resize and black padding to {SIZE}; no lens warp or crop',
              'camera_calibration_limit': 'Physical view mapping only; lens FOV and side extrinsics do not match NVIDIA calibration',
              'missing_camera_policy': 'Omit a side view if any required history frame is absent or older than the limit; no black model inputs',
              'trajectory_axes': 'forward/left/up', 'small_model_source_axes': 'forward/right/down',
              'future_times_s': FUTURE.tolist(), 'motion_source': 'deviceMotion.timestamp; causal history, last velocity propagated <=100ms',
              'actual_path_limit': 'Integrated recorded motion estimate, not independent ground truth; never supplied as future model input',
              'rlog_sha256': hashes, 'log_message_counts': counts, 'camera_manifest': read_json(args.export/'cameras.json'),
              'valid_samples': len(arrays), 'requested_samples': len(schedule), 'schedule': schedule}
  overlay_motion, full_calibrations, wide_camera = overlay_logs(args.logs)
  np.savez_compressed(args.output/'render-motion.npz', ns=overlay_motion.ns, p=overlay_motion.p, r=overlay_motion.r, v=overlay_motion.v)
  write_json(args.output/'render-calibrations.json', full_calibrations)
  manifest['render_openpilot_frames'] = save_render_models(args.output, args.export, models, calibrations)
  manifest['wide_projection'] = wide_camera
  for row in schedule:
    row['navigation_text'] = args.navigation
  write_json(args.output/'manifest.json', manifest)
  print(f'Prepared {args.output / "manifest.json"}', flush=True)


def overlay_logs(root):
  """Retain recorded poses and full camera calibration for video projection."""
  from openpilot.cereal import log
  import zstandard
  motions, calibrations, devices, sensors = {}, [], set(), set()
  for path in sorted(Path(root).glob('*/rlog.zst')):
    with path.open('rb') as stream, zstandard.ZstdDecompressor().stream_reader(stream) as decoder:
      raw = decoder.read()
    for event in log.Event.read_multiple_bytes(raw):
      if not event.valid:
        continue
      kind = event.which()
      message = getattr(event, kind)
      if kind == 'deviceMotion' and message.timestamp > 0 and message.inputsOK and message.orientationNED.valid and message.velocityDevice.valid:
        motions[int(message.timestamp)] = ([getattr(message.orientationNED, a) for a in 'xyz'],
                                          [getattr(message.velocityDevice, a) for a in 'xyz'])
      elif kind == 'extrinsicsCalibration' and str(message.calStatus) == 'calibrated':
        calibrations.append({'ns': int(event.logMonoTime), **message.to_dict()})
      elif kind == 'deviceState':
        devices.add(str(message.deviceType))
      elif kind == 'wideRoadCameraState':
        sensors.add(str(message.sensor))
  if len(devices) != 1 or len(sensors) != 1 or not calibrations:
    raise ValueError('Ambiguous or missing recorded camera configuration')
  from openpilot.common.transformations.camera import DEVICE_CAMERAS
  camera = DEVICE_CAMERAS[(devices.pop(), sensors.pop())].wide_road
  return Motion(motions), sorted(calibrations, key=lambda c: c['ns']), {
    'width': camera.width, 'height': camera.height, 'intrinsics': camera.intrinsics.tolist(),
    'projection': 'openpilot wide-camera pinhole projection with recorded extrinsics and camera height; no measured lens distortion correction'}


def navigation_text(seconds, distance, turns, arrival):
  for turn in turns:
    if seconds > turn['end_s']:
      continue
    remaining = float(np.interp(turn['start_s'], distance[0], distance[1])-np.interp(seconds, distance[0], distance[1]))
    if remaining > 250:
      return 'Follow the road.'
    action = 'Turn '+turn['direction']
    if remaining < 8 or seconds >= turn['start_s']:
      return action+'.'
    return f'{action} in {max(10, round(remaining/10)*10)}m.'
  if seconds >= arrival:
    return 'You have arrived at your destination.'
  remaining = float(np.interp(arrival, distance[0], distance[1])-np.interp(seconds, distance[0], distance[1]))
  return f'Continue to the destination in {max(5, round(remaining/5)*5)}m.'


def navigate(args):
  """Create a separate run with reviewed route instructions, retaining baseline inputs."""
  check_mount(args.output, args.recording_uuid)
  if (args.output/'manifest.json').exists():
    raise FileExistsError('Use a new output directory')
  manifest = read_json(args.source/'manifest.json')
  annotations = read_json(args.turns)
  if not 0 <= args.trim_end_seconds < manifest['duration_s']:
    raise ValueError('Trim must leave a positive recording duration')
  if args.trim_end_seconds:
    manifest['original_duration_s'] = manifest['duration_s']
    manifest['duration_s'] -= args.trim_end_seconds
    manifest['schedule'] = [row for row in manifest['schedule'] if row['display_time_s'] < manifest['duration_s']]
    manifest['valid_samples'] = sum('sample' in row for row in manifest['schedule'])
    manifest['requested_samples'] = len(manifest['schedule'])
  turns = annotations['turns']
  if annotations['route'] != manifest['route'] or not turns:
    raise ValueError('Maneuvers must describe this route')
  end = -1
  for turn in turns:
    if turn['direction'] not in ('left', 'right') or not end < turn['start_s'] < turn['end_s']:
      raise ValueError('Maneuvers must be ordered, non-overlapping left or right turns')
    end = turn['end_s']
  if not end < annotations['arrival_s'] < manifest.get('original_duration_s', manifest['duration_s']):
    raise ValueError('Arrival must follow the last turn within the recording')
  motion, calibrations, camera = overlay_logs(args.logs)
  seconds = (motion.ns-manifest['capture_start_ns'])/1e9
  speed = np.linalg.norm(motion.v, axis=1)
  distance = np.concatenate(([0.], np.cumsum(np.diff(seconds)*(speed[1:]+speed[:-1])/2)))
  args.output.mkdir(parents=True, exist_ok=True)
  args.samples.mkdir(parents=True, exist_ok=True)
  for row in manifest['schedule']:
    text = navigation_text((row['query_ns']-manifest['capture_start_ns'])/1e9, (seconds, distance), turns, annotations['arrival_s'])
    row['navigation_text'] = text
    if 'sample' in row:
      source = Path(row['sample'])
      target = args.samples/source.name
      target.mkdir(exist_ok=False)
      for path in source.glob('*.npy'):
        if path.name != 'nav_text.npy':
          os.link(path, target/path.name)
      np.save(target/'nav_text.npy', np.array(text), allow_pickle=False)
      row['sample'] = str(target.resolve())
  manifest.update(navigation='Per-sample reconstructed route instructions',
                  navigation_source='Reviewed recorded turns and integrated travel distance; retrospective route-intent replay, not logged live navigation',
                  navigation_annotations=annotations, baseline_manifest=str((args.source/'manifest.json').resolve()),
                  baseline_manifest_sha256=hashlib.sha256((args.source/'manifest.json').read_bytes()).hexdigest(),
                  wide_projection=camera)
  # These poses are exclusively for rendering and route annotation. They never
  # replace causal model history or provide a future trajectory to the model.
  np.savez_compressed(args.output/'render-motion.npz', ns=motion.ns, p=motion.p, r=motion.r, v=motion.v)
  write_json(args.output/'render-calibrations.json', calibrations)
  write_json(args.output/'navigation.json', annotations)
  _, model_calibrations, models, _, _ = read_logs(args.logs)
  manifest['render_openpilot_frames'] = save_render_models(args.output, Path(manifest['export']), models, model_calibrations)
  write_json(args.output/'manifest.json', manifest)
  print(f'Prepared {manifest["valid_samples"]} navigation inputs in {args.output}', flush=True)


def infer(args):
  here = Path(__file__).resolve().parent
  manifest = read_json(args.output/'manifest.json')
  groups = defaultdict(list)
  for row in manifest['schedule']:
    if 'sample' in row:
      groups[tuple(row['available_views'])].append(row['sample'])
  samples = [p for group in groups.values() for p in group]
  persistent = getattr(args, 'persistent_runtime', False)
  if persistent:
    groups = {('persistent',): samples}
  if not samples:
    raise ValueError('No prepared inputs')
  config = read_json(here/'best-config.json')
  revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=args.tinygrad, text=True).strip()
  if revision != config['tinygrad_revision']:
    raise ValueError('Use the saved tinygrad revision')
  vocab = here/'coc_vocab.json'
  if hashlib.sha256(vocab.read_bytes()).hexdigest() != config['vocabulary_sha256']:
    raise ValueError('Vocabulary checksum differs')
  noise = np.load(args.noise, allow_pickle=False)
  if noise.shape != (1, 64, 2) or noise.dtype != np.float32 or hashlib.sha256(noise.tobytes()).hexdigest() != config['initial_noise_sha256']:
    raise ValueError('Initial diffusion noise differs from the saved configuration')
  env = dict(os.environ, PROFILE='0', PYTHONPATH=f'{args.tinygrad.resolve()}:{here.parent}',
             ALPAMAYO_FP8_CACHE=str(args.fp8_cache.resolve()), CACHEDB=str(args.cache.resolve()))
  for key, value in config['environment'].items():
    if value is None:
      env.pop(key, None)
    else:
      env[key] = value
  args.work.mkdir(parents=True, exist_ok=True)
  prompt_adapter = None
  if args.group_prompts:
    # Only CPU tokenization runs here. Each subprocess still owns the GPU.
    import importlib.util
    import torch
    torch.set_num_threads(4)
    sys.path[:0] = [str(args.tinygrad.resolve()), str(here.parent)]
    spec = importlib.util.spec_from_file_location('route_fast_inputs', here/'fast-inputs.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    prompt_adapter = module.FastInputs(args.checkpoint, camera_profile='nvidia', allow_missing_cameras=True)
    prompt_sizes = {}
    def prompt_size(path):
      key = (tuple(np.load(Path(path)/'camera_indices.npy').tolist()), str(np.load(Path(path)/'nav_text.npy').item()))
      if key not in prompt_sizes:
        data = {p.stem: [str(np.load(p, allow_pickle=False).item())] if p.stem == 'nav_text'
                else torch.from_numpy(np.load(p, allow_pickle=False)) for p in Path(path).glob('*.npy')}
        _, _, prepared = prompt_adapter.prepare(data)
        prompt_sizes[key] = int(prepared['tokenized_data']['input_ids'].shape[-1])
      return prompt_sizes[key]
  entries, summaries = [], {}
  # Each camera layout gets its own process. This keeps captured GPU graphs and
  # their KV buffers isolated while preserving chronological order per layout.
  for cameras, paths in sorted(groups.items()):
    if args.limit:
      paths = paths[:args.limit]
    label = '-'.join(map(str, cameras))
    completed = {}
    for journal in sorted(args.work.glob('views-*.jsonl' if persistent else f'views-{label}*.jsonl')):
      previous = [json.loads(line) for line in journal.read_text().splitlines()]
      for entry in previous:
        if entry['sample'] not in paths or entry['sample'] in completed:
          raise ValueError(f'Duplicate or unexpected saved prediction: {journal}')
        predicted_xyz(entry)
        completed[entry['sample']] = entry
      report_path = journal.with_suffix('.json')
      summaries[journal.stem] = read_json(report_path)['summary'] if report_path.exists() else {
        'interrupted': True, 'saved_predictions': len(previous)}
    entries.extend(completed.values())
    paths = [path for path in paths if path not in completed]
    if not paths:
      continue
    pending_groups = defaultdict(list)
    if persistent and prompt_adapter:
      paths.sort(key=lambda path: (prompt_size(path), tuple(np.load(Path(path)/'camera_indices.npy').tolist())))
    for path in paths:
      pending_groups[prompt_size(path) if prompt_adapter and not persistent else 0].append(path)
    retained = len(completed)
    for size, pending_paths in sorted(pending_groups.items()):
      batch_size = len(pending_paths) if persistent else args.batch_size or len(pending_paths)
      for begin in range(0, len(pending_paths), batch_size):
        batch = pending_paths[begin:begin+batch_size]
        part = retained
        report = args.work/f'views-{label}-part-{part:05d}.json'
        # An interrupted attempt can leave an empty journal. Preserve it.
        while report.exists() or report.with_suffix('.jsonl').exists():
          part += 1
          report = args.work/f'views-{label}-part-{part:05d}.json'
        arguments = list(config['arguments'])
        if persistent:
          arguments.remove('--warmup-graphs')
          arguments[arguments.index('--vision-group-limit')+1] = '1'
        warmup = min(8, len(batch)-1)
        for flag, value in (('--warmup', warmup), ('--runs', len(batch)-warmup)):
          arguments[arguments.index(flag)+1] = str(value)
        command = [sys.executable, str(here/'benchmark-pipeline.py'), '--checkpoint', str(args.checkpoint),
                   '--samples', *batch, '--initial-noise', str(args.noise), '--reduced-vocab', str(vocab),
                   '--output', str(report), '--allow-missing-cameras', *arguments]
        if args.collect_garbage:
          command.append('--collect-garbage')
        if persistent:
          command += ['--gc-interval', '32', '--minimum-cache-capacity', '3584']
          # Controls use existing inputs and never become duplicate route outputs.
          # Alternate prompt/layout shapes to check shared KV buffers before new work.
          candidates = [entry for entry in completed.values() if entry['prediction']['reasoning'].startswith('<NOMINAL>')]
          controls = {}
          for camera_count in (22, 16, 22, 16):
            for entry in candidates:
              path = entry['sample']
              if path not in controls and len(np.load(Path(path)/'camera_indices.npy')) == camera_count:
                controls[path] = entry
                break
          if controls:
            reference = args.work/f'validation-{part:05d}.json'
            write_json(reference, controls)
            command += ['--validation-reference', str(reference)]
        write_json(args.work/f'invocation-{label}-{part:05d}.json', {
          'command': command, 'environment': {k: env.get(k) for k in config['environment']},
          'sample_count': len(batch), 'resumed_predictions': retained, 'prompt_tokens': size or None,
          'one_pass': True, 'config': config})
        print(f'Replaying cameras {cameras}: {len(batch)} snapshots; retained {retained}; prompt tokens {size or "mixed"}', flush=True)
        subprocess.run(command, env=env, check=True)
        saved = read_json(report)
        batch_entries = saved['warmup']+saved['runs']
        if len(batch_entries) != len(batch) or {entry['sample'] for entry in batch_entries} != set(batch):
          raise ValueError('Batch coverage differs from requested inputs')
        entries.extend(batch_entries)
        retained += len(batch_entries)
        summaries[report.stem] = saved['summary']
  if args.limit:
    return
  check_mount(args.output, args.recording_uuid)
  by_sample = {e['sample']: e for e in entries}
  if len(entries) != len(samples) or set(by_sample) != set(samples):
    raise ValueError('Inference coverage differs from requested inputs')
  predictions = []
  for row in manifest['schedule']:
    if 'sample' not in row:
      continue
    entry = by_sample[row['sample']]
    predicted_xyz(entry)
    predictions.append({'tick': row['tick'], 'query_ns': row['query_ns'],
                        'navigation_text': row.get('navigation_text', manifest['navigation']), **entry})
  (args.output/'predictions.jsonl').write_text(''.join(json.dumps(p, allow_nan=False)+'\n' for p in predictions))
  write_json(args.output/'inference-summary.json', summaries)
  print(f'Saved {len(predictions)} route predictions', flush=True)


def render(args):
  from route_video import render_route
  check_mount(args.output, args.recording_uuid)
  manifest = read_json(args.output/'manifest.json')
  if not 1 <= args.zoom <= 2:
    raise ValueError('Use a display zoom between 1 and 2')
  render_route(args, manifest)


def timed_infer(args):
  """Measure the complete command, including restarts and resumed work."""
  args.work.mkdir(parents=True, exist_ok=True)
  def saved_samples():
    return {json.loads(line)['sample'] for path in args.work.glob('views-*.jsonl') for line in path.read_text().splitlines()}
  before = saved_samples()
  started_utc = datetime.now(timezone.utc).isoformat()
  started = time.perf_counter()
  succeeded = False
  try:
    infer(args)
    succeeded = True
  finally:
    elapsed = time.perf_counter()-started
    added = len(saved_samples()-before)
    measurement = {
      'started_utc': started_utc, 'finished_utc': datetime.now(timezone.utc).isoformat(),
      'succeeded': succeeded, 'wall_seconds': elapsed, 'new_predictions': added,
      'effective_predictions_per_second': added/elapsed, 'resumed_predictions': len(before),
      'grouped_by_camera_layout': True, 'grouped_by_prompt_length': args.group_prompts,
       'batch_size': args.batch_size,
       'persistent_runtime': getattr(args, 'persistent_runtime', False),
      'boundary': 'Complete inference command: input processing, model loading, graph compilation, and process restarts; excludes prior image preparation and video rendering',
    }
    write_json(args.work/f'command-timing-{time.time_ns()}.json', measurement)
    print('Inference wall time: '+json.dumps(measurement), flush=True)


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('action', choices=('prepare', 'navigate', 'infer', 'render'))
  parser.add_argument('--output', type=Path, required=True)
  parser.add_argument('--recording-uuid', help='Require the output filesystem UUID before recording writes')
  parser.add_argument('--export', type=Path)
  parser.add_argument('--logs', type=Path)
  parser.add_argument('--samples', type=Path)
  parser.add_argument('--source', type=Path, help='Prepared baseline run for a new navigation replay')
  parser.add_argument('--turns', type=Path, help='Reviewed maneuver annotations relative to capture start')
  parser.add_argument('--trim-end-seconds', type=float, default=0.)
  parser.add_argument('--zoom', type=float, default=1., help='Wide camera display zoom; model images are unchanged')
  parser.add_argument('--compare-output', type=Path, nargs='+', help='Overlay other completed camera configurations at matching timestamps')
  parser.add_argument('--big-model-output', type=Path, help='Overlay a completed continuous openpilot big-model replay')
  parser.add_argument('--hz', type=float, default=1.)
  parser.add_argument('--max-frame-age-ms', type=float, default=150.)
  parser.add_argument('--navigation', default='Follow the road.')
  for name in ('checkpoint', 'tinygrad', 'noise', 'fp8-cache', 'cache', 'work'):
    parser.add_argument('--'+name, type=Path)
  parser.add_argument('--limit', type=int, default=0, help='Smoke replay only; does not replace full predictions')
  parser.add_argument('--batch-size', type=int, default=0, help='Maximum snapshots per GPU process; zero keeps each layout together')
  parser.add_argument('--group-prompts', action='store_true', help='Group equal prompt lengths to limit GPU graph reallocations')
  parser.add_argument('--collect-garbage', action='store_true', help='Collect released input graphs between offline predictions')
  parser.add_argument('--persistent-runtime', action='store_true', help='One model, bounded single-image vision graphs, shared KV allocation')
  args = parser.parse_args()
  if args.batch_size < 0:
    parser.error('--batch-size must be nonnegative')
  if not 0 < args.hz <= 20 or not 0 < args.max_frame_age_ms <= 150:
    parser.error('Use a positive cadence <=20 Hz and frame age <=150 ms')
  required = {'prepare': ('export', 'logs', 'samples'), 'navigate': ('source', 'turns', 'logs', 'samples'),
              'infer': ('checkpoint', 'tinygrad', 'noise', 'fp8_cache', 'cache', 'work'), 'render': ()}
  for name in required[args.action]:
    if getattr(args, name) is None:
      parser.error(f'--{name.replace("_", "-")} is required for {args.action}')
  (timed_infer if args.action == 'infer' else globals()[args.action])(args)


if __name__ == '__main__':
  main()
