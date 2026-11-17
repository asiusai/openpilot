"""Render timestamp-aligned trajectories on the recorded wide camera."""
from collections import defaultdict
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
import subprocess
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.spatial.transform import Rotation

AXES = np.diag([1., -1., -1.])
FUTURE = np.arange(1, 65)*.1
COLORS = ('#ffba58', '#62c9ff')
TWO_CAMERA_COLOR = '#ed80df'
CAMERA_COLORS = {2: TWO_CAMERA_COLOR, 4: '#78df9b', 6: COLORS[0]}
ACTUAL_COLOR = '#b2b8c0'
BIG_COLOR = '#ff6363'
MAIN_SIZE = (1600, 900)


@lru_cache(maxsize=16)
def font(size):
  return ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', size)


def lines(draw, text, width, size):
  output, line = [], ''
  for word in text.split():
    trial = (line+' '+word).strip()
    if line and draw.textlength(trial, font=font(size)) > width:
      output.append(line)
      line = word
    else:
      line = trial
  return output+[line] if line else output


def navigation_caption(draw, text):
  """Display the exact navigation meaning with a compact direction symbol."""
  import re
  text = text.strip()
  match = re.fullmatch(r'Turn (right|left)(?: in (\d+)m)?\.', text)
  direction = match.group(1) if match else None
  title = f'Turn {direction}' if match else text.rstrip('.')
  detail = f'{match.group(2)} m' if match and match.group(2) else ('Now' if match else '')
  draw.rounded_rectangle((24, 920, 404, 1056), radius=18, fill='#1b242e', outline='#35414e', width=1)
  draw.rounded_rectangle((42, 943, 114, 1015), radius=16, fill='#db514b')
  if direction:
    coords = [(66, 998), (66, 969), (92, 969)]
    head = [(83, 959), (94, 969), (83, 979)]
    if direction == 'left':
      coords = [(156-x, y) for x, y in coords]
      head = [(156-x, y) for x, y in head]
    draw.line(coords, fill='white', width=6, joint='curve')
    draw.line(head, fill='white', width=6, joint='curve')
  else:
    draw.line([(78, 999), (78, 960)], fill='white', width=6)
    draw.line([(67, 972), (78, 960), (89, 972)], fill='white', width=6, joint='curve')
  draw.text((132, 936), 'NAVIGATION', font=font(16), fill='#a8b4c1')
  size = next(s for s in range(29, 17, -1) if len(lines(draw, title, 250, s)) <= 2)
  wrapped = lines(draw, title, 250, size)
  for i, value in enumerate(wrapped):
    draw.text((132, 960+i*(size+3)), value, font=font(size), fill='#ffffff')
  if detail:
    draw.text((132, 1002), detail, font=font(24), fill='#c3ced9')


def display_reasoning(prediction):
  """Format captions without modifying the saved model output."""
  reason = prediction['prediction']['reasoning'].replace('<|traj_future_start|>', '').strip() if prediction else ''
  if reason == '<NOMINAL>':
    return 'No additional reasoning', True
  # Token 124549 is '<N'. Some saved generations continue with prose instead
  # of completing NOMINAL. Remove only this exact, incomplete leading marker.
  if reason.startswith('<N') and not reason.startswith('<NOMINAL>'):
    reason = reason[2:].lstrip()
  return reason or 'Prediction unavailable', False


def reasoning_caption(draw, prediction):
  draw.rounded_rectangle((422, 920, 1896, 1056), radius=18, fill='#171f29', outline='#35414e', width=1)
  draw.rounded_rectangle((442, 942, 447, 1034), radius=2, fill=COLORS[0])
  draw.text((465, 935), 'ALPAMAYO', font=font(16), fill=COLORS[0])
  reason, nominal = display_reasoning(prediction)
  size = next((s for s in range(28, 15, -1) if len(lines(draw, reason, 1400, s))*(s+6) <= 84), None)
  if size is None:
    raise ValueError('Reasoning exceeds caption area; do not truncate model messages')
  for i, value in enumerate(lines(draw, reason, 1400, size)):
    draw.text((465, 963+i*(size+6)), value, font=font(size), fill='#9caab8' if nominal else '#f0f3f7')


def trajectories(row, prediction):
  alp = np.asarray(prediction['prediction']['xyz']).reshape(-1, 3)
  if alp.shape != (64, 3) or not np.isfinite(alp).all():
    raise ValueError('Invalid trajectory')
  pos = row['small_model']['model']['position']
  small = np.stack([np.interp(FUTURE, pos['t'], pos[a]) for a in 'xyz'], axis=-1) @ AXES
  return alp, small


def comparison_captions(draw, runs):
  height = (158-6*(len(runs)-1))//len(runs)
  for index, (cameras, prediction, _) in enumerate(runs):
    top, color = 916+index*(height+6), CAMERA_COLORS[cameras]
    draw.rounded_rectangle((422, top, 1896, top+height), radius=12, fill='#171f29', outline='#35414e', width=1)
    draw.rounded_rectangle((438, top+10, 443, top+height-10), radius=2, fill=color)
    draw.text((455, top+(height-20)//2), f'Alpamayo {cameras} cam', font=font(16), fill=color)
    reason, nominal = display_reasoning(prediction)
    size = next((s for s in range(24, 10, -1) if len(lines(draw, reason, 1260, s))*(s+4) <= height-10), None)
    if size is None:
      raise ValueError('Reasoning exceeds comparison caption area; do not truncate model messages')
    wrapped = lines(draw, reason, 1260, size)
    y = top+(height-len(wrapped)*(size+4))//2
    for i, value in enumerate(wrapped):
      draw.text((610, y+i*(size+4)), value, font=font(size), fill='#9caab8' if nominal else '#f0f3f7')


class Geometry:
  def __init__(self, output, manifest):
    data = np.load(output/'render-motion.npz', allow_pickle=False)
    self.ns, self.p, self.r, self.v = (data[key] for key in ('ns', 'p', 'r', 'v'))
    self.calibrations = json.loads((output/'render-calibrations.json').read_text())
    self.cal_ns = np.array([c['ns'] for c in self.calibrations], dtype=np.int64)
    self.intrinsics = np.array(manifest['wide_projection']['intrinsics'])
    self.size = (manifest['wide_projection']['width'], manifest['wide_projection']['height'])

  def pose(self, ns):
    i = int(np.searchsorted(self.ns, ns, side='right'))-1
    if i < 0 or not 0 <= ns-int(self.ns[i]) <= 150_000_000:
      raise ValueError('No fresh recorded pose for overlay')
    return self.p[i]+self.v[i]*(ns-int(self.ns[i]))/1e9, self.r[i]

  def calibration(self, ns):
    i = int(np.searchsorted(self.cal_ns, ns, side='right'))-1
    if i < 0 or ns-int(self.cal_ns[i]) > 1_000_000_000:
      raise ValueError('No fresh camera calibration for overlay')
    return self.calibrations[i]

  def project(self, path, row, ns):
    origin, rotation = self.pose(row['query_ns'])
    vehicle = rotation @ Rotation.from_euler('xyz', row['calibration_rpy']).as_matrix()
    world = path @ AXES @ vehicle.T + origin
    position, device = self.pose(ns)
    cal = self.calibration(ns)
    calib = Rotation.from_euler('xyz', cal['rpyCalib']).as_matrix()
    wide = Rotation.from_euler('xyz', cal['wideFromDeviceEuler']).as_matrix()
    if not cal['height'] or not .5 < cal['height'][0] < 3:
      raise ValueError('No valid recorded camera height')
    # Match openpilot's road-plane height offset, then transform into the
    # selected wide camera. Motion compensation keeps each saved path registered
    # to every 20 Hz video frame without generating intermediate predictions.
    in_device = (world-position) @ device + np.array([0., 0., cal['height'][0]]) @ calib.T
    in_wide = in_device @ wide.T
    view = in_wide[:, [1, 2, 0]]
    # Clip crossing segments in 3D before projection. Dropping their first
    # vertex makes a path endpoint jump as it crosses the near plane.
    clipped = []
    near = .5
    for index, point in enumerate(view):
      if index and (view[index-1, 2] > near) != (point[2] > near):
        previous = view[index-1]
        crossing = previous+(point-previous)*((near-previous[2])/(point[2]-previous[2]))
        crossing[2] = near
        clipped.append(crossing)
      clipped.append(point)
    view = np.array(clipped)
    pixel = view @ self.intrinsics.T
    with np.errstate(divide='ignore', invalid='ignore'):
      pixel = pixel[:, :2]/pixel[:, 2:3]
    # This is the spatial path. Keep genuine points still ahead of the camera,
    # even when their predicted arrival time has passed. No cosmetic anchoring.
    valid = (view[:, 2] >= near) & np.isfinite(pixel).all(axis=1)
    return pixel, valid

  def actual(self, ns):
    wanted = ns+np.rint(FUTURE*1e9).astype(np.int64)
    first = int(np.searchsorted(self.ns, ns, side='right'))-1
    last = int(np.searchsorted(self.ns, wanted[-1]))
    if first < 0 or last >= len(self.ns) or np.any(np.diff(self.ns[first:last+1]) > 150_000_000):
      return None
    origin, rotation = self.pose(ns)
    cal = self.calibration(ns)
    vehicle = rotation @ Rotation.from_euler('xyz', cal['rpyCalib']).as_matrix()
    future = np.stack([np.interp(wanted-self.ns[0], self.ns-self.ns[0], self.p[:, a]) for a in range(3)], axis=-1)
    return (future-origin) @ vehicle @ AXES, {'query_ns': ns, 'calibration_rpy': cal['rpyCalib']}


class LoggedModels:
  def __init__(self, output, filename='render-openpilot.npz'):
    data = np.load(output/filename, allow_pickle=False)
    self.ns, self.rpy, self.xyz = (data[key] for key in ('ns', 'rpy', 'xyz'))
    if (not len(self.ns) or np.any(np.diff(self.ns) <= 0) or self.xyz.shape != (len(self.ns), 64, 3)
        or not np.isfinite(self.xyz).all() or not np.isfinite(self.rpy).all()):
      raise ValueError('Invalid logged openpilot timeline')
    self.selected = set()
    self.missing = 0

  def at(self, ns):
    index = int(np.searchsorted(self.ns, ns, side='right'))-1
    if index < 0 or ns-int(self.ns[index]) > 150_000_000:
      self.missing += 1
      return None
    self.selected.add(index)
    return self.xyz[index], {'query_ns': int(self.ns[index]), 'calibration_rpy': self.rpy[index]}


def crop_box(size, zoom):
  width, height = size
  h = min(height, width*MAIN_SIZE[1]/MAIN_SIZE[0])/zoom
  w = h*MAIN_SIZE[0]/MAIN_SIZE[1]
  return ((width-w)/2, (height-h)/2, (width+w)/2, (height+h)/2)


def draw_path(draw, pixel, valid, box, color, width):
  left, top, right, bottom = box
  xy = (pixel-np.array([left, top]))*np.array([MAIN_SIZE[0]/(right-left), MAIN_SIZE[1]/(bottom-top)])
  # Clip individual segments, never connect separate offscreen runs.
  for index in range(1, len(xy)):
    if not valid[index-1:index+1].all():
      continue
    a, b = xy[index-1].copy(), xy[index].copy()
    delta = b-a
    lo, hi = 0., 1.
    for axis, maximum in ((0, MAIN_SIZE[0]-1), (1, MAIN_SIZE[1]-1)):
      if abs(delta[axis]) < 1e-8:
        if not 0 <= a[axis] <= maximum:
          hi = -1
          break
      else:
        t0, t1 = sorted(((0-a[axis])/delta[axis], (maximum-a[axis])/delta[axis]))
        lo, hi = max(lo, t0), min(hi, t1)
    if lo <= hi:
      start, finish = tuple(a+lo*delta), tuple(a+hi*delta)
      draw.line((start, finish), fill='#15191e', width=width+2)
      draw.line((start, finish), fill=color, width=width)


def legend_statistics(output, manifest, models, big_output=None):
  """Keep measured throughput, recorded cadence, and trajectory error distinct."""
  comparison_path = output/'camera-comparison.json'
  if not comparison_path.exists():
    return {}
  comparison = json.loads(comparison_path.read_text())
  metrics = comparison['summary']['all']['models']
  timing_path = output/'matched-camera-timing-warm.json'
  if not timing_path.exists():
    timing_path = output/'matched-camera-timing.json'
  timing = json.loads(timing_path.read_text()) if timing_path.exists() else None
  stats = {}
  for name, values in metrics.items():
    error = float(values['ade_m'])
    if not math.isfinite(error) or error < 0:
      raise ValueError(f'Invalid mean path error: {name}')
    stats[name] = {'mean_error_m': error, 'error_horizon_s': comparison['horizon_s'],
                   'error_samples': comparison['summary']['all']['samples']}
    if name.endswith('cam'):
      source = Path(comparison['outputs'][name])
      profile = json.loads((source/'manifest.json').read_text()).get('camera_input_profile', 'baseline')
      stats[name]['input_profile'] = profile
      if profile == 'calibrated_vehicle_history':
        stats[name]['input_label'] = 'Calibration + learned history'
    if name.endswith('cam') and timing is not None:
      if timing['results'][name[:-3]].get('profile', profile) != profile:
        raise ValueError(f'Timing measures another input profile: {name}')
      rate = timing['results'][name[:-3]]['summary']['throughput_fps']
      stats[name].update(hz=rate, hz_label='replay', hz_source=str(timing_path))
  selected = models.ns[(models.ns >= manifest['capture_start_ns']) &
                       (models.ns <= manifest['capture_start_ns']+manifest['duration_s']*1e9)]
  if len(selected) < 2:
    raise ValueError('Insufficient small-model timestamps for its recorded rate')
  stats['openpilot'].update(hz=(len(selected)-1)*1e9/float(selected[-1]-selected[0]), hz_label='recorded',
                            hz_source='Frame-matched logged modelV2 capture timestamps within the displayed recording')
  if big_output is not None:
    report = json.loads((big_output/'inference-summary.json').read_text())
    wall = report['wall_seconds']-report['load_and_warmup_seconds']
    stats['openpilot_big'].update(hz=report['frames']/wall, hz_label='replay',
                                  hz_source='Continuous big-model replay after loading; includes decoding, preparation and saving')
  for name, entry in stats.items():
    if 'hz' in entry and (not math.isfinite(entry['hz']) or entry['hz'] <= 0):
      raise ValueError(f'Invalid measured rate: {name}')
  if big_output is not None and any('hz' not in entry for entry in stats.values()):
    raise ValueError('Final comparison requires a measured or recorded rate for every model')
  return stats


def compose(wide, grid, row, prediction, geometry, ns, zoom, small_model=None, route_map=None, seconds=0., display_views=None,
            comparisons=None, big_model=None, statistics=None):
  image = Image.new('RGB', (1920, 1080), '#10151c')
  box = crop_box(wide.size, zoom)
  main = wide.transform(MAIN_SIZE, Image.Transform.EXTENT, box, Image.Resampling.BICUBIC)
  draw = ImageDraw.Draw(main)
  paths = []
  runs = sorted([(len(display_views or [0, 1, 2, 3, 5, 6]), prediction, row), *(comparisons or [])], reverse=True,
                key=lambda run: run[0])
  if len({run[0] for run in runs}) != len(runs) or any(run[0] not in CAMERA_COLORS for run in runs):
    raise ValueError('Use one run for each supported camera configuration')
  actual = geometry.actual(ns)
  if actual is not None:
    paths.append((*actual, ACTUAL_COLOR, 13 if big_model is not None else 11 if comparisons else 5))
  if small_model is not None:
    paths.append((*small_model, COLORS[1], 11 if big_model is not None else 9 if comparisons else 7))
  if big_model is not None:
    paths.append((*big_model, BIG_COLOR, 9))
  for index, (cameras, saved, origin) in enumerate(runs):
    if saved is not None:
      paths.append((trajectories(origin, saved)[0], origin, CAMERA_COLORS[cameras], 3+2*(len(runs)-1-index)))
  for path, origin, color, width in paths:
    # The measured ego origin at inference time is known, not a new prediction.
    path = np.concatenate((np.zeros((1, 3)), path))
    pixel, valid = geometry.project(path, origin, ns)
    draw_path(draw, pixel, valid, box, color, width)
  legend = [(f'Alpamayo {cameras} cam', CAMERA_COLORS[cameras], f'{cameras}cam') for cameras, _, _ in runs]
  legend += [('openpilot small', COLORS[1], 'openpilot')]
  if big_model is not None:
    legend += [('openpilot big', BIG_COLOR, 'openpilot_big')]
  legend += [('Recorded', ACTUAL_COLOR, None)]
  details = []
  for _, _, key in legend:
    entry = (statistics or {}).get(key, {})
    lines = [entry['input_label']] if 'input_label' in entry else []
    lines += ([f'{entry["hz"]:.2f} Hz {entry["hz_label"]}'] if 'hz' in entry else [])
    if 'mean_error_m' in entry:
      lines.append(f'{entry["mean_error_m"]:.2f} m mean error')
    details.append(lines)
  legend_size = next(size for size in range(22, 13, -1)
                      if sum(max([draw.textlength(text, font=font(size)),
                                  *[draw.textlength(line, font=font(16)) for line in lines]])+24
                             for (text, _, _), lines in zip(legend, details)) <= 1200)
  widths = [max([draw.textlength(text, font=font(legend_size)),
                 *[draw.textlength(line, font=font(16)) for line in lines]])+24
            for (text, _, _), lines in zip(legend, details)]
  x = 1584-sum(widths)
  bottom = 59+20*max(map(len, details), default=0)
  draw.rounded_rectangle((x-16, 16, 1584, bottom), radius=12, fill='#10151c')
  for (text, color, _), width, lines in zip(legend, widths, details):
    draw.text((x, 23), text, font=font(legend_size), fill=color)
    for line_number, line in enumerate(lines):
      draw.text((x, 53+line_number*20), line, font=font(16), fill='#d1d8e2')
    x += width
  two_front_views = display_views == [1, 6] and not comparisons
  image.paste(main, (160 if two_front_views else 0, 0))
  if route_map is not None:
    inset = route_map.frame(seconds)
    image.paste(inset, (24, 24), inset)
  # Reuse verified, upright synchronized grid pixels; omit its label strips.
  tiles = (4,) if two_front_views else (0, 2, 4, 3, 5)
  for index, tile in enumerate(tiles):
    x, y = tile%3*576, tile//3*354+30
    small = grid.crop((x, y, x+576, y+324)).resize((320, 180), Image.Resampling.LANCZOS)
    image.paste(small, (1576, 80) if two_front_views else (1600, index*180))
  draw = ImageDraw.Draw(image)
  navigation_caption(draw, row['navigation_text'])
  if comparisons:
    comparison_captions(draw, runs)
  else:
    reasoning_caption(draw, prediction)
  return image


class WideReader:
  def __init__(self, path, records):
    import av
    self.container = av.open(str(path))
    self.container.streams.video[0].thread_type = 'AUTO'
    self.frames = iter(self.container.decode(video=0))
    self.ns = np.array([r['capture_ns'] for r in records], dtype=np.int64)
    self.index, self.image = -1, None

  def at(self, ns):
    target = int(np.searchsorted(self.ns, ns, side='right'))-1
    if target < 0 or ns-int(self.ns[target]) > 150_000_000:
      raise ValueError('Missing main camera frame')
    while self.index < target:
      frame = next(self.frames)
      self.index += 1
      if self.index == target:
        self.image = frame.to_image()
    return self.image, int(self.ns[target])

  def close(self):
    self.container.close()


def render_route(args, manifest):
  import av
  predictions = {p['tick']: p for line in (args.output/'predictions.jsonl').read_text().splitlines() if (p := json.loads(line))}
  expected = {r['tick']: r['query_ns'] for r in manifest['schedule'] if 'sample' in r}
  if {tick: p['query_ns'] for tick, p in predictions.items()} != expected:
    raise ValueError('Incomplete prediction coverage')
  schedule = {r['tick']: r for r in manifest['schedule']}
  references = []
  for compare_output in getattr(args, 'compare_output', None) or []:
    reference_manifest = json.loads((compare_output/'manifest.json').read_text())
    for key in ('route', 'sample_hz', 'capture_start_ns', 'capture_end_ns', 'future_times_s'):
      if reference_manifest[key] != manifest[key]:
        raise ValueError(f'Comparison timeline differs: {key}')
    reference_schedule = {r['tick']: r for r in reference_manifest['schedule']}
    reference_predictions = {p['tick']: p for line in (compare_output/'predictions.jsonl').read_text().splitlines()
                             if (p := json.loads(line))}
    if {tick: p['query_ns'] for tick, p in reference_predictions.items()} != expected:
      raise ValueError('Comparison predictions do not match the primary timestamps')
    for tick, row in schedule.items():
      for key in ('navigation_text', 'ego_history_xyz', 'ego_history_rot', 'calibration_rpy'):
        if reference_schedule[tick][key] != row[key]:
          raise ValueError(f'Comparison non-camera input differs: {key}')
    references.append({'output': compare_output, 'cameras': len(reference_manifest.get('camera_subset', [0, 1, 2, 3, 5, 6])),
                       'input_profile': reference_manifest.get('camera_input_profile', 'baseline'),
                       'schedule': reference_schedule, 'predictions': reference_predictions, 'selected': set()})
  if manifest.get('camera_input_profile') == 'calibrated_vehicle_history' and any(
      r['input_profile'] != 'calibrated_vehicle_history' for r in references):
    raise ValueError('Calibration comparison captions require matching input profiles')
  export = Path(manifest['export'])
  grid_meta = json.loads((export/'grid-summary.json').read_text())
  count_expected = min(grid_meta['frames'], math.ceil(manifest['duration_s']*grid_meta['fps']))
  native = json.loads((export/'clips/native-timestamps.json').read_text())['views']['1']
  geometry = Geometry(args.output, manifest)
  models = LoggedModels(args.output)
  big_output = getattr(args, 'big_model_output', None)
  big_models = None
  if big_output is not None:
    big_manifest = json.loads((big_output/'manifest.json').read_text())
    if big_manifest['route'] != manifest['route'] or big_manifest['capture_start_ns'] != manifest['capture_start_ns']:
      raise ValueError('Big-model replay belongs to another recording')
    big_models = LoggedModels(big_output, 'render-openpilot-big.npz')
  statistics = legend_statistics(args.output, manifest, models, big_output)
  from route_map import RouteMap
  route_map = RouteMap(args.output) if (args.output/'map-layout.json').exists() else None
  reader = WideReader(export/'clips/front-wide.mp4', native)
  output, partial = args.output/'comparison.mp4', args.output/'comparison.partial.mp4'
  if output.exists() or partial.exists():
    raise FileExistsError(output if output.exists() else partial)
  command = ['ffmpeg', '-v', 'error', '-n', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', '1920x1080',
             '-r', str(grid_meta['fps']), '-i', '-', '-an', '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '21',
             '-threads', '4', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(partial)]
  encoder = subprocess.Popen(command, stdin=subprocess.PIPE)
  count = 0
  selected_alpamayo = set()
  try:
    with av.open(str(export/'six-camera-drive.mp4')) as container:
      for index, frame in enumerate(container.decode(video=0)):
        if index >= count_expected:
          break
        ns = manifest['capture_start_ns']+round(index/grid_meta['fps']*1e9)
        tick = min(int(index/grid_meta['fps']*manifest['sample_hz']), max(schedule))
        if tick in predictions:
          selected_alpamayo.add(predictions[tick]['query_ns'])
        for reference in references:
          if tick in reference['predictions']:
            reference['selected'].add(reference['predictions'][tick]['query_ns'])
        wide, camera_ns = reader.at(ns)
        big_model = big_models.at(camera_ns) if big_models is not None else None
        if big_models is not None and big_model is None:
          raise ValueError(f'Missing big-model prediction at video frame {index}')
        image = compose(wide, frame.to_image(), schedule[tick], predictions.get(tick), geometry, camera_ns, args.zoom,
                        models.at(camera_ns), route_map, (camera_ns-manifest['capture_start_ns'])/1e9,
                         manifest.get('display_views'),
                           [(r['cameras'], r['predictions'].get(tick), r['schedule'][tick]) for r in references], big_model, statistics)
        encoder.stdin.write(image.tobytes())
        if index in [round(grid_meta['fps']*t) for t in (10, 16, 20, 24, 45, 116, 121, 150, 163, 221, 231, 244)]:
          image.save(args.output/f'preview-{round(index/grid_meta["fps"]):03d}.jpg', quality=93)
        count += 1
    encoder.stdin.close()
    if encoder.wait() != 0 or count != count_expected:
      raise RuntimeError(f'Video encode or frame count failure: {count}/{count_expected}')
    subprocess.run(['ffmpeg', '-v', 'error', '-xerror', '-i', str(partial), '-f', 'null', '-'], check=True)
    partial.replace(output)
  finally:
    reader.close()
    if encoder.poll() is None:
      encoder.kill()
      encoder.wait()
  errors = defaultdict(list)
  for row in manifest['schedule']:
    if row.get('recorded_motion_future_xyz') is None or row['tick'] not in predictions:
      continue
    actual = np.array(row['recorded_motion_future_xyz'])
    for name, path in zip(('alpamayo', 'openpilot'), trajectories(row, predictions[row['tick']])):
      error = np.linalg.norm(path[:, :2]-actual[:, :2], axis=-1)
      errors[name+'_ade_m'].append(float(error.mean()))
      errors[name+'_fde_m'].append(float(error[-1]))
  summary = {'video': output.name, 'frames': count, 'fps': grid_meta['fps'], 'duration_s': count/grid_meta['fps'],
             'predictions': len(predictions), 'navigation_source': manifest['navigation_source'],
              'display_zoom': args.zoom, 'model_input_profile': manifest.get('camera_input_profile', 'baseline'),
               'wide_projection': manifest['wide_projection'],
                'display_views': [0, 1, 2, 3, 5, 6] if references else manifest.get('display_views', [0, 1, 2, 3, 5, 6]),
               'model_camera_subset': manifest.get('camera_subset'),
              'overlay_alignment': 'Transform each saved prediction with measured ego poses into each displayed wide-camera frame',
                'openpilot_overlay': {'source': 'Every frame-matched logged modelV2, selected by camera capture time; independent of Alpamayo updates',
                                      'unique_predictions': len(models.selected), 'missing_video_frames': models.missing},
                'openpilot_big_overlay': None if big_models is None else {
                  'source': str(big_output), 'unique_predictions': len(big_models.selected),
                  'missing_video_frames': big_models.missing, 'pre_roll_s': big_manifest['pre_roll_s'],
                  'artifacts': big_manifest['artifacts'],
                  'prediction_sha256': hashlib.sha256((big_output/'predictions.jsonl').read_bytes()).hexdigest(),
                  'timing': 'Continuous offline recurrent replay at native camera cadence; selected by capture time'},
                'alpamayo_overlay': {'unique_predictions': len(selected_alpamayo), 'requested_sample_hz': manifest['sample_hz'],
                                     'source': 'Fresh saved predictions, held until the next prediction; no prediction interpolation'},
                 'alpamayo_reference_overlays': [
                    {'source': str(r['output']), 'configured_cameras': r['cameras'], 'unique_predictions': len(r['selected']),
                     'model_input_profile': r['input_profile'],
                    'prediction_sha256': hashlib.sha256((r['output']/'predictions.jsonl').read_bytes()).hexdigest(),
                    'camera_limit': 'Configured cameras; unavailable side views were omitted'} for r in references],
               'gray_path': 'Future integrated recorded motion at each displayed frame; not independent ground truth',
               'gps_map': None if route_map is None else {'fixes': len(route_map.times), 'missing_marker_frames': route_map.missing,
                                                         'timing': route_map.data['timing'], 'attribution': route_map.data['map']},
              'metric_reference': manifest['actual_path_limit'], 'legend_statistics': statistics,
             'metrics': {key: {'mean': float(np.mean(v)), 'median': float(np.median(v)), 'samples': len(v)} for key, v in errors.items()},
             'video_sha256': hashlib.sha256(output.read_bytes()).hexdigest()}
  (args.output/'comparison-summary.json').write_text(json.dumps(summary, indent=2)+'\n')
  configured = len(manifest.get('camera_subset', [0, 1, 2, 3, 5, 6]))
  color_names = {2: 'Magenta', 4: 'Green', 6: 'Orange'}
  caption = ('Comparing Alpamayo and openpilot on recorded driving footage.\n\n'
             f'{color_names[configured]}: Alpamayo. Blue: openpilot. Gray: recorded motion.\n'
             'These predictions are replayed offline. Alpamayo did not control the vehicle. '
              'Driver assistance requires an attentive driver.\n')
  if references:
    counts = sorted([len(manifest['camera_subset']), *[r['cameras'] for r in references]])
    labels = {2: 'Magenta: two device cameras.', 4: 'Green: two device cameras plus two front-side cameras.',
              6: 'Orange: up to six cameras.'}
    caption = ('Comparing Alpamayo camera configurations '+', '.join(map(str, counts))+' on the same recorded drive.\n\n'
                +' '.join(labels[count] for count in counts)+' '
               'Blue: openpilot. Gray: integrated recorded motion. Missing side views are omitted.\n'
                'All Alpamayo runs use the same navigation, motion history, and nominal 120° wide / 30° road input crops.\n'
               'These predictions are replayed offline. Alpamayo did not control the vehicle. '
               'Driver assistance requires an attentive driver.\n')
  elif manifest.get('camera_subset') == [1, 6]:
    caption += '\nAlpamayo receives only the device front-wide and road cameras, with nominal 120° and 30° model-input crops.\n'
  if manifest.get('camera_input_profile') == 'calibrated_vehicle_history':
    caption += ('\nBoth device-camera inputs use recorded camera calibration with nominal 120° wide / 30° road framing. '
                'Alpamayo receives causal motion history from recorded speed, steering, and learned vehicle parameters. '
                'Future recorded motion is only used for comparison. Side-camera poses remain uncalibrated.\n')
    caption += f'\nAlpamayo predictions advance at {manifest["sample_hz"]:g} saved timestamps per recorded second.\n'
  if big_models is not None:
    caption = caption.replace('Blue: openpilot.', 'Blue: logged small openpilot. Red: offline big openpilot.')
    caption += '\nThe big openpilot replay uses continuous frame history and standard openpilot camera warps, without Alpamayo input crops.\n'
  if statistics:
    caption += ('\nMean error is average horizontal path distance over 6.4 seconds, using identical recorded times for every model. '
                'Recorded motion is an integrated estimate, not independent ground truth. '
                'Alpamayo Hz uses matched warmed replay tests. Big openpilot Hz includes continuous replay preparation and saving. '
                'Small openpilot Hz is its recorded update rate. These rates use different execution boundaries.\n')
  if route_map is not None:
    caption += '\nMap data © OpenStreetMap contributors: https://www.openstreetmap.org/copyright\n'
  (args.output/'post-caption.txt').write_text(caption)
  print(json.dumps(summary, indent=2), flush=True)
