"""Causal, bounded two-camera inputs shared by live capture and replay."""
from bisect import bisect_right
from collections import OrderedDict, deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
import copy
import hashlib
import json
import re
import threading

import numpy as np

CAMERAS = (1, 6)
OFFSETS_NS = (-2_000_000_000, -1_000_000_000, -500_000_000, -100_000_000, 0)
HALF_SECOND_NS = 500_000_000
FRAME_TOLERANCE_NS = 150_000_000
STATE_LIMITS = {'extrinsicsCalibration': 1_000_000_000, 'carState': 150_000_000,
                'vehicleParameters': 1_000_000_000, 'carParams': 120_000_000_000,
                'deviceMotion': 100_000_000, 'navInstruction': 2_000_000_000}


class Unavailable(ValueError):
  """A complete causal input is unavailable."""


class Priming(Unavailable):
  """A real anchor pair was retained, but the two-second history is incomplete."""
  def __init__(self, reason, anchors):
    super().__init__(reason)
    self.anchors = anchors


@dataclass(frozen=True)
class Frame:
  camera: int
  frame_id: int
  sof_ns: int
  eof_ns: int
  received_ns: int
  ready_ns: int
  pixels: np.ndarray
  format: str = 'rgb'

  def reference(self):
    return {'view': self.camera, 'frame_id': self.frame_id, 'capture_ns': self.sof_ns, 'eof_ns': self.eof_ns,
            'received_ns': self.received_ns, 'ready_ns': self.ready_ns, 'source_format': self.format}


@dataclass(frozen=True)
class State:
  stamp: int
  received: int
  value: object
  valid: bool = True


@dataclass(frozen=True)
class Selection:
  query_ns: int
  selected_ns: int
  frames: tuple[Frame, ...]
  states: dict[str, tuple[State, ...]]
  small_model: dict
  camera_gaps: dict = field(default_factory=dict)


def required(rows, stamp, max_age):
  index = bisect_right([r.stamp for r in rows], stamp)-1
  if index < 0 or stamp-rows[index].stamp > max_age or not rows[index].valid:
    raise Unavailable('Missing, invalid, or stale state')
  return rows[index].value, rows[index].stamp


class Store:
  def __init__(self, frame_limit=64, byte_limit=256*1024**2, state_limit=512):
    if frame_limit < 5 or byte_limit <= 0 or state_limit < 40:
      raise ValueError('Insufficient positive history bounds')
    self.frame_limit, self.byte_limit, self.state_limit = frame_limit, byte_limit, state_limit
    self.frames = {camera: deque() for camera in CAMERAS}
    self.camera_gaps = dict.fromkeys(CAMERAS, 0)
    self.states = {kind: [] for kind in (*STATE_LIMITS, 'modelV2')}
    self.bytes = 0
    self.lock = threading.Lock()

  def add_frame(self, frame):
    if (frame.camera not in CAMERAS or frame.frame_id < 0 or
        not 0 < frame.sof_ns <= frame.eof_ns <= frame.received_ns <= frame.ready_ns or
        frame.pixels.dtype != np.uint8 or not frame.pixels.flags.c_contiguous or frame.pixels.flags.writeable or
        frame.pixels.nbytes > min(self.byte_limit, 8*1024**2) or
        frame.format not in ('rgb', 'nv12') or
        (frame.format == 'rgb' and (frame.pixels.ndim != 3 or frame.pixels.shape[0] != 3)) or
        (frame.format == 'nv12' and (frame.pixels.ndim != 2 or frame.pixels.shape[0] % 3 or frame.pixels.shape[1] % 2))):
      raise ValueError('Invalid frame or mutable pixel storage')
    with self.lock:
      ring = self.frames[frame.camera]
      if ring and (frame.sof_ns <= ring[-1].sof_ns or frame.frame_id <= ring[-1].frame_id):
        return False
      if ring:
        self.camera_gaps[frame.camera] += max(0, frame.frame_id-ring[-1].frame_id-1)
      ring.append(frame)
      self.bytes += frame.pixels.nbytes
      for candidate in self.frames.values():
        while candidate and (len(candidate) > self.frame_limit or candidate[0].sof_ns < frame.sof_ns-3_300_000_000):
          self.bytes -= candidate.popleft().pixels.nbytes
      while self.bytes > self.byte_limit:
        oldest = min((r for r in self.frames.values() if r), key=lambda r: r[0].sof_ns)
        self.bytes -= oldest.popleft().pixels.nbytes
    return True

  def add_state(self, kind, stamp, received, value, valid=True):
    if kind not in self.states or not 0 < stamp <= received:
      raise ValueError('Invalid state timestamps')
    if valid and len(json.dumps(value, allow_nan=False).encode()) > 64*1024:
      raise ValueError('State exceeds 64 KiB')
    row = State(stamp, received, copy.deepcopy(value) if valid else None, valid)
    with self.lock:
      rows = self.states[kind]
      index = bisect_right([r.stamp for r in rows], stamp)
      if index and rows[index-1].stamp == stamp:
        if received <= rows[index-1].received:
          return
        rows[index-1] = row
      else:
        rows.insert(index, row)
      horizon = 121_000_000_000 if kind == 'carParams' else 3_300_000_000
      while rows and (len(rows) > self.state_limit or rows[0].stamp < received-horizon):
        rows.pop(0)

  def select(self, now, last_query=0, last_current=None, *, half_second_slot=None):
    with self.lock:
      rings = {c: tuple(f for f in ring if f.ready_ns <= now) for c, ring in self.frames.items()}
      states = {k: tuple(r for r in rows if r.received <= now) for k, rows in self.states.items()}
      gaps = self.camera_gaps.copy()
    # Wait for the newest matching small-model output, rather than chasing a
    # camera that is always ahead of modeld. The query keeps its actual SOF.
    matches = [(frame, row) for frame in reversed(rings[6]) if half_second_slot is None or frame.sof_ns <= half_second_slot
               for row in reversed(states['modelV2'])
               if row.valid and row.value['frameId'] == frame.frame_id and row.value['timestampEof'] == frame.eof_ns]
    if not matches:
      raise Unavailable('No exact small-model frame match')
    current, model = matches[0]
    query = current.sof_ns
    if query <= last_query or not 0 <= now-query <= 250_000_000:
      raise Unavailable('No fresh query')
    if half_second_slot is not None and not (query <= half_second_slot <= now and half_second_slot-query <= FRAME_TOLERANCE_NS):
      raise Unavailable('No fresh half-second anchor')
    offsets = OFFSETS_NS if half_second_slot is None else OFFSETS_NS[-2:]
    frames = []
    for camera in CAMERAS:
      ring = rings[camera]
      times = [f.sof_ns for f in ring]
      for offset in offsets:
        desired = query+offset
        index = bisect_right(times, desired)-1
        if index < 0 or desired-times[index] > FRAME_TOLERANCE_NS:
          raise Unavailable('Missing causal camera history')
        frames.append(ring[index])
      if len({f.sof_ns for f in frames[-len(offsets):]}) != len(offsets) or frames[-1].sof_ns <= (last_current or {}).get(camera, 0):
        raise Unavailable('Repeated or stalled camera history')
    # Only the matched small-model output is consumed. Retain the complete vehicle state history.
    del states['modelV2']
    return Selection(query, now, tuple(frames), states, {'log_mono_time': model.stamp, 'model': model.value}, gaps)


def navigation(rows, query):
  # A received cancellation or route replacement invalidates older guidance,
  # even when its publication follows the selected camera's exposure.
  if rows and not rows[-1].valid:
    return '', {'source': 'unavailable', 'log_mono_time': rows[-1].stamp}
  try:
    value, stamp = required(rows, query, STATE_LIMITS['navInstruction'])
  except Unavailable:
    return '', {'source': 'unavailable', 'log_mono_time': None}
  text = ' '.join(value['maneuverPrimaryText'].split())
  secondary = ' '.join(value.get('maneuverSecondaryText', '').split())
  kind, modifier = value.get('maneuverType', ''), value.get('maneuverModifier', '')
  distance = value['maneuverDistance']
  if (not np.isfinite(distance) or distance < 0 or
      rows[-1].value.get('routeRevision') != value.get('routeRevision')):
    return '', {'source': 'unavailable', 'log_mono_time': stamp}
  direction = {'left': 'left', 'right': 'right', 'slight left': 'slightly left', 'slight right': 'slightly right',
               'sharp left': 'sharply left', 'sharp right': 'sharply right', 'straight': 'straight'}.get(modifier, '')
  action = {'turn': 'Turn', 'end of road': 'At the end of the road, turn', 'fork': 'Keep', 'merge': 'Merge',
            'on ramp': 'Take the ramp', 'off ramp': 'Take the exit', 'continue': 'Continue', 'new name': 'Continue',
            'depart': 'Depart', 'arrive': 'Arrive at the destination', 'roundabout': 'Enter the roundabout',
            'rotary': 'Enter the roundabout', 'roundabout turn': 'At the roundabout, turn',
            'exit roundabout': 'Exit the roundabout', 'exit rotary': 'Exit the roundabout'}.get(kind, '')
  if modifier == 'uturn':
    action, direction = 'Make a U-turn', ''
  if action:
    # Preserve maneuver, direction, exit and distance within the model's short
    # navigation budget. Full road names and sign text remain in the audit.
    text = action+(f' {direction}' if direction and kind != 'arrive' else '')
    if kind in ('roundabout', 'rotary', 'roundabout turn'):
      banner = value['maneuverPrimaryText']+' '+secondary
      exit_match = re.search(r'\b(\d{1,2})(?:st|nd|rd|th)?\s+exit\b|\bexit\s+(\d{1,2})\b', banner, re.IGNORECASE)
      if exit_match:
        text = f'At the roundabout, take exit {exit_match.group(1) or exit_match.group(2)}'
  if not text:
    return '', {'source': 'unavailable', 'log_mono_time': stamp}
  text = f'{text} in {round(distance)}m'
  if len(text) > 1024:
    raise Unavailable('Navigation instruction exceeds 1024 characters')
  return text, {'source': 'navInstruction', 'log_mono_time': stamp, 'instruction': value, 'model_instruction': text}


class Preparer:
  def __init__(self, geometry, cache_limit=48, sampler=None):
    if cache_limit < 0:
      raise ValueError('Negative image cache capacity')
    if any(tuple(geometry['cameras'][str(c)]['prepared_size']) != (768, 432) for c in CAMERAS):
      raise ValueError('Keep the full calibrated reference resolution: 768x432')
    self.geometry = copy.deepcopy(geometry)
    self.cache, self.cache_limit = OrderedDict(), cache_limit
    self.sampler = sampler
    self.anchors = deque(maxlen=4)

  def prepare(self, selected, half_second_slot=None):
    from openpilot.selfdrive.alpamayo.history import ego_history
    query, states = selected.query_ns, selected.states
    if half_second_slot is not None:
      views, hits = self.rolling_images(selected, half_second_slot)
    cal, cal_ns = required(states['extrinsicsCalibration'], query, STATE_LIMITS['extrinsicsCalibration'])
    xyz, rot, history = ego_history(states, query, cal['rpyCalib'])
    cp, cp_ns = required(states['carParams'], query, STATE_LIMITS['carParams'])
    history['car_params'] = {'source_ns': cp_ns, 'age_ms': (query-cp_ns)/1e6, 'values': copy.deepcopy(cp)}
    history['learned_parameters'] = []
    for sample_ns in query+np.arange(-38, 1, dtype=np.int64)*50_000_000:
      parameters, stamp = required(states['vehicleParameters'], int(sample_ns), STATE_LIMITS['vehicleParameters'])
      history['learned_parameters'].append({'sample_ns': int(sample_ns), 'source_ns': stamp, 'age_ms': (int(sample_ns)-stamp)/1e6,
                                            'values': copy.deepcopy(parameters),
                                            'applied_stiffness_factor': max(parameters['stiffnessFactor'], .1),
                                            'applied_steer_ratio': max(parameters['steerRatio'], .1)})
    if half_second_slot is None:
      views, hits = self.project(selected.frames, states)
    images, refs = zip(*views, strict=True)
    refs = [dict(ref, requested_ns=query+offset) for ref, offset in zip(refs, OFFSETS_NS*2, strict=True)]
    nav, nav_meta = navigation(states['navInstruction'], query)
    arrays = {'image_frames': np.stack(images), 'camera_indices': np.array([1]*5+[6]*5, dtype=np.int64),
              'absolute_timestamps': np.array([ref['capture_ns']//1000 for ref in refs], dtype=np.int64),
              'ego_history_xyz': xyz, 'ego_history_rot': rot, 'nav_text': [nav]}
    metadata = {'query_ns': query, 'selected_ns': selected.selected_ns, 'frames': refs, 'small_model': selected.small_model,
                'calibration_rpy': cal['rpyCalib'], 'calibration': copy.deepcopy(cal), 'calibration_log_mono_time': cal_ns,
                'ego_history_xyz': xyz.tolist(),
                'ego_history_rot': rot.tolist(), 'vehicle_history': history, 'navigation_text': nav, 'navigation': nav_meta,
                'camera_profile': 'calibrated_vehicle_history', 'image_cache_hits': hits, 'camera_gaps': selected.camera_gaps}
    return arrays, metadata

  def rolling_images(self, selected, opening):
    """Only retain owned, calibrated pixels. The worker unmaps the input after each slot."""
    query, frames = selected.query_ns, selected.frames
    if not (len(frames) == 4 and opening <= selected.selected_ns < opening+HALF_SECOND_NS and
            0 <= opening-query <= FRAME_TOLERANCE_NS and frames[-1].sof_ns == query):
      raise Unavailable('Front query is outside its slot')
    reason = 'Priming front-camera anchors'
    if self.anchors and opening <= self.anchors[-1][0]:
      raise Unavailable('Front slot did not advance')
    if self.anchors and opening != self.anchors[-1][0]+HALF_SECOND_NS:
      self.anchors.clear()
      reason = 'Repriming after a missed slot'
    for i, camera in enumerate(CAMERAS):
      prior, current = frames[2*i:2*i+2]
      if not (prior.camera == current.camera == camera and prior.frame_id < current.frame_id and prior.sof_ns < current.sof_ns and
              0 <= query-100_000_000-prior.sof_ns <= FRAME_TOLERANCE_NS and 0 <= query-current.sof_ns <= FRAME_TOLERANCE_NS and
              max(prior.ready_ns, current.ready_ns) <= selected.selected_ns):
        raise Unavailable('Missing, repeated or stale front exposure')
      if self.anchors:
        _, last_pair = self.anchors[-1]
        _, last = last_pair[i]
        if last['capture_ns'] >= prior.sof_ns or last['frame_id'] >= prior.frame_id:
          raise Unavailable('Front exposure did not advance')
    older = []
    if len(self.anchors) == 4:
      older = [[self.anchors[-lag][1][i] for lag in (4, 2, 1)] for i in range(len(CAMERAS))]
      if any(abs(query+offset-ref['capture_ns']) > FRAME_TOLERANCE_NS for view in older for (_, ref), offset in zip(view, OFFSETS_NS[:3], strict=True)):
        self.anchors.clear()
        older = []
        reason = 'Repriming after a capture gap'
    fresh, hits = self.project(frames if older else frames[1::2], selected.states)
    current_views = fresh[1::2] if older else fresh
    self.anchors.append((opening, [(image, dict(ref, anchor_slot_ns=opening, anchor_query_ns=query, anchor_selected_ns=selected.selected_ns))
                                  for image, ref in current_views]))
    if not older:
      raise Priming(f'{reason} ({len(self.anchors)}/4 slots)', self.anchors[-1][1])
    return [frame for i, view in enumerate(older) for frame in (*view, *fresh[2*i:2*i+2])], hits+6

  def project(self, frames, states):
    from openpilot.selfdrive.alpamayo.geometry import calibrated_image
    refs, keys, missing, hits = [], [], {}, 0
    for frame in frames:
      calibration, stamp = required(states['extrinsicsCalibration'], frame.sof_ns, STATE_LIMITS['extrinsicsCalibration'])
      # Content identity survives the process boundary and invalidates reused
      # frame IDs, changed pixels, and changed calibration without approximation.
      key = (frame.camera, frame.pixels.shape, frame.format, hashlib.sha256(frame.pixels).digest(), json.dumps(calibration, sort_keys=True))
      if key in self.cache or key in missing:
        hits += 1
      else:
        missing[key] = frame, calibration
      keys.append(key)
      refs.append({**frame.reference(), 'image_calibration_ns': stamp, 'image_calibration': copy.deepcopy(calibration)})
    def warp(item):
      frame, calibration = item
      pixels = frame.pixels
      if frame.format == 'nv12':
        import av
        pixels = np.ascontiguousarray(av.VideoFrame.from_ndarray(pixels, format='nv12').to_ndarray(format='rgb24').transpose(2, 0, 1))
      image = (self.sampler or calibrated_image)(pixels, frame.camera, self.geometry, calibration)
      if np.shares_memory(image, frame.pixels):
        image = image.copy()  # The source mapping closes when this snapshot finishes.
      image.flags.writeable = False
      return image
    if self.sampler is not None and len(missing) > 1:
      # Both workers finish before mapped frame storage is released. Only this thread mutates the LRU.
      with ThreadPoolExecutor(max_workers=2) as pool:
        self.cache.update(zip(missing, pool.map(warp, missing.values()), strict=True))
    else:
      self.cache.update((key, warp(item)) for key, item in missing.items())
    images = [self.cache[key][None] for key in keys]
    for key in keys:
      self.cache.move_to_end(key)
    while len(self.cache) > self.cache_limit:
      self.cache.popitem(last=False)
    return list(zip(images, refs, strict=True)), hits
