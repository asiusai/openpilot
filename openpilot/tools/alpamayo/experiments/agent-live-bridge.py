"""Bounded, read-only live input bridge. No model, GPU, publisher, or device startup.

Import this module from a persistent model owner. Call Bridge.run_once with a
synchronous consumer which returns a decoded result and exposes synchronize().
The default consumer measures CPU input construction only and predicts nothing.
"""
from bisect import bisect_right
from collections import Counter, OrderedDict, deque
from dataclasses import dataclass
import copy
import json
import math
import threading
import time

import numpy as np

from calibration_trials import asof, calibrated_image, vehicle_history
from camera_profiles import replay_module

OFFSETS_NS = (-2_000_000_000, -1_000_000_000, -500_000_000, -100_000_000, 0)
CAMERAS = (1, 6)
STATE_LIMITS = {'extrinsicsCalibration': 1_000_000_000, 'carState': 150_000_000,
                'vehicleParameters': 1_000_000_000, 'carParams': 120_000_000_000,
                'deviceMotion': 100_000_000}


class Unavailable(ValueError):
  """A fresh, complete, causal input is unavailable."""


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
  copy_ns: int = 0

  @property
  def capture_ns(self):
    # Preserve the source SOF. V0 estimates it from DMA EOF minus readout time.
    return self.sof_ns

  def reference(self):
    return dict(view=self.camera, frame_id=self.frame_id, capture_ns=self.capture_ns,
                sof_ns=self.sof_ns, eof_ns=self.eof_ns, received_ns=self.received_ns,
                ready_ns=self.ready_ns, copy_ns=self.copy_ns, source_shape=list(self.pixels.shape),
                source_format=self.format)


@dataclass(frozen=True)
class State:
  stamp: int
  received: int
  value: object
  valid: bool


@dataclass(frozen=True)
class Selection:
  query_ns: int
  selected_ns: int
  frames: tuple
  states: dict
  ingest_cpu_ns: int
  small_model: object


def storage_ok(total, free, upcoming=0):
  """Reserve both logger deletion thresholds, plus every prospective write."""
  return total > 0 and upcoming >= 0 and free-upcoming >= max(math.ceil(total*.10), 5*1024**3)


class Store:
  """Native frame ring and source-timestamped state history, with fixed ceilings."""
  def __init__(self, frame_limit=64, byte_limit=256*1024**2, state_limit=512, history_ns=3_300_000_000):
    if frame_limit < 5 or byte_limit <= 0 or state_limit < 40 or history_ns < 2_200_000_000:
      raise ValueError('Insufficient positive history bounds')
    self.frame_limit, self.byte_limit, self.state_limit, self.history_ns = frame_limit, byte_limit, state_limit, history_ns
    self.frames = {camera: deque() for camera in CAMERAS}
    self.states = {kind: [] for kind in (*STATE_LIMITS, 'modelV2')}
    self.lock = threading.Lock()
    self.bytes = self.peak_bytes = self.ingest_cpu_ns = 0
    self.counts = Counter()

  def add_frame(self, frame):
    if (frame.camera not in CAMERAS or frame.frame_id < 0 or
        not 0 < frame.sof_ns <= frame.eof_ns <= frame.received_ns <= frame.ready_ns or
        frame.eof_ns-frame.sof_ns > 100_000_000 or frame.format not in ('rgb', 'nv12') or
        frame.pixels.dtype != np.uint8 or not frame.pixels.flags.c_contiguous or frame.pixels.flags.writeable or
        (frame.format == 'rgb' and (frame.pixels.ndim != 3 or frame.pixels.shape[0] != 3)) or
        (frame.format == 'nv12' and (frame.pixels.ndim != 2 or frame.pixels.shape[0] % 3 or frame.pixels.shape[1] % 2))):
      raise ValueError('Invalid native frame or mutable storage')
    if frame.pixels.nbytes > min(self.byte_limit, 8*1024**2):
      raise ValueError('Native frame exceeds the byte ceiling')
    with self.lock:
      ring = self.frames[frame.camera]
      if ring and (frame.capture_ns <= ring[-1].capture_ns or frame.frame_id <= ring[-1].frame_id):
        self.counts['late_or_duplicate_frames'] += 1
        return False
      if ring:
        self.counts['source_frame_id_gaps'] += max(0, frame.frame_id-ring[-1].frame_id-1)
      ring.append(frame)
      self.bytes += frame.pixels.nbytes
      self.counts['received_frames'] += 1
      for candidate in self.frames.values():
        while candidate and (len(candidate) > self.frame_limit or candidate[0].capture_ns < frame.capture_ns-self.history_ns):
          self.bytes -= candidate.popleft().pixels.nbytes
          self.counts['frame_evictions'] += 1
      while self.bytes > self.byte_limit:
        oldest = min((r for r in self.frames.values() if r), key=lambda r: r[0].capture_ns)
        self.bytes -= oldest.popleft().pixels.nbytes
        self.counts['byte_evictions'] += 1
      self.peak_bytes = max(self.peak_bytes, self.bytes)
      return True

  def add_state(self, kind, stamp, received, value, valid=True, ingest_cpu_ns=0):
    begin_cpu = time.thread_time_ns()
    if kind not in self.states or not 0 < stamp <= received or ingest_cpu_ns < 0:
      raise ValueError('Invalid state source/receive timestamp')
    if valid and len(json.dumps(value, allow_nan=False).encode()) > 64*1024:
      raise ValueError('State exceeds the fixed 64 KiB serialized record ceiling')
    # The caller owns no mutable reference after ingestion. Invalid samples are
    # tombstones: an invalid update must not uncover an older valid sample.
    entry = State(stamp, received, copy.deepcopy(value) if valid else None, valid)
    with self.lock:
      rows = self.states[kind]
      index = bisect_right([r.stamp for r in rows], stamp)
      if index and rows[index-1].stamp == stamp:
        self.counts['duplicate_states'] += 1
        return False
      rows.insert(index, entry)
      horizon = 121_000_000_000 if kind == 'carParams' else self.history_ns
      maximum = 16 if kind == 'carParams' else self.state_limit
      while rows and (len(rows) > maximum or rows[0].stamp < received-horizon):
        rows.pop(0)
        self.counts['state_evictions'] += 1
      self.ingest_cpu_ns += ingest_cpu_ns+time.thread_time_ns()-begin_cpu
      self.counts['received_states'] += 1
      return True

  def select(self, now, last_query, last_current, max_input_age_ns, require_small_model):
    with self.lock:
      rings = {c: tuple(f for f in ring if f.ready_ns <= now) for c, ring in self.frames.items()}
      states = {k: tuple(r for r in rows if r.received <= now) for k, rows in self.states.items()}
      ingest = self.ingest_cpu_ns
    if any(not ring for ring in rings.values()):
      raise Unavailable('missing_front_camera')
    query = rings[6][-1].capture_ns
    if query <= last_query:
      raise Unavailable('no_new_query')
    if not 0 <= now-query <= max_input_age_ns:
      raise Unavailable('stale_query')
    selected = []
    for camera in CAMERAS:
      ring = rings[camera]
      times = [f.capture_ns for f in ring]
      for offset in OFFSETS_NS:
        desired = query+offset
        index = bisect_right(times, desired)-1
        if index < 0 or desired-times[index] > 150_000_000:
          raise Unavailable('missing_or_delayed_front_history')
        selected.append(ring[index])
      if selected[-1].capture_ns <= last_current.get(camera, 0):
        raise Unavailable('stale_current_camera')
      if len({f.capture_ns for f in selected[-5:]}) != 5:
        raise Unavailable('repeated_history_frame')
    small = next((r.value for r in reversed(states['modelV2']) if r.valid and
                  r.value['frameId'] == selected[-1].frame_id and r.value['timestampEof'] == selected[-1].eof_ns), None)
    if require_small_model and small is None:
      raise Unavailable('missing_matching_small_model')
    return Selection(query, now, tuple(selected), states, ingest, small)

  def bounds(self):
    with self.lock:
      return dict(frame_bytes=self.bytes, peak_frame_bytes=self.peak_bytes, frame_byte_limit=self.byte_limit,
                  frames={str(c): len(r) for c, r in self.frames.items()}, frame_limit_per_camera=self.frame_limit,
                  states={k: len(v) for k, v in self.states.items()}, state_limit=self.state_limit,
                  counts=dict(self.counts))


def records(rows, query):
  causal = [r for r in rows if r.stamp <= query]
  return np.array([r.stamp for r in causal], dtype=np.int64), [r.value if r.valid else None for r in causal]


def required(record, stamp, age):
  try:
    value, selected = asof(record, stamp, age)
  except ValueError as exc:
    raise Unavailable('missing_or_stale_state') from exc
  if value is None:
    raise Unavailable('invalid_state_update')
  return value, selected


class Preparer:
  """Canonical nominal 120/30 projection plus measured and learned history."""
  def __init__(self, source, target, warp=calibrated_image, cache_limit=32, cache_bytes=48*1024**2, clock=time.monotonic_ns):
    if cache_limit < 0 or cache_bytes < 0:
      raise ValueError('Negative image cache limit')
    if any(tuple(source['cameras'][str(c)]['prepared_size']) != (768, 432) for c in CAMERAS):
      raise ValueError('Keep the calibrated reference image resolution: 768x432')
    self.source, self.target, self.warp, self.clock = copy.deepcopy(source), copy.deepcopy(target), warp, clock
    self.cache_limit, self.cache_bytes = cache_limit, cache_bytes
    self.cache, self.bytes = OrderedDict(), 0
    self.Motion = replay_module().Motion

  def rgb(self, frame):
    if frame.format == 'rgb':
      return frame.pixels
    # This uses the existing route decoder's PyAV dependency. Live color-range
    # parity with recorded video is a separate device qualification checkpoint.
    import av
    rgb = av.VideoFrame.from_ndarray(frame.pixels, format='nv12').to_ndarray(format='rgb24')
    return np.ascontiguousarray(rgb.transpose(2, 0, 1))

  def prepare(self, selection):
    begin = self.clock()
    query = selection.query_ns
    state = {k: records(v, query) for k, v in selection.states.items() if k in STATE_LIMITS}
    calibration, calibration_ns = required(state['extrinsicsCalibration'], query, STATE_LIMITS['extrinsicsCalibration'])
    motion_values = {r.stamp: r.value for r in selection.states['deviceMotion'] if r.stamp <= query and r.valid}
    if not motion_values or any(not r.valid and query-2_050_000_000 <= r.stamp <= query for r in selection.states['deviceMotion']):
      raise Unavailable('missing_or_invalid_motion')
    try:
      xyz, rot, future, motion_age_ms = self.Motion(motion_values).sample(query, calibration['rpyCalib'])
    except ValueError as exc:
      raise Unavailable('missing_or_stale_motion_history') from exc
    if future is not None:
      raise AssertionError('Future motion entered live preparation')
    required(state['carParams'], query, STATE_LIMITS['carParams'])
    for stamp in query+np.arange(-38, 1, dtype=np.int64)*50_000_000:
      for kind in ('carState', 'vehicleParameters'):
        required(state[kind], int(stamp), STATE_LIMITS[kind])
    row = dict(query_ns=query, ego_history_xyz=xyz, ego_history_rot=rot)
    xyz, rot, audit = vehicle_history(row, state)
    history_done = self.clock()
    images, refs, hits = [], [], 0
    for frame, offset in zip(selection.frames, OFFSETS_NS*2, strict=True):
      cal, cal_ns = required(state['extrinsicsCalibration'], frame.capture_ns, STATE_LIMITS['extrinsicsCalibration'])
      key = (frame.camera, frame.frame_id, frame.capture_ns, cal_ns, json.dumps(cal, sort_keys=True))
      cached = self.cache.get(key)
      # Source objects are immutable and retained by this bounded cache. This
      # identity check also prevents reusing pixels after a capture restart.
      if cached is not None and cached[0] is frame:
        image = cached[1]
        hits += 1
        self.cache.move_to_end(key)
      else:
        image = self.warp(self.rgb(frame), frame.camera, self.source, self.target, cal)
        if image.dtype != np.uint8 or image.shape != (3, 432, 768):
          raise ValueError('Calibrated frame lost the full reference resolution')
        image.flags.writeable = False
        if key in self.cache:
          self.bytes -= self.cache.pop(key)[2]
        size = image.nbytes+frame.pixels.nbytes
        if self.cache_limit and size <= self.cache_bytes:
          self.cache[key] = (frame, image, size)
          self.bytes += size
          while len(self.cache) > self.cache_limit or self.bytes > self.cache_bytes:
            self.bytes -= self.cache.popitem(last=False)[1][2]
      images.append(image[None])
      refs.append({**frame.reference(), 'requested_ns': query+offset, 'image_calibration_ns': cal_ns})
    arrays = dict(image_frames=np.stack(images), camera_indices=np.array([f.camera for f in selection.frames], dtype=np.int64),
                  absolute_timestamps=np.array([f.capture_ns//1000 for f in selection.frames], dtype=np.int64),
                  ego_history_xyz=xyz, ego_history_rot=rot)
    end = self.clock()
    detail = dict(frames=refs, calibration_rpy=calibration['rpyCalib'], calibration_log_mono_time=calibration_ns,
                  vehicle_history_audit=audit, motion_extrapolation_ms=motion_age_ms,
                  history_build_ms=(history_done-begin)/1e6, color_warp_stack_ms=(end-history_done)/1e6,
                  image_cache_hits=hits, image_cache_bytes=self.bytes, image_cache_entries=len(self.cache),
                  image_cache_byte_limit=self.cache_bytes, small_model=selection.small_model,
                  camera_profile='calibrated_vehicle_history', history_policy='causal_floor_all_received_native_frames')
    return arrays, detail


class Bridge:
  """One active query, zero queued plans. Select newest input only when idle."""
  def __init__(self, store, preparer, hz=5., clock=time.monotonic_ns, max_input_age_ns=250_000_000,
               require_small_model=False, navigation='Follow the road.'):
    if not 0 < hz <= 20 or max_input_age_ns <= 0 or not navigation or len(navigation) > 1024:
      raise ValueError('Invalid scheduler configuration')
    self.store, self.preparer, self.clock = store, preparer, clock
    self.period_ns, self.max_input_age_ns = round(1e9/hz), max_input_age_ns
    self.require_small_model, self.navigation = require_small_model, navigation
    self.last_query, self.last_current, self.deadline = 0, {}, 0
    self.active = threading.Lock()
    self.counts = Counter()
    self.first_dispatch = self.last_completion = self.previous_ingest = None
    self.observation_start = None
    self.last_frame_id = None

  def run_once(self, consumer=None):
    # Consumer contract: predict(arrays, metadata) -> {query_ns, prediction}.
    # synchronize() MUST wait for all work that owns the decoded trajectory.
    # No fallback to the previous result is accepted. CPU mode counts zero plans.
    if not self.active.acquire(blocking=False):
      self.counts['busy_calls'] += 1
      return None
    try:
      begin = self.clock()
      if self.observation_start is None:
        self.observation_start = begin
      if begin < self.deadline:
        return None
      try:
        selected = self.store.select(begin, self.last_query, self.last_current, self.max_input_age_ns, self.require_small_model)
        arrays, detail = self.preparer.prepare(selected)
      except Unavailable as exc:
        self.counts[str(exc)] += 1
        return None
      prepared = self.clock()
      self.deadline = begin+self.period_ns
      self.last_query = selected.query_ns
      self.last_current = {c: selected.frames[i*5+4].capture_ns for i, c in enumerate(CAMERAS)}
      frame_id = selected.frames[-1].frame_id
      if self.last_frame_id is not None:
        self.counts['road_frame_ids_skipped_between_queries'] += max(0, frame_id-self.last_frame_id-1)
      self.last_frame_id = frame_id
      arrays['nav_text'] = [self.navigation]
      meta = dict(query_ns=selected.query_ns, selected_ns=begin, prepared_ns=prepared, **detail,
                  navigation_text=self.navigation, navigation_source='explicit fixed test instruction')
      result = None
      if consumer is not None:
        result = consumer.predict(arrays, meta)
        consumer.synchronize()
        if result['query_ns'] != selected.query_ns:
          raise ValueError('Consumer returned a stale or unrelated plan')
        xyz = np.asarray(result['prediction']['xyz'])
        if xyz.shape not in ((1, 64, 3), (64, 3)) or not np.isfinite(xyz).all():
          raise ValueError('Consumer must return a complete decoded trajectory')
      completed = self.clock()
      if completed < prepared:
        raise ValueError('Measurement clock moved backward')
      current = (selected.frames[4], selected.frames[9])
      timing = dict(service_including_history_ms=(completed-begin)/1e6, prepare_ms=(prepared-begin)/1e6,
                    consumer_and_completion_ms=(completed-prepared)/1e6,
                    query_to_output_ms=(completed-selected.query_ns)/1e6,
                    oldest_current_sof_to_output_ms=max(completed-f.sof_ns for f in current)/1e6,
                    camera=[dict(view=f.camera, sof_to_received_ms=(f.received_ns-f.sof_ns)/1e6,
                                 eof_to_received_ms=(f.received_ns-f.eof_ns)/1e6,
                                 receive_to_ready_ms=(f.ready_ns-f.received_ns)/1e6,
                                 sof_age_at_dispatch_ms=(begin-f.sof_ns)/1e6,
                                 sof_to_output_ms=(completed-f.sof_ns)/1e6,
                                 eof_to_output_ms=(completed-f.eof_ns)/1e6) for f in current],
                    state_ingest_cpu_since_previous_dispatch_ms=None if self.previous_ingest is None else
                      (selected.ingest_cpu_ns-self.previous_ingest)/1e6,
                    sensor_acquisition_latency_verified=False)
      self.previous_ingest = selected.ingest_cpu_ns
      self.first_dispatch = begin if self.first_dispatch is None else self.first_dispatch
      self.last_completion = completed
      self.counts['input_completions'] += 1
      self.counts['model_completions'] += int(consumer is not None)
      self.counts['missed_service_periods'] += max(0, (completed-begin)//self.period_ns)
      return dict(kind='model_completion' if consumer is not None else 'cpu_input_only', input=meta,
                  prediction=None if result is None else result['prediction'], completed_ns=completed, timing=timing,
                  limits=self.store.bounds())
    finally:
      self.active.release()

  def summary(self):
    # Include missing-camera and state stalls after the last result. A held plan
    # must never imply ongoing throughput after input delivery stops.
    wall = 0 if self.observation_start is None else (self.clock()-self.observation_start)/1e9
    return dict(counts=dict(self.counts), requested_hz=1e9/self.period_ns,
                model_completion_hz=self.counts['model_completions']/wall if wall > 0 else 0.,
                measurement_wall_s=wall, queued_queries=0, maximum_active_queries=1,
                limit='CPU input completions are not model predictions; source SOF is estimated on v0')
