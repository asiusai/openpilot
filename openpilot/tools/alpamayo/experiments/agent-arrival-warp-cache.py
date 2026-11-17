"""Optional CPU-only preparation of received RGB frames; no model or GPU work."""
from collections import Counter, OrderedDict
import json
import threading
import time
import weakref


class ArrivalWarp:
  def __init__(self, warp, bridge, source, target, *, max_entries=128, max_bytes=128*1024**2):
    if max_entries < 1 or max_bytes < 3*432*768:
      raise ValueError('Insufficient bounded output cache')
    self.warp, self.bridge, self.source, self.target = warp, bridge, source, target
    self.max_entries, self.max_bytes = max_entries, max_bytes
    self.cache, self.counts, self.bytes, self.peak_bytes = OrderedDict(), Counter(), 0, 0
    self.lock = threading.Lock()

  def __call__(self, array, camera, source, target, calibration):
    # Mutable arrays cannot use an identity cache. Keep the original validation.
    if array.flags.writeable:
      return self.warp(array, camera, source, target, calibration)
    key = (id(array), camera, json.dumps((source, target, calibration), sort_keys=True, allow_nan=False))
    with self.lock:
      cached = self.cache.get(key)
      if cached is not None and cached[0]() is array:
        self.cache.move_to_end(key)
        self.counts['hits'] += 1
        return cached[1]
      self.counts['misses'] += 1
      result = self.warp(array, camera, source, target, calibration)
      result.flags.writeable = False
      if result.nbytes <= self.max_bytes:
        if cached is not None:
          self.bytes -= self.cache.pop(key)[1].nbytes
        # Do not retain raw frames beyond the native frame ring's own lifetime.
        self.cache[key] = (weakref.ref(array), result)
        self.bytes += result.nbytes
        while len(self.cache) > self.max_entries or self.bytes > self.max_bytes:
          self.bytes -= self.cache.popitem(last=False)[1][1].nbytes
          self.counts['evictions'] += 1
        self.peak_bytes = max(self.peak_bytes, self.bytes)
      return result

  def received(self, store, frame, now):
    """Call after native admission. Only already received calibration is eligible."""
    if frame.ready_ns > now:
      raise ValueError('Cannot prepare an unreceived frame')
    if frame.format != 'rgb':
      self.counts['non_rgb_not_prepared'] += 1
      return False
    with store.lock:
      rows = tuple(r for r in store.states['extrinsicsCalibration'] if r.received <= now)
    try:
      calibration, _ = self.bridge.required(self.bridge.records(rows, frame.capture_ns),
        frame.capture_ns, self.bridge.STATE_LIMITS['extrinsicsCalibration'])
    except self.bridge.Unavailable:
      self.counts['unavailable_calibration'] += 1
      return False
    begin = time.perf_counter_ns()
    self(frame.pixels, frame.camera, self.source, self.target, calibration)
    self.counts['arrival_prepare_ns'] += time.perf_counter_ns()-begin
    self.counts['arrival_frames_prepared'] += 1
    return True

  def state(self):
    with self.lock:
      return dict(counts=dict(self.counts), bytes=self.bytes, peak_bytes=self.peak_bytes,
        entries=len(self.cache), max_entries=self.max_entries, max_bytes=self.max_bytes,
        retained_raw_frames=0)

  def close(self):
    with self.lock:
      self.cache.clear()
      self.bytes = 0
