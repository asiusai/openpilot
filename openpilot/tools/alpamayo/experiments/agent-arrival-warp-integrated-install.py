"""Post-admission CPU prewarp. No Producer, query selection, or model code copy."""
from contextlib import ExitStack
import importlib.util
from pathlib import Path
import threading
import time


class Integration:
  def __init__(self, common, stack):
    self.common, self.stack = common, stack
    self.cache = self.preparer = self.producer = None
    self.sampler_lock = threading.RLock()
    self.rows = []
    self.info = dict(installed=False, closed=False, producer_binding_restored=False,
      no_frame_exclusions=True, extra_prediction_queue=0, gpu_features_precomputed=0,
      nv12_qualified=False, active_clock_pauses=0, cache_cleanup_bytes=None)

  def attach_preparer(self, preparer, bridge):
    assert self.cache is None and self.preparer is None
    original = preparer.warp
    def serialized(*args, **kwargs):
      # Core's mutable-input fallback does not acquire its cache lock.
      # Serialize every selected sampler call, including that fallback.
      with self.sampler_lock:
        return original(*args, **kwargs)
    module = self.common.load(self.common.C.EXP/'agent-arrival-warp-cache.py', 'arrival_integrated_cache')
    self.cache = module.ArrivalWarp(serialized, bridge, preparer.source, preparer.target)
    self.preparer = preparer
    preparer.warp = self.cache
    self.info['installed'] = True
    def restore():
      assert preparer.warp is self.cache
      preparer.warp = original
    self.stack.callback(restore)

  def attach_producer(self, producer):
    assert self.producer is None and 'add_frame' not in producer.store.__dict__
    self.producer = producer
    original = producer.store.add_frame
    owner = threading.get_ident()
    def admitted(frame):
      # Reject foreign callers, but allow canonical main-thread prime or source thread.
      expected = owner if producer.thread is None else producer.thread.ident
      assert threading.get_ident() == expected
      accepted = original(frame)
      if not accepted:
        return False
      assert self.cache is not None and not self.info['closed']
      # Store admission has released Store.lock. Never hold it while warping.
      now = producer.clock()
      assert frame.ready_ns <= now and (producer.clock.anchor is None or now >= producer.trace.start)
      before = time.perf_counter_ns()
      prepared = self.cache.received(producer.store, frame, now)
      elapsed = time.perf_counter_ns()-before
      after = producer.clock()
      assert after >= now
      assert len(self.rows) < 480, 'Bound the compact arrival audit, never retain raw arrays'
      self.rows.append(dict(view=frame.camera, frame_id=frame.frame_id, capture_ns=frame.capture_ns,
        ready_ns=frame.ready_ns, started_ns=now, finished_ns=after,
        primed=producer.clock.anchor is None, prepared=prepared, wall_ns=elapsed))
      return True
    producer.store.add_frame = admitted
    def restore():
      assert producer.store.add_frame is admitted
      del producer.store.add_frame
      assert producer.store.add_frame == original
      self.info['producer_binding_restored'] = True
    self.stack.callback(restore)

  def close(self):
    if self.info['closed']:
      return
    # Parent hook exits only after the canonical observe/finally has stopped Producer.
    # Initialization failures can leave an unstarted Producer; close that too.
    if self.producer is not None:
      self.producer.close()
      assert self.producer.thread is None or not self.producer.thread.is_alive()
    if self.cache is not None:
      self.info['cache_before_cleanup'] = self.cache.state()
      self.cache.close()
      self.info['cache_cleanup_bytes'] = self.cache.state()['bytes']
      assert self.info['cache_cleanup_bytes'] == 0
    self.info['closed'] = True

  def report(self):
    result = dict(self.info, arrivals=list(self.rows))
    for name, primed in (('prime', True), ('active', False)):
      rows = [row for row in self.rows if row['primed'] is primed]
      result[name] = dict(frames=len(rows), prepared=sum(r['prepared'] for r in rows),
        wall_ms=sum(r['wall_ns'] for r in rows)/1e6, max_wall_ms=max((r['wall_ns']/1e6 for r in rows), default=0))
    result['first_arrival_wall_ms'] = self.rows[0]['wall_ns']/1e6 if self.rows else None
    result['first_prepared_wall_ms'] = next((r['wall_ns']/1e6 for r in self.rows if r['prepared']), None)
    return result


def factory(common):
  C = common.C
  original = C.load(C.EXP/'agent-live-cpu-integrated-install.py', 'arrival_integrated_selected_install')
  Base = original.factory(C)
  class Hooks(Base):
    def __init__(self, mode):
      assert mode == 'paced'
      super().__init__(mode)
      self.arrival = Integration(common, self.stack)

    def bridge(self, module):
      super().bridge(module)
      original_init = module.Preparer.__init__
      def initialize(preparer, *args, **kwargs):
        begin = time.perf_counter_ns()
        original_init(preparer, *args, **kwargs)
        self.arrival.info['preparer_and_sampler_initialization_ms'] = (time.perf_counter_ns()-begin)/1e6
        self.arrival.attach_preparer(preparer, module)
      self.patch(module.Preparer, '__init__', initialize)

    def __enter__(self):
      value = super().__enter__()
      previous = importlib.util.spec_from_file_location
      hooks = self
      def spec(name, location, *args, **kwargs):
        result = previous(name, location, *args, **kwargs)
        if result is None or location is None or Path(location).resolve() != C.N.LIVE/'experiments/agent-paced-replay.py':
          return result
        loader = result.loader
        class Loader:
          def create_module(self, spec):
            return loader.create_module(spec)
          def exec_module(self, module):
            loader.exec_module(module)
            original_init = module.Producer.__init__
            def initialize(producer, *a, **kw):
              original_init(producer, *a, **kw)
              hooks.arrival.attach_producer(producer)
            hooks.patch(module.Producer, '__init__', initialize)
        result.loader = Loader()
        return result
      self.patch(importlib.util, 'spec_from_file_location', spec)
      return value

    def __exit__(self, *error):
      try:
        if self.sampler is not None:
          self.arrival.info['sampler_before_cleanup'] = self.sampler.state()
        self.arrival.close()
      finally:
        result = super().__exit__(*error)
      return result

    def coverage(self):
      result = super().coverage()
      assert self.arrival.cache is not None and self.arrival.producer is not None
      assert self.arrival.rows and self.arrival.preparer.warp is self.arrival.cache
      assert self.arrival.cache.state()['counts']['hits'] > 0
      return result
  return Hooks
