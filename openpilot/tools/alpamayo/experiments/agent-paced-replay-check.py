"""CPU scheduling checks with real timestamp/state traces and explicit GPU-free doubles."""
import argparse
from collections import OrderedDict
import copy
from dataclasses import replace
import hashlib
import importlib.abc
import importlib.util
import json
import os
from pathlib import Path
import socket
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

os.environ['HF_HUB_OFFLINE'] = os.environ['TRANSFORMERS_OFFLINE'] = '1'
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import numpy as np


def local(name):
  spec = importlib.util.spec_from_file_location(name.replace('-', '_'), ROOT/(name+'.py'))
  value = importlib.util.module_from_spec(spec)
  sys.modules[spec.name] = value
  spec.loader.exec_module(value)
  return value


def forbidden(*a, **kw):
  raise AssertionError('No GPU, device, or network work is permitted by these CPU checks')


class BlockGPU(importlib.abc.MetaPathFinder):
  def find_spec(self, fullname, *a, **kw):
    if fullname == 'tinygrad' or fullname.startswith('tinygrad.'):
      forbidden()
    return None


class HostClock:
  def __init__(self): self.now = 1_000_000_000
  def __call__(self): return self.now
  def advance(self, ns): self.now += int(ns)


class RGBDouble:
  def __init__(self): self.reads = 0
  def __call__(self, ref):
    self.reads += 1
    value = np.zeros((3, 2, 2), np.uint8)
    value.reshape(-1)[:4] = np.frombuffer(np.uint32(ref['frame_id']).tobytes(), np.uint8)
    value[1].fill(ref['view'])
    value.flags.writeable = False
    return value
  def report(self): return dict(synthetic_identity_images=True, rgb_reads=self.reads, rgb_bytes=12*self.reads, rgb_load_ms=None)


def fake_warp(array, camera, source, target, calibration):
  result = np.zeros((3, 432, 768), np.uint8)
  result.reshape(-1)[:12] = array.reshape(-1)
  result.reshape(-1)[12:36] = np.frombuffer(np.asarray(calibration['rpyCalib'], np.float64).tobytes(), np.uint8)
  return result


class Checks(unittest.TestCase):
  @classmethod
  def setUpClass(cls):
    cls.paced = local('agent-paced-replay')
    cls.bridge = cls.paced.bridge
    cls.trace = cls.paced.Trace(ARGS.manifest, ARGS.logs)
    cls.scenarios, cls.parity = {}, []

  def scenario(self, latency_ms, *, trace=None, sync_error=False, stall_wide=False, budget=180., after_completion_ms=0):
    trace = trace or self.trace
    if stall_wide:
      trace = copy.copy(trace)
      trace.events = [e for e in trace.events if not (e.kind == 'frame' and e.value['view'] == 1 and e.release_ns > trace.start+2_000_000_000)]
    host = HostClock()
    clock = self.paced.Clock(trace.start, host)
    store = self.bridge.Store(frame_limit=64, byte_limit=384*1024**2)
    rgb = RGBDouble()
    producer = self.paced.Producer(trace, store, rgb, clock)
    prep = self.bridge.Preparer(trace.manifest['source_camera_geometry'], trace.manifest['target_camera_geometry'], warp=fake_warp, clock=clock)
    selected = []
    owner = self
    class CheckedPreparer:
      def prepare(self, selection):
        owner.assertTrue(all(r.received <= selection.selected_ns for rows in selection.states.values() for r in rows))
        owner.assertTrue(all(f.eof_ns <= f.received_ns <= f.ready_ns <= selection.selected_ns for f in selection.frames))
        arrays, detail = prep.prepare(selection)
        selected.append(selection)
        return arrays, detail
    nav = self.paced.Navigation(CheckedPreparer(), trace)
    runner = self.bridge.Bridge(store, nav, hz=5., clock=clock, require_small_model=False)
    nav.runner = runner
    vision = SimpleNamespace(hits=0, misses=0, cache=OrderedDict())
    class Consumer:
      def __init__(self):
        self.inputs = SimpleNamespace(uploaded_pixel_bytes=None)
        self.calls, self.synchronized, self.failed = 0, 0, False
      def predict(self, arrays, metadata):
        self.calls += 1
        owner.assertEqual(arrays['image_frames'].shape, (10, 1, 3, 432, 768))
        owner.assertEqual(arrays['nav_text'], [trace.nav(metadata['query_ns'])])
        owner.assertEqual(metadata['navigation_source'], trace.manifest['navigation_source'])
        owner.assertEqual(metadata['measurement_label'], self_label)
        before = hashlib.sha256(arrays['image_frames'].tobytes()).hexdigest()
        keys = [hashlib.sha256(a.tobytes()).hexdigest() for a in arrays['image_frames']]
        vision.misses = sum(key not in vision.cache for key in keys)
        vision.hits = 10-vision.misses
        for key in keys:
          vision.cache[key] = None
          vision.cache.move_to_end(key)
        while len(vision.cache) > 144:
          vision.cache.popitem(last=False)
        host.advance(round(latency_ms*1e6))
        # Arrival processing continues while the synchronous consumer owns inputs.
        producer.pump()
        owner.assertEqual(hashlib.sha256(arrays['image_frames'].tobytes()).hexdigest(), before)
        return dict(query_ns=metadata['query_ns'], prediction={'xyz': np.zeros((1, 64, 3)).tolist(), 'cpu_double': True})
      def synchronize(self):
        if sync_error:
          self.failed = True
          raise OSError('independent failed completion')
        self.synchronized += 1
    self_label = self.paced.LABEL
    consumer = Consumer()
    components = SimpleNamespace(vision_cache=vision, decoder=SimpleNamespace())
    producer.prime_history()
    self.assertLessEqual(producer.prime['latest_release_ns'], trace.start)
    clock.start()
    emitted = []
    def idle():
      now = clock()
      future = [trace.end]
      if runner.deadline > now: future.append(runner.deadline)
      if producer.index < len(trace.events) and trace.events[producer.index].release_ns > now:
        future.append(trace.events[producer.index].release_ns)
      host.advance(max(1, min(future)-now))
    summary = self.paced.observe(trace, producer, runner, consumer, components, clock=clock, budget_seconds=budget,
      emit=emitted.append, idle=idle, before_tick=producer.pump,
      collect=(lambda completed: host.advance(round(after_completion_ms*1e6))) if after_completion_ms else None)
    rows = [r for r in emitted if r['kind'] == 'model_completion']
    return SimpleNamespace(summary=summary, rows=rows, selections=selected, store=store, prep=prep,
      producer=producer, host=host, clock=clock, consumer=consumer, vision=vision, trace=trace)

  def check_progression(self, result):
    rows, trace = result.rows, result.trace
    self.assertGreater(len(rows), 0)
    queries = [r['input']['query_ns'] for r in rows]
    self.assertTrue(all(b > a for a, b in zip(queries, queries[1:])))
    self.assertTrue(all(trace.start <= q < trace.end for q in queries))
    self.assertEqual(result.consumer.calls, result.consumer.synchronized)
    self.assertEqual(len(rows), result.summary['completed_queries'])
    self.assertEqual(result.summary['admitted_road_queries'], len(rows)+result.summary['unqueued_obsolete_road_queries']+result.summary['unselected_arrived_tail'])
    self.assertEqual(result.summary['native_road_queries_in_window'], result.summary['admitted_road_queries']+result.summary['unadmitted_road_queries'])
    for row in rows:
      meta = row['input']
      available = [r for r in trace.frames if r['release_ns'] <= meta['selected_ns']]
      latest = max(r['capture_ns'] for r in available if r['view'] == 6)
      self.assertEqual(meta['query_ns'], latest)
      for ref, offset in zip(meta['frames'], self.bridge.OFFSETS_NS*2, strict=True):
        floor = max((r for r in available if r['view'] == ref['view'] and r['capture_ns'] <= meta['query_ns']+offset),
                    key=lambda r: r['capture_ns'])
        self.assertEqual(ref['frame_id'], floor['frame_id'])
        self.assertEqual(ref['capture_ns'], floor['capture_ns'])
        self.assertEqual(ref['eof_ns'], floor['eof_ns'])
        self.assertLessEqual(floor['published_ns'], meta['selected_ns'])
        self.assertLessEqual(ref['capture_ns'], ref['requested_ns'])
      self.assertEqual(row['completed_ns']-meta['selected_ns'], round(row['timing']['service_including_history_ms']*1e6))
      self.assertEqual(row['timing']['query_to_output_ms'], (row['completed_ns']-meta['query_ns'])/1e6)
      self.assertEqual(row['paced_consumer']['vision_hits']+row['paced_consumer']['vision_misses'], 10)
    expected_skip = sum(b['input']['frames'][-1]['frame_id']-a['input']['frames'][-1]['frame_id']-1 for a, b in zip(rows, rows[1:]))
    self.assertEqual(result.summary['bridge_counts']['road_frame_ids_skipped_between_queries'], expected_skip)
    self.assertLessEqual(result.store.peak_bytes, result.store.byte_limit)
    self.assertTrue(all(len(v) <= 64 for v in result.store.frames.values()))
    self.assertTrue(all(len(v) <= (16 if k == 'carParams' else 512) for k, v in result.store.states.items()))
    self.assertLessEqual(result.prep.bytes, result.prep.cache_bytes)
    self.assertLessEqual(len(result.vision.cache), 144)

  def test_elapsed_service_selects_newest_real_20hz_queries(self):
    for latency in (80, 229, 400):
      result = self.scenario(latency)
      self.check_progression(result)
      self.__class__.scenarios[latency] = result
    fast, medium, slow = (self.scenarios[n].summary for n in (80, 229, 400))
    self.assertGreater(fast['completed_queries'], medium['completed_queries'])
    self.assertGreater(medium['completed_queries'], slow['completed_queries'])
    self.assertGreater(slow['unqueued_obsolete_road_queries'], medium['unqueued_obsolete_road_queries'])
    self.assertGreater(medium['unqueued_obsolete_road_queries'], fast['unqueued_obsolete_road_queries'])

  def test_clock_excludes_loading_and_never_pauses_during_execution(self):
    host = HostClock()
    clock = self.paced.Clock(self.trace.start, host)
    host.advance(99_000_000_000)
    self.assertEqual(clock(), self.trace.start)
    clock.start()
    host.advance(229_000_000)
    self.assertEqual(clock(), self.trace.start+229_000_000)
    with self.assertRaisesRegex(RuntimeError, 'cannot restart'):
      clock.start()

  def test_elapsed_nonmodel_work_also_changes_next_query(self):
    result = self.scenario(80, after_completion_ms=150)
    self.check_progression(result)
    self.assertLess(result.summary['completed_queries'], 40)
    self.assertGreater(result.summary['unqueued_obsolete_road_queries'], 117)

  def test_full_native_rgb_ring_retains_floor_history_with_explicit_bound(self):
    pixels = np.zeros((3, 760, 1344), np.uint8)
    pixels.flags.writeable = False
    for limit, full_slots in ((256*1024**2, False), (384*1024**2, True)):
      store = self.bridge.Store(frame_limit=64, byte_limit=limit)
      for ref in self.trace.frames:
        store.add_frame(self.bridge.Frame(ref['view'], ref['frame_id'], ref['capture_ns'], ref['eof_ns'],
          ref['published_ns'], ref['release_ns'], pixels))
      self.assertLessEqual(store.peak_bytes, limit)
      self.assertTrue(all(len(r) <= 64 for r in store.frames.values()))
      self.assertEqual(len(store.select(self.trace.end, 0, {}, 250_000_000, False).frames), 10)
      if full_slots:
        self.assertTrue(all(len(r) == 64 for r in store.frames.values()))
        self.assertEqual(store.bytes, 128*pixels.nbytes)
      else:
        self.assertLess(sum(map(len, store.frames.values())), 128)
        self.assertGreater(store.counts['byte_evictions'], 0)

  def test_eof_and_original_log_order_bound_admission(self):
    events = self.trace.events
    self.assertTrue(all(a.ordinal < b.ordinal for a, b in zip(events, events[1:])))
    self.assertTrue(all(a.release_ns <= b.release_ns for a, b in zip(events, events[1:])))
    self.assertTrue(all(e.published_ns <= e.release_ns for e in events))
    self.assertTrue(all(e.value['capture_ns'] <= e.value['eof_ns'] <= e.published_ns for e in events if e.kind == 'frame'))
    host = HostClock()
    clock = self.paced.Clock(self.trace.start, host)
    store = self.bridge.Store()
    producer = self.paced.Producer(self.trace, store, RGBDouble(), clock)
    producer.prime_history()
    pending = events[producer.index]
    self.assertGreater(pending.release_ns, clock())
    before = store.bounds()
    producer.pump()
    self.assertEqual(store.bounds(), before)
    self.assertTrue(all(r.received <= clock() for rows in store.states.values() for r in rows))

  def test_background_producer_uses_elapsed_host_clock_and_stops(self):
    trace = copy.copy(self.trace)
    trace.end = trace.start+200_000_000
    clock = self.paced.Clock(trace.start)
    producer = self.paced.Producer(trace, self.bridge.Store(), RGBDouble(), clock)
    producer.prime_history()
    primed = len(producer.arrivals)
    clock.start()
    producer.start()
    producer.thread.join(timeout=1)
    self.assertFalse(producer.thread.is_alive())
    producer.close()
    arrived = producer.arrivals[primed:]
    self.assertGreater(len(arrived), 0)
    self.assertTrue(all(a['published_ns'] <= a['release_ns'] <= a['received_ns'] <= a['ready_ns'] for a in arrived))
    self.assertTrue(all(a['release_ns'] < trace.end for a in arrived))
    self.assertTrue(all(a['ordinal'] < b['ordinal'] for a, b in zip(arrived, arrived[1:])))

  def test_future_state_values_cannot_change_current_arrays(self):
    result = self.scenarios.get(229) or self.scenario(229)
    selected = next(s for s in result.selections if any(r.stamp > s.query_ns for r in s.states['carState']))
    states = dict(selected.states)
    changed = 0
    rows = []
    for row in states['carState']:
      if row.stamp > selected.query_ns and row.valid:
        row = replace(row, value=dict(row.value, vEgo=999., steeringAngleDeg=85.))
        changed += 1
      rows.append(row)
    states['carState'] = tuple(rows)
    prep = self.bridge.Preparer(self.trace.manifest['source_camera_geometry'], self.trace.manifest['target_camera_geometry'], warp=fake_warp)
    before, _ = prep.prepare(selected)
    after, _ = prep.prepare(replace(selected, states=states))
    self.assertGreater(changed, 0)
    for key in before:
      self.assertEqual(before[key].tobytes(), after[key].tobytes(), key)

  def test_hard_budget_uses_kernel_termination_and_restores_alarm(self):
    import signal
    with patch.object(signal, 'getitimer', return_value=(0., 0.)), patch.object(signal, 'signal', return_value='previous') as handler, \
         patch.object(signal, 'setitimer') as alarm:
      with self.paced.hard_budget(180):
        self.assertEqual(alarm.call_args.args, (signal.ITIMER_REAL, 180))
        self.assertEqual(handler.call_args.args, (signal.SIGALRM, signal.SIG_DFL))
      self.assertEqual(alarm.call_args.args, (signal.ITIMER_REAL, 0))
      self.assertEqual(handler.call_args.args, (signal.SIGALRM, 'previous'))
    with self.assertRaisesRegex(ValueError, '180 seconds'):
      with self.paced.hard_budget(181):
        self.fail('Unbounded inference was admitted')

  def test_stalled_camera_does_not_republish_a_plan(self):
    result = self.scenario(229, stall_wide=True)
    queries = [r['input']['query_ns'] for r in result.rows]
    self.assertEqual(len(queries), len(set(queries)))
    self.assertTrue(all(r['input']['selected_ns'] < self.trace.start+2_300_000_000 for r in result.rows))
    self.assertGreater(result.summary['bridge_counts'].get('stale_current_camera', 0)+
                       result.summary['bridge_counts'].get('missing_or_delayed_front_history', 0), 0)
    self.assertGreater(result.summary['unselected_arrived_tail'], 0)

  def test_failed_sync_and_budget_never_count_completion(self):
    with self.assertRaisesRegex(OSError, 'failed completion'):
      self.scenario(80, sync_error=True)
    with self.assertRaisesRegex(TimeoutError, 'hard completion budget'):
      self.scenario(400, budget=.2)

  def test_no_catchup_after_slow_inference_exceeds_source_end(self):
    result = self.scenario(9000)
    self.assertEqual(result.consumer.calls, 1)
    self.assertEqual(result.summary['completed_queries'], 1)
    self.assertGreater(result.summary['final_completion_ns'], self.trace.end)
    self.assertGreater(result.summary['unselected_arrived_tail'], 0)
    self.assertEqual(result.summary['queued_prediction_snapshots'], 0)

  def test_prepared_arrays_match_canonical_helpers_at_selected_queries(self):
    from calibration_trials import asof, calibrated_image, vehicle_history
    from camera_profiles import replay_module
    for latency in (80, 229, 400):
      result = self.scenarios.get(latency) or self.scenario(latency)
      checked = 0
      for selection in result.selections:
        refs = {(r['view'], r['frame_id']): r for r in self.trace.frames}
        paths = [ARGS.raw_frames/str(f.camera)/f'{refs[f.camera, f.frame_id]["video_index"]}.npy' for f in selection.frames]
        if not all(p.is_file() for p in paths):
          continue
        frames = []
        for frame, path in zip(selection.frames, paths, strict=True):
          raw = np.ascontiguousarray(np.load(path, allow_pickle=False))
          raw.flags.writeable = False
          frames.append(replace(frame, pixels=raw))
        native_selection = replace(selection, frames=tuple(frames))
        prep = self.bridge.Preparer(self.trace.manifest['source_camera_geometry'], self.trace.manifest['target_camera_geometry'])
        actual, details = prep.prepare(native_selection)
        q = selection.query_ns
        # Independent direct canonical-helper assembly. No bridge/preparer calls.
        states = {k: (np.asarray([r.stamp for r in rows if r.stamp <= q], np.int64),
                      [r.value if r.valid else None for r in rows if r.stamp <= q])
                  for k, rows in selection.states.items() if k in self.bridge.STATE_LIMITS}
        cal, _ = asof(states['extrinsicsCalibration'], q, 1_000_000_000)
        motion = {r.stamp: r.value for r in selection.states['deviceMotion'] if r.stamp <= q and r.valid}
        xyz, rot, future, _ = replay_module().Motion(motion).sample(q, cal['rpyCalib'])
        self.assertIsNone(future)
        xyz, rot, audit = vehicle_history(dict(query_ns=q, ego_history_xyz=xyz, ego_history_rot=rot), states)
        expected = []
        for frame in frames:
          calibration, stamp = asof(states['extrinsicsCalibration'], frame.sof_ns, 1_000_000_000)
          expected.append(calibrated_image(frame.pixels, frame.camera, self.trace.manifest['source_camera_geometry'],
                                           self.trace.manifest['target_camera_geometry'], calibration)[None])
        np.testing.assert_array_equal(actual['image_frames'], np.stack(expected))
        np.testing.assert_array_equal(actual['camera_indices'], [f.camera for f in frames])
        np.testing.assert_array_equal(actual['absolute_timestamps'], [f.sof_ns//1000 for f in frames])
        self.assertEqual(actual['ego_history_xyz'].tobytes(), xyz.tobytes())
        self.assertEqual(actual['ego_history_rot'].tobytes(), rot.tobytes())
        self.assertEqual(details['vehicle_history_audit'], audit)
        self.__class__.parity.append(dict(fake_service_ms=latency, query_ns=q, full_images=10, exact_arrays=True,
          capture_timestamps=[f.sof_ns for f in frames], maximum_source_state_stamp_ns=max(motion)))
        checked += 1
        if checked == 2:
          break
      self.assertEqual(checked, 2, f'Need two complete original raw-cache controls for {latency} ms')

  def test_missing_rgb_cache_is_explicit_and_not_replaced(self):
    with self.assertRaisesRegex(ValueError, 'Incomplete native 20 Hz'):
      self.paced.RGB(ARGS.raw_frames, self.trace.frames, verify=False)

  def test_real_gpu_gate_cannot_be_replaced_by_cpu_evidence(self):
    cli = SimpleNamespace(consumer_parity_report=Path('/virtual/cpu-only.json'))
    with patch.object(Path, 'read_text', return_value=json.dumps({'mode': 'consumer', 'gpu_executed': False})):
      with self.assertRaisesRegex(ValueError, 'real-GPU'):
        self.paced.model_run(cli, self.trace)


def main():
  global ARGS
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--manifest', type=Path, default=Path('/mnt/nvme-256/footage/f9ad9a96/exports/0000008e--321b2ddc03/alpamayo-4hz-calibrated-two-six/manifest.json'))
  parser.add_argument('--logs', type=Path, default=Path('/mnt/nvme-256/footage/f9ad9a96/realdata'))
  parser.add_argument('--raw-frames', type=Path, default=Path('/mnt/nvme-500/tmp/alpamayo-route-4hz-20261116/native-frames'))
  parser.add_argument('--output', type=Path, required=True)
  ARGS = parser.parse_args()
  sys.meta_path.insert(0, BlockGPU())
  with patch.object(socket.socket, 'connect', forbidden), patch.object(socket.socket, 'connect_ex', forbidden), \
       patch.object(socket, 'create_connection', forbidden):
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Checks))
  report = dict(cpu_only=True, tests=result.testsRun, failures=len(result.failures), errors=len(result.errors),
    real_gpu_or_model_execution=False, canonical_array_parity=Checks.parity, trace=Checks.trace.provenance,
    simulations={str(k): {name: v.summary[name] for name in ('completed_queries', 'admitted_road_queries',
      'unqueued_obsolete_road_queries', 'unselected_arrived_tail', 'source_frame_id_gaps', 'bridge_counts')}
      for k, v in Checks.scenarios.items()},
    limit='Injected 80/229/400 ms clocks, synthetic identity pixels and fake feature caches test scheduling only; no measured end-to-end rate')
  with ARGS.output.open('x') as stream:
    json.dump(report, stream, indent=2)
    stream.write('\n')
  if not result.wasSuccessful():
    raise SystemExit(1)


if __name__ == '__main__':
  main()
