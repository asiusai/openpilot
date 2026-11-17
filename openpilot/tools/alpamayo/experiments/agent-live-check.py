"""Independent CPU event/clock checks and optional recorded-input parity.

No hardware, message sockets, model imports, compiler, or GPU execution.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace
import unittest

import numpy as np


def local(name):
  spec = importlib.util.spec_from_file_location(name.replace('-', '_'), Path(__file__).with_name(name+'.py'))
  module = importlib.util.module_from_spec(spec)
  sys.modules[spec.name] = module
  spec.loader.exec_module(module)
  return module


live = local('agent-live-bridge')
capture = local('agent-live-capture')
MS = 1_000_000
BASE = 10_000*MS
CP = dict(rotationalInertiaPerMass=2., wheelbase=2.7, centerToFront=1.2,
          tireStiffnessFrontPerMass=60., tireStiffnessRearPerMass=70., steerRatio=15.)
CAL = dict(rpyCalib=[0., 0., 0.], wideFromDeviceEuler=[0., 0., 0.])
LP = dict(stiffnessFactor=1., steerRatio=15., angleOffsetDeg=0., roll=0.)


def frame(camera, tick, step=100*MS, value=0):
  stamp = BASE+tick*step
  pixels = np.full((3, 16, 24), value, dtype=np.uint8)
  pixels.flags.writeable = False
  return live.Frame(camera, tick, stamp, stamp+5*MS, stamp+10*MS, stamp+11*MS, pixels)


def populated():
  store = live.Store()
  store.add_state('carParams', BASE-MS, BASE, CP)
  for tick in range(61):
    stamp = BASE+tick*50*MS
    store.add_state('deviceMotion', stamp, stamp+MS, [[0., 0., 0.], [10., 0., 0.]])
    store.add_state('carState', stamp, stamp+MS, dict(vEgo=10., steeringAngleDeg=0., aEgo=0.))
    store.add_state('vehicleParameters', stamp, stamp+MS, LP)
    if tick % 5 == 0:
      store.add_state('extrinsicsCalibration', stamp, stamp+MS, CAL)
    if tick % 2 == 0:
      for c in live.CAMERAS:
        store.add_frame(frame(c, tick//2))
  return store


class Clock:
  def __init__(self, now=BASE+3015*MS):
    self.now = now
  def __call__(self):
    return self.now
  def advance(self, ns):
    self.now += ns


class StubPrepare:
  def __init__(self, clock, work_ms=6):
    self.clock, self.work_ms = clock, work_ms
  def prepare(self, selected):
    self.clock.advance(self.work_ms*MS)
    return {}, {'frames': [f.reference() for f in selected.frames]}


class Consumer:
  def __init__(self, clock, ms=229):
    self.clock, self.ms = clock, ms
    self.synchronizations = 0
  def predict(self, arrays, meta):
    self.clock.advance(self.ms*MS)
    return dict(query_ns=meta['query_ns'], prediction=dict(xyz=np.zeros((64, 3)).tolist(), reasoning='synthetic timing check'))
  def synchronize(self):
    self.synchronizations += 1
    self.clock.advance(MS)


def geometry():
  return {'cameras': {
    '1': dict(prepared_size=[768, 432], content_box=[0, 0, 768, 432], focal_pixels=[240., 240.],
              principal_point=[383.5, 215.5], projection='equidistant'),
    '6': dict(prepared_size=[768, 432], content_box=[0, 0, 768, 432], focal_pixels=[650., 650.],
              principal_point=[383.5, 215.5], projection='pinhole')}}


class Checks(unittest.TestCase):
  def test_actual_capture_floor_and_full_five(self):
    store = populated()
    query = BASE+3000*MS
    selected = store.select(query+15*MS, 0, {}, 250*MS, False)
    self.assertEqual(selected.query_ns, query)
    self.assertEqual([f.capture_ns for f in selected.frames], [query+o for o in live.OFFSETS_NS]*2)
    for f, offset in zip(selected.frames, live.OFFSETS_NS*2):
      self.assertLessEqual(f.capture_ns, query+offset)

  def test_missing_delayed_or_repeated_history_rejected(self):
    for remove in ([10, 9], list(range(27, 31)), list(range(20, 31))):
      store = populated()
      store.frames[1] = type(store.frames[1])(f for f in store.frames[1] if f.frame_id not in remove)
      with self.assertRaises(live.Unavailable):
        store.select(BASE+3015*MS, 0, {}, 250*MS, False)
    store = populated()
    self.assertFalse(store.add_frame(frame(1, 25)))
    self.assertEqual(store.counts['late_or_duplicate_frames'], 1)

  def test_one_camera_stall_and_duplicate_query(self):
    store, clock = populated(), Clock()
    runner = live.Bridge(store, StubPrepare(clock), clock=clock, hz=10)
    first = runner.run_once()
    self.assertIsNotNone(first)
    clock.advance(100*MS)
    self.assertIsNone(runner.run_once())
    store.add_frame(frame(6, 31))
    self.assertIsNone(runner.run_once())
    self.assertEqual(runner.counts['stale_current_camera'], 1)

  def test_busy_does_not_queue_or_repeat_plan(self):
    store, clock = populated(), Clock()
    runner = live.Bridge(store, StubPrepare(clock), clock=clock)
    consumer = Consumer(clock)
    original = consumer.predict
    def nested(arrays, meta):
      self.assertIsNone(runner.run_once(consumer))
      return original(arrays, meta)
    consumer.predict = nested
    row = runner.run_once(consumer)
    self.assertEqual(row['timing']['service_including_history_ms'], 236)
    self.assertEqual(row['timing']['query_to_output_ms'], 251)
    self.assertEqual(row['timing']['camera'][0]['sof_to_received_ms'], 10)
    self.assertEqual(row['timing']['camera'][0]['eof_to_received_ms'], 5)
    self.assertEqual(consumer.synchronizations, 1)
    self.assertEqual(runner.summary()['queued_queries'], 0)
    self.assertEqual(runner.counts['model_completions'], 1)

  def test_slow_inference_latest_drop_and_real_completion_rate(self):
    store, clock = populated(), Clock()
    runner = live.Bridge(store, StubPrepare(clock), clock=clock, hz=10)
    consumer = Consumer(clock)
    last_tick, queries = 30, []
    for _ in range(80):
      # Independent camera event sequence: frames arrive while inference owns
      # the worker. Only the latest completed acquisition is dispatched next.
      latest_tick = (clock.now-BASE-11*MS)//(100*MS)
      for tick in range(last_tick+1, latest_tick+1):
        for camera in live.CAMERAS:
          store.add_frame(frame(camera, tick))
      last_tick = latest_tick
      row = runner.run_once(consumer)
      self.assertIsNotNone(row)
      queries.append(row['input']['query_ns'])
      self.assertLessEqual(row['timing']['camera'][1]['sof_age_at_dispatch_ms'], 111)
    self.assertEqual(len(set(queries)), 80)
    self.assertTrue(all(b-a >= 200*MS for a, b in zip(queries, queries[1:])))
    self.assertAlmostEqual(runner.summary()['model_completion_hz'], 1000/236)
    self.assertLess(runner.summary()['model_completion_hz'], 5)
    self.assertLessEqual(store.peak_bytes, store.byte_limit)

  def test_cpu_completion_is_never_a_prediction(self):
    clock = Clock()
    runner = live.Bridge(populated(), StubPrepare(clock), clock=clock)
    row = runner.run_once()
    self.assertEqual(row['kind'], 'cpu_input_only')
    self.assertIsNone(row['prediction'])
    self.assertEqual(runner.summary()['model_completion_hz'], 0)

  def test_stalled_tail_counts_against_throughput(self):
    clock = Clock()
    runner = live.Bridge(populated(), StubPrepare(clock), clock=clock)
    runner.run_once(Consumer(clock))
    clock.advance(10_000*MS)
    self.assertIsNone(runner.run_once(Consumer(clock)))
    self.assertLess(runner.summary()['model_completion_hz'], .1)

  def test_wrong_plan_identity_rejected(self):
    clock = Clock()
    runner = live.Bridge(populated(), StubPrepare(clock), clock=clock)
    consumer = Consumer(clock)
    consumer.predict = lambda arrays, meta: dict(query_ns=0, prediction=dict(xyz=np.zeros((64, 3))))
    with self.assertRaisesRegex(ValueError, 'stale'):
      runner.run_once(consumer)
    self.assertEqual(runner.counts['model_completions'], 0)

  def test_bounded_memory_under_producer_overrun(self):
    store = live.Store(frame_limit=48, byte_limit=12*1152, state_limit=64)
    for tick in range(2500):
      for camera in live.CAMERAS:
        store.add_frame(frame(camera, tick))
      stamp = BASE+tick*100*MS
      store.add_state('carState', stamp, stamp+MS, dict(vEgo=10.))
      store.add_state('carParams', stamp, stamp+MS, CP)
    state = store.bounds()
    self.assertLessEqual(state['frame_bytes'], 12*1152)
    self.assertLessEqual(sum(state['frames'].values()), 12)
    self.assertLessEqual(state['states']['carState'], 64)
    self.assertLessEqual(state['states']['carParams'], 16)
    self.assertGreater(store.counts['byte_evictions'], 0)
    with self.assertRaisesRegex(ValueError, 'record ceiling'):
      store.add_state('modelV2', BASE, BASE+MS, {'oversize': 'x'*(65*1024)})

  def test_future_receive_and_delayed_state_are_causal(self):
    store = populated()
    q = BASE+3000*MS
    store.add_state('carState', q-25*MS, q+50*MS, dict(vEgo=999., steeringAngleDeg=90.))
    selected = store.select(q+15*MS, 0, {}, 250*MS, False)
    self.assertFalse(any(r.value and r.value.get('vEgo') == 999 for r in selected.states['carState']))
    selected = store.select(q+60*MS, 0, {}, 250*MS, False)
    self.assertTrue(any(r.value and r.value.get('vEgo') == 999 for r in selected.states['carState']))
    self.assertEqual(live.required(live.records(selected.states['carState'], q), q-30*MS, 150*MS)[0]['vEgo'], 10)

  def test_real_helpers_straight_history_and_full_images(self):
    store = populated()
    selection = store.select(BASE+3015*MS, 0, {}, 250*MS, False)
    prep = live.Preparer(geometry(), {})
    arrays, detail = prep.prepare(selection)
    self.assertEqual(arrays['image_frames'].shape, (10, 1, 3, 432, 768))
    np.testing.assert_array_equal(arrays['image_frames'], 0)
    np.testing.assert_allclose(arrays['ego_history_xyz'][0, :, 0], np.arange(-19, 1), atol=1e-5)
    np.testing.assert_array_equal(arrays['ego_history_xyz'][0, :, 1:], 0)
    np.testing.assert_allclose(arrays['ego_history_rot'][0], np.broadcast_to(np.eye(3), (20, 3, 3)), atol=1e-7)
    self.assertEqual(detail['vehicle_history_audit']['max_car_state_age_ms'], 0)
    repeated, second = prep.prepare(selection)
    np.testing.assert_array_equal(arrays['image_frames'], repeated['image_frames'])
    self.assertEqual(second['image_cache_hits'], 10)
    self.assertLessEqual(prep.bytes, prep.cache_bytes)

  def test_age_limits_invalid_updates_and_future_exclusion(self):
    q = BASE+3000*MS
    for kind, boundary in (('carState', 150*MS), ('vehicleParameters', 1000*MS),
                           ('extrinsicsCalibration', 1000*MS), ('carParams', 120000*MS)):
      record = (np.array([q-boundary]), [{'ok': True}])
      self.assertEqual(live.required(record, q, boundary)[1], q-boundary)
      with self.assertRaises(live.Unavailable):
        live.required(record, q+1, boundary)
    prep = live.Preparer(geometry(), {})
    for kind in ('carState', 'vehicleParameters', 'extrinsicsCalibration', 'deviceMotion'):
      store = populated()
      store.add_state(kind, q-MS, q+2*MS, None, valid=False)
      if kind != 'deviceMotion':
        # Ensure the invalid update is newest at the query.
        store.states[kind] = [r for r in store.states[kind] if r.stamp != q]
      with self.assertRaises(live.Unavailable):
        prep.prepare(store.select(q+15*MS, 0, {}, 250*MS, False))
    store = populated()
    store.add_state('carState', q+MS, q+2*MS, dict(vEgo=9999., steeringAngleDeg=90.))
    arrays, _ = prep.prepare(store.select(q+15*MS, 0, {}, 250*MS, False))
    np.testing.assert_allclose(arrays['ego_history_xyz'][0, :, 0], np.arange(-19, 1), atol=1e-5)

  def test_calibration_selected_at_each_capture_not_query(self):
    store = populated()
    change = BASE+2250*MS
    for i, record in enumerate(store.states['extrinsicsCalibration']):
      if record.stamp >= change:
        store.states['extrinsicsCalibration'][i] = live.State(record.stamp, record.received,
          dict(rpyCalib=[0., 0., .02], wideFromDeviceEuler=[0., 0., 0.]), True)
    observed = []
    def inspect(array, camera, source, target, cal):
      observed.append(cal['rpyCalib'][2])
      return np.zeros((3, 432, 768), dtype=np.uint8)
    prep = live.Preparer(geometry(), {}, warp=inspect)
    prep.prepare(store.select(BASE+3015*MS, 0, {}, 250*MS, False))
    self.assertEqual(observed, [0., 0., .02, .02, .02]*2)

  def test_image_cache_bounds_and_changed_source_identity(self):
    store = populated()
    prep = live.Preparer(geometry(), {}, cache_limit=3, cache_bytes=3*1024**2,
                        warp=lambda *args: np.zeros((3, 432, 768), dtype=np.uint8))
    selection = store.select(BASE+3015*MS, 0, {}, 250*MS, False)
    prep.prepare(selection)
    self.assertLessEqual(len(prep.cache), 3)
    self.assertLessEqual(prep.bytes, 3*1024**2)
    prep = live.Preparer(geometry(), {}, warp=lambda array, *args: np.full((3, 432, 768), array[0, 0, 0], np.uint8))
    prep.prepare(selection)
    changed = list(selection.frames)
    changed[-1] = frame(6, 30, value=17)
    changed = live.Selection(selection.query_ns, selection.selected_ns, tuple(changed), selection.states, 0, None)
    arrays, detail = prep.prepare(changed)
    self.assertEqual(detail['image_cache_hits'], 9)
    np.testing.assert_array_equal(arrays['image_frames'][-1], 17)

  def test_real_history_left_turn_and_missing_motion(self):
    store = populated()
    for record in store.states['carState']:
      record.value['steeringAngleDeg'] = 10.
    prep = live.Preparer(geometry(), {}, warp=lambda *args: np.zeros((3, 432, 768), dtype=np.uint8))
    arrays, meta = prep.prepare(store.select(BASE+3015*MS, 0, {}, 250*MS, False))
    self.assertGreater(meta['vehicle_history_audit']['left_yaw_rate_rad_s'], 0)
    self.assertGreater(arrays['ego_history_xyz'][0, 0, 1], 0)
    store.states['deviceMotion'] = [r for r in store.states['deviceMotion'] if r.stamp < BASE+2800*MS]
    with self.assertRaises(live.Unavailable):
      prep.prepare(store.select(BASE+3015*MS, 0, {}, 250*MS, False))

  def test_native_stride_copy_and_buffer_lifetime(self):
    h, w, stride = 760, 1344, 1408
    uv_offset = h*stride+128
    raw = np.full(uv_offset+h//2*stride, 239, np.uint8)
    y = np.ndarray((h, w), np.uint8, buffer=raw, strides=(stride, 1))
    uv = np.ndarray((h//2, w), np.uint8, buffer=raw, offset=uv_offset, strides=(stride, 1))
    y[:] = 80
    uv[:] = 128
    client = SimpleNamespace(frame_id=8, timestamp_sof=BASE, timestamp_eof=BASE+5*MS)
    buf = SimpleNamespace(width=w, height=h, stride=stride, uv_offset=uv_offset, data=raw)
    clock = Clock(BASE+10*MS)
    copied = capture.copy_vision_buffer(1, client, buf, clock)
    raw[:] = 0
    np.testing.assert_array_equal(copied.pixels[:h], 80)
    np.testing.assert_array_equal(copied.pixels[h:], 128)
    self.assertFalse(copied.pixels.flags.writeable)
    self.assertEqual((copied.sof_ns, copied.eof_ns), (BASE, BASE+5*MS))
    rgb = live.Preparer(geometry(), {}).rgb(copied)
    self.assertEqual(rgb.shape, (3, h, w))
    self.assertEqual(int(rgb.max()), int(rgb.min()))

  def test_storage_reserves_both_thresholds(self):
    gib = 1024**3
    self.assertFalse(live.storage_ok(100*gib, 9*gib))
    self.assertFalse(live.storage_ok(32*gib, 4*gib))
    self.assertTrue(live.storage_ok(100*gib, 10*gib))
    self.assertFalse(live.storage_ok(100*gib, 10*gib, 1))
    self.assertFalse(live.storage_ok(32*gib, 5*gib, 1))
    self.assertTrue(live.storage_ok(32*gib, 6*gib, gib))


def recorded_parity(manifest_path, raw_frames, logs, calibrations):
  """Four route windows, actual raw RGB, real states, original source timestamps.

  Event.logMonoTime stands in for receipt; rlogs do not record this subscriber's
  receipt time. This is CPU input parity, never a capture-latency measurement.
  """
  from openpilot.cereal import log
  import zstandard
  manifest = json.loads(manifest_path.read_text())
  rows = [manifest['schedule'][i] for i in (0, 80, 160, 240)]
  expected_hashes = manifest['rlog_sha256']
  states = {k: [] for k in live.STATE_LIMITS}
  for relative, expected in expected_hashes.items():
    path = logs/relative
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
      raise ValueError('Fixture rlog checksum changed')
    with path.open('rb') as stream, zstandard.ZstdDecompressor().stream_reader(stream) as decoder:
      raw = decoder.read()
    for event in log.Event.read_multiple_bytes(raw):
      kind = event.which()
      if kind not in states:
        continue
      msg = getattr(event, kind)
      stamp = int(msg.timestamp) if kind == 'deviceMotion' else int(event.logMonoTime)
      horizon = 121_000_000_000 if kind == 'carParams' else 3_300_000_000
      if stamp <= 0 or not any(row['query_ns']-horizon <= stamp <= row['query_ns'] for row in rows):
        continue
      value, valid = capture.state_value(kind, msg)
      states[kind].append((stamp, int(event.logMonoTime), value, bool(event.valid and valid)))
  cal_rows = json.loads(calibrations.read_text())
  controls = []
  for row in rows:
    store = live.Store()
    q = row['query_ns']
    # Old state is loaded first; invalid states retain their source ordering.
    for kind, entries in states.items():
      horizon = 121_000_000_000 if kind == 'carParams' else 3_300_000_000
      for stamp, received, value, valid in sorted(entries):
        if q-horizon <= stamp <= q:
          store.add_state(kind, stamp, max(stamp, received), value, valid)
    for ref in row['frames']:
      array = np.ascontiguousarray(np.load(raw_frames/str(ref['view'])/f'{ref["video_index"]}.npy', allow_pickle=False))
      array.flags.writeable = False
      stamp = ref['capture_ns']
      # EOF/receive are synthetic here. Actual SOF is retained byte-for-byte.
      store.add_frame(live.Frame(ref['view'], ref['frame_id'], stamp, stamp+5*MS, stamp+10*MS, stamp+11*MS, array))
    selected = store.select(q+100*MS, 0, {}, 250*MS, False)
    prep = live.Preparer(manifest['source_camera_geometry'], manifest['target_camera_geometry'])
    arrays, detail = prep.prepare(selected)
    sample = Path(row['sample'])
    errors = {}
    for key in ('image_frames', 'camera_indices', 'absolute_timestamps', 'ego_history_xyz', 'ego_history_rot'):
      expected = np.load(sample/f'{key}.npy', allow_pickle=False)
      actual = arrays[key]
      if key in ('ego_history_xyz', 'ego_history_rot'):
        # Bounded odometry integrates from a local origin, removing large global
        # sums. Preserve a strict numeric tolerance for vertical/tilt roundoff.
        np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-6)
        errors[key] = float(np.max(np.abs(actual-expected)))
      else:
        np.testing.assert_array_equal(actual, expected)
    expected_cal = [live.asof((np.array([c['ns'] for c in cal_rows]), cal_rows), f['capture_ns'], 1_000_000_000)[1]
                    for f in row['frames']]
    if [f['image_calibration_ns'] for f in detail['frames']] != expected_cal:
      raise AssertionError('Image calibration timestamp differs from recorded controls')
    controls.append(dict(query_ns=q, display_time_s=row['display_time_s'], full_images=10,
                         exact_pixels_timestamps_camera_ids=True, history_max_abs_error=errors))
  return dict(controls=controls, rlog_files_sha256_verified=len(expected_hashes),
              limit='Real recorded inputs; synthetic receive/EOF clocks; no live latency or GPU measurement')


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--output', type=Path)
  parser.add_argument('--manifest', type=Path)
  parser.add_argument('--raw-frames', type=Path)
  parser.add_argument('--logs', type=Path)
  parser.add_argument('--calibrations', type=Path)
  args = parser.parse_args()
  started = time.monotonic()
  result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Checks))
  if not result.wasSuccessful():
    raise SystemExit(1)
  report = dict(cpu_checks=result.testsRun, passed=True, hardware_operations=False, gpu_operations=False,
                simulated_model_latency_ms=229, simulation_is_not_model_performance=True)
  if any((args.manifest, args.raw_frames, args.logs, args.calibrations)):
    if not all((args.manifest, args.raw_frames, args.logs, args.calibrations)):
      parser.error('Recorded checks require all four fixture paths')
    report['recorded_parity'] = recorded_parity(args.manifest, args.raw_frames, args.logs, args.calibrations)
  report['wall_seconds'] = time.monotonic()-started
  report['sources_sha256'] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('agent-live-*.py')}
  import inspect
  from opendbc.car.vehicle_model import VehicleModel
  helpers = [Path(__file__).with_name(name) for name in ('calibration_trials.py', 'camera_profiles.py', 'replay-route.py')]
  helpers.append(Path(inspect.getfile(VehicleModel)))
  report['canonical_helpers_sha256'] = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in helpers}
  if any(name == 'tinygrad' or name.startswith('tinygrad.') for name in sys.modules):
    raise RuntimeError('The CPU bridge check unexpectedly imported the GPU runtime')
  if args.output:
    with args.output.open('x') as stream:
      stream.write(json.dumps(report, indent=2)+'\n')
  print(json.dumps(report, indent=2))


if __name__ == '__main__':
  main()
