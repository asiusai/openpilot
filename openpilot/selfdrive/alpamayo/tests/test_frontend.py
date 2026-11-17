import ast
from dataclasses import replace
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from openpilot.selfdrive.alpamayo.capture import copy_vision_buffer, healthy, navigation_ready, state_value
from openpilot.selfdrive.alpamayo.feature_cache import FeatureCache
from openpilot.selfdrive.alpamayo.frontend import CAMERAS, OFFSETS_NS, Frame, Preparer, State, Store, Unavailable, navigation, required
from openpilot.selfdrive.alpamayo.geometry import calibrated_image
from openpilot.selfdrive.alpamayo.history import measured_history
from openpilot.selfdrive.alpamayo.runtime import atomic_json, load_selection, private_output, ram_handoff, run_prediction, save_selection, storage_ok

BASE, MS = 10_000_000_000, 1_000_000
CAL = {'rpyCalib': [0., 0., 0.], 'wideFromDeviceEuler': [0., 0., 0.]}
CP = {'rotationalInertiaPerMass': 2., 'wheelbase': 2.7, 'centerToFront': 1.2,
      'tireStiffnessFrontPerMass': 60., 'tireStiffnessRearPerMass': 70., 'steerRatio': 15.}
LP = {'stiffnessFactor': 1., 'steerRatio': 15., 'angleOffsetDeg': 0., 'roll': 0.}


def geometry():
  return {'cameras': {
    '1': {'prepared_size': [768, 432], 'content_box': [0, 0, 768, 432], 'focal_pixels': [240., 240.],
          'principal_point': [383.5, 215.5], 'projection': 'equidistant'},
    '6': {'prepared_size': [768, 432], 'content_box': [0, 0, 768, 432], 'focal_pixels': [650., 650.],
          'principal_point': [383.5, 215.5], 'projection': 'pinhole'}}}


def populated():
  store = Store()
  store.add_state('carParams', BASE, BASE, CP)
  for tick in range(61):
    stamp = BASE+tick*50*MS
    store.add_state('deviceMotion', stamp, stamp+MS, [[0., 0., 0.], [10., 0., 0.]])
    store.add_state('carState', stamp, stamp+MS, {'vEgo': 10., 'steeringAngleDeg': 0., 'aEgo': 0.})
    store.add_state('vehicleParameters', stamp, stamp+MS, LP)
    store.add_state('extrinsicsCalibration', stamp, stamp+MS, CAL)
    if tick % 2 == 0:
      for camera in CAMERAS:
        pixels = np.full((3, 16, 24), tick, dtype=np.uint8)
        pixels.flags.writeable = False
        store.add_frame(Frame(camera, tick//2, stamp, stamp+5*MS, stamp+10*MS, stamp+11*MS, pixels))
      store.add_state('modelV2', stamp+12*MS, stamp+13*MS, {'frameId': tick//2, 'timestampEof': stamp+5*MS})
  return store


def prediction():
  return {'xyz': np.zeros((1, 64, 3)).tolist(), 'rot': np.tile(np.eye(3), (1, 64, 1, 1)).tolist(),
          'reasoning': 'CPU fixture', 'generated_token_ids': [1, 2]}


class FrontendTests(unittest.TestCase):
  def test_publisher_health_failures_invalidate_ego_history(self):
    from openpilot.cereal import log
    motion = log.DeviceMotion.new_message(timestamp=BASE+3000*MS, inputsOK=True, sensorsOK=True, posenetOK=True,
                                         orientationNED={'valid': True}, velocityDevice={'x': 10., 'valid': True})
    parameters = log.VehicleParameters.new_message(**LP, valid=True, sensorValid=True)
    for kind, message, fields in (('deviceMotion', motion, ('inputsOK', 'sensorsOK', 'posenetOK')),
                                  ('vehicleParameters', parameters, ('valid', 'sensorValid'))):
      self.assertTrue(state_value(kind, message)[1])
      for flag in fields:
        with self.subTest(kind=kind, flag=flag):
          store = populated()
          setattr(message, flag, False)
          value, valid = state_value(kind, message)
          self.assertFalse(valid)
          store.add_state(kind, BASE+3000*MS, BASE+3002*MS, value, valid)
          self.assertIsNone(store.states[kind][-1].value)
          with self.assertRaises(Unavailable):
            Preparer(geometry()).prepare(store.select(BASE+3015*MS))
          setattr(message, flag, True)

  def test_newer_received_duplicate_timestamp_replaces_validity(self):
    store = Store()
    store.add_state('deviceMotion', 100, 110, 'valid')
    store.add_state('deviceMotion', 100, 120, 'invalid', False)
    rows = store.states['deviceMotion']
    self.assertEqual(rows, [State(100, 120, None, False)])
    with self.assertRaises(Unavailable):
      required(rows, 125, 100)
    for received in (110, 120):
      store.add_state('deviceMotion', 100, received, 'old valid')
    self.assertEqual(rows, [State(100, 120, None, False)])
    store.add_state('deviceMotion', 105, 125, 'next')
    store.add_state('deviceMotion', 100, 130, 'recovered')
    self.assertEqual(rows, [State(100, 130, 'recovered'), State(105, 125, 'next')])
    self.assertEqual(required(rows, 100, 100), ('recovered', 100))

  def test_full_causal_history_and_model_match(self):
    store = populated()
    selected = store.select(BASE+3015*MS)
    self.assertEqual([f.sof_ns for f in selected.frames], [selected.query_ns+o for o in OFFSETS_NS]*2)
    self.assertEqual(selected.small_model['model']['frameId'], selected.frames[-1].frame_id)
    # A newer frame without its model output must not starve the older match.
    latest = store.frames[6][-1]
    store.add_frame(replace(latest, frame_id=31, sof_ns=BASE+3100*MS, eof_ns=BASE+3105*MS,
                            received_ns=BASE+3110*MS, ready_ns=BASE+3111*MS))
    self.assertEqual(store.select(BASE+3115*MS).query_ns, selected.query_ns)
    with self.assertRaises(Unavailable):
      store.select(BASE+3115*MS, selected.query_ns)

  def test_missing_or_future_camera_is_unavailable(self):
    store = populated()
    store.frames[1] = type(store.frames[1])(f for f in store.frames[1] if f.frame_id not in (9, 10))
    with self.assertRaises(Unavailable):
      store.select(BASE+3015*MS)
    with self.assertRaises(Unavailable):
      populated().select(BASE+4000*MS)

  def test_prepare_preserves_history_navigation_and_calibration(self):
    store = populated()
    store.add_state('navInstruction', BASE+2900*MS, BASE+2901*MS,
                    {'maneuverPrimaryText': 'Turn right', 'maneuverDistance': 42.})
    arrays, meta = Preparer(geometry()).prepare(store.select(BASE+3015*MS))
    self.assertEqual(arrays['image_frames'].shape, (10, 1, 3, 432, 768))
    np.testing.assert_allclose(arrays['ego_history_xyz'][0, :, 0], np.arange(-19, 1), atol=1e-5)
    self.assertEqual(arrays['nav_text'], ['Turn right in 42m'])
    self.assertTrue(all(f['image_calibration_ns'] <= f['capture_ns'] <= f['requested_ns'] for f in meta['frames']))
    store.states['carState'][-1] = replace(store.states['carState'][-1], value=None, valid=False)
    with self.assertRaises(Unavailable):
      Preparer(geometry()).prepare(store.select(BASE+3015*MS))

  def test_navigation_never_fills_missing_or_invalid_guidance(self):
    value = {'maneuverPrimaryText': 'Turn left', 'maneuverDistance': 20.}
    for rows in ([], [State(BASE+1, BASE+1, value)], [State(BASE, BASE, value), State(BASE+1, BASE+1, None, False)]):
      query = BASE+2 if len(rows) == 2 else BASE
      self.assertEqual(navigation(rows, query)[0], '')

  def test_navigation_retains_mapbox_turn_and_current_route(self):
    from openpilot.cereal import log
    message = log.NavInstruction.new_message(maneuverPrimaryText='Main Street', maneuverSecondaryText='Centre',
                                            maneuverType='turn', maneuverModifier='right', maneuverDistance=120.)
    value, valid = state_value('navInstruction', message)
    self.assertTrue(valid)
    value['routeRevision'] = 'current'
    rows = [State(BASE, BASE, value)]
    self.assertEqual(navigation(rows, BASE)[0], 'Turn right in 120m')
    self.assertEqual(navigation(rows, BASE)[1]['instruction'], value)
    rows.append(State(BASE+1, BASE+1, None, False))
    self.assertEqual(navigation(rows, BASE)[0], '')
    rows[-1] = State(BASE+1, BASE+1, dict(value, routeRevision='replacement'))
    self.assertEqual(navigation(rows, BASE)[0], '')
    self.assertEqual(navigation(rows, BASE+3_000*MS)[0], '')
    ready = {'status': 'ready', 'lastKnown': False, 'revision': 'current', 'publishedAt': BASE/1e9-1}
    params = SimpleNamespace(get_bool=lambda key: True)
    self.assertTrue(navigation_ready(params, SimpleNamespace(state=lambda: ready), BASE)[0])
    for changes in ({'status': 'loading'}, {'status': 'idle'}, {'lastKnown': True}, {'publishedAt': BASE/1e9+1}):
      self.assertFalse(navigation_ready(params, SimpleNamespace(state=lambda changes=changes: dict(ready, **changes)), BASE)[0])
    self.assertFalse(navigation_ready(SimpleNamespace(get_bool=lambda key: False), SimpleNamespace(state=lambda: ready), BASE)[0])

  def test_navigation_compacts_signs_without_losing_maneuvers(self):
    value = {'maneuverPrimaryText': 'Highway A1 / Warsaw / Lodz / Gdansk', 'maneuverSecondaryText': 'Follow signs for Centrum',
             'maneuverType': 'on ramp', 'maneuverModifier': 'slight right', 'maneuverDistance': 250.}
    self.assertEqual(navigation([State(BASE, BASE, value)], BASE)[0], 'Take the ramp slightly right in 250m')
    value.update(maneuverType='roundabout', maneuverPrimaryText='Take the 3rd exit onto Main Street', maneuverModifier='right')
    self.assertEqual(navigation([State(BASE, BASE, value)], BASE)[0], 'At the roundabout, take exit 3 in 250m')
    value.update(maneuverType='turn', maneuverModifier='uturn')
    self.assertEqual(navigation([State(BASE, BASE, value)], BASE)[0], 'Make a U-turn in 250m')

  def test_measured_history_axes_and_no_future(self):
    for yaw in (0., .7, -2.4):
      rows = [State(BASE+i*50*MS, BASE+i*50*MS, [[0., 0., yaw], [10., 2., 0.]]) for i in range(61)]
      xyz, rot, age = measured_history(rows, BASE+3000*MS, [0., 0., 0.])
      expected = np.stack((np.arange(-19, 1), -.2*np.arange(-19, 1), np.zeros(20)), axis=-1)
      np.testing.assert_allclose(xyz[0], expected, atol=1e-5)
      np.testing.assert_allclose(rot[0], np.tile(np.eye(3), (20, 1, 1)), atol=1e-6)
      rows.append(State(BASE+4000*MS, BASE+4000*MS, [[0., 0., 0.], [9999., 0., 0.]]))
      np.testing.assert_array_equal(measured_history(rows, BASE+3000*MS, [0., 0., 0.])[0], xyz)
      self.assertEqual(age, 0.)
    with self.assertRaises(Unavailable):
      measured_history(rows[:10]+rows[40:], BASE+3000*MS, [0., 0., 0.])

  def test_pixel_and_feature_cache_invalidation(self):
    selected = populated().select(BASE+3015*MS)
    preparer = Preparer(geometry())
    original, _ = preparer.prepare(selected)
    self.assertEqual(preparer.prepare(selected)[1]['image_cache_hits'], 10)
    pixels = selected.frames[0].pixels.copy()
    pixels[:] = 250
    pixels.flags.writeable = False
    changed = replace(selected, frames=(replace(selected.frames[0], pixels=pixels), *selected.frames[1:]))
    actual, meta = preparer.prepare(changed)
    self.assertEqual(meta['image_cache_hits'], 9)
    self.assertFalse(np.array_equal(actual['image_frames'][0], original['image_frames'][0]))
    buffer = np.zeros(3)
    def encode(p, v, g):
      buffer[:] = p.sum()+v.sum()+g.sum()
      return buffer
    cache = FeatureCache(encode, np.copy, capacity=2)
    p, v, g = np.ones(2), np.zeros(2), np.array([1, 2])
    first = cache(p, v, g)
    np.testing.assert_array_equal(cache(p.copy(), v, g), first)
    cache(p*2, v, g)
    np.testing.assert_array_equal(first, [5., 5., 5.])
    cache(p, v+1, g)
    cache(p, v, g+1)
    self.assertEqual((cache.hits, cache.misses, len(cache.cache)), (1, 4, 2))

  def test_calibration_and_learned_history_values_trace_saved_inputs(self):
    store = populated()
    gradient = np.tile(np.arange(24, dtype=np.uint8)*10, (3, 16, 1))
    gradient.flags.writeable = False
    for camera in CAMERAS:
      store.frames[camera] = type(store.frames[camera])(replace(frame, pixels=gradient) for frame in store.frames[camera])
    for row in store.states['carState']:
      row.value['steeringAngleDeg'] = 5.
    preparer = Preparer(geometry())
    original, _ = preparer.prepare(store.select(BASE+3015*MS))
    calibration = {'rpyCalib': [0., .03, .02], 'wideFromDeviceEuler': [0., -.01, .03]}
    parameters = dict(LP, stiffnessFactor=.8, steerRatio=12., angleOffsetDeg=1., roll=.01)
    store.add_state('extrinsicsCalibration', BASE+3000*MS, BASE+3002*MS, calibration)
    store.add_state('vehicleParameters', BASE+3000*MS, BASE+3002*MS, parameters)
    selected = store.select(BASE+3015*MS)
    actual, metadata = preparer.prepare(selected)
    self.assertEqual(metadata['image_cache_hits'], 8)  # Same frame IDs and pixels, two changed current-frame calibrations.
    for i in (4, 9):
      self.assertFalse(np.array_equal(actual['image_frames'][i], original['image_frames'][i]))
      np.testing.assert_array_equal(actual['image_frames'][i, 0], calibrated_image(gradient, CAMERAS[i//5], geometry(), calibration))
    self.assertFalse(np.array_equal(actual['ego_history_xyz'][..., 2], original['ego_history_xyz'][..., 2]))
    self.assertFalse(np.array_equal(actual['ego_history_xyz'][..., :2], original['ego_history_xyz'][..., :2]))
    self.assertEqual(actual['nav_text'], [''])
    adapter = SimpleNamespace(predict=lambda a, m: {'query_ns': m['query_ns'], 'prediction': prediction()}, synchronize=lambda: None)
    with tempfile.TemporaryDirectory() as directory, patch('openpilot.selfdrive.alpamayo.runtime.storage_ok', return_value=True):
      path = Path(directory)/'prediction.json'
      atomic_json(path, run_prediction(adapter, actual, metadata))
      saved = json.loads(path.read_text())['input']
    self.assertEqual(saved['calibration'], calibration)
    for i, frame in enumerate(saved['frames']):
      self.assertEqual(frame['image_calibration'], calibration if i in (4, 9) else CAL)
      self.assertLessEqual(frame['image_calibration_ns'], frame['capture_ns'])
    audit = saved['vehicle_history']['learned_parameters']
    self.assertEqual(len(audit), 39)
    for row in audit:
      self.assertEqual(row['values'], parameters if row['sample_ns'] == selected.query_ns else LP)
      self.assertEqual(row['age_ms'], (row['sample_ns']-row['source_ns'])/1e6)
    self.assertEqual(saved['vehicle_history']['car_params']['values'], CP)
    np.testing.assert_array_equal(saved['ego_history_xyz'], actual['ego_history_xyz'])
    np.testing.assert_array_equal(saved['ego_history_rot'], actual['ego_history_rot'])

  def test_slot_cadence_cache_and_missed_query_keep_full_history(self):
    for period in (250, 500):
      store, cache = Store(), FeatureCache(lambda p, v, g: p.copy(), np.copy)
      next_frame, misses = 0, []
      for tick in range(24):
        now = BASE+(2000+tick*period)*MS+15*MS
        while BASE+next_frame*100*MS+15*MS <= now:
          stamp = BASE+next_frame*100*MS
          for camera in CAMERAS:
            pixels = np.full((3, 1, 1), next_frame, dtype=np.uint8)
            pixels.flags.writeable = False
            store.add_frame(Frame(camera, next_frame, stamp, stamp+5*MS, stamp+10*MS, stamp+11*MS, pixels))
          store.add_state('modelV2', stamp+12*MS, stamp+13*MS, {'frameId': next_frame, 'timestampEof': stamp+5*MS})
          next_frame += 1
        if tick == 20:  # An occupied inference slot skips this otherwise available query.
          continue
        selected = store.select(now)
        self.assertEqual(len(selected.frames), 10)
        before = cache.misses
        for frame, offset in zip(selected.frames, OFFSETS_NS*2, strict=True):
          self.assertLessEqual(frame.sof_ns, selected.query_ns+offset)
          cache(frame.pixels, np.array([frame.camera]), np.ones((1, 3), dtype=np.int64))
        misses.append((tick, cache.misses-before))
      self.assertTrue(all(count == 4 for tick, count in misses if 10 <= tick < 20), (period, misses))
      self.assertTrue(any(count > 4 for tick, count in misses if tick > 20), (period, misses))

  def test_projection_checks_coverage(self):
    image = np.full((3, 16, 24), 123, dtype=np.uint8)
    np.testing.assert_array_equal(calibrated_image(image, 1, geometry(), CAL), np.full((3, 432, 768), 123, dtype=np.uint8))
    with self.assertRaises(ValueError):
      calibrated_image(image, 1, geometry(), dict(CAL, rpyCalib=[0., 0., 2.]))

  def test_nv12_copy_owns_padded_source(self):
    w, h, stride = 1344, 760, 1408
    raw = np.zeros((h*3//2, stride), dtype=np.uint8)
    raw[:h, :w] = 21
    raw[h:, :w] = 128
    client = SimpleNamespace(frame_id=1, timestamp_sof=BASE, timestamp_eof=BASE+MS)
    buf = SimpleNamespace(width=w, height=h, stride=stride, uv_offset=h*stride, data=raw)
    frame = copy_vision_buffer(1, client, buf, clock=lambda: BASE+2*MS)
    raw[:] = 0
    self.assertEqual(frame.pixels.shape, (1140, 1344))
    self.assertTrue(np.all(frame.pixels[:h] == 21))
    self.assertFalse(frame.pixels.flags.writeable)

  def test_snapshot_roundtrip_and_complete_shadow_output(self):
    selected = populated().select(BASE+3015*MS)
    with tempfile.TemporaryDirectory() as folder, ram_handoff() as ram, patch('openpilot.selfdrive.alpamayo.runtime.storage_ok', return_value=True):
      path = Path(ram)/'pending.bin'
      save_selection(path, selected, {'route': 'synthetic', 'boot_id': 'synthetic'})
      with load_selection(path) as (loaded, context):
        self.assertEqual(loaded.query_ns, selected.query_ns)
        self.assertEqual(context['route'], 'synthetic')
        np.testing.assert_array_equal(loaded.frames[0].pixels, selected.frames[0].pixels)
        arrays, metadata = Preparer(geometry()).prepare(loaded)
        adapter = SimpleNamespace(predict=lambda a, m: {'query_ns': m['query_ns'], 'prediction': prediction()}, synchronize=lambda: None)
        result = run_prediction(adapter, arrays, metadata)
      output = Path(folder)/f'{selected.query_ns}.json'
      atomic_json(output, result)
      saved = json.loads(output.read_text())
      self.assertEqual(saved['input']['small_model'], selected.small_model)
      self.assertEqual(saved['prediction']['generated_token_ids'], [1, 2])
      adapter.predict = lambda a, m: {'query_ns': m['query_ns']-1, 'prediction': prediction()}
      with self.assertRaises(ValueError):
        run_prediction(adapter, arrays, metadata)

  def test_health_and_memory_bounds(self):
    state = {'time_ns': BASE, 'model_alive': True, 'device_valid': True, 'model_age_ns': 0, 'thermal_status': 'green'}
    self.assertTrue(healthy(state, BASE))
    self.assertFalse(healthy(state, BASE+501*MS))
    self.assertFalse(healthy(dict(state, thermal_status='red'), BASE))
    store = Store(byte_limit=3000)
    for frame in populated().frames[1]:
      store.add_frame(frame)
      self.assertLessEqual(store.bytes, 3000)

  def test_private_output_and_recording_storage_reserve(self):
    with tempfile.TemporaryDirectory() as folder, patch.dict(os.environ, {'LOG_ROOT': folder}):
      with self.assertRaises(ValueError):
        private_output(Path(folder)/'shadow')
      self.assertEqual(private_output(Path(folder).parent/'separate'), Path(folder).parent/'separate')
    for free, upcoming, expected in ((11, 0, True), (9, 0, False), (11, 2, False)):
      usage = SimpleNamespace(total=100*1024**3, free=free*1024**3)
      with patch('shutil.disk_usage', return_value=usage):
        self.assertEqual(storage_ok('.', upcoming*1024**3), expected)

  def test_source_frame_gaps_remain_in_output_metadata(self):
    store = populated()
    latest = store.frames[1][-1]
    store.add_frame(replace(latest, frame_id=33, sof_ns=BASE+3100*MS, eof_ns=BASE+3105*MS,
                            received_ns=BASE+3110*MS, ready_ns=BASE+3111*MS))
    selected = store.select(BASE+3115*MS)
    self.assertEqual(selected.camera_gaps[1], 2)
    self.assertEqual(Preparer(geometry()).prepare(selected)[1]['camera_gaps'][1], 2)


class DefaultProcessTests(unittest.TestCase):
  def test_default_and_shadow_never_probe_big_model(self):
    root = Path(__file__).resolve().parents[2]
    tree = ast.parse((root/'modeld/modeld.py').read_text())
    main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'main')
    assignment = next(n for n in main.body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'CHESTNUT' for t in n.targets))
    expr = compile(ast.Expression(assignment.value), '<model policy>', 'eval')
    def forbidden():
      self.fail('Small-model policy probed Chestnut')
    for env in ({}, {'ALPAMAYO_SHADOW': '1'}, {'DISABLE_CHESTNUT': '0', 'ALPAMAYO_SHADOW': '1'}, {'DISABLE_CHESTNUT': '1'}):
      with patch.dict(os.environ, env, clear=True):
        self.assertFalse(eval(expr, {'os': os, 'chestnut_compiled': forbidden}))
    with patch.dict(os.environ, {'DISABLE_CHESTNUT': '0'}, clear=True):
      self.assertTrue(eval(expr, {'os': os, 'chestnut_compiled': lambda: True, 'ASIUS_HARDWARE': True,
                                 'chestnut_present': lambda **kwargs: True}))

  def test_manager_only_starts_opted_in_onroad_shadow(self):
    path = Path(__file__).resolve().parents[3]/'system/manager/process_config.py'
    tree = ast.parse(path.read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'alpamayo_shadow')
    for arg in node.args.args:
      arg.annotation = None
    node.returns = None
    namespace = {'os': os}
    exec(compile(ast.Module([node], []), str(path), 'exec'), namespace)
    for env, expected in (({}, False), ({'ALPAMAYO_SHADOW': '0'}, False), ({'ALPAMAYO_SHADOW': '1'}, True)):
      with patch.dict(os.environ, env, clear=True):
        self.assertFalse(namespace['alpamayo_shadow'](False, None, None))
        self.assertEqual(namespace['alpamayo_shadow'](True, None, None), expected)
    from openpilot.selfdrive.alpamayo import shadowd
    with patch.dict(os.environ, {}, clear=True), patch('subprocess.Popen', side_effect=AssertionError('Unrequested worker')):
      shadowd.main()


if __name__ == '__main__':
  unittest.main()
