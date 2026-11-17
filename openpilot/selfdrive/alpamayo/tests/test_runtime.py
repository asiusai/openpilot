from dataclasses import replace
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, Mock, patch

import numpy as np

from openpilot.selfdrive.alpamayo import capture, frontend, inference, runtime, shadowd
from openpilot.selfdrive.alpamayo.tests.test_frontend import BASE, MS, geometry, populated, prediction


class HandoffTests(unittest.TestCase):
  def test_binary_snapshot_rejects_corruption_before_mapping(self):
    selected = populated().select(BASE+3015*MS)
    with runtime.ram_handoff() as directory:
      path = Path(directory)/'pending.bin'
      runtime.save_selection(path, selected, {'nested': {'navigation': 'Turn left', 'missing': None}})
      raw = path.read_bytes()
      _, length = runtime.INPUT_PREFIX.unpack(raw[:runtime.INPUT_PREFIX.size])
      header = raw[runtime.INPUT_PREFIX.size:runtime.INPUT_PREFIX.size+length]
      payload = raw[runtime.INPUT_PREFIX.size+length:]
      malformed = [b'PK'+raw[2:], raw[:5], raw[:runtime.INPUT_PREFIX.size+length-1], raw[:-1], raw+b'X',
                   runtime.INPUT_PREFIX.pack(runtime.INPUT_MAGIC, runtime.HEADER_LIMIT+1)+b'{}']
      edits = [lambda m: m.update(frames=m['frames'][:-1]), lambda m: m.update(context=[]), lambda m: m.update(query_ns='bad'),
               lambda m: m['states'].update(deviceMotion=[{'wrong': 1}]),
               lambda m: m['frames'][0].update(dtype='float32'), lambda m: m['frames'][0].update(dtype='object'),
               lambda m: m['frames'][0].update(shape=[3, -1, 24]), lambda m: m['frames'][0].update(shape=[3, 0, 24]),
               lambda m: m['frames'][0].update(shape=[3, True, 24]), lambda m: m['frames'][0].update(shape=[3, 2**64, 24]),
               lambda m: m['frames'][0].update(shape=[3, 16, 24, 1]), lambda m: m['frames'][0].update(nbytes=1),
               lambda m: m['frames'][0].update(nbytes=-1), lambda m: m['frames'][0].update(nbytes=1152.0),
               lambda m: m['frames'][0].update(source_format='nv12'), lambda m: m['frames'][0].update(view=6),
               lambda m: m['frames'][0].update(capture_ns=-1), lambda m: m['context'].update(nonfinite=float('nan'))]
      for edit in edits:
        metadata = json.loads(header)
        edit(metadata)
        encoded = json.dumps(metadata).encode()
        malformed.append(runtime.INPUT_PREFIX.pack(runtime.INPUT_MAGIC, len(encoded))+encoded+payload)
      for index, blob in enumerate(malformed):
        with self.subTest(case=index), patch.object(np, 'memmap', side_effect=AssertionError('Malformed data reached mmap')):
          path.write_bytes(blob)
          with self.assertRaises(ValueError), runtime.load_selection(path):
            self.fail('Malformed snapshot accepted')
      with path.open('wb') as stream:
        stream.truncate(runtime.INPUT_LIMIT+1)
      with self.assertRaises(ValueError), runtime.load_selection(path):
        self.fail('Oversized snapshot accepted')

  def test_mapping_closes_on_exception_and_next_snapshot_owns_new_storage(self):
    class Finished(Exception):
      pass
    selected = populated().select(BASE+3015*MS)
    with runtime.ram_handoff() as directory:
      pending, working = Path(directory)/'pending.bin', Path(directory)/'working.bin'
      runtime.save_selection(pending, selected, {'nested': {'value': [1, None, 'unchanged']}})
      pending.replace(working)
      with self.assertRaises(Finished), runtime.load_selection(working) as (loaded, context):
        first = loaded.frames[0].pixels._mmap
        self.assertFalse(first.closed)
        self.assertEqual(context, {'nested': {'value': [1, None, 'unchanged']}})
        raise Finished
      self.assertTrue(first.closed)
      working.unlink()
      changed = np.full_like(selected.frames[0].pixels, 199)
      changed.flags.writeable = False
      second = replace(selected, frames=(replace(selected.frames[0], pixels=changed), *selected.frames[1:]))
      runtime.save_selection(pending, second, {})
      with runtime.load_selection(pending) as (loaded, _):
        mapping = loaded.frames[0].pixels._mmap
        self.assertIsNot(mapping, first)
        np.testing.assert_array_equal(loaded.frames[0].pixels, changed)
        self.assertFalse(mapping.closed)
      self.assertTrue(mapping.closed)

  def test_writer_rejects_mutable_noncontiguous_or_wrong_dtype_frames(self):
    selected = populated().select(BASE+3015*MS)
    mutable = selected.frames[0].pixels.copy()
    wrong_type = selected.frames[0].pixels.astype(np.float32)
    wrong_type.flags.writeable = False
    strided = selected.frames[0].pixels[:, :, ::2]
    with runtime.ram_handoff() as directory:
      for invalid in (mutable, wrong_type, strided):
        changed = replace(selected, frames=(replace(selected.frames[0], pixels=invalid), *selected.frames[1:]))
        with self.assertRaises(ValueError):
          runtime.save_selection(Path(directory)/'pending.bin', changed, {})
      self.assertEqual(list(Path(directory).iterdir()), [])

  def test_full_native_rgb_roundtrip_preserves_prepared_inputs_and_complete_output(self):
    selected = populated().select(BASE+3015*MS)
    frames = []
    for i, frame in enumerate(selected.frames):
      pixels = np.full((3, 760, 1344), i*20, dtype=np.uint8)
      pixels.flags.writeable = False
      frames.append(replace(frame, pixels=pixels, format='rgb'))
    selected = replace(selected, frames=tuple(frames))
    expected, expected_metadata = frontend.Preparer(geometry()).prepare(selected)
    adapter = SimpleNamespace(predict=lambda arrays, meta: {'query_ns': meta['query_ns'], 'prediction': prediction()}, synchronize=lambda: None)
    baseline = runtime.run_prediction(adapter, expected, expected_metadata, clock=lambda: BASE+3130*MS)
    with runtime.ram_handoff() as directory:
      path = Path(directory)/'pending.bin'
      runtime.save_selection(path, selected, {})
      self.assertGreater(path.stat().st_size, 29*1024**2)
      self.assertLessEqual(path.stat().st_size, runtime.INPUT_LIMIT)
      with runtime.load_selection(path) as (loaded, _):
        actual, actual_metadata = frontend.Preparer(geometry()).prepare(loaded)
        for name in expected:
          np.testing.assert_array_equal(actual[name], expected[name])
        result = runtime.run_prediction(adapter, actual, actual_metadata, clock=lambda: BASE+3130*MS)
        self.assertEqual(result, baseline)

  def test_native_snapshot_uses_one_private_ram_slot(self):
    selected = populated().select(BASE+3015*MS)
    frames = []
    for i, frame in enumerate(selected.frames):
      pixels = np.full((1140, 1344), i, dtype=np.uint8)
      pixels.flags.writeable = False
      frames.append(replace(frame, pixels=pixels, format='nv12'))
    selected = replace(selected, frames=tuple(frames))
    with runtime.ram_handoff() as directory:
      root = Path(directory)
      self.assertEqual(root.parent, Path('/dev/shm'))
      self.assertEqual(root.stat().st_mode & 0o777, 0o700)
      pending, working = root/'pending.bin', root/'working.bin'
      runtime.save_selection(pending, selected, {'boot_id': 'synthetic'})
      self.assertLess(pending.stat().st_size, runtime.INPUT_LIMIT)
      self.assertGreater(pending.stat().st_size, 14*1024**2)
      self.assertEqual(list(root.iterdir()), [pending])
      with self.assertRaisesRegex(RuntimeError, 'occupied'):
        runtime.save_selection(pending, selected, {})
      pending.replace(working)
      with self.assertRaisesRegex(RuntimeError, 'occupied'):
        runtime.save_selection(pending, selected, {})
      memmap_type = np.memmap
      with patch.object(np, 'memmap', wraps=memmap_type) as map_file, runtime.load_selection(working) as (loaded, context):
        self.assertEqual(context, {'boot_id': 'synthetic'})
        self.assertEqual(loaded.query_ns, selected.query_ns)
        self.assertEqual(loaded.selected_ns, selected.selected_ns)
        self.assertEqual(loaded.small_model, selected.small_model)
        self.assertEqual(loaded.camera_gaps, selected.camera_gaps)
        self.assertEqual(loaded.states, selected.states)
        mapping = loaded.frames[0].pixels._mmap
        for original, actual in zip(selected.frames, loaded.frames, strict=True):
          self.assertEqual(actual.reference(), original.reference())
          np.testing.assert_array_equal(actual.pixels, original.pixels)
          self.assertIsInstance(actual.pixels, memmap_type)
          self.assertIs(actual.pixels._mmap, mapping)
          self.assertFalse(actual.pixels.flags.owndata)
          self.assertFalse(actual.pixels.flags.writeable)
          with self.assertRaises(ValueError):
            actual.pixels.flags.writeable = True
          with self.assertRaises(ValueError):
            actual.pixels.flat[0] = 255
        self.assertFalse(mapping.closed)
        map_file.assert_called_once()
      self.assertTrue(mapping.closed)
      working.unlink()
      runtime.save_selection(pending, selected, {})
    self.assertFalse(root.exists())

  def test_ram_capacity_and_memory_reserves_do_not_use_disk_threshold(self):
    usage = SimpleNamespace(total=64*1024**2, free=40*1024**2)
    with patch('shutil.disk_usage', return_value=usage), patch.object(runtime, 'memory_ok', return_value=True):
      self.assertTrue(runtime.input_storage_ok('/dev/shm', runtime.INPUT_LIMIT))
      self.assertFalse(runtime.storage_ok('/dev/shm', runtime.INPUT_LIMIT))
      usage.free -= 1
      self.assertFalse(runtime.input_storage_ok('/dev/shm', runtime.INPUT_LIMIT))
    with patch.object(runtime, 'memory_ok', return_value=False):
      with self.assertRaises(RuntimeError):
        runtime.ram_handoff()
    for available, upcoming, expected in ((800, 32, True), (800, 33, False), (767, 0, False)):
      with patch.object(Path, 'read_text', return_value=f'MemAvailable: {available*1024} kB\n'):
        self.assertEqual(runtime.memory_ok(upcoming*1024**2), expected)

  def test_no_disk_fallback_oversize_or_partial_snapshot(self):
    selected = populated().select(BASE+3015*MS)
    with tempfile.TemporaryDirectory() as directory:
      with self.assertRaisesRegex(RuntimeError, 'RAM'):
        runtime.save_selection(Path(directory)/'pending.bin', selected, {})
      self.assertEqual(list(Path(directory).iterdir()), [])
    with patch.object(Path, 'read_text', return_value='1 0 0:1 / /dev/shm rw - ext4 /dev/test rw\n'):
      with self.assertRaises(RuntimeError):
        runtime.ram_handoff()
    with runtime.ram_handoff() as directory:
      root = Path(directory)
      with patch.object(runtime, 'INPUT_LIMIT', 1):
        with self.assertRaisesRegex(RuntimeError, 'RAM'):
          runtime.save_selection(root/'pending.bin', selected, {})
      original_open = Path.open
      def failing_open(path, mode='r', *args, **kwargs):
        stream = original_open(path, mode, *args, **kwargs)
        if mode != 'xb':
          return stream
        wrapped = MagicMock(wraps=stream)
        wrapped.__enter__.return_value = wrapped
        wrapped.__exit__.side_effect = lambda *_: stream.close()
        def write(data):
          if isinstance(data, memoryview):
            self.assertIs(data.obj, selected.frames[0].pixels)
            stream.write(data[:7])
            raise OSError('tmpfs full')
          return stream.write(data)
        wrapped.write.side_effect = write
        return wrapped
      with patch.object(Path, 'open', failing_open):
        with self.assertRaisesRegex(OSError, 'tmpfs full'):
          runtime.save_selection(root/'pending.bin', selected, {})
      self.assertEqual(list(root.iterdir()), [])

  def test_shutdown_kills_worker_and_cleans_ram_but_keeps_outputs_and_cache(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      output, cache = root/'outputs', root/'cache'
      output.mkdir()
      cache.mkdir()
      (cache/'staged-firmware').write_bytes(b'preserve')
      python = root/'isolated-python'
      python.symlink_to(sys.executable)
      config = root/'config.json'
      config.write_text(json.dumps({'python': str(python), 'output': str(output), 'cache': str(cache)}))
      handed_off = []
      def interrupt(_config, _config_path, _python, session, handoff, _cache, _boot):
        handed_off.append(handoff)
        (handoff/'pending.tmp').write_bytes(b'partial input')
        (session/'prediction.json').write_text('{}')
        raise KeyboardInterrupt
      old_umask = os.umask(0o077)
      try:
        with patch.dict(os.environ, {'ALPAMAYO_SHADOW': '1', 'ALPAMAYO_CONFIG': str(config)}), \
             patch.object(runtime, 'storage_ok', return_value=True), patch.object(signal, 'signal'), \
             patch.object(shadowd, 'supervise', side_effect=interrupt):
          with self.assertRaises(KeyboardInterrupt):
            shadowd.main()
      finally:
        os.umask(old_umask)
      self.assertEqual(len(handed_off), 1)
      self.assertFalse(handed_off[0].exists())
      self.assertEqual(len(list(output.glob('*/prediction.json'))), 1)
      self.assertEqual((cache/'staged-firmware').read_bytes(), b'preserve')
    code = 'import signal; signal.signal(signal.SIGTERM, signal.SIG_IGN); print("ready", flush=True); signal.pause()'
    child = subprocess.Popen([sys.executable, '-c', code], stdout=subprocess.PIPE, text=True, start_new_session=True)
    try:
      self.assertEqual(child.stdout.readline(), 'ready\n')
      shadowd.stop_worker(child)
      self.assertEqual(child.returncode, -signal.SIGKILL)
    finally:
      if child.poll() is None:
        child.kill()
        child.wait(timeout=1)
      child.stdout.close()

  def test_persistent_cache_validation_and_worker_environment(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      output, cache, routes = root/'output', root/'cache', root/'routes'
      for path in (output, cache, routes):
        path.mkdir()
      (cache/'compiler-cache.db').write_bytes(b'existing-cache')
      with patch.dict(os.environ, {'LOG_ROOT': str(routes), 'XDG_CACHE_HOME': '/unrelated', 'CACHEDB': '/unrelated.db'}):
        self.assertEqual(shadowd.cache_directory({'cache': str(cache)}, output), cache)
        for invalid in ('relative', str(root/'absent'), str(output), str(routes), str(root)):
          with self.assertRaises(ValueError):
            shadowd.cache_directory({'cache': invalid}, output)
        with patch.object(tempfile, 'TemporaryFile', side_effect=PermissionError('read only')):
          with self.assertRaises(PermissionError):
            shadowd.cache_directory({'cache': str(cache)}, output)
        env = shadowd.worker_environment({'tinygrad': '/isolated/tinygrad'}, cache)
        self.assertEqual(env['XDG_CACHE_HOME'], str(cache))
        self.assertEqual(env['CACHEDB'], str(cache/'compiler-cache.db'))
        self.assertEqual(env['HF_HUB_OFFLINE'], '1')
        repository = Path(shadowd.__file__).resolve().parents[3]
        self.assertEqual(env['PYTHONPATH'].split(os.pathsep), ['/isolated/tinygrad', str(repository), str(repository/'opendbc_repo')])
        self.assertEqual(env['PYTHONNOUSERSITE'], '1')
        self.assertEqual(env, shadowd.worker_environment({'tinygrad': '/isolated/tinygrad'}, cache))
      self.assertEqual((cache/'compiler-cache.db').read_bytes(), b'existing-cache')
      self.assertEqual(len(list(cache.iterdir())), 1)


class LiveReviewTests(unittest.TestCase):
  def test_selected_worker_injects_native_sampler_and_default_keeps_scipy(self):
    from functools import partial
    from openpilot.selfdrive.alpamayo.selected.camera import calibrated_image
    class Finished(Exception):
      pass
    with tempfile.TemporaryDirectory() as directory, runtime.ram_handoff() as handoff:
      for variant in ('default', 'selected'):
        with patch.object(frontend, 'Preparer', side_effect=Finished) as prepare:
          with self.assertRaises(Finished):
            runtime.worker({'inference': {'variant': variant}, 'geometry': geometry(), 'cache': directory}, directory, handoff)
        sampler = prepare.call_args.kwargs['sampler']
        if variant == 'default':
          self.assertIsNone(sampler)
        else:
          self.assertIsInstance(sampler, partial)
          self.assertIs(sampler.func, calibrated_image)
          self.assertEqual(sampler.keywords, {'cache_dir': directory})

  def test_prompt_rejection_releases_mapping_and_same_worker_accepts_next_input(self):
    from openpilot.selfdrive.alpamayo.selected.policy import PromptTooLong
    class Finished(Exception):
      pass
    store = populated()
    first = store.select(BASE+3015*MS)
    for camera in frontend.CAMERAS:
      frame = store.frames[camera][-1]
      store.add_frame(replace(frame, frame_id=31, sof_ns=BASE+3100*MS, eof_ns=BASE+3105*MS,
                              received_ns=BASE+3110*MS, ready_ns=BASE+3111*MS))
    store.add_state('modelV2', BASE+3112*MS, BASE+3113*MS, {'frameId': 31, 'timestampEof': BASE+3105*MS})
    second = store.select(BASE+3115*MS)
    mappings, calls = [], []
    memmap_type, atomic_json, run_prediction = np.memmap, runtime.atomic_json, runtime.run_prediction
    def map_file(*args, **kwargs):
      self.assertTrue(all(m._mmap.closed for m in mappings))
      mapped = memmap_type(*args, **kwargs)
      mappings.append(mapped)
      return mapped
    def predict(arrays, metadata):
      self.assertFalse(mappings[-1]._mmap.closed)
      calls.append(metadata['query_ns'])
      if len(calls) == 1:
        raise PromptTooLong('Prompt uses 2049 tokens; selected capacity is 2048')
      return {'query_ns': metadata['query_ns'], 'prediction': prediction()}
    adapter = SimpleNamespace(manifest={}, predict=predict, synchronize=lambda: None, close=Mock())
    with tempfile.TemporaryDirectory() as directory, runtime.ram_handoff() as handoff:
      root, slot = Path(directory), Path(handoff)
      runtime.save_selection(slot/'pending.bin', first, {'boot_id': 'synthetic'})
      def idle(_):
        self.assertEqual(len(calls), 1)
        self.assertTrue(mappings[-1]._mmap.closed)
        self.assertFalse((slot/'working.bin').exists())
        (slot/'skipped').unlink()
        runtime.save_selection(slot/'pending.bin', second, {'boot_id': 'synthetic'})
      def save(path, value):
        size = atomic_json(path, value)
        if path.name == f'{second.query_ns}.json':
          self.assertFalse(mappings[-1]._mmap.closed)
          raise Finished
        return size
      def execute(*args):
        return run_prediction(*args, clock=lambda: BASE+3130*MS)
      with patch.object(inference, 'ShadowAdapter', return_value=adapter) as construct, patch.object(runtime, 'storage_ok', return_value=True), \
           patch.object(runtime.time, 'monotonic_ns', return_value=BASE+3130*MS), patch.object(runtime.time, 'sleep', side_effect=idle), \
           patch.object(np, 'memmap', side_effect=map_file), patch.object(runtime, 'run_prediction', side_effect=execute), \
           patch.object(runtime, 'atomic_json', side_effect=save):
        with self.assertRaises(Finished):
          runtime.worker({'geometry': geometry(), 'inference': {}}, root, slot)
      construct.assert_called_once()
      self.assertEqual(calls, [first.query_ns, second.query_ns])
      rejected = json.loads((root/'unavailable.json').read_text())
      self.assertEqual(rejected['query_ns'], first.query_ns)
      self.assertEqual(rejected['type'], 'PromptTooLong')
      self.assertIn('2049', rejected['reason'])
      self.assertEqual(json.loads((root/f'{second.query_ns}.json').read_text())['prediction']['generated_token_ids'], [1, 2])
      self.assertTrue(all(m._mmap.closed for m in mappings))
      self.assertEqual(list(slot.iterdir()), [])
      adapter.close.assert_called_once()

  def test_worker_does_not_mask_unrelated_value_errors(self):
    selected = populated().select(BASE+3015*MS)
    adapter = SimpleNamespace(manifest={}, predict=Mock(side_effect=ValueError('unexpected model failure')), close=Mock())
    with tempfile.TemporaryDirectory() as directory, runtime.ram_handoff() as handoff:
      root, slot = Path(directory), Path(handoff)
      runtime.save_selection(slot/'pending.bin', selected, {})
      with patch.object(inference, 'ShadowAdapter', return_value=adapter), patch.object(runtime, 'storage_ok', return_value=True), \
           patch.object(runtime.time, 'monotonic_ns', return_value=BASE+3130*MS):
        with self.assertRaisesRegex(ValueError, 'unexpected model failure'):
          runtime.worker({'geometry': geometry(), 'inference': {}}, root, slot)
      self.assertFalse((root/'unavailable.json').exists())
      self.assertEqual(list(slot.iterdir()), [])
      adapter.close.assert_called_once()

  def test_explicit_adapter_dispatch_maps_selected_cache_and_rejects_extra_fields(self):
    from openpilot.selfdrive.alpamayo import selected
    for variant in (None, 'default'):
      settings = {'checkpoint': '/path/to/checkpoint', 'initial_noise': '/path/to/noise.npy', 'cache_vision': False}
      if variant is not None:
        settings['variant'] = variant
      with patch.object(inference, 'ShadowAdapter') as baseline, patch.object(selected, 'ShadowAdapter') as optimized:
        runtime.load_adapter({'inference': settings})
        baseline.assert_called_once_with({k: v for k, v in settings.items() if k != 'variant'})
        optimized.assert_not_called()
    settings = {'variant': 'selected', 'checkpoint': '/path/to/checkpoint', 'fp8_cache': '/path/to/fp8-cache',
                'prefill_calibration': '/path/to/prefill-scales.json', 'initial_noise': '/path/to/noise.npy'}
    with tempfile.TemporaryDirectory() as directory, patch.object(inference, 'ShadowAdapter') as baseline, \
         patch.object(selected, 'ShadowAdapter', side_effect=lambda fields: selected.Config(**fields)) as optimized:
      # Construct the real CPU-only config type to catch leaked selectors or unsupported adapter fields.
      config = {'inference': settings, 'cache': directory}
      result = runtime.load_adapter(config)
      self.assertEqual(result.cache_dir, Path(directory))
      self.assertEqual(result.device_timeout_ms, 10000)
      self.assertEqual(result.initial_noise, Path(settings['initial_noise']))
      self.assertEqual(settings['variant'], 'selected')
      self.assertNotIn('variant', optimized.call_args.args[0])
      self.assertNotIn('cache_dir', settings)
      baseline.assert_not_called()
      with self.assertRaisesRegex(ValueError, 'Unexpected selected inference fields'):
        runtime.load_adapter(dict(config, inference=dict(settings, cache_vision=True)))
      with self.assertRaisesRegex(ValueError, 'Unknown Alpamayo variant'):
        runtime.load_adapter(dict(config, inference=dict(settings, variant='unavailable')))

  def test_supervisor_reads_health_before_evaluation_clock(self):
    class Finished(Exception):
      pass
    events, current = [], [100]
    class Producer:
      error = None
      start, close = Mock(), Mock()
      @property
      def health(self):
        current[0] += 1
        events.append('health')
        return {'time_ns': current[0], 'model_alive': True, 'device_valid': True, 'model_age_ns': 0, 'thermal_status': 'green'}
    producer = Producer()
    def clock():
      events.append('clock')
      return current[0]
    def wait(interval):
      self.assertEqual(interval, .02)  # A healthy publication must reach the ordinary loop with an existing child.
      if len(events) == 4:
        raise Finished
    producer.stop = SimpleNamespace(wait=wait)
    child = SimpleNamespace(poll=lambda: None)
    with tempfile.TemporaryDirectory() as directory, runtime.ram_handoff() as handoff, \
         patch.dict(sys.modules, {'openpilot.common.params': SimpleNamespace(Params=lambda: None)}), \
         patch.object(capture, 'NativeProducer', return_value=producer), \
         patch.object(shadowd.time, 'monotonic_ns', side_effect=clock), \
         patch.object(runtime, 'memory_ok', return_value=True), patch.object(runtime, 'storage_ok', return_value=True), \
         patch.object(shadowd.subprocess, 'Popen', return_value=child) as spawn, patch.object(shadowd, 'stop_worker') as stop:
      with self.assertRaises(Finished):
        shadowd.supervise({'tinygrad': '/isolated/tinygrad'}, Path('config'), Path('python'), Path(directory),
                          Path(handoff), Path(directory)/'cache', 'synthetic')
      self.assertEqual(events, ['health', 'clock', 'health', 'clock'])
      spawn.assert_called_once()
      stop.assert_called_once_with(child)
      producer.close.assert_called_once()
    self.assertFalse(capture.healthy(producer.health, 100))  # Keep rejection of genuinely future health timestamps.

  def test_live_timing_keeps_copy_handoff_and_output_age_separate(self):
    selected = populated().select(BASE+3015*MS)
    fields = runtime.live_timing(selected, BASE+3050*MS, BASE+3130*MS)
    self.assertEqual(fields['live_processing_started_ns'], BASE+3010*MS)
    self.assertEqual(fields['selection_started_ns'], selected.selected_ns)
    self.assertEqual(fields['live_processing_ms'], 120.)
    self.assertEqual(fields['worker_processing_ms'], 80.)
    self.assertEqual(fields['input_age_ms'], 130.)
    self.assertNotIn('full_pipeline_ms', fields)
    frames = list(selected.frames)
    frames[4] = replace(frames[4], received_ns=BASE+3006*MS)
    self.assertEqual(runtime.live_timing(replace(selected, frames=tuple(frames)), BASE+3050*MS, BASE+3130*MS)['live_processing_ms'], 124.)
    with self.assertRaises(ValueError):
      runtime.live_timing(selected, BASE+3014*MS, BASE+3130*MS)
    with self.assertRaises(ValueError):
      runtime.live_timing(selected, BASE+3050*MS, BASE+3049*MS)

  def test_worker_logs_inclusive_live_timing_after_real_ram_handoff(self):
    class Finished(Exception):
      pass
    selected = populated().select(BASE+3015*MS)
    adapter = SimpleNamespace(manifest={'variant': 'cpu-test'}, close=Mock(), synchronize=lambda: None,
                              predict=lambda arrays, meta: {'query_ns': meta['query_ns'], 'prediction': prediction()})
    atomic_json, run_prediction = runtime.atomic_json, runtime.run_prediction
    def save(path, value):
      size = atomic_json(path, value)
      if path.name == f'{selected.query_ns}.json':
        raise Finished
      return size
    prediction_clock = iter((BASE+3080*MS, BASE+3120*MS))
    def predict(*args):
      return run_prediction(*args, clock=lambda: next(prediction_clock))
    with tempfile.TemporaryDirectory() as directory, runtime.ram_handoff() as handoff:
      root, slot = Path(directory), Path(handoff)
      runtime.save_selection(slot/'pending.bin', selected, {'boot_id': 'synthetic'})
      worker_clock = iter((BASE+3050*MS, BASE+3060*MS, BASE+3070*MS, BASE+3130*MS))
      def clock():
        self.assertTrue((slot/'working.bin').exists())  # The snapshot must be claimed before the worker interval starts.
        return next(worker_clock)
      with patch.object(inference, 'ShadowAdapter', return_value=adapter), patch.object(runtime, 'atomic_json', side_effect=save), \
           patch.object(runtime, 'storage_ok', return_value=True), patch.object(runtime, 'run_prediction', side_effect=predict), \
           patch.object(runtime.time, 'monotonic_ns', side_effect=clock):
        with self.assertRaises(Finished):
          runtime.worker({'geometry': geometry(), 'inference': {}}, root, slot)
      result = json.loads((root/f'{selected.query_ns}.json').read_text())
      self.assertEqual(result['input']['boot_id'], 'synthetic')
      self.assertEqual(result['live_processing_ms'], 120.)
      self.assertEqual(result['worker_processing_ms'], 80.)
      self.assertEqual(result['input_age_ms'], 130.)
      self.assertEqual(result['inference_ms'], 40.)
      self.assertEqual(result['completed_ns'], BASE+3120*MS)
      self.assertEqual(result['output_ready_ns'], BASE+3130*MS)
      self.assertEqual(result['prediction']['generated_token_ids'], [1, 2])
      self.assertNotIn('full_pipeline_ms', result)
      self.assertEqual(list(slot.iterdir()), [])
      adapter.close.assert_called_once()


class SlotTests(unittest.TestCase):
  def test_fixed_periods_skip_busy_missing_and_late_slots_without_catchup(self):
    self.assertEqual(runtime.FixedSlots().period_ms, 500)
    for invalid in (0, 200, 500., True, '500'):
      with self.assertRaises(ValueError):
        runtime.FixedSlots(invalid)
    for period in (100, 250, 500):
      slots, p = runtime.FixedSlots(period), period*MS
      self.assertIsNone(slots.take(1))
      self.assertEqual(slots.take(p), {'slot_start_ns': p, 'slot_deadline_ns': 2*p, 'period_ms': period, 'missed_slots': 0})
      slots.skip()  # Busy: this slot is consumed without creating a pending item.
      self.assertIsNone(slots.take(p+1))
      self.assertEqual(slots.take(2*p)['missed_slots'], 1)
      slots.skip()  # Missing causal input also gets one attempt only.
      late = slots.take(5*p+1)
      self.assertEqual(late['slot_start_ns'], 5*p)
      self.assertEqual(late['missed_slots'], 4)
      self.assertIsNone(slots.take(5*p+1))
      self.assertIsNone(slots.take(6*p-1))
      self.assertEqual(slots.take(6*p)['slot_start_ns'], 6*p)

  def test_busy_supervisor_waits_for_next_500ms_slot_and_counts_worker_rejection(self):
    class Finished(Exception):
      pass
    store, current, calls, snapshots = populated(), [BASE+3000*MS], [0], []
    producer = SimpleNamespace(error=None, start=Mock(), close=Mock())
    def update_health():
      producer.health = {'time_ns': current[0], 'model_alive': True, 'device_valid': True,
                         'model_age_ns': 0, 'thermal_status': 'green'}
    update_health()
    with tempfile.TemporaryDirectory() as directory, runtime.ram_handoff() as handoff:
      root, slot = Path(directory), Path(handoff)
      (root/'ready.json').write_text('{}')
      def wait(interval):
        self.assertEqual(interval, .02)
        if calls[0] in (0, 3):
          with runtime.load_selection(slot/'pending.bin') as (loaded, context):
            snapshots.append((loaded.query_ns, context))
        if calls[0] == 3:
          raise Finished
        if calls[0] == 2:
          (slot/'pending.bin').unlink()
          (slot/'skipped').touch()  # The worker rejected its now-stale input.
        calls[0] += 1
        current[0] += 500*MS
        last = store.frames[6][-1]
        for tick in range(1, 6):
          stamp, frame_id = last.sof_ns+tick*100*MS, last.frame_id+tick
          for camera in frontend.CAMERAS:
            frame = store.frames[camera][-1]
            store.add_frame(replace(frame, frame_id=frame_id, sof_ns=stamp, eof_ns=stamp+5*MS,
                                    received_ns=stamp+10*MS, ready_ns=stamp+11*MS))
          store.add_state('modelV2', stamp+12*MS, stamp+13*MS, {'frameId': frame_id, 'timestampEof': stamp+5*MS})
        update_health()
      producer.stop = SimpleNamespace(wait=wait)
      with patch.dict(sys.modules, {'openpilot.common.params': SimpleNamespace(Params=lambda: SimpleNamespace(get=lambda _: b'synthetic'))}), \
           patch.object(capture, 'NativeProducer', return_value=producer), patch.object(frontend, 'Store', return_value=store), \
           patch.object(shadowd.time, 'monotonic_ns', side_effect=lambda: current[0]), \
           patch.object(runtime, 'memory_ok', return_value=True), patch.object(runtime, 'storage_ok', return_value=True), \
           patch.object(shadowd.subprocess, 'Popen', return_value=SimpleNamespace(poll=lambda: None)), patch.object(shadowd, 'stop_worker'):
        with self.assertRaises(Finished):
          shadowd.supervise({'tinygrad': '/isolated/tinygrad'}, Path('config'), Path('python'), root, slot, root/'cache', 'synthetic')
      self.assertEqual([meta['slot_start_ns'] for _, meta in snapshots], [BASE+3000*MS, BASE+4500*MS])
      self.assertEqual([meta['missed_slots'] for _, meta in snapshots], [0, 3])
      self.assertEqual([query for query, _ in snapshots], [BASE+2900*MS, BASE+4400*MS])
      self.assertEqual(list(slot.iterdir()), [slot/'pending.bin'])
      producer.close.assert_called_once()

  def test_late_serialization_and_expired_worker_slot_never_start_inference(self):
    class Finished(Exception):
      pass
    selected = populated().select(BASE+3015*MS)
    context = {'slot_start_ns': BASE+3000*MS, 'slot_deadline_ns': BASE+3500*MS, 'period_ms': 500, 'missed_slots': 0}
    adapter = SimpleNamespace(manifest={}, close=Mock(), predict=Mock())
    with tempfile.TemporaryDirectory() as directory, runtime.ram_handoff() as handoff:
      root, slot = Path(directory), Path(handoff)
      with patch.object(runtime.time, 'monotonic_ns', return_value=context['slot_deadline_ns']):
        with self.assertRaises(frontend.Unavailable):
          runtime.save_selection(slot/'pending.bin', selected, context)
      self.assertEqual(list(slot.iterdir()), [])
      with patch.object(runtime.time, 'monotonic_ns', return_value=BASE+3020*MS):
        runtime.save_selection(slot/'pending.bin', selected, context)
      with patch.object(inference, 'ShadowAdapter', return_value=adapter), patch.object(runtime, 'storage_ok', return_value=True), \
           patch.object(runtime.time, 'monotonic_ns', side_effect=[BASE+3490*MS, BASE+3500*MS, BASE+3501*MS]), \
           patch.object(runtime.time, 'sleep', side_effect=Finished):
        with self.assertRaises(Finished):
          runtime.worker({'geometry': geometry(), 'inference': {}}, root, slot)
      adapter.predict.assert_not_called()
      adapter.close.assert_called_once()
      self.assertEqual(list(slot.iterdir()), [slot/'skipped'])
      self.assertFalse((root/f'{selected.query_ns}.json').exists())

  def test_slot_output_records_deadline_lateness_and_input_age(self):
    selected = populated().select(BASE+3015*MS)
    slot = {'slot_start_ns': BASE+3000*MS, 'slot_deadline_ns': BASE+3500*MS, 'period_ms': 500, 'missed_slots': 2}
    for completed, met, lateness in ((3499, True, 0.), (3500, True, 0.), (3510, False, 10.)):
      result = runtime.live_timing(selected, BASE+3050*MS, BASE+completed*MS, slot)
      self.assertEqual(result['slot_deadline_met'], met)
      self.assertEqual(result['completion_lateness_ms'], lateness)
      self.assertEqual(result['selection_lateness_ms'], 15.)
      self.assertEqual(result['input_age_at_selection_ms'], 15.)
      self.assertEqual(result['input_age_ms'], float(completed-3000))
      self.assertEqual(result['missed_slots'], 2)


if __name__ == '__main__':
  unittest.main()
