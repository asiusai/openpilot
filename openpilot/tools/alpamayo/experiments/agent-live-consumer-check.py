"""Actual CPU adapters/CompactInputs methods with an explicitly fake GPU owner.

Every device open and socket connect is blocked. No inference result is computed.
The fake runner checks all upload data and forwards supplied decoded fixtures.
"""
import argparse
from collections import OrderedDict
import copy
import functools
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import socket
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

os.environ['DEV'] = 'CPU'
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
sys.dont_write_bytecode = True

import numpy as np
import torch
from tinygrad import Device


def forbidden(*args, **kwargs):
  raise AssertionError('CPU consumer checks must not open devices or network connections')


def local(name):
  spec = importlib.util.spec_from_file_location(name.replace('-', '_'), Path(__file__).with_name(name+'.py'))
  module = importlib.util.module_from_spec(spec)
  sys.modules[spec.name] = module
  spec.loader.exec_module(module)
  return module


class HostBuffer:
  """Storage double only. Actual CompactInputs.update writes into this memory."""
  def __init__(self, values):
    self.value = np.ascontiguousarray(values)
    self.shape, self.dtype = self.value.shape, self.value.dtype
    self.assignments = 0
  def _buffer(self):
    return self
  def as_memoryview(self, allow_zero_copy=False):
    return memoryview(self.value).cast('B')
  def to(self, destination):
    if destination != 'CPU':
      raise AssertionError('Fake host buffer was sent to a real device')
    return self
  def assign(self, value):
    np.copyto(self.value, value.value, casting='no')
    self.assignments += 1
    return self
  def realize(self):
    return self


class Clock:
  def __init__(self):
    self.now = 1
  def __call__(self):
    self.now += 1000
    return self.now


def exact(left, right):
  if isinstance(left, torch.Tensor):
    if left.shape != right.shape or left.dtype != right.dtype:
      raise AssertionError('Tensor shape/dtype mismatch')
    np.testing.assert_array_equal(left.contiguous().view(torch.uint8).numpy(), right.contiguous().view(torch.uint8).numpy())
  elif isinstance(left, np.ndarray):
    if left.shape != right.shape or left.dtype != right.dtype:
      raise AssertionError('Array shape/dtype mismatch')
    np.testing.assert_array_equal(left.view(np.uint8).reshape(-1), right.view(np.uint8).reshape(-1))
  elif isinstance(left, dict):
    if left.keys() != right.keys():
      raise AssertionError('Dictionary keys differ')
    for key in left:
      exact(left[key], right[key])
  else:
    if left != right:
      raise AssertionError('Values differ')


def prepare_reference(adapter, arrays, settings, compact, padding, length):
  data = {key: value[:] if key == 'nav_text' else torch.from_numpy(value) for key, value in arrays.items()}
  config, _, prepared = adapter.prepare(data)
  packed = adapter.native_inputs(prepared)
  prepared, packed = padding(prepared, packed, length)
  return prepared, packed, compact.arrays(settings, prepared, packed), compact.signature(settings, config, prepared, packed)


def load_case(row):
  arrays = {p.stem: [str(np.load(p, allow_pickle=False).item())] if p.stem == 'nav_text'
            else np.ascontiguousarray(np.load(p, allow_pickle=False)) for p in Path(row['sample']).glob('*.npy')}
  return arrays, copy.deepcopy(row)


def shifted(case, step):
  arrays, meta = copy.deepcopy(case)
  delta = step*100_000_000
  arrays['absolute_timestamps'] += delta//1000
  meta['query_ns'] += delta
  for frame in meta['frames']:
    for key in ('capture_ns', 'requested_ns', 'image_calibration_ns'):
      frame[key] += delta
  return arrays, meta


def fake_model(clock):
  class Model:
    def __init__(self):
      self.padded_prefill = SimpleNamespace()
      self.cache = [SimpleNamespace(uop=SimpleNamespace(base=object()))]
      self.minimum_cache_capacity = 0
      self.trajectory_jits = {}
      self.syncs = 0
      self.fail_sync = False
      self.suffix = 'unwritten'
    def synchronize(self):
      self.syncs += 1
      clock.now += 1_000_000
      if self.fail_sync:
        raise OSError('synthetic completion failure')
    def encode(self, *args):
      raise AssertionError('No model architecture runs in CPU consumer checks')
  return Model()


def cpu_inputs_class(compact):
  class CPUInputs(compact):
    """Only device allocation is replaced. Arrays/signature/update/upload are real."""
    positions = {}
    def __init__(self, args, config, prepared, packed, arrays=None):
      self._signature = self.signature(args, config, prepared, packed)
      self.layout_key = (str(config._name_or_path), prepared['tokenized_data']['image_grid_thw'].numpy().tobytes())
      if self.layout_key not in self.positions:
        if len(self.positions) >= 4:
          self.positions.pop(next(iter(self.positions)))
        self.positions[self.layout_key] = torch.from_numpy(packed[3]).to(torch.bfloat16).view(torch.uint16).numpy().copy()
      self.cpu = [HostBuffer(a.copy()) for a in arrays]
      self.cache_rotary_inputs = args.cache_rotary_inputs
      self.gpu_rotary = {i: HostBuffer(a.copy()) for i, a in enumerate(arrays) if 3 <= i < len(arrays)-1}
      self.rotary_values = {i: a.copy() for i, a in enumerate(arrays) if i in self.gpu_rotary}
      self.rotary_uploaded_bytes = 0
      self.noise = self.cpu[-1].value
      grids = prepared['tokenized_data']['image_grid_thw'].numpy()
      ends = np.r_[0, np.cumsum(grids.prod(axis=1))]
      self.batches = tuple((int(a), int(b), 1) for a, b in zip(ends[:-1], ends[1:]))
      image_ids = np.flatnonzero(packed[0][0] == config.vlm_config.image_token_id)
      self.spans = tuple((int(span[0]), int(span[-1])+1) for span in np.split(image_ids, np.flatnonzero(np.diff(image_ids) != 1)+1))
      self.selective_cache = None
      self.selective_jits = {}
      self.pixel_storage = HostBuffer(np.zeros(self.cpu[0].shape, dtype=np.uint16))
      self.uploaded_pixel_bytes = 0
    @staticmethod
    def rows(tensor, begin, end):
      return HostBuffer(tensor.value[begin:end])
  return CPUInputs


class Checks(unittest.TestCase):
  @classmethod
  def setUpClass(cls):
    cls.live = local('agent-live-consumer')
    cls.compact = local('compact-inputs').CompactInputs
    cls.Vision = local('vision-cache').VisionCache
    cls.pad = staticmethod(local('padded-prefill').pad_inputs)
    cls.policy = local('agent-speculation-policy')
    cls.spec = local('agent-speculation-prefill')
    cls.warmup = local('agent-pipeline-warmup')
    kind = local('fast-inputs').FastInputs
    opts = dict(camera_profile='nvidia', allow_missing_cameras=True, cache_resize=True, direct_byte_inputs=True)
    cls.adapter = kind(ARGS.checkpoint, **opts)
    cls.reference_adapter = kind(ARGS.checkpoint, **opts)
    cls.manifest = json.loads(ARGS.manifest.read_text())
    cls.cases = [load_case(cls.manifest['schedule'][i]) for i in (0, 1, 2, 80, 160, 240)]
    cls.saved_prediction = json.loads(ARGS.prediction_report.read_text())['runs'][0]['prediction']
    cls.parity_calls = cls.upload_arrays_checked = cls.selective_upload_calls = 0
    cls.case_results = []
    cls.provenance_example = None

  def setup_runtime(self, *, speculation=False, active=None, length=2112, budgets=None, old_tokens=None):
    clock = Clock()
    model = fake_model(clock)
    vision = self.Vision(model, capacity=48)
    settings = SimpleNamespace(max_tokens=64, vision_batch=1, byte_pixels=True, cache_rotary_inputs=True,
                               coarse_timing=True, greedy=True, diffusion_steps=3,
                               initial_noise=np.linspace(-.3, .7, 128, dtype=np.float32).reshape(1, 64, 2))
    CPUInputs = cpu_inputs_class(self.compact)
    current = self
    class Decoder:
      def __init__(self):
        self.model = model
        self.history, self.statistics = [], []
        self.jits, self.verifiers, self.select_jits = {}, {}, {}
        self.warmup_graphs = False
        self.calls = 0
        self.original_callback_calls = 0
        self.before_diffusion = self.offline_callback
        self.expected = None
        self.output = copy.deepcopy(current.saved_prediction)
        self.targets = np.array([*current.policy.CHECKPOINT_DRAFT_IDS, *([1]*6)], np.int32)
        self.failure = None
        self.nested = None
      def offline_callback(self):
        self.original_callback_calls += 1
        raise AssertionError('An offline sample prefetch callback must not run')
      def read_tokens(self, payload):
        return payload
      def generate(self, m, args, config, tokenizer, logits, dc, ds, prompt):
        m.suffix = 'serial overwritten suffix'
        result = [42, config.traj_ids['future_start']]
        self.history = (self.history+[result])[-64:]
        self.statistics.append({'serial_tokens': len(result)})
        return result
      def run(self, m, args, config, tokenizer, prepared, packed, seed, inputs):
        self.calls += 1
        if self.nested is not None:
          self.nested()
        if self.failure is not None:
          raise self.failure
        expected_prepared, expected_packed, expected_upload, signature = self.expected
        exact(prepared, expected_prepared)
        for a, b in zip(packed, expected_packed, strict=True):
          exact(a, b)
        if signature != inputs._signature:
          raise AssertionError('Input signature differs from direct recorded helpers')
        for a, b in zip(inputs.cpu, expected_upload, strict=True):
          exact(a.value, b)
        expected_positions = torch.from_numpy(expected_packed[3]).to(torch.bfloat16).view(torch.uint16).numpy()
        exact(inputs.positions[inputs.layout_key], expected_positions)
        current.__class__.upload_arrays_checked += len(expected_upload)
        current.__class__.parity_calls += 1
        if vision.keys != current.Vision.input_keys(expected_prepared) or inputs.selective_cache is not vision:
          raise AssertionError('The canonical content-verified vision cache was bypassed')
        missing = tuple(i for i, key in enumerate(vision.keys) if vision.verify or key not in vision.cache)
        self.last_missing = missing
        def upload_selected(*sources):
          if len(sources) != len(missing)+len(inputs.cpu)-1:
            raise AssertionError('Selective source count changed')
          for i, source in zip(missing, sources[:len(missing)], strict=True):
            begin, end, _ = inputs.batches[i]
            exact(source.value, expected_upload[0][begin:end])
            inputs.pixel_storage.value[begin:end] = current.compact.pixel_table[source.value]
          for index, source in enumerate(sources[len(missing):], 1):
            expected = settings.initial_noise if index == len(inputs.cpu)-1 else expected_upload[index]
            exact(source.value, expected)
          return inputs.pixel_storage
        inputs.selective_jits[missing] = upload_selected
        # Execute the actual CompactInputs.upload selection and source assembly.
        inputs.upload(settings.initial_noise)
        current.__class__.selective_upload_calls += 1
        expected_bytes = sum((inputs.batches[i][1]-inputs.batches[i][0])*inputs.cpu[0].shape[1] for i in missing)
        if inputs.uploaded_pixel_bytes != expected_bytes:
          raise AssertionError('Selective upload byte accounting changed')
        for key in vision.keys:
          vision.cache[key] = 'fake feature payload'
          vision.cache.move_to_end(key)
        while len(vision.cache) > vision.capacity:
          vision.cache.popitem(last=False)
        if speculation:
          model.padded_prefill.last_prompt = prepared['actual_prompt_length']
          model.padded_prefill.last_targets = self.targets if prepared['prefill_draft_eligible'] else None
          model.suffix = 'accepted speculative suffix'
          generated = self.generate(m, args, config, tokenizer, object(), object(), object(), prepared['actual_prompt_length'])
          self.output = dict(self.output, generated_token_ids=generated, generated_tokens=len(generated),
                             reasoning=tokenizer.decode(generated))
        else:
          self.statistics.append({'ordinary_stub': True})
        self.before_diffusion()
        clock.now += 4_000_000
        return self.output
    if active is not None:
      startup = self.warmup.StartupWarmup(lambda: active[0])
      startup.patch_decoder(Decoder)
    decoder = Decoder()
    padding = self.pad
    if speculation:
      model.padded_prefill = SimpleNamespace(last_prompt=None, last_targets=None, prefix_audit=None, audit=False)
      self.spec.install_decoder(decoder, self.policy.CHECKPOINT_DRAFT_IDS, self.adapter.config.traj_ids['future_start'])
      padding = functools.partial(self.policy.padded_inputs, draft=self.policy.CHECKPOINT_DRAFT_IDS,
                                  anchor=self.adapter.config.traj_ids['future_start'])
    previous_inputs = None
    if old_tokens is not None:
      old_settings = SimpleNamespace(**{**vars(settings), 'max_tokens': old_tokens})
      p, k, a, _ = prepare_reference(self.reference_adapter, self.cases[0][0], old_settings, self.compact, padding, length)
      previous_inputs = CPUInputs(old_settings, self.adapter.config, p, k, arrays=a)
    consumer = self.live.Consumer(adapter=self.adapter, model=model, runner=decoder.run, settings=settings,
      config=self.adapter.config, input_class=CPUInputs, vision_cache=vision, pad_inputs=padding, pad_length=length,
      decoder=decoder, inputs=previous_inputs, speculation_policy=self.policy if speculation else None,
      speculation_draft=self.policy.CHECKPOINT_DRAFT_IDS if speculation else None,
      initializer_provenance={'test_only': True, 'gpu': 'fake', 'checkpoint': str(ARGS.checkpoint),
                              'config_sha256': hashlib.sha256((ARGS.checkpoint/'config.json').read_bytes()).hexdigest()},
      budgets=budgets or self.live.Budgets(), clock=clock)
    return SimpleNamespace(consumer=consumer, model=model, vision=vision, decoder=decoder, clock=clock,
                            settings=settings, padding=padding, length=length)

  def run_case(self, runtime, case, *, complete=True):
    arrays, metadata = copy.deepcopy(case)
    runtime.clock.now = max(runtime.clock.now, metadata['query_ns']+30_000_000)
    metadata.update(selected_ns=runtime.clock.now-2_000_000, prepared_ns=runtime.clock.now-1_000_000,
                    navigation_text=arrays['nav_text'][0])
    runtime.decoder.expected = prepare_reference(self.reference_adapter, arrays, runtime.settings, self.compact,
                                                 runtime.padding, runtime.length)
    before_output = runtime.decoder.output
    with patch.object(np, 'load', side_effect=AssertionError('Consumer attempted filesystem sample replay')):
      result = runtime.consumer.predict(arrays, metadata)
      if complete:
        runtime.consumer.synchronize()
    self.assertIs(result['prediction'], runtime.decoder.output)
    if runtime.consumer.policy is None:
      self.assertIs(result['prediction'], before_output)
    self.assertIs(metadata['live_consumer'], result['live_consumer'])
    self.assertEqual(result['query_ns'], metadata['query_ns'])
    self.assertEqual(result['live_consumer']['source_capture_timestamps_ns'], [ref['capture_ns'] for ref in metadata['frames']])
    return result, metadata

  def test_recorded_windows_complete_upload_parity_and_output_passthrough(self):
    runtime = self.setup_runtime()
    for case in self.cases:
      result, meta = self.run_case(runtime, case)
      self.assertTrue(result['live_consumer']['completed'])
      self.assertEqual(result['prediction'], self.saved_prediction)
      self.assertGreater(result['live_consumer']['query_to_completion_ms'], 0)
      self.case_results.append({'query_ns': meta['query_ns'], 'all_upload_arrays_exact': True})
    self.__class__.provenance_example = runtime.consumer.provenance
    self.assertEqual(runtime.decoder.original_callback_calls, 0)
    self.assertGreater(runtime.consumer.reused_inputs, 0)
    self.assertEqual(runtime.model.syncs, len(self.cases)+1)
    self.assertEqual(runtime.consumer.records[-1]['owned_input_objects'], 1)

  def test_same_timestamp_content_mutation_invalidates_exact_image_cache(self):
    runtime = self.setup_runtime()
    self.run_case(runtime, self.cases[0])
    old_keys = runtime.vision.keys
    original = self.cases[0][1]['frames']
    changed = copy.deepcopy(self.cases[1])
    shared = next((i, j) for i, new in enumerate(changed[1]['frames']) for j, old in enumerate(original)
                  if new['view'] == old['view'] and new['capture_ns'] == old['capture_ns'])
    i, j = shared
    changed[0]['image_frames'][i, 0, :, 80:180, 180:280] ^= 255
    self.run_case(runtime, changed)
    self.assertNotEqual(runtime.vision.keys[i], old_keys[j])
    self.assertIn(i, runtime.decoder.last_missing)
    self.assertEqual(changed[1]['frames'][i]['capture_ns'], original[j]['capture_ns'])

  def test_live_calibration_history_navigation_changes(self):
    runtime = self.setup_runtime()
    self.run_case(runtime, self.cases[0])
    changed = copy.deepcopy(self.cases[1])
    changed[0]['nav_text'] = ['Turn left at the next intersection, then continue for 200 meters.']
    changed[0]['ego_history_xyz'][0, 0, 0] -= .8
    from scipy.spatial.transform import Rotation
    changed[0]['ego_history_rot'][0, 0] = Rotation.from_euler('z', .05).as_matrix().astype(np.float32)
    from calibration_trials import calibrated_image
    ref = changed[1]['frames'][-1]
    raw = np.ascontiguousarray(np.load(ARGS.raw_frames/str(ref['view'])/f'{ref["video_index"]}.npy', allow_pickle=False))
    calibration = {'rpyCalib': [0., -.04, -.07], 'wideFromDeviceEuler': [0., 0., 0.]}
    changed[0]['image_frames'][-1, 0] = calibrated_image(raw, 6, self.manifest['source_camera_geometry'],
                                                       self.manifest['target_camera_geometry'], calibration)
    ref['image_calibration_ns'] = ref['capture_ns']-1
    result, _ = self.run_case(runtime, changed)
    self.assertIn(9, runtime.decoder.last_missing)
    self.assertTrue(result['live_consumer']['completed'])

  def test_incompatible_existing_signature_and_same_size_wrong_dtype(self):
    runtime = self.setup_runtime(old_tokens=128)
    previous = runtime.consumer.inputs
    first, _ = self.run_case(runtime, self.cases[0])
    self.assertFalse(first['live_consumer']['input_reused'])
    self.assertIsNot(runtime.consumer.inputs, previous)
    # Exercise the additional dtype guard before canonical raw-memory update.
    runtime.consumer.inputs.cpu[2] = HostBuffer(runtime.consumer.inputs.cpu[2].value.astype(np.float32))
    current = runtime.consumer.inputs
    second, _ = self.run_case(runtime, self.cases[1])
    self.assertFalse(second['live_consumer']['input_reused'])
    self.assertIsNot(runtime.consumer.inputs, current)
    from tinygrad import dtypes
    for value, expected in ((dtypes.uint8, 'uint8'), (dtypes.uint16, 'uint16'),
                            (dtypes.int32, 'int32'), (dtypes.float32, 'float32')):
      self.assertEqual(self.live.Consumer._cpu_dtype(SimpleNamespace(dtype=value)), expected)

  def test_optional_actual_cpu_cache_adapter_preserves_uploads(self):
    cached = local('agent-cpu-inputs-adapter').FastInputs(ARGS.checkpoint, camera_profile='nvidia',
      allow_missing_cameras=True, cache_resize=True, direct_byte_inputs=True)
    with patch.object(self, 'adapter', cached):
      runtime = self.setup_runtime()
      self.run_case(runtime, self.cases[0])
      changed = copy.deepcopy(self.cases[1])
      changed[0]['image_frames'][0, 0, 1, 40:90, 40:90] ^= 255
      changed[0]['nav_text'] = ['Stop at the next safe location and wait for further instructions.']
      self.run_case(runtime, changed)
      self.assertGreater(cached.cpu_input_statistics['patch_hits'], 0)

  def test_completion_ownership_and_idempotent_sync(self):
    runtime = self.setup_runtime()
    result, _ = self.run_case(runtime, self.cases[0], complete=False)
    self.assertFalse(result['live_consumer']['completed'])
    with self.assertRaisesRegex(RuntimeError, 'still active'):
      self.run_case(runtime, self.cases[1])
    runtime.consumer.synchronize()
    self.assertTrue(result['live_consumer']['completed'])
    syncs = runtime.model.syncs
    runtime.consumer.synchronize()
    self.assertEqual(runtime.model.syncs, syncs)
    self.assertGreaterEqual(result['live_consumer']['completion_sync_ms'], 1)

  def test_stale_queries_and_mismatched_source_metadata_fail_closed(self):
    runtime = self.setup_runtime()
    self.run_case(runtime, self.cases[0])
    with self.assertRaisesRegex(ValueError, 'stale'):
      self.run_case(runtime, self.cases[0])
    with self.assertRaisesRegex(RuntimeError, 'failed'):
      self.run_case(runtime, self.cases[1])
    runtime = self.setup_runtime()
    changed = copy.deepcopy(self.cases[0])
    changed[1]['frames'][0]['capture_ns'] += 1_000_000
    with self.assertRaisesRegex(ValueError, 'Frame timestamp'):
      self.run_case(runtime, changed)
    self.assertEqual(runtime.decoder.calls, 0)

  def test_settings_change_and_runner_errors_propagate_with_sync(self):
    runtime = self.setup_runtime()
    runtime.settings.cache_rotary_inputs = False
    with self.assertRaisesRegex(ValueError, 'settings'):
      self.run_case(runtime, self.cases[0])
    runtime = self.setup_runtime()
    failure = ArithmeticError('deliberate runner failure')
    runtime.decoder.failure = failure
    with self.assertRaises(ArithmeticError) as caught:
      self.run_case(runtime, self.cases[0])
    self.assertIs(caught.exception, failure)
    self.assertEqual(runtime.model.syncs, 2)
    self.assertFalse(runtime.consumer.records[-1]['completed'])

  def test_adapter_error_and_secondary_completion_error_preserve_primary_exception(self):
    runtime = self.setup_runtime()
    failure = LookupError('deliberate live adapter failure')
    runtime.model.fail_sync = True
    with patch.object(self.adapter, 'prepare', side_effect=failure):
      with self.assertRaises(LookupError) as caught:
        self.run_case(runtime, self.cases[0])
    self.assertIs(caught.exception, failure)
    self.assertTrue(any('synchronization also failed' in note for note in failure.__notes__))
    self.assertEqual(runtime.decoder.calls, 0)

  def test_premature_or_foreign_thread_synchronization_is_rejected(self):
    runtime = self.setup_runtime()
    def nested():
      with self.assertRaisesRegex(RuntimeError, 'Complete predict'):
        runtime.consumer.synchronize()
    runtime.decoder.nested = nested
    self.run_case(runtime, self.cases[0], complete=False)
    import threading
    caught = []
    def foreign():
      try:
        runtime.consumer.synchronize()
      except RuntimeError as error:
        caught.append(str(error))
    thread = threading.Thread(target=foreign)
    thread.start()
    thread.join(timeout=2)
    self.assertFalse(thread.is_alive())
    self.assertEqual(len(caught), 1)
    runtime.consumer.synchronize()
    self.assertTrue(runtime.consumer.records[-1]['completed'])

  def test_actual_bridge_run_once_consumer_contract(self):
    bridge = local('agent-live-bridge')
    runtime = self.setup_runtime()
    arrays, row = copy.deepcopy(self.cases[0])
    store = bridge.Store()
    for i, ref in enumerate(row['frames']):
      pixels = arrays['image_frames'][i, 0].copy()
      pixels.flags.writeable = False
      stamp = ref['capture_ns']
      store.add_frame(bridge.Frame(ref['view'], ref['frame_id'], stamp, stamp+5_000_000,
                                  stamp+10_000_000, stamp+11_000_000, pixels))
    class PreparedLive:
      def prepare(self, selection):
        runtime.clock.now += 6_000_000
        return arrays, {'frames': row['frames']}
    runtime.clock.now = row['query_ns']+30_000_000
    runtime.decoder.expected = prepare_reference(self.reference_adapter, arrays, runtime.settings, self.compact,
                                                 runtime.padding, runtime.length)
    scheduler = bridge.Bridge(store, PreparedLive(), clock=runtime.clock, navigation=arrays['nav_text'][0])
    with patch.object(np, 'load', side_effect=AssertionError('Bridge/consumer attempted file replay')):
      result = scheduler.run_once(runtime.consumer)
    self.assertIsNotNone(result)
    self.assertIs(result['prediction'], runtime.decoder.output)
    self.assertTrue(result['input']['live_consumer']['completed'])
    self.assertEqual(result['input']['query_ns'], row['query_ns'])
    self.assertGreater(result['timing']['query_to_output_ms'], result['input']['live_consumer']['query_to_completion_ms'])
    self.assertIsNone(scheduler.run_once(runtime.consumer))
    self.assertEqual(runtime.decoder.calls, 1)

  def test_completion_failure_retains_owner_and_blocks_reuse(self):
    runtime = self.setup_runtime()
    self.run_case(runtime, self.cases[0], complete=False)
    runtime.model.fail_sync = True
    with self.assertRaisesRegex(OSError, 'completion'):
      runtime.consumer.synchronize()
    self.assertFalse(runtime.consumer.records[-1]['completed'])
    owner = runtime.consumer.inputs
    runtime.consumer.close()
    self.assertIs(runtime.consumer.inputs, owner)
    with self.assertRaisesRegex(RuntimeError, 'closed or failed'):
      self.run_case(runtime, self.cases[1])

  def test_fixed_vision_positions_content_change_is_rejected(self):
    runtime = self.setup_runtime()
    self.run_case(runtime, self.cases[0])
    original = self.adapter.native_inputs
    def changed(prepared):
      ids, pos, delta, vision = original(prepared)
      vision = vision.copy()
      vision.flat[0] += 1
      return ids, pos, delta, vision
    with patch.object(self.adapter, 'native_inputs', changed):
      with self.assertRaisesRegex(ValueError, 'vision-position contents'):
        self.run_case(runtime, self.cases[1])

  def test_actual_speculative_hook_acceptance_and_every_rejection(self):
    runtime = self.setup_runtime(speculation=True)
    for index, mismatch in enumerate((None, *range(6), 'finite')):
      runtime.decoder.targets = np.array([*self.policy.CHECKPOINT_DRAFT_IDS, *([1]*6)], np.int32)
      if mismatch == 'finite':
        runtime.decoder.targets[6] = 0
      elif mismatch is not None:
        runtime.decoder.targets[mismatch] += 1
      result, _ = self.run_case(runtime, shifted(self.cases[0], index))
      self.assertEqual(result['live_consumer']['speculation']['accepted'], mismatch is None)
      self.assertEqual(runtime.model.suffix, 'accepted speculative suffix' if mismatch is None else 'serial overwritten suffix')
    self.assertEqual(runtime.decoder.original_callback_calls, 0)

  def test_speculation_ineligible_padding_preserves_original_fallback(self):
    runtime = self.setup_runtime()
    self.run_case(runtime, self.cases[0])
    actual = runtime.consumer.records[-1]['actual_prompt_tokens']
    runtime = self.setup_runtime(speculation=True, length=actual+3)
    result, _ = self.run_case(runtime, self.cases[0])
    self.assertFalse(result['live_consumer']['speculation']['eligible'])
    self.assertEqual(result['prediction']['generated_token_ids'], [42, self.adapter.config.traj_ids['future_start']])

  def test_discovered_startup_gate_blocks_speculative_kv_overwrite(self):
    active = [True]
    with self.assertRaisesRegex(RuntimeError, 'StartupWarmup'):
      self.setup_runtime(speculation=True, active=active)
    active[0] = False
    runtime = self.setup_runtime(speculation=True, active=active)
    self.assertGreater(len(runtime.consumer.startup_helpers), 0)
    self.run_case(runtime, self.cases[0])
    self.assertEqual(runtime.model.suffix, 'accepted speculative suffix')
    active[0] = True
    with self.assertRaisesRegex(RuntimeError, 'StartupWarmup'):
      self.run_case(runtime, self.cases[1])
    self.assertEqual(runtime.decoder.calls, 1)
    self.assertEqual(runtime.model.suffix, 'accepted speculative suffix')

  def test_wrong_padding_hook_and_external_prefetch_are_rejected(self):
    runtime = self.setup_runtime(speculation=True)
    runtime.consumer.pad_inputs = self.pad
    with self.assertRaisesRegex(ValueError, 'speculative pad_inputs'):
      self.run_case(runtime, self.cases[0])
    runtime = self.setup_runtime()
    runtime.decoder.before_diffusion = runtime.decoder.offline_callback
    with self.assertRaisesRegex(RuntimeError, 'before_diffusion'):
      self.run_case(runtime, self.cases[0])
    self.assertEqual(runtime.decoder.original_callback_calls, 0)

  def test_diagnostic_records_bounded_without_truncating_text_history(self):
    runtime = self.setup_runtime(speculation=True, budgets=self.live.Budgets(records=2))
    for index in range(5):
      self.run_case(runtime, shifted(self.cases[0], index))
    self.assertEqual(len(runtime.consumer.records), 2)
    self.assertEqual(len(runtime.decoder.statistics), 2)
    self.assertEqual(len(runtime.decoder.prefill_draft_records), 2)
    self.assertEqual(runtime.decoder.history, [list(self.policy.CHECKPOINT_DRAFT_IDS)])
    self.assertEqual(runtime.consumer.constructed_inputs, 1)
    runtime.vision.encoders = dict.fromkeys(range(17))
    with self.assertRaisesRegex(RuntimeError, 'declared live bound'):
      self.run_case(runtime, shifted(self.cases[0], 6))

  def test_kv_replacement_blocks_reusing_captured_graphs(self):
    runtime = self.setup_runtime()
    self.run_case(runtime, self.cases[0])
    runtime.model.cache = [SimpleNamespace(uop=SimpleNamespace(base=object()))]
    with self.assertRaisesRegex(ValueError, 'KV allocation changed'):
      self.run_case(runtime, self.cases[1])
    self.assertEqual(runtime.decoder.calls, 1)

  def test_prediction_remains_untouched_and_incomplete_output_is_rejected(self):
    runtime = self.setup_runtime()
    before = copy.deepcopy(runtime.decoder.output)
    result, _ = self.run_case(runtime, self.cases[0])
    self.assertEqual(result['prediction'], before)
    self.assertNotIn('live_consumer', result['prediction'])
    runtime.decoder.output = dict(before, xyz=[[[float('nan')]*3]*64])
    with self.assertRaisesRegex(ValueError, 'decoded trajectory'):
      self.run_case(runtime, self.cases[1])


def main():
  global ARGS
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--checkpoint', type=Path, required=True)
  parser.add_argument('--manifest', type=Path, required=True)
  parser.add_argument('--raw-frames', type=Path, required=True)
  parser.add_argument('--prediction-report', type=Path, required=True)
  parser.add_argument('--output', type=Path)
  ARGS = parser.parse_args()
  torch.set_num_threads(2)
  torch.set_num_interop_threads(1)
  with patch.object(type(Device), '__getitem__', forbidden), patch.object(socket.socket, 'connect', forbidden), \
       patch.object(socket.socket, 'connect_ex', forbidden):
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Checks))
  if not result.wasSuccessful():
    raise SystemExit(1)
  if Device._opened_devices:
    raise AssertionError('CPU checks unexpectedly initialized a tinygrad backend')
  report = dict(passed=True, tests=result.testsRun, cpu_only=True, initialized_tinygrad_devices=[],
                network_connections=False, actual_model_inference=False,
                actual_adapter_calls_with_exact_parity=Checks.parity_calls,
                exact_upload_arrays=Checks.upload_arrays_checked, actual_compact_selective_upload_calls=Checks.selective_upload_calls,
                recorded_windows=Checks.case_results, speculation='actual installed decoder wrapper; fake target payload and KV storage',
                unchanged_prediction='Real recorded decoded result passed through fake runner by identity; no model-output parity claim',
                fixtures={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in (ARGS.manifest, ARGS.prediction_report)},
                consumer_provenance_example=Checks.provenance_example,
                sources_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('agent-live-consumer*.py')})
  if ARGS.output:
    with ARGS.output.open('x') as stream:
      stream.write(json.dumps(report, indent=2)+'\n')
  print(json.dumps(report, indent=2))


if __name__ == '__main__':
  main()
