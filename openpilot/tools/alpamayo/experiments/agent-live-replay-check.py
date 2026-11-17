"""CPU checks for offline qualification; every device open and socket is blocked."""
import argparse
from contextlib import ExitStack, redirect_stdout
import copy
import importlib.machinery
import importlib.util
import io
import json
import os
from pathlib import Path
import socket
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

os.environ['DEV'] = 'CPU'
os.environ['HF_HUB_OFFLINE'] = os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import numpy as np
from tinygrad import Device


def local(name):
  spec = importlib.util.spec_from_file_location(name.replace('-', '_'), ROOT/(name+'.py'))
  module = importlib.util.module_from_spec(spec)
  sys.modules[spec.name] = module
  spec.loader.exec_module(module)
  return module


def forbidden(*a, **kw):
  raise AssertionError('CPU qualification cannot open any device or network connection')


class Checks(unittest.TestCase):
  @classmethod
  def setUpClass(cls):
    cls.replay = local('agent-live-replay')
    cls.runtime = local('agent-live-runtime')
    cls.pipeline = local('benchmark-pipeline')
    cls.factory_checks = local('agent-live-runtime-check')
    cls.dataset = cls.replay.Dataset(ARGS.samples_json, ARGS.source_manifest)
    cls.normalized, cls.args, cls.protocol = cls.replay.normalize(ARGS.invocation, cls.dataset, cls.pipeline, Path('/virtual/report.benchmark.json'))
    cls.helpers = local('agent-live-consumer-check')
    cls.helpers.ARGS = SimpleNamespace(checkpoint=ARGS.checkpoint)
    base = cls.helpers.Checks
    base.live = local('agent-live-consumer')
    base.compact = local('compact-inputs').CompactInputs
    base.Vision = local('vision-cache').VisionCache
    base.pad = staticmethod(local('padded-prefill').pad_inputs)
    base.policy, base.spec, base.warmup = local('agent-speculation-policy'), local('agent-speculation-prefill'), local('agent-pipeline-warmup')
    kind = local('fast-inputs').FastInputs
    opts = dict(camera_profile='nvidia', allow_missing_cameras=True, cache_resize=True, direct_byte_inputs=True)
    base.adapter, base.reference_adapter = kind(ARGS.checkpoint, **opts), kind(ARGS.checkpoint, **opts)
    base.cases = [(cls.dataset.load(row), copy.deepcopy(row)) for row in cls.dataset.rows[:3]]
    base.saved_prediction = json.loads(ARGS.prediction_report.read_text())['runs'][0]['prediction']
    base.parity_calls = base.upload_arrays_checked = base.selective_upload_calls = 0
    cls.checker = base(methodName='test_recorded_windows_complete_upload_parity_and_output_passthrough')
    cls.events = []

  def fixture(self, cases):
    arrays = [case[0] for case in cases]
    rows = [copy.deepcopy(case[1]) for case in cases]
    for i, row in enumerate(rows):
      row['sample'] = f'<independent CPU event {i}>'
    return SimpleNamespace(rows=rows, samples=[r['sample'] for r in rows], inputs=[self.replay.tree(a) for a in arrays],
                           load=lambda row: arrays[rows.index(row)])

  def bound(self, stack, dataset, *, speculative=False, old_tokens=None):
    fake = self.checker.setup_runtime(speculation=speculative, old_tokens=old_tokens)
    prior_inputs = fake.consumer.inputs
    fake.consumer.close()
    components = SimpleNamespace(adapter=self.checker.adapter, model=fake.model, runner=fake.decoder.run,
      settings=fake.settings, config=self.checker.adapter.config, tokenizer=self.checker.adapter.tokenizer,
      input_class=fake.consumer.input_class, vision_cache=fake.vision, pad_inputs=fake.padding,
      inputs=prior_inputs, decoder=fake.decoder, startup=None, startup_active=[False], profiler=None,
      provenance={'cpu_test': True}, vision_provenance={'gpu': 'fake'}, report={'source_sha256': {'cpu_test': True}})
    args = copy.copy(self.args)
    args.coarse_timing = args.cache_rotary_inputs = True
    args.collect_garbage = False
    args.verify_vision_cache = args.verify_vision_warmup = False
    # Match the canonical noise in the real invocation, not a guessed seed.
    fake.settings.initial_noise = np.load(args.initial_noise, allow_pickle=False)
    original_prepare = components.adapter.prepare
    stack.enter_context(patch.object(components.adapter, 'prepare', original_prepare))
    audit = self.replay.Audit(dataset)
    original_load = dataset.load
    def load(row):
      arrays = original_load(row)
      fake.decoder.expected = self.helpers.prepare_reference(self.checker.reference_adapter, arrays, fake.settings,
        self.checker.compact, fake.padding, args.pad_prompt_to)
      return arrays
    stack.enter_context(patch.object(dataset, 'load', load))
    audit.install(components)
    clock = self.replay.ReplayClock()
    consumer = self.runtime.bind_components(components, self.replay.binding_args(args), consumer_module=self.checker.live,
      speculation_policy=self.checker.policy if speculative else None,
      speculation_draft=self.checker.policy.CHECKPOINT_DRAFT_IDS if speculative else None, clock=clock)
    return SimpleNamespace(fake=fake, components=components, args=args, audit=audit, consumer=consumer, clock=clock)

  def run_bound(self, runtime, dataset):
    with patch.object(np, 'load', side_effect=AssertionError('Consumer replay must consume current in-memory arrays')):
      self.replay.replay_consumer(runtime.components, runtime.args, dataset, runtime.audit,
                                 consumer=runtime.consumer, clock=runtime.clock)

  def test_original_64_floor_metadata_and_exact_history(self):
    self.assertEqual(len(self.dataset.inputs), 64)
    self.assertEqual(len({r['query_ns'] for r in self.dataset.rows}), 64)
    self.assertTrue(all(a['query_ns'] < b['query_ns'] for a, b in zip(self.dataset.rows, self.dataset.rows[1:])))
    self.assertTrue(all(len(r['frames']) == 10 for r in self.dataset.rows))
    self.assertNotIn('recorded_motion_future_xyz', self.dataset.rows[0])
    self.assertEqual(self.protocol['files']['initial_noise']['sha256'], self.replay.sha(self.args.initial_noise))

  def test_only_overlap_and_startup_flags_are_disabled(self):
    flags = {d['flag'] for d in self.protocol['disabled_flags']}
    self.assertEqual(flags, {'--pipeline-inputs', '--freeze-warmup-gc'})
    self.assertEqual((self.args.warmup, self.args.runs, self.args.steps), (2, 62, 3))
    self.assertEqual(self.args.pad_prompt_to, 2112)
    self.assertTrue(self.args.mmap_inputs)
    changed = copy.deepcopy(self.normalized)
    changed['command'] += ['--verify-vision-warmup', '--verify-vision-cache']
    with patch.object(Path, 'read_text', return_value=json.dumps(changed)):
      _, args, _ = self.replay.normalize(ARGS.invocation, self.dataset, self.pipeline, Path('/virtual/other.json'))
    self.assertTrue(args.verify_vision_cache)
    self.assertTrue(args.verify_vision_warmup)

  def test_missing_native_metadata_and_nonfloor_frame_fail(self):
    with self.assertRaises((KeyError, ValueError)):
      self.replay.ReplayClock().metadata({'frames': []})
    from calibration_trials import asof
    row = copy.deepcopy(self.dataset.rows[0])
    native = {c: (np.array([r['capture_ns'] for r in row['frames'] if r['view'] == c]),
                  [dict(r) for r in row['frames'] if r['view'] == c]) for c in (1, 6)}
    for i, ref in enumerate(row['frames']):
      ref['video_index'] = i % 5
    stamps = sorted({r['image_calibration_ns'] for r in row['frames']})
    calibrations = (np.asarray(stamps), [dict(ns=s) for s in stamps])
    self.replay.Dataset.validate_native(row, native, calibrations, asof)
    row['frames'][0]['capture_ns'] += 1
    with self.assertRaisesRegex(ValueError, 'causal floor'):
      self.replay.Dataset.validate_native(row, native, calibrations, asof)
    row = copy.deepcopy(self.dataset.rows[0])
    del row['frames'][0]['image_calibration_ns']
    with self.assertRaises((KeyError, ValueError, IndexError)):
      self.replay.Dataset.validate_native(row, native, calibrations, asof)

  def test_reject_ring_or_altered_original_manifest(self):
    with patch.object(self.replay, 'sha', return_value='not the pinned source'):
      with self.assertRaisesRegex(ValueError, 'exact original'):
        self.replay.Dataset(ARGS.samples_json, ARGS.source_manifest)

  def test_actual_consumer_arrays_and_canonical_output_are_unchanged(self):
    dataset = self.fixture(self.checker.cases)
    with ExitStack() as stack:
      runtime = self.bound(stack, dataset)
      before = copy.deepcopy(runtime.fake.decoder.output)
      self.run_bound(runtime, dataset)
      self.assertEqual(len(runtime.audit.rows), 3)
      self.assertEqual(runtime.fake.decoder.output, before)
      self.assertEqual([r['query_ns'] for r in runtime.audit.rows], [r['query_ns'] for r in dataset.rows])
      self.assertTrue(all(r['completed'] for r in runtime.audit.rows))
      self.assertTrue(all(r['prediction'] == before for r in runtime.audit.rows))
      self.assertNotIn('query_to_completion_ms', runtime.audit.rows[-1]['consumer_protocol'])
      self.assertEqual(runtime.consumer.last_query_ns, dataset.rows[-1]['query_ns'])

  def test_changed_navigation_history_and_content_are_current(self):
    original = copy.deepcopy(self.checker.cases[0])
    changed = copy.deepcopy(original)
    arrays, row = changed
    row['query_ns'] += 100_000_000
    for ref in row['frames']:
      ref['requested_ns'] += 100_000_000
    row['frames'][-1]['capture_ns'] = row['query_ns']
    arrays['absolute_timestamps'][-1] = row['query_ns']//1000
    row['navigation_text'] = 'Turn left at the next intersection.'
    arrays['nav_text'] = [row['navigation_text']]
    arrays['ego_history_xyz'][0, 0, 0] -= .7
    arrays['image_frames'][0, 0, :, 20:90, 20:90] ^= 255
    row['frames'][0]['image_calibration_ns'] += 1
    dataset = self.fixture([original, changed])
    with ExitStack() as stack:
      runtime = self.bound(stack, dataset)
      self.run_bound(runtime, dataset)
      a, b = runtime.audit.rows
      self.assertNotEqual(a['vision_keys'][0], b['vision_keys'][0])
      self.assertNotEqual(a['cpu_buffers'][0], b['cpu_buffers'][0])
      self.assertNotEqual(a['cpu_buffers'][2], b['cpu_buffers'][2])
      self.assertEqual(original[1]['frames'][0]['capture_ns'], changed[1]['frames'][0]['capture_ns'])

  def test_incompatible_input_signature_reallocates_once(self):
    dataset = self.fixture(self.checker.cases[:2])
    with ExitStack() as stack:
      runtime = self.bound(stack, dataset, old_tokens=128)
      old = runtime.consumer.inputs
      self.run_bound(runtime, dataset)
      self.assertFalse(runtime.audit.rows[0]['consumer_protocol']['input_reused'])
      self.assertTrue(runtime.audit.rows[1]['consumer_protocol']['input_reused'])
      self.assertEqual(runtime.consumer.constructed_inputs, 1)
      self.assertIsNot(runtime.consumer.inputs, old)

  def test_failed_completion_never_accepts_output_or_reuses_owner(self):
    dataset = self.fixture(self.checker.cases[:2])
    with ExitStack() as stack:
      runtime = self.bound(stack, dataset)
      runtime.fake.model.fail_sync = True
      with self.assertRaisesRegex(OSError, 'completion'):
        self.run_bound(runtime, dataset)
      self.assertEqual(runtime.audit.rows, [])
      self.assertFalse(runtime.audit.pending['completed'])
      owner = runtime.consumer.inputs
      runtime.consumer.close()
      self.assertIs(runtime.consumer.inputs, owner)
      self.assertEqual(runtime.fake.decoder.calls, 1)

  def test_runner_error_propagates_without_an_accepted_row(self):
    dataset = self.fixture(self.checker.cases[:1])
    with ExitStack() as stack:
      runtime = self.bound(stack, dataset)
      runtime.fake.decoder.failure = ArithmeticError('independent decoder failure')
      with self.assertRaisesRegex(ArithmeticError, 'independent'):
        self.run_bound(runtime, dataset)
      self.assertEqual(runtime.audit.rows, [])
      self.assertFalse(runtime.audit.pending['completed'])

  def test_kv_reallocation_fails_completion_and_blocks_result(self):
    dataset = self.fixture(self.checker.cases[:1])
    with ExitStack() as stack:
      runtime = self.bound(stack, dataset)
      runtime.fake.decoder.nested = lambda: setattr(runtime.fake.model, 'cache', [SimpleNamespace(uop=SimpleNamespace(base=object()))])
      with self.assertRaisesRegex(ValueError, 'reallocated existing KV'):
        self.run_bound(runtime, dataset)
      self.assertEqual(runtime.audit.rows, [])
      self.assertFalse(runtime.audit.pending['completed'])

  def test_changed_snapshot_after_preflight_is_rejected(self):
    case = copy.deepcopy(self.checker.cases[0])
    dataset = self.fixture([case])
    case[0]['image_frames'][0, 0, 0, 30:50, 30:50] ^= 255
    with ExitStack() as stack:
      runtime = self.bound(stack, dataset)
      with self.assertRaisesRegex(ValueError, 'exact original ordered snapshots'):
        self.run_bound(runtime, dataset)
      self.assertEqual(runtime.fake.decoder.calls, 0)
      self.assertEqual(runtime.audit.rows, [])

  def test_actual_verified_speculation_acceptance_and_fallback(self):
    for accepted in (True, False):
      dataset = self.fixture(self.checker.cases[:2])
      with ExitStack() as stack:
        runtime = self.bound(stack, dataset, speculative=True)
        if not accepted:
          runtime.fake.decoder.targets[2] += 1
        self.run_bound(runtime, dataset)
        self.assertEqual([r['speculation']['accepted'] for r in runtime.audit.rows], [accepted]*2)
        self.assertEqual(runtime.fake.model.suffix, 'accepted speculative suffix' if accepted else 'serial overwritten suffix')
        self.assertTrue(all(r['history_after']['history'] for r in runtime.audit.rows))
        self.assertEqual(runtime.fake.decoder.original_callback_calls, 0)

  def test_ineligible_speculation_keeps_original_decoder(self):
    # Canonical padding policy decides eligibility; no fabricated KV is supplied.
    runtime = self.checker.setup_runtime()
    self.checker.run_case(runtime, self.checker.cases[0])
    actual = runtime.consumer.records[-1]['actual_prompt_tokens']
    runtime.consumer.close()
    runtime = self.checker.setup_runtime(speculation=True, length=actual)
    result, _ = self.checker.run_case(runtime, self.checker.cases[0])
    self.assertFalse(result['live_consumer']['speculation']['eligible'])
    self.assertEqual(runtime.decoder.prefill_draft_records[-1]['generated'], [42, self.checker.adapter.config.traj_ids['future_start']])

  def test_exact_comparison_checks_every_output_and_verification_field(self):
    prediction = copy.deepcopy(self.checker.saved_prediction)
    rows = [dict(index=i, sample=str(i), query_ns=i+1, completed=True, prediction=prediction,
                 history_before=[], history_after=[], decoder_statistics={'serial_tokens': 2},
                 speculation={'accepted': False}) for i in range(64)]
    reference = dict(mode='reference', protocol=self.protocol, decoder='speculative', prefill_attention='baseline', initialized={}, rows=rows)
    candidate = copy.deepcopy(reference)
    candidate['mode'] = 'consumer'
    self.assertTrue(self.replay.compare(reference, candidate)['exact'])
    for field in ('xyz', 'rot', 'reasoning', 'generated_token_ids'):
      changed = copy.deepcopy(candidate)
      p = changed['rows'][0]['prediction']
      if field == 'xyz': p[field][0][0][0] += 1e-9
      elif field == 'rot': p[field][0][0][0][0] += 1e-9
      elif field == 'reasoning': p[field] += ' changed'
      else: p[field][0] += 1
      self.assertFalse(self.replay.compare(reference, changed)['exact'], field)
    for field in ('history_after', 'decoder_statistics', 'speculation'):
      changed = copy.deepcopy(candidate)
      changed['rows'][0][field] = {'changed': True}
      self.assertFalse(self.replay.compare(reference, changed)['exact'], field)
    candidate['rows'][0]['completed'] = False
    with self.assertRaisesRegex(ValueError, 'unsynchronized'):
      self.replay.compare(reference, candidate)

  def test_original_benchmark_loop_and_consumer_compare_all_64_with_gpu_double(self):
    # Execute the unmodified benchmark main(), its real loader and input logic.
    # Only prepared GPU components and artifact writes are replaced.
    reports = []
    for mode in ('reference', 'consumer'):
      with ExitStack() as stack:
        pipeline = local('benchmark-pipeline')
        fake = self.checker.setup_runtime()
        fake.consumer.close()
        fake.settings.coarse_timing = fake.settings.cache_rotary_inputs = False
        fake.settings.initial_noise = np.load(self.args.initial_noise, allow_pickle=False)
        BaseInputs = fake.consumer.input_class
        class Inputs(BaseInputs):
          def __init__(self, settings, config, prepared, packed, arrays=None):
            super().__init__(settings, config, prepared, packed,
                             arrays=self.arrays(settings, prepared, packed) if arrays is None else arrays)
            if not settings.cache_rotary_inputs:
              self.gpu_rotary, self.rotary_values = {}, {}
        adapter = self.checker.adapter
        actual_prepare = adapter.prepare
        def checked_prepare(data):
          arrays = {k: list(v) if k == 'nav_text' else v.numpy() for k, v in data.items()}
          fake.decoder.expected = self.helpers.prepare_reference(self.checker.reference_adapter, arrays, fake.settings,
            self.checker.compact, fake.padding, self.args.pad_prompt_to)
          return actual_prepare(data)
        stack.enter_context(patch.object(adapter, 'prepare', checked_prepare))
        components = SimpleNamespace(adapter=adapter, model=fake.model, runner=fake.decoder.run,
          settings=fake.settings, config=adapter.config, tokenizer=adapter.tokenizer, input_class=Inputs,
          vision_cache=fake.vision, decoder=fake.decoder, pad_inputs=fake.padding, startup=None,
          startup_active=[False], profiler=None, inputs=None, vision_provenance={'gpu': 'fake'},
          benchmark=object(), Device=Device, GlobalCounters=SimpleNamespace(mem_used_per_device={}),
          report={'source_sha256': {'cpu_test': True}, 'runtime_environment': {}, 'warmup': [], 'runs': []},
          provenance={'cpu_test': True})
        initialize = stack.enter_context(patch.object(pipeline, 'initialize', return_value=components))
        stack.enter_context(patch.object(pipeline, 'snapshot_sources', return_value=({'cpu_test': True}, Path('/virtual/sources'))))
        actual_local = self.replay.local
        stack.enter_context(patch.object(self.replay, 'local', side_effect=lambda name:
          pipeline if name == 'benchmark-pipeline' else actual_local(name)))
        command = copy.deepcopy(self.normalized)
        command['environment'] = {}
        invocation = Path('/virtual/current-invocation.json')
        original_read = Path.read_text
        stack.enter_context(patch.object(Path, 'read_text', lambda path, *a, **kw:
          json.dumps(command) if path == invocation else original_read(path, *a, **kw)))
        original_open, original_write, original_exists = Path.open, Path.write_text, Path.exists
        stack.enter_context(patch.object(Path, 'open', lambda path, *a, **kw:
          io.StringIO() if str(path).startswith('/virtual/') else original_open(path, *a, **kw)))
        stack.enter_context(patch.object(Path, 'write_text', lambda path, *a, **kw:
          0 if str(path).startswith('/virtual/') else original_write(path, *a, **kw)))
        stack.enter_context(patch.object(Path, 'exists', lambda path:
          False if str(path).startswith('/virtual/') else original_exists(path)))
        stack.enter_context(redirect_stdout(io.StringIO()))
        result = self.replay.execute_existing_launcher(invocation, self.args, mode=mode, decoder_mode='ordinary', dataset=self.dataset)
        self.assertEqual(initialize.call_count, 1)
        self.assertEqual(fake.decoder.calls, 64)
        self.assertEqual([r['query_ns'] for r in result['rows']], [r['query_ns'] for r in self.dataset.rows])
        reports.append(dict(result, mode=mode, decoder='ordinary', prefill_attention='baseline', protocol=self.protocol))
    self.assertTrue(self.replay.compare(*reports)['exact'])

  def test_original_launcher_hooks_reach_current_factory_once(self):
    # Real launcher's hook machinery and current factory; only heavy installers
    # and final inference/file writes are doubled. No copied initializer runs.
    for mode in ('ordinary', 'speculative'):
      pipeline = local('benchmark-pipeline')
      args = copy.copy(self.args)
      mock = self.factory_checks.MockInitialization(pipeline, args)
      original_loader = pipeline.local_module
      real_spec = importlib.util.spec_from_file_location
      actual_speculation = local('agent-speculation-prefill')
      def prefill(model, draft, anchor, audit):
        mock.events.append('hook:prefill')
        model.padded_prefill = SimpleNamespace(last_prompt=None, last_targets=None, audit=audit)
      replacements = {
        'padded-prefill.py': mock.load('padded-prefill'),
        'lookup-decode.py': SimpleNamespace(LookupDecoder=mock.Decoder),
        'agent-speculation-prefill.py': SimpleNamespace(install_prefill=prefill, install_decoder=actual_speculation.install_decoder)}
      def supplied_spec(name, location, *a, **kw):
        replacement = replacements.get(Path(location).name)
        if replacement is None:
          return real_spec(name, location, *a, **kw)
        class Loader:
          def create_module(self, spec): return None
          def exec_module(self, module): module.__dict__.update(vars(replacement))
        return importlib.machinery.ModuleSpec(name, Loader())
      def load(name):
        return original_loader(name) if name in ('padded-prefill', 'lookup-decode') else mock.load(name)
      actual_local = self.replay.local
      def local_override(name):
        return pipeline if name == 'benchmark-pipeline' else actual_local(name)
      old_read_text = mock.read_text
      invocation = Path('/virtual/replay-invocation.json')
      command = copy.deepcopy(self.normalized)
      command['environment'] = {}  # Never apply the recorded GPU environment in CPU tests.
      mock.read_text = lambda path, *a, **kw: json.dumps(command) if path == invocation else old_read_text(path, *a, **kw)
      # The tokenizer reads the actual local checkpoint before the doubled factory.
      from transformers import AutoTokenizer
      tokenizer = AutoTokenizer.from_pretrained(str(ARGS.checkpoint), local_files_only=True, padding_side='left')
      actual_json = json.loads((ARGS.checkpoint/'config.json').read_text())
      previous_read = mock.read_text
      mock.read_text = lambda path, *a, **kw: json.dumps(actual_json) if path == ARGS.checkpoint/'config.json' else previous_read(path, *a, **kw)
      seen = []
      def consume(components, binding, **kwargs):
        self.assertEqual(hasattr(components.decoder, 'prefill_draft_records'), mode == 'speculative')
        self.assertEqual(hasattr(components.model.padded_prefill, 'last_targets'), mode == 'speculative')
        arrays = self.checker.cases[0][0]
        prepared, packed, _, _ = self.helpers.prepare_reference(self.checker.reference_adapter, arrays,
          components.settings, self.checker.compact, components.pad_inputs, binding.pad_prompt_to)
        self.assertEqual('prefill_draft_eligible' in prepared, mode == 'speculative')
        if mode == 'speculative':
          actual = prepared['actual_prompt_length']
          self.assertEqual(tuple(packed[0][0, actual:actual+6]), self.checker.policy.CHECKPOINT_DRAFT_IDS)
        seen.append(components)
        return SimpleNamespace(provenance={'cpu_test': True})
      def fake_replay(components, effective, dataset, audit, **kwargs):
        audit.rows = [dict(completed=True)]*64
      fake_config = SimpleNamespace(to_dict=lambda: {'cpu_test': True})
      mock.adapter.config = fake_config
      # initialize() does not use traj_ids in these doubled installer paths.
      with mock.context(load), patch.object(importlib.util, 'spec_from_file_location', supplied_spec), \
           patch.object(pipeline, 'snapshot_sources', return_value=({'cpu_test': True}, Path('/virtual/sources'))), \
           patch.object(self.replay, 'local', side_effect=local_override), \
           patch.object(self.replay.Audit, 'install', lambda a, c: None), \
           patch.object(self.replay, 'replay_consumer', fake_replay), \
           patch.object(self.runtime, 'bind_components', side_effect=consume), \
           patch.object(AutoTokenizer, 'from_pretrained', return_value=tokenizer), \
           patch.object(Path, 'exists', return_value=False), patch.object(Path, 'write_text', return_value=0):
        def selected(name):
          if name == 'agent-live-runtime': return self.runtime
          return local_override(name)
        with patch.object(self.replay, 'local', side_effect=selected):
          result = self.replay.execute_existing_launcher(invocation, args, mode='consumer', decoder_mode=mode, dataset=self.dataset)
      self.assertEqual(len(seen), 1)
      self.assertEqual(mock.events.count('model'), 1)
      self.assertEqual(mock.events.count('hook:prefill'), int(mode == 'speculative'))
      self.assertEqual(len(result['rows']), 64)
      self.assertLess(mock.events.index('install:quantize-packed'), mock.events.index('cache'))
      self.assertNotIn('warm:decoder', mock.events)
      self.events.append(dict(mode=mode, factory_calls=1, events=mock.events))


def main():
  global ARGS
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--checkpoint', type=Path, default=Path('/tmp/opencode/alpamayo-asius-20261114/model'))
  parser.add_argument('--samples-json', type=Path, default=Path('/mnt/nvme-500/tmp/alpamayo-speed-20261116/quality-samples.json'))
  parser.add_argument('--source-manifest', type=Path, default=Path('/mnt/nvme-256/footage/f9ad9a96/exports/0000008e--321b2ddc03/alpamayo-4hz-calibrated-two-six/manifest.json'))
  parser.add_argument('--invocation', type=Path, default=Path('/mnt/nvme-500/tmp/alpamayo-speed-20261116/full-fixed-vision-quality/invocation.json'))
  parser.add_argument('--prediction-report', type=Path, default=Path('/mnt/nvme-500/tmp/alpamayo-speed-20261116/full-fixed-vision-quality/report.json'))
  parser.add_argument('--output', type=Path, required=True)
  ARGS = parser.parse_args()
  with patch.object(type(Device), '__getitem__', forbidden), patch.object(socket.socket, 'connect', forbidden), \
       patch.object(socket.socket, 'connect_ex', forbidden), patch.object(socket, 'create_connection', forbidden):
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Checks))
  report = dict(cpu_only=True, gpu_inference=False, tests=result.testsRun, failures=len(result.failures), errors=len(result.errors),
    original_queries_preflighted=64, actual_canonical_upload_arrays=Checks.helpers.Checks.upload_arrays_checked,
    actual_canonical_upload_calls=Checks.helpers.Checks.selective_upload_calls, initializer_events=Checks.events,
    limits='GPU factory/Consumer prediction parity and native latency remain unverified; output fixtures are forwarded by a GPU double')
  with ARGS.output.open('x') as stream:
    json.dump(report, stream, indent=2)
    stream.write('\n')
  if not result.wasSuccessful():
    raise SystemExit(1)


if __name__ == '__main__':
  main()
