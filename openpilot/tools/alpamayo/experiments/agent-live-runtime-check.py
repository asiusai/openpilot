"""CPU-only factory extraction, import-hook, prepared-component, and guard checks."""
import argparse
import ast
from contextlib import ExitStack, redirect_stdout
import copy
import hashlib
import importlib.machinery
import importlib.util
import io
import json
import os
from pathlib import Path
import socket
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

os.environ['DEV'] = 'CPU'
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
sys.dont_write_bytecode = True

import numpy as np
import torch
from tinygrad import Device


ROOT = Path(__file__).parent


def local(name):
  spec = importlib.util.spec_from_file_location(name.replace('-', '_'), ROOT/(name+'.py'))
  value = importlib.util.module_from_spec(spec)
  sys.modules[spec.name] = value
  spec.loader.exec_module(value)
  return value


def forbidden(*args, **kwargs):
  raise AssertionError('Runtime CPU checks must not open a GPU or network connection')


def block_hash(nodes):
  return hashlib.sha256('\n'.join(ast.dump(n, include_attributes=False) for n in nodes).encode()).hexdigest()


def target(node):
  return node.targets[0].id if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name) else None


class MockInitialization:
  """Installer/storage doubles; no alternative model architecture or weight loader."""
  def __init__(self, pipeline, args):
    self.pipeline, self.args, self.events, self.calls = pipeline, args, [], []
    self.padding = local('padded-prefill')
    self.policy = local('agent-speculation-policy')
    owner = self

    class Model:
      def __init__(self, checkpoint, precision):
        owner.events.append('model')
        owner.calls.append(('model', checkpoint, precision))
        self.config = {'vlm_config': {'vision_config': {'num_hidden_layers': 4}}}
        self.padded_prefill = None
      def synchronize(self):
        owner.events.append('sync')

    class Decoder:
      def __init__(self, model, *args):
        self.model = model
        self.before_diffusion = lambda: None
        self.history, self.statistics = [], []
        self.warmup_graphs = False
      def run(self, *args, **kwargs):
        raise AssertionError('Factory must not run inference')
      def generate(self, *args, **kwargs):
        raise AssertionError('Factory must not generate tokens')

    class FakeDevice:
      DEFAULT = 'FAKE'
      def __getitem__(self, key):
        owner.events.append('device:'+key)
        return SimpleNamespace(arch='fake-gfx1200')

    self.Model, self.Decoder = Model, Decoder
    self.adapter = SimpleNamespace(config=SimpleNamespace(traj_ids={'future_start': 135084}), tokenizer=object(), pixel_bounds={1: 720, 6: 720})
    self.input_class = type('MockInputs', (), {})
    self.modules = {
      'benchmark': SimpleNamespace(EdgeModel=Model, NativeInputs=type('NativeInputs', (), {})),
      'tinygrad': SimpleNamespace(Device=FakeDevice()),
      'tinygrad.helpers': SimpleNamespace(GlobalCounters=SimpleNamespace(mem_used_per_device={})),
      'tinygrad.device': SimpleNamespace(Compiled=type('Compiled', (), {})),
      'wgp_compiler': SimpleNamespace(install=lambda dev: owner.events.append('install:wgp')),
    }
    self.noise = np.linspace(-.3, .7, 128, dtype=np.float32).reshape(1, 64, 2)
    self.fixed = {'scales': {'prefill': .125}}
    self.vision = {'checkpoint_provenance': {
      'config_sha256': hashlib.sha256(b'config').hexdigest(), 'weight_index_sha256': hashlib.sha256(b'index').hexdigest(),
      'bf16_vision_layers': args.bf16_vision_layers},
      'scales': {f'vlm.model.visual.encoder.layers.{layer}/{i}': .25 for layer in range(args.bf16_vision_layers, 4) for i in range(4)}}
    self.real_read_text, self.real_read_bytes = Path.read_text, Path.read_bytes

  def load(self, name):
    self.events.append('load:'+name)
    def action(label, result=None):
      def call(*args, **kwargs):
        self.events.append(label)
        self.calls.append((label, args, kwargs))
        return result
      return call
    if name == 'fast-inputs':
      return SimpleNamespace(FastInputs=action('adapter', self.adapter))
    if name == 'compact-inputs':
      return SimpleNamespace(CompactInputs=self.input_class)
    if name == 'restore-precision':
      return SimpleNamespace(restore=action('restore', {'restored': True}))
    if name == 'padded-prefill':
      return SimpleNamespace(install=action('install:padding'), pad_inputs=self.padding.pad_inputs)
    if name in ('draft-decode', 'lookup-decode', 'block-decode'):
      def make(*args):
        self.events.append('decoder:'+name)
        return self.Decoder(*args)
      return SimpleNamespace(**{dict(zip(('draft-decode', 'lookup-decode', 'block-decode'), ('DraftDecoder', 'LookupDecoder', 'BlockDecoder')))[name]: make})
    if name == 'vision-cache':
      def create(model, **kwargs):
        self.events.append('cache')
        self.calls.append(('cache', (), kwargs))
        return SimpleNamespace(model=model, warmup_graphs=False, **kwargs)
      return SimpleNamespace(VisionCache=create)
    if name == 'profile-kernels':
      return SimpleNamespace(Recorder=action('profiler', object()))
    if name == 'agent-pipeline-warmup':
      owner = self
      class Startup:
        def __init__(self, active, record, graph_budget):
          self.active = active
          self.record = record
          self.graph_budget = graph_budget
        def patch_decoder(self, decoder): self.record('decoder'); owner.events.append('warm:decoder')
        def patch_inputs(self, inputs): self.record('inputs'); owner.events.append('warm:inputs')
        def patch_vision(self, vision): self.record('vision'); owner.events.append('warm:vision')
      return SimpleNamespace(StartupWarmup=Startup)
    return SimpleNamespace(install=action('install:'+name, ['mock:'+name]),
                           fix_dispatch_packet=action('fix_dispatch'), download=lambda *args: forbidden())

  def read_bytes(self, path, *args, **kwargs):
    if str(path) == str(Path(self.args.checkpoint)/'config.json'):
      return b'config'
    if str(path) == str(Path(self.args.checkpoint)/'model.safetensors.index.json'):
      return b'index'
    if path in (self.args.prefill_calibration, self.args.vision_calibration):
      return self.read_text(path).encode()
    return self.real_read_bytes(path, *args, **kwargs)

  def read_text(self, path, *args, **kwargs):
    if self.args.prefill_calibration is not None and path == self.args.prefill_calibration:
      return json.dumps(self.fixed)
    if self.args.vision_calibration is not None and path == self.args.vision_calibration:
      return json.dumps(self.vision)
    return self.real_read_text(path, *args, **kwargs)

  def context(self, loader=None):
    stack = ExitStack()
    stack.enter_context(patch.dict(sys.modules, self.modules))
    stack.enter_context(patch.object(self.pipeline, 'local_module', loader or self.load))
    stack.enter_context(patch.object(self.pipeline.torch, 'set_num_threads', lambda n: self.events.append('threads:'+str(n))))
    stack.enter_context(patch.object(Path, 'read_text', lambda path, *a, **kw: self.read_text(path, *a, **kw)))
    stack.enter_context(patch.object(Path, 'read_bytes', lambda path, *a, **kw: self.read_bytes(path, *a, **kw)))
    stack.enter_context(patch.object(self.pipeline.np, 'load', lambda *a, **kw: self.noise.copy()))
    stack.enter_context(redirect_stdout(io.StringIO()))
    return stack

  def initialize(self):
    return self.pipeline.initialize(self.args, sources={'benchmark-pipeline.py': 'test-source'}, source_dir=Path('/virtual/sources'))


class Checks(unittest.TestCase):
  @classmethod
  def setUpClass(cls):
    cls.pipeline = local('benchmark-pipeline')
    cls.runtime = local('agent-live-runtime')
    cls.baseline = json.loads((ROOT/'agent-live-runtime-baseline.json').read_text())
    cls.consumer_checks = local('agent-live-consumer-check')
    cls.consumer_checks.ARGS = SimpleNamespace(checkpoint=ARGS.checkpoint, manifest=ARGS.manifest,
                                             raw_frames=ARGS.raw_frames, prediction_report=ARGS.prediction_report)
    cls.consumer_checks.Checks.setUpClass()

  def args(self, *extra):
    return self.pipeline.parse_args(['--checkpoint', '/virtual/checkpoint', '--samples', '<live>', '--output', '/virtual/out.json',
      '--camera-profile', 'nvidia', '--allow-missing-cameras', '--compact-inputs', '--byte-pixels', '--selective-pixels',
      '--vision-cache', '--vision-batch', '1', '--steps', '3', '--cache-resize', '--direct-byte-inputs', '--packed-quantize',
      '--quantize-reduction', '--prefill-calibration', '/virtual/prefill.json', '--pad-prompt-to', '2112',
      '--initial-noise', '/virtual/noise.npy', '--warmup', '0', '--runs', '0', *extra])

  def test_mechanical_extraction_preserves_original_statement_blocks(self):
    tree = ast.parse((ROOT/'benchmark-pipeline.py').read_text())
    functions = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
    parser = functions['parse_args'].body
    parser = parser[:next(i for i, n in enumerate(parser) if target(n) == 'args')]
    initializer = copy.deepcopy(functions['initialize'].body)
    begin = next(i for i, n in enumerate(initializer) if isinstance(n, ast.Import) and n.names[0].name == 'benchmark')
    end = next(i for i, n in enumerate(initializer) if target(n) == 'report')
    init = initializer[begin:end]
    init = [n for n in init if target(n) not in ('padding', 'startup')]
    for node in init:
      if isinstance(node, ast.If) and ast.unparse(node.test) == 'args.pad_prompt_to is not None':
        self.assertEqual([target(n) for n in node.body], [None, 'padding', None])
        self.assertEqual(ast.unparse(node.body[-1]), 'padding.install(model)')
        node.body = [node.body[0], ast.parse("local_module('padded-prefill').install(model)").body[0]]
    report = initializer[end:-1]
    main = functions['main'].body
    tail = main[next(i for i, n in enumerate(main) if target(n) == 'inputs'):]
    for name, nodes in [('parser', parser), ('initialization', init), ('report_and_noise', report), ('replay_tail', tail)]:
      self.assertEqual(block_hash(nodes), self.baseline['ast_sha256'][name], name)

  def test_existing_model_and_live_sources_are_unchanged(self):
    for name, expected in self.baseline['immutable_sources'].items():
      self.assertEqual(hashlib.sha256((ROOT.parent/name).read_bytes()).hexdigest(), expected, name)

  def test_selected_installer_order_and_report_provenance(self):
    args = self.args('--llvm-wgp', '--bf16-vision-layers', '2', '--bf16-expert-projections', '--packed-bf16-vision',
      '--materialize-quant', '--grouped-quantize-reduction', '--reduced-vocab', '/virtual/vocab.json', '--expert-kernels',
      '--online-attention', '--fused-norm', '--packed-projector', '--batched-vision', '--image-gemm', '--group-vision',
      '--vision-calibration', '/virtual/vision.json', '--draft-previous', '--lookup-draft', '--lookup-carry',
      '--phrase-drafts', '--draft-attention', '--compact-draft-io', '--direct-readback', '--cache-rotary-inputs', '--verify-vision-cache')
    mock = MockInitialization(self.pipeline, args)
    with mock.context():
      components = mock.initialize()
    events = [e for e in mock.events if not e.startswith('load:')]
    self.assertEqual(events, ['threads:4', 'device:FAKE', 'install:wgp', 'adapter', 'model', 'restore',
      'fix_dispatch', 'install:bf16-linear', 'install:fast-quantize', 'fix_dispatch', 'install:quantize-reduction',
      'install:reduced-vocab', 'install:expert-kernels', 'fix_dispatch', 'install:online-attention', 'fix_dispatch',
      'install:fused-norm', 'install:quantize-packed', 'install:padding', 'install:projector-kernels',
      'install:batched-vision', 'install:vision-calibration', 'cache', 'sync', 'decoder:draft-decode',
      'decoder:lookup-decode', 'install:phrase-drafts', 'device:FAKE'])
    self.assertIs(components.runner.__self__, components.decoder)
    self.assertIs(components.config, components.adapter.config)
    self.assertTrue(components.vision_cache.verify)
    np.testing.assert_array_equal(components.settings.initial_noise, mock.noise)
    self.assertNotIn('initial_noise', components.report['settings'])
    self.assertEqual(components.report['initial_noise_sha256'], hashlib.sha256(mock.noise.tobytes()).hexdigest())
    self.assertEqual(components.provenance, components.report)
    components.report['runs'].append({'later': True})
    self.assertEqual(components.provenance['runs'], [])

  def test_parser_preserves_warmup_flag_and_factory_helper_order(self):
    args = self.args('--warmup', '2', '--lookup-draft', '--draft-previous', '--warm-current-shapes')
    self.assertTrue(args.warmup_graphs)
    mock = MockInitialization(self.pipeline, args)
    with mock.context():
      result = mock.initialize()
    self.assertEqual([e for e in mock.events if e.startswith('warm:')], ['warm:decoder', 'warm:inputs', 'warm:vision'])
    self.assertFalse(result.startup.active())
    result.startup_active[0] = True
    self.assertTrue(result.startup.active())
    with self.assertRaisesRegex(ValueError, 'startup warming'):
      self.runtime.validate_live_args(args)

  def test_static_prefill_calibration_checks_are_retained(self):
    for scales in ({}, {'a': 0.}, {'a': float('nan')}, {'a': -1.}):
      mock = MockInitialization(self.pipeline, self.args())
      mock.fixed = {'scales': scales}
      with mock.context(), self.assertRaises(AssertionError):
        mock.initialize()
      self.assertNotIn('install:quantize-packed', mock.events)

  def test_static_vision_provenance_and_complete_scale_checks_are_retained(self):
    for defect in ('checkpoint', 'missing', 'nonfinite'):
      args = self.args('--grouped-quantize-reduction', '--batched-vision', '--group-vision', '--vision-calibration', '/virtual/vision.json')
      mock = MockInitialization(self.pipeline, args)
      if defect == 'checkpoint': mock.vision['checkpoint_provenance']['config_sha256'] = 'wrong'
      elif defect == 'missing': mock.vision['scales'].pop(next(iter(mock.vision['scales'])))
      else: mock.vision['scales'][next(iter(mock.vision['scales']))] = float('inf')
      with mock.context(), self.assertRaises(ValueError):
        mock.initialize()
      self.assertNotIn('install:vision-calibration', mock.events)

  def test_block_runner_without_draft_remains_bound_and_hooked_padding_is_retained(self):
    mock = MockInitialization(self.pipeline, self.args())
    with mock.context():
      result = mock.initialize()
    self.assertIsNone(result.decoder)
    self.assertIs(result.runner.__self__.model, result.model)
    self.assertIs(result.pad_inputs, mock.padding.pad_inputs)
    self.assertIsNone(result.inputs)

  def test_current_speculation_launcher_import_hooks_survive_factory(self):
    # Execute the actual launcher's hook setup. Replace only model/kernel installers
    # and final filesystem writes; its policy and hooked padding remain real.
    launcher = local('agent-speculation-run')
    args = self.args('--draft-previous', '--lookup-draft', '--reduced-vocab', str(ROOT/'coc_vocab.json'))
    mock = MockInitialization(self.pipeline, args)
    real_spec = importlib.util.spec_from_file_location
    policy = mock.policy
    baseline_pad = mock.load('padded-prefill')
    actual_speculation = local('agent-speculation-prefill')
    def fake_prefill(model, draft, anchor, audit):
      mock.events.append('hook:prefill')
      model.padded_prefill = SimpleNamespace(last_prompt=None, last_targets=None, audit=audit)
    spec_impl = SimpleNamespace(install_prefill=fake_prefill, install_decoder=actual_speculation.install_decoder)
    replacements = {'padded-prefill.py': baseline_pad, 'lookup-decode.py': SimpleNamespace(LookupDecoder=mock.Decoder),
                    'agent-speculation-prefill.py': spec_impl}
    def supplied_spec(name, location, *a, **kw):
      replacement = replacements.get(Path(location).name)
      if replacement is None:
        return real_spec(name, location, *a, **kw)
      class Loader:
        def create_module(self, spec): return None
        def exec_module(self, module): module.__dict__.update(vars(replacement))
      return importlib.machinery.ModuleSpec(name, Loader())
    original_loader = self.pipeline.local_module
    def load(name):
      return original_loader(name) if name in ('padded-prefill', 'lookup-decode') else mock.load(name)
    invocation = {'command': ['python', str(ROOT/'benchmark-pipeline.py'), '--checkpoint', str(ARGS.checkpoint),
      '--prefill-calibration', 'unused', '--packed-quantize', '--compact-inputs', '--lookup-draft',
      '--reduced-vocab', str(ROOT/'coc_vocab.json')], 'environment': {}}
    actual_read_text = mock.read_text
    def read_text(path, *a, **kw):
      return json.dumps(invocation) if str(path) == '/virtual/invocation.json' else actual_read_text(path, *a, **kw)
    mock.read_text = read_text
    captured = []
    def intercepted_run(*a, **kw):
      captured.append(mock.initialize())
    import runpy
    import transformers
    with mock.context(loader=load), patch.object(importlib.util, 'spec_from_file_location', supplied_spec), \
         patch.object(transformers.AutoTokenizer, 'from_pretrained', return_value=SimpleNamespace(encode=lambda *a, **kw: list(policy.CHECKPOINT_DRAFT_IDS))), \
         patch.object(runpy, 'run_path', intercepted_run), patch.object(Path, 'write_text', return_value=1), \
         patch.object(sys, 'argv', ['agent-speculation-run.py', '--invocation', '/virtual/invocation.json', '--output', '/virtual/hook.json']):
      launcher.main()
    self.assertEqual(len(captured), 1)
    component = captured[0]
    self.assertIn('hook:prefill', mock.events)
    self.assertTrue(hasattr(component.decoder, 'prefill_draft_records'))
    packed = (np.array([[1, 2, 3]], dtype=np.int64), np.zeros((3, 1, 3), dtype=np.int64), 0, np.ones((1,), dtype=np.float32))
    prepared, padded = component.pad_inputs({}, packed, 16)
    self.assertTrue(prepared['prefill_draft_eligible'])
    self.assertEqual(tuple(padded[0][0, 3:9]), policy.CHECKPOINT_DRAFT_IDS)

  def prepared(self, speculation=False):
    check = self.consumer_checks.Checks()
    runtime = check.setup_runtime(speculation=speculation)
    previous = runtime.consumer
    previous.close()
    components = SimpleNamespace(adapter=previous.adapter, model=previous.model, settings=previous.settings,
      config=previous.config, tokenizer=previous.tokenizer, input_class=previous.input_class,
      vision_cache=previous.vision_cache, pad_inputs=previous.pad_inputs, decoder=previous.decoder,
      runner=previous.runner, inputs=None, startup=None, startup_active=[False], profiler=None,
      provenance={'cpu_mock': True}, vision_provenance={'cpu_mock': True})
    args = self.args('--cache-rotary-inputs', '--coarse-timing', '--max-tokens', '64')
    return check, runtime, components, args

  def test_prepared_real_adapter_compact_arrays_and_runner_accept_live_queries(self):
    check, runtime, components, args = self.prepared()
    runtime.consumer = self.runtime.bind_components(components, args, consumer_module=self.consumer_checks.Checks.live, clock=runtime.clock)
    for case in self.consumer_checks.Checks.cases[:2]:
      result, _ = check.run_case(runtime, case)
      self.assertTrue(result['live_consumer']['completed'])
    with self.assertRaisesRegex(RuntimeError, 'already have'):
      self.runtime.bind_components(components, args)

  def test_prepared_speculative_components_require_explicit_policy_and_accept_verified_draft(self):
    check, runtime, components, args = self.prepared(speculation=True)
    with self.assertRaisesRegex(ValueError, 'Declare target-verified'):
      self.runtime.bind_components(components, args, consumer_module=self.consumer_checks.Checks.live, clock=runtime.clock)
    runtime.consumer = self.runtime.bind_components(components, args, consumer_module=self.consumer_checks.Checks.live,
      speculation_policy=self.consumer_checks.Checks.policy, speculation_draft=self.consumer_checks.Checks.policy.CHECKPOINT_DRAFT_IDS,
      clock=runtime.clock)
    result, _ = check.run_case(runtime, self.consumer_checks.Checks.cases[0])
    self.assertTrue(result['live_consumer']['speculation']['accepted'])
    self.assertEqual(runtime.model.suffix, 'accepted speculative suffix')

  def test_prepared_identity_verification_and_active_startup_fail_before_sync(self):
    for defect in ('identity', 'verification', 'startup', 'settings'):
      _, runtime, components, args = self.prepared()
      before = runtime.model.syncs
      if defect == 'identity': components.config = object()
      elif defect == 'verification': components.vision_cache.verify = not args.verify_vision_cache
      elif defect == 'startup': components.startup_active[0] = True
      else: args.max_tokens = 128
      with self.assertRaises((ValueError, RuntimeError)):
        self.runtime.bind_components(components, args, consumer_module=self.consumer_checks.Checks.live, clock=runtime.clock)
      self.assertEqual(runtime.model.syncs, before)

  def test_unsupported_flags_fail_before_initializer_and_no_verification_is_disabled(self):
    for field in ('warmup_graphs', 'warm_current_shapes', 'verify_vision_warmup', 'pipeline_inputs', 'capture_prefill_scales'):
      args = self.args()
      setattr(args, field, True)
      with patch.object(self.pipeline, 'initialize', side_effect=AssertionError('Must reject before initialization')):
        with self.assertRaises(ValueError): self.runtime.initialize_consumer(self.pipeline, args)
      self.assertTrue(getattr(args, field))
    args = self.args('--verify-vision-cache')
    self.runtime.validate_live_args(args)
    self.assertTrue(args.verify_vision_cache)

  def test_factory_called_once_and_exact_returned_objects_are_bound(self):
    _, prepared, components, args = self.prepared()
    with patch.object(self.pipeline, 'initialize', return_value=components) as factory:
      result, consumer = self.runtime.initialize_consumer(self.pipeline, args, sources={'test': 'hash'},
        consumer_module=self.consumer_checks.Checks.live, clock=prepared.clock)
    factory.assert_called_once_with(args, sources={'test': 'hash'}, source_dir='<live source hashes; no snapshot files>')
    self.assertIs(result, components)
    self.assertIs(consumer.model, components.model)
    self.assertIs(consumer.runner, components.runner)
    self.assertIs(consumer.pad_inputs, components.pad_inputs)

  def test_options_do_not_accept_replay_or_silently_override_startup(self):
    for options in (['--samples=x'], ['--warmup', '12'], ['--native-frame-cache', '/data/cache']):
      with self.assertRaisesRegex(ValueError, 'model options only'):
        self.runtime.parse_pipeline_options(self.pipeline, options, Path('/virtual/output'))

  def test_example_options_parse_without_initializing_or_changing_model_flags(self):
    options = json.loads((ROOT/'agent-live-runtime-options.example.json').read_text())
    with patch.object(self.pipeline, 'initialize', side_effect=AssertionError('Parsing must not initialize')):
      args = self.runtime.parse_pipeline_options(self.pipeline, options, Path('/virtual/output'))
    self.assertEqual((args.warmup, args.runs), (0, 0))
    self.assertEqual((args.max_tokens, args.pad_prompt_to, args.steps), (64, 2112, 3))
    self.assertTrue(args.lookup_draft and args.lookup_carry and args.coarse_timing)

  def test_cli_guard_precedes_factory_and_cleanup_preserves_primary_failure(self):
    options = (ROOT/'agent-live-runtime-options.example.json').read_text()
    manifest = json.dumps(dict(camera_input_profile='calibrated_vehicle_history', camera_subset=[1, 6],
                               source_camera_geometry={}, target_camera_geometry={}))
    for fail in (False, True):
      events = []
      producer = SimpleNamespace(start=lambda: events.append('producer:start'), close=lambda: events.append('producer:close'))
      bridge = SimpleNamespace(Store=lambda: object(), calibrated_image=object(), storage_ok=lambda *a: True,
        Preparer=lambda *a, **kw: object(), Bridge=lambda *a, **kw: object())
      capture = SimpleNamespace(bridge=bridge, NativeProducer=lambda store: producer)
      class Guard:
        def __init__(self, *a): self.initializing = True
        def resources(self, **kw): events.append('guard:resources')
        def health_ready(self): return True
        def start(self): events.append('guard:start')
        def check(self): events.append('guard:check')
        def close(self): events.append('guard:close')
      class Output:
        def __init__(self, *a): events.append('output:open')
        def save(self, row): events.append('save:'+row['kind'])
        def close(self): events.append('output:close')
      consumer = SimpleNamespace(provenance={}, close=lambda: events.append('consumer:close'))
      failure = LookupError('deliberate initializer failure')
      def initialize(*a, **kw):
        events.append('factory')
        if fail: raise failure
        return object(), consumer
      def read_text(path, *a, **kw):
        return {'/sys/devices/soc0/serial_number': 'test-serial', '/virtual/geometry': manifest, '/virtual/options': options}[str(path)]
      with patch.object(self.runtime, 'local', side_effect=lambda name: self.pipeline if name == 'benchmark-pipeline' else capture), \
           patch.object(self.runtime, 'Guard', Guard), patch.object(self.runtime, 'Output', Output), \
           patch.object(self.runtime, 'initialize_consumer', initialize), patch.object(self.runtime, 'source_hashes', return_value={}), \
           patch.object(self.runtime, 'observe', return_value={'kind': 'summary'}), patch.object(self.runtime.os, 'getuid', return_value=1000), \
           patch.object(self.runtime.os, 'umask', return_value=0), patch.object(Path, 'read_text', read_text), \
           patch.object(Path, 'read_bytes', lambda path: read_text(path).encode()):
        argv = ['--live-model', '--serial', 'test-serial', '--model-pid', '1', '--manager-pid', '2',
          '--geometry-manifest', '/virtual/geometry', '--pipeline-options', '/virtual/options', '--output', '/virtual/output',
          '--warp', 'scipy', '--navigation-text', 'Follow the road.']
        if fail:
          with self.assertRaises(LookupError) as caught: self.runtime.main(argv)
          self.assertIs(caught.exception, failure)
        else:
          self.runtime.main(argv)
      self.assertLess(events.index('guard:start'), events.index('factory'))
      self.assertLess(events.index('producer:start'), events.index('factory'))
      self.assertEqual(events.count('factory'), 1)
      self.assertIn('guard:close', events)
      self.assertIn('producer:close', events)
      self.assertEqual(events[-1], 'output:close')
      self.assertEqual('consumer:close' in events, not fail)

  def test_observe_checks_both_sides_of_inference_and_counts_no_stale_rows(self):
    events, rows = [], []
    now = [0.]
    class Bridge:
      store = SimpleNamespace(bounds=lambda: {'bounded': True})
      def run_once(self, consumer):
        events.append('infer')
        now[0] += .25
        return {'kind': 'model_completion'} if now[0] == .25 else None
      def summary(self): return {'model_completion_hz': 1/now[0]}
    guard = SimpleNamespace(check=lambda: events.append('guard'))
    producer = SimpleNamespace(stop=SimpleNamespace(wait=lambda seconds: None))
    report = self.runtime.observe(Bridge(), object(), producer, guard, SimpleNamespace(save=rows.append), .75, clock=lambda: now[0])
    self.assertEqual(events, ['guard', 'infer', 'guard']*3)
    self.assertEqual(len(rows), 1)
    self.assertTrue(rows[0]['runtime_guard_passed'])
    self.assertEqual(report['observation_seconds'], .75)
    self.assertTrue(report['cold_graph_stalls_included'])
    guard.check = lambda: (_ for _ in ()).throw(RuntimeError('lost small model'))
    with self.assertRaisesRegex(RuntimeError, 'lost small model'):
      self.runtime.observe(Bridge(), object(), producer, guard, SimpleNamespace(save=rows.append), .75, clock=lambda: now[0])
    self.assertEqual(len(rows), 1)

  def test_guard_detects_stale_small_model_storage_memory_and_producer_errors(self):
    bridge = local('agent-live-bridge')
    now = 1_000_000_000
    healthy = dict(time_ns=now, model_alive=True, device_valid=True, model_age_ns=50_000_000, thermal_status='green')
    producer = SimpleNamespace(error=None, health=healthy, stop=SimpleNamespace(set=lambda: None))
    capture = SimpleNamespace(verify_small_model=lambda *a: ('model', 'manager'), bridge=bridge)
    args = SimpleNamespace(model_pid=1, manager_pid=2, output=Path('/virtual/output'), initialization_reserve_mib=64)
    with patch.dict(os.environ, {'ALPAMAYO_FP8_CACHE': '', 'CACHEDB': ''}), \
         patch.object(Path, 'read_text', return_value='MemAvailable: 2048000 kB\n'), \
         patch.object(self.runtime.shutil, 'disk_usage', return_value=SimpleNamespace(total=100*1024**3, free=30*1024**3)):
      guard = self.runtime.Guard(capture, producer, args, clock=lambda: now)
      guard.check()
      producer.health = dict(healthy, model_age_ns=300_000_000)
      with self.assertRaisesRegex(RuntimeError, 'health'): guard.check()
      producer.health = healthy
      with patch.object(Path, 'read_text', return_value='MemAvailable: 1000 kB\n'):
        with self.assertRaisesRegex(RuntimeError, 'memory'): guard.check()
      with patch.object(self.runtime.shutil, 'disk_usage', return_value=SimpleNamespace(total=100*1024**3, free=6*1024**3)):
        with self.assertRaisesRegex(RuntimeError, 'Storage'): guard.check()
      producer.error = LookupError('camera gap')
      with self.assertRaisesRegex(RuntimeError, 'producer'): guard.check()

  def test_bounded_output_preserves_thresholds_and_never_overwrites(self):
    bridge = local('agent-live-bridge')
    stream = io.StringIO()
    with patch.object(Path, 'open', return_value=stream) as opened, \
         patch.object(self.runtime.shutil, 'disk_usage', return_value=SimpleNamespace(total=100*1024**3, free=30*1024**3)):
      output = self.runtime.Output(Path('/virtual/output'), bridge.storage_ok)
      opened.assert_called_once_with('x')
      output.save({'real_completion': True})
      self.assertEqual(output.written, len(stream.getvalue().encode()))
      output.limit = output.written
      with self.assertRaisesRegex(RuntimeError, 'limit'): output.save({'another': True})
      with self.assertRaises(ValueError): output.save({'invalid': float('nan')})


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
  if not result.wasSuccessful(): raise SystemExit(1)
  if Device._opened_devices: raise AssertionError('Unexpected tinygrad initialization')
  report = dict(passed=True, tests=result.testsRun, cpu_only=True, initialized_tinygrad_devices=[],
    actual_model_inference=False, network_connections=False,
    actual_upload_parity_calls=Checks.consumer_checks.Checks.parity_calls,
    actual_upload_arrays=Checks.consumer_checks.Checks.upload_arrays_checked,
    preserved_baseline=Checks.baseline,
    source_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in [ROOT/'benchmark-pipeline.py', *ROOT.glob('agent-live-runtime*.py')]})
  if ARGS.output:
    with ARGS.output.open('x') as stream: stream.write(json.dumps(report, indent=2)+'\n')
  print(json.dumps(report, indent=2))


if __name__ == '__main__': main()
