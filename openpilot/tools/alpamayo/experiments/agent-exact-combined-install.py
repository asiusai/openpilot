"""Compose frozen exact kernels while preserving installer order and cleanup."""
from contextlib import ExitStack
import functools
import importlib.util
import inspect
from pathlib import Path
import sys
from unittest.mock import patch


class Composition:
  TARGETS = {'quantize-packed.py', 'batched-vision.py', 'vision-cache.py', 'draft-decode.py', 'block-decode.py', 'stream-inputs.py'}

  def __init__(self, common, manifest, stage, candidate, *, device_factory=None, mlp_dispatch=None):
    self.common, self.manifest, self.stage, self.candidate = common, manifest, stage, candidate
    self.device_factory, self.mlp_dispatch = device_factory, mlp_dispatch
    self.stack = ExitStack()
    self.packed = self.model = self.vision_info = self.session = self.cache = self.sampler = None
    self.events, self.restorations = [], []
    self.sampler_calls = 0
    self.arm_calls = 0
    self.modules = []
    self.hook = common.load(common.MLP/'experiments/agent-fused-mlp.py', 'combined_frozen_mlp')
    self.kernel = common.load(common.MLP/'experiments/agent-packed-epilogue.py', 'combined_frozen_fc1')
    self.vision = common.load(common.VISION/'experiments/agent-vision-tiles.py', 'combined_frozen_vision')

  def __enter__(self):
    if getattr(importlib.util, '_agent_exact_combined_active', False): raise ValueError('Nested composition rejected')
    for name in ('model', 'benchmark', 'inputs', 'native_warp', 'stream_inputs'):
      if name in sys.modules: raise ValueError(f'Install before factory and input imports: {name}')
    self.original_spec = importlib.util.spec_from_file_location
    self.stack.enter_context(patch.object(importlib.util, '_agent_exact_combined_active', True, create=True))
    self.stack.enter_context(patch.object(importlib.util, 'spec_from_file_location', self.spec_hook))
    self.events.append('before_factory_and_input_imports')
    return self

  def __exit__(self, *error):
    return self.stack.__exit__(*error)

  def patched(self, obj, name, value):
    original = getattr(obj, name)
    setattr(obj, name, value)
    def restore():
      if getattr(obj, name) is not value:
        self.restorations.append(dict(name=name, restored=False))
        raise RuntimeError(f'Hook replacement conflict: {name}')
      setattr(obj, name, original)
      self.restorations.append(dict(name=name, restored=True))
    self.stack.callback(restore)

  def spec_hook(self, name, location, *values, **options):
    spec = self.original_spec(name, location, *values, **options)
    if spec is None or location is None: return spec
    path = Path(location).resolve()
    if path.name not in self.TARGETS: return spec
    self.require_path(path)
    loader = spec.loader
    owner = self
    class Loader:
      def create_module(self, value): return loader.create_module(value)
      def exec_module(self, module):
        loader.exec_module(module)
        owner.after_load(module, path)
    spec.loader = Loader()
    return spec

  def require_path(self, path):
    if path.parent != self.common.MLP/'experiments' or str(path) not in self.manifest['files']:
      raise ValueError(f'Unpinned installer import: {path}')
    if self.common.sha(path) != self.manifest['files'][str(path)]: raise ValueError(f'Installer bytes changed: {path}')

  def after_load(self, module, path):
    self.require_path(path)
    self.modules.append(module)
    if path.name == 'quantize-packed.py':
      original = module.install
      @functools.wraps(original)
      def install(model, *values, **options):
        if self.packed is not None: raise ValueError('Duplicate packed quantizer installation')
        args = inspect.signature(original).bind(model, *values, **options)
        args.apply_defaults()
        if args.arguments['capture_scales'] or not args.arguments['fixed_scales']:
          raise ValueError('Require the original complete fixed language scales')
        result = original(model, *values, **options)
        self.model, self.packed = model, (module, dict(args.arguments['fixed_scales']))
        self.events.append('packed_quantizer')
        return result
      self.patched(module, 'install', install)
    elif path.name == 'batched-vision.py':
      original = module.install
      @functools.wraps(original)
      def install(model, *values, **options):
        if self.model is not model or self.packed is None or 'batched_vision' in self.events:
          raise ValueError('Batched vision must follow one packed quantizer')
        if getattr(model, 'encode_jits', {}) or any(j.cnt or j.captured is not None for _, j in self.hook.prefill_jits(model)):
          raise ValueError('Install before model graph tracing')
        result = original(model, *values, **options)
        self.events.append('batched_vision')
        if self.candidate: self.install_vision(model)
        return result
      self.patched(module, 'install', install)
    elif path.name == 'vision-cache.py':
      original = module.VisionCache.__init__
      @functools.wraps(original)
      def initialize(cache, model, *values, **options):
        if model is not self.model or 'batched_vision' not in self.events or self.cache is not None:
          raise ValueError('Vision cache installation order changed')
        original(cache, model, *values, **options)
        if cache.cache or cache.encoders or cache.stitchers: raise ValueError('Vision cache must start empty')
        self.cache = cache
        self.events.append('vision_cache')
        if self.candidate: self.stack.callback(self.clear_vision_graphs)
      self.patched(module.VisionCache, '__init__', initialize)
    elif path.name in ('draft-decode.py', 'block-decode.py'):
      cls = module.DraftDecoder if path.name == 'draft-decode.py' else module.BlockDecoder
      original = cls.__init__
      @functools.wraps(original)
      def initialize(decoder, model, *values, **options):
        if self.cache is None or model is not self.model: raise ValueError('Decoder must follow the vision cache')
        original(decoder, model, *values, **options)
        self.arm(model)
      self.patched(cls, '__init__', initialize)
    elif path.name == 'stream-inputs.py':
      if self.stage != 'native' or not self.arm_calls: raise ValueError('Native preparation started before composed factory completion')
      if self.candidate:
        self.install_sampler(module)
      self.events.append('native_preparation_module')

  def install_vision(self, model):
    if self.session is not None: raise ValueError('Vision must precede language fusion')
    original = model.linear
    existed, old_info = 'agent_vision_tiles' in vars(model), vars(model).get('agent_vision_tiles')
    self.vision_info = self.vision.install(model)
    installed = model.linear
    self.events.append('vision_tiles')
    def restore():
      if model.linear is not installed: raise RuntimeError('Vision method replacement conflict')
      model.linear = original
      if existed: model.agent_vision_tiles = old_info
      else: delattr(model, 'agent_vision_tiles')
      self.restorations.append(dict(name='vision_model_linear_and_marker', restored=True))
    self.stack.callback(restore)

  def arm(self, model):
    if self.arm_calls:
      if self.model is not model: raise ValueError('Multiple composed models rejected')
      return
    if self.packed is None or self.cache is None or (self.candidate and self.vision_info is None):
      raise ValueError('Language fusion requires packed quantization and vision installation first')
    self.arm_calls += 1
    if self.candidate:
      if self.device_factory is None:
        from tinygrad import Device
        device = Device[Device.DEFAULT]
      else: device = self.device_factory()
      module, scales = self.packed
      self.session = self.stack.enter_context(self.hook.install(model, module, self.kernel, device, scales,
        qualified_rows=(2048,), dispatch=self.mlp_dispatch))
      self.events.append('language_fusion')
    self.events.append('factory_ready')

  def install_sampler(self, stream):
    if self.sampler is not None: raise ValueError('Duplicate native sampler installation')
    implementation = self.common.load(self.common.SAMPLER/'agent-warp-fast-sampler.py', 'combined_frozen_sampler')
    implementation.ARTIFACTS = self.common.HERE/'agent-exact-combined-artifacts'
    check = self.common.read(self.common.SAMPLER/'agent-warp-fast-validated-portable-v6.json')
    config = check['variants'][0]
    self.sampler = implementation.Sampler(config['kind'], config['workers'],
      Path(config['provenance']['command'][0]).name, config['provenance']['native'], config['max_maps'], config['max_bytes'])
    self.stack.callback(self.sampler.close)
    if self.sampler.lib.provenance['library_sha256'] != config['provenance']['library_sha256']:
      raise ValueError('Sampler binary differs from CPU qualification')
    def sample(*values, **options):
      self.sampler_calls += 1
      return self.sampler.calibrated_image(*values, **options)
    self.patched(stream, 'calibrated_image', sample)
    self.patched(stream, 'library', lambda: self.sampler.lib)
    self.events.append('qualified_cpu_sampler')

  def clear_vision_graphs(self):
    for group in (self.cache.encoders, self.cache.stitchers):
      for jit in group.values(): jit.reset()
      group.clear()
    self.cache.cache.clear()
    self.restorations.append(dict(name='candidate_vision_graphs_and_features_cleared', restored=True))

  @staticmethod
  def names(jit):
    from tinygrad.uop.ops import Ops
    captured = getattr(jit, 'captured', None)
    if captured is None: return []
    names = []
    for call in captured._linear.src:
      call = call.without_after
      aux = getattr(call.arg, 'aux', None)
      if hasattr(aux, 'kernels'): names.extend(str(k[1]) for k in aux.kernels)
      elif call.op is Ops.CALL and call.body.op is Ops.PROGRAM: names.append(call.body.src[0].arg.name)
    return names

  def coverage(self):
    if self.arm_calls != 1 or self.cache is None: raise ValueError('Expected one complete composed factory')
    language = [{'jit':name, 'fused_fc1':sum(n.startswith('agent_packed_epilogue') for n in self.names(jit))}
                for name, jit in self.hook.prefill_jits(self.model)]
    vision = []
    for shape, jit in self.cache.encoders.items():
      names = self.names(jit)
      vision.append(dict(shape=[list(s) for s in shape], fc1=sum(n.startswith('agent_vision_tiles_3x3') for n in names),
                         fc2=sum(n.startswith('agent_vision_tiles_3x6') for n in names)))
    result = dict(language_graphs=language, vision_graphs=vision, events=self.events,
                  sampler_calls=self.sampler_calls, sampler=None if self.sampler is None else self.sampler.state())
    if not self.candidate:
      if any(r['fused_fc1'] for r in language) or any(r['fc1']+r['fc2'] for r in vision):
        raise ValueError('Reference graph contains candidate kernels')
      return result
    stats = self.session.stats
    expected_layers = {self.hook.PREFIX+str(i) for i in range(28)}
    if set(stats['layer_hits']) != expected_layers or len(set(stats['layer_hits'].values())) != 1:
      raise ValueError('Incomplete 28-layer MLP coverage')
    if not (stats['python_trace_fc1'] == stats['python_trace_fc2'] == stats['python_trace_fused_quantize'] == stats['layer_completions'] > 0):
      raise ValueError('Language dispatch counts differ')
    if stats['failed_text_layers'] or set(stats['shape_hits']) != {'2048'}:
      raise ValueError('Unexpected language shape or failure')
    if not any(r['fused_fc1'] == 28 for r in language): raise ValueError('Missing captured 28-kernel MLP graph')
    traces = self.vision_info['trace_calls']
    batches = (1, 2) if self.stage == 'quality' else (2,)
    for batch in batches:
      required = {f'vlm.model.visual.encoder.layers.{i}.mlp.fc{fc}/batch{batch}' for i in range(2,27) for fc in (1,2)}
      if not required <= {name for name,count in traces.items() if count > 0}:
        raise ValueError(f'Missing 50 vision projections for batch {batch}')
    if not any(r['fc1'] == r['fc2'] == 25 for r in vision): raise ValueError('Missing captured 50-kernel vision graph')
    if self.stage == 'native' and (self.sampler is None or self.sampler_calls <= 0 or self.sampler.misses <= 0):
      raise ValueError('Native preparation did not use the qualified sampler')
    if self.stage == 'quality' and self.sampler_calls: raise ValueError('Prepared quality unexpectedly used native sampling')
    result.update(language_stats=stats, vision_hook=self.vision_info,
                  qualification='Original micro gates retained; combined GPU exactness requires this paired route')
    return result
