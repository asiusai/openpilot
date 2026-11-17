#!/usr/bin/env python3
"""Focused CPU-only runner gates, lifecycle, paired-control, and full-cohort mock checks."""
from contextlib import ExitStack
import copy
import hashlib
import importlib.machinery
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

EXP = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('int4model_runner', EXP / 'agent-int4model-run.py')
r = importlib.util.module_from_spec(spec)
spec.loader.exec_module(r)
p = r.ptq


def require(condition, message):
  if not condition:
    raise AssertionError(message)


def rejects(function, exception=ValueError):
  try:
    function()
  except exception:
    return
  raise AssertionError('Expected rejection')


class Jit:
  def __init__(self, function):
    self.function = function
    self.resets = 0

  def reset(self):
    self.resets += 1

  def __call__(self, *args, **kwargs):
    return self.function(*args, **kwargs)


class Model:
  def __init__(self, layers=1):
    self.config = {'vlm_config': {'text_config': {'hidden_act': 'relu2', 'mlp_bias': False,
                   'num_hidden_layers': layers, 'hidden_size': 32, 'intermediate_size': 32}}}
    self.weights, self.weight_scales = {}, {}
    for layer in range(layers):
      name = f'vlm.model.language_model.layers.{layer}.mlp.fc2.weight'
      self.weights[name] = np.eye(32, dtype=np.float32)
      self.weight_scales[name] = np.ones(1, np.float32)
    self.prefill_jit = Jit(self._language)
    self.decode_jit, self.greedy_decode_jit, self.trajectory_jits = Jit(lambda: None), Jit(lambda: None), {}
    self.fp8_calls, self.operation_calls = 0, []
    self.test_name = None

  def quantize(self, x):
    self.fp8_calls += 1
    return x.copy(), np.ones(1, np.float32)

  def linear(self, x, name, quantized_input=None, heads=None):
    self.operation_calls.append(name)
    q, _ = self.quantize(x) if quantized_input is None else quantized_input
    return q.copy()

  def text_layer(self, x, cos, sin, prefix, cache=None, mask=None, cache_pos=None):
    return self.linear(x, self.test_name or prefix + '.mlp.fc2')

  def _language(self, x):
    for layer in range(self.config['vlm_config']['text_config']['num_hidden_layers']):
      x = self.text_layer(x, None, None, f'vlm.model.language_model.layers.{layer}', cache_pos=0)
    return x

  def synchronize(self):
    pass


def prediction():
  return dict(xyz=np.zeros((1, 64, 3)).tolist(), rot=np.broadcast_to(np.eye(3), (1, 64, 3, 3)).tolist(),
              generated_tokens=2, generated_token_ids=[42, 135084], reasoning='nominal', timings_ms={'total': 1.})


def synthetic_reference(model):
  ref = dict(schema=p.SCHEMA, checkpoint=dict(config=model.config,
             canonical_config_sha256=hashlib.sha256(p.canonical(model.config)).hexdigest()),
             fp8_cache=dict(weights_sha256='mock'), settings=dict(precision='fp8', candidate_installed_during_collection=False))
  ref['identity_sha256'] = hashlib.sha256(p.canonical(ref)).hexdigest()
  return ref


def test_guards():
  rng = np.random.default_rng(445)
  model, control = Model(), Model()
  # Simulate an existing packed-quantization install that intercepts every linear.
  before = model.linear
  def packed_linear(x, name, quantized_input=None, heads=None):
    return before(x, name, quantized_input, heads)
  model.linear = packed_linear
  config_hash = hashlib.sha256(p.canonical(model.config)).hexdigest()
  name = 'vlm.model.language_model.layers.0.mlp.fc2'
  w = np.eye(32, dtype=np.float32)
  qw, sw = p.quantize_weight(w)
  weights = {name: dict(packed=p.pack_codes(qw, True), scale=sw, shape=w.shape)}
  manifest = dict(schema=p.SCHEMA, complete=True, layout=p.LAYOUT, equalization=None, clipping=None,
    outlier_correction=None, checkpoint=dict(canonical_config_sha256=config_hash), records=[dict(name=name, shape=list(w.shape))])
  counts = {}
  def quantizer(x):
    q, s = p.quantize_activation(x)
    return p.pack_codes(q, False), s
  def kernel(a, b, scales, *, rows, columns, bias, dtype):
    return p.scaled_reference(p.unpack_codes(a, rows, False), p.unpack_codes(b, columns, True), *scales)
  x = p.round_bf16(np.abs(rng.normal(size=(1, 160, 32))).astype(np.float32))
  original = model.linear, model.text_layer, model.quantize
  with ExitStack() as stack:
    r.install_candidate_guards(model, manifest, weights, quantizer, kernel, stack, counts)
    result = model._language(x)
    qa, sa = p.quantize_activation(x.reshape(160, 32))
    np.testing.assert_array_equal(result.reshape(160, 32), p.scaled_reference(qa, qw, sa, sw))
    require(model.fp8_calls == 0 and counts[name] == 1, 'fc2 was quantized twice')
    # Independent unchanged-operation checks use the same raw inputs and an unwrapped control.
    operations = [
      ('vlm.model.language_model.layers.0', name, 12, 160),
      ('vlm.model.language_model.layers.0', name, None, 160),
      ('vlm.model.language_model.layers.0', name, 0, 1),
      ('vlm.model.language_model.layers.0', 'vlm.model.language_model.layers.0.mlp.fc1', 0, 160),
      ('vlm.model.language_model.layers.0', 'vlm.model.language_model.layers.0.self_attn.q_proj', 0, 160),
      ('vlm.model.visual.encoder.layers.0', 'vlm.model.visual.encoder.layers.0.mlp.fc2', 0, 160),
      ('expert.expert.layers.0', 'expert.expert.layers.0.mlp.fc2', None, 64),
    ]
    for prefix, operation, pos, rows in operations:
      model.test_name = control.test_name = operation
      a = model.text_layer(x[:, :rows], None, None, prefix, cache_pos=pos)
      b = control.text_layer(x[:, :rows], None, None, prefix, cache_pos=pos)
      np.testing.assert_array_equal(a, b)
    require(model.fp8_calls == len(operations) and counts[name] == 1, 'Non-target operation dispatch changed')
  require((model.linear, model.text_layer, model.quantize) == original, 'Candidate guards were not restored')
  # A broken A4 callback entering original quantization must fail instead of running twice.
  model.test_name = None
  with ExitStack() as stack:
    def broken_quantizer(x):
      model.quantize(x)
      return quantizer(x)
    r.install_candidate_guards(model, manifest, weights, broken_quantizer, kernel, stack, {})
    rejects(lambda: model._language(x))


def test_gate(root):
  sources = {str(EXP / name): p.sha_file(EXP / name) for name in
             ('agent-int4-scaled-kernel.py', 'agent-int4k32-kernel.py', 'wgp_compiler.py')}
  mode = dict(complete=True, all_pass=True, activation_tail_exact=True,
    small_controls=[dict(name=n, shape=[2112, 2048, 9216], raw_all_exact=True, scaled_bias_and_no_bias_all_exact=True)
                    for n in ('original', 'activation-outlier', 'weight-tail')],
    captured_mutations=[dict(change=n, all_bits_exact=True) for n in ('original', 'activation', 'weight-scale-bias')])
  report = dict(cpu_only=False, complete=True, all_pass=True, modes={'cu': mode}, source_hashes=sources)
  file = root / 'mock-gate.json'
  p.json_write(file, report)
  r.check_kernel_gate(file, 'cu', sources.copy())
  for key in ('cpu_only', 'complete', 'all_pass'):
    bad = copy.deepcopy(report)
    bad[key] = True if key == 'cpu_only' else False
    p.json_write(file, bad)
    rejects(lambda: r.check_kernel_gate(file, 'cu', sources.copy()))
  bad = copy.deepcopy(report)
  bad['source_hashes'][str(EXP / 'agent-int4-scaled-kernel.py')] = 'old'
  p.json_write(file, bad)
  rejects(lambda: r.check_kernel_gate(file, 'cu', sources.copy()))
  p.json_write(file, report)
  rejects(lambda: r.check_kernel_gate(file, 'wgp', sources.copy()))
  tested, active = root / 'tested-tree' / 'experiments', root / 'active-tree' / 'experiments'
  tested.mkdir(parents=True)
  active.mkdir(parents=True)
  relocated = copy.deepcopy(report)
  relocated['source_hashes'] = {}
  for source, digest in sources.items():
    name = Path(source).name
    for destination in (tested / name, active / name):
      destination.write_bytes(Path(source).read_bytes())
    relocated['source_hashes'][str(tested / name)] = digest
  p.json_write(file, relocated)
  previous = r.EXP
  try:
    r.EXP = active
    active_sources = {str(active / Path(name).name): digest for name, digest in sources.items()}
    gate = r.check_kernel_gate(file, 'cu', active_sources.copy())
    require(gate['active_sources'] == active_sources, 'Frozen-tree relocation lost source identity')
    changed = active / 'agent-int4-scaled-kernel.py'
    original = changed.read_bytes()
    changed.write_bytes(original + b'\n# mutated active source\n')
    rejects(lambda: r.check_kernel_gate(file, 'cu', active_sources.copy()))
    changed.write_bytes(original)
    (tested / changed.name).write_bytes(original + b'\n# mutated tested source\n')
    rejects(lambda: r.check_kernel_gate(file, 'cu', active_sources.copy()))
  finally:
    r.EXP = previous


def test_verification(root):
  cache = root / 'cache.bin'
  payload = bytes(range(64))
  cache.write_bytes(payload)
  index = {'records': [dict(name='weight', kind='fp8', packed=True, offset=3, nbytes=25),
                       dict(name='scale', kind='scale', packed=False, offset=32, nbytes=4)]}
  calls = []
  def reader(record, start, count):
    calls.append(count)
    return payload[record['offset'] + start:record['offset'] + start + count]
  result = r.verify_loaded_records(index, cache, reader, chunk_bytes=7)
  require(result['tensor_bytes'] == 29 and max(calls) <= 7, 'Readbacks were not bounded')
  rejects(lambda: r.verify_loaded_records(index, cache, lambda rec, start, count: bytes(count), chunk_bytes=7))
  rejects(lambda: r.verify_loaded_records(index, cache, reader, chunk_bytes=32 * 1024**2))


def test_controls():
  good = prediction()
  r.finite_prediction(good)
  for key in ('xyz', 'rot'):
    bad = copy.deepcopy(good)
    array = np.asarray(bad[key]); array.flat[0] = np.nan; bad[key] = array.tolist()
    rejects(lambda: r.finite_prediction(bad))
  before = dict(sample='sample', prepared={'sha256': 'same'}, prediction=good)
  after = copy.deepcopy(before)
  require(r.compare_control(before, after)['xyz_max_delta_m'] == 0, 'Identical control drift')
  after['prediction']['xyz'][0][0][1] = 1
  require(r.compare_control(before, after)['xyz_max_delta_m'] == 1, 'Control comparison missed a changed coordinate')
  after['prepared']['sha256'] = 'different'
  rejects(lambda: r.compare_control(before, after))


def test_loaded_constructor(root):
  cache = root / 'constructor-cache'
  cache.mkdir()
  (cache / 'weights.bin').write_bytes(b'original-fp8-scale-bytes')
  index = {'records': [dict(name='w', kind='fp8', packed=True, offset=0, nbytes=12)]}
  for corrupt in (False, True):
    class Loaded:
      def __init__(self): self.origin = 'default'
    original = Loaded.__init__
    def bulk(self): self.origin = 'bulk'
    Loaded.__init__ = bulk  # The existing benchmark-fast import has this side effect.
    module = SimpleNamespace(EdgeModel=Loaded, original_init=original)
    args = SimpleNamespace(mode='collect')
    verified = SimpleNamespace(cache=cache, cache_index=index)
    payload = (cache / 'weights.bin').read_bytes()
    def reader(model):
      require(model.origin == 'bulk', 'Verification ran before original model loading')
      return lambda rec, start, count: (b'X' * count if corrupt else payload[start:start + count])
    with ExitStack() as stack, patch.object(r, 'gpu_record_reader', reader):
      hooks = r.RunnerHooks(args, verified, stack)
      hooks.patch_module(module, EXP / 'benchmark-fast.py')
      if corrupt:
        rejects(Loaded)
      else:
        Loaded()
        require(hooks.loaded_verification['all_bytes_equal'], 'Loaded weight proof was not retained')
    require(Loaded.__init__ is original, 'benchmark-fast initializer hook leaked')


def test_settings():
  settings = json.loads((EXP / 'agent-int4model-reference-settings.json').read_text())
  file = Path('/mnt/nvme-500/tmp/alpamayo-speed-20261116/full-fixed-vision-quality/invocation.json')
  saved = json.loads(file.read_text())
  reference = json.loads((EXP / 'agent-int4model-artifacts/reference-provenance.json').read_text())['reference']
  blocks, _ = r.option_blocks(saved['command'])
  r.validate_reference_settings(blocks, reference)
  for flag, values in (('--steps', ['2']), ('--bf16-vision-layers', ['0']), ('--history-pixel-scale', ['.5']),
                       ('--front-history-indices', ['0,2,4']), ('--row-quantize', [])):
    bad = copy.deepcopy(blocks); bad[flag] = values
    rejects(lambda: r.validate_reference_settings(bad, reference))
  dynamic = copy.deepcopy(blocks); del dynamic['--prefill-calibration']
  rejects(lambda: r.validate_reference_settings(dynamic, reference))
  swapped = copy.deepcopy(blocks); swapped['--prefill-calibration'] = blocks['--vision-calibration']
  rejects(lambda: r.validate_reference_settings(swapped, reference))
  require(settings['diffusion_steps'] == 3, 'Frozen reference changed')


def test_full_collection(root, fail=False):
  """Exercise actual frozen Collector for exactly 32x28 captures through fake imports."""
  base = root / ('failure' if fail else 'full')
  base.mkdir()
  samples = []
  for i in range(32):
    directory = base / f'sample-{i}'
    directory.mkdir()
    for filename in p.INPUT_FILES:
      np.save(directory / filename, np.array([i], np.int32), allow_pickle=False)
    samples.append(str(directory.resolve()))
  calibration, reserved = base / 'allowed.json', base / 'reserved.json'
  p.json_write(calibration, samples); p.json_write(reserved, [str(base / 'not-readable')])
  selection = p.CalibrationSelection(calibration, reserved)
  model = Model(28)
  class Padded:
    def __init__(self):
      def forbidden_replay(*args):
        raise AssertionError('Collector replayed a stale padded prefill JIT')
      self.jit = Jit(forbidden_replay)
    def language(self, x):
      return model._language(x)
    def __call__(self, x):
      return self.jit(x)
  model.padded_prefill = Padded()
  original_padded_jit = model.padded_prefill.jit
  reference = synthetic_reference(model)
  args = SimpleNamespace(mode='collect', output=base / 'output.json', capture_dir=base / 'capture', kernel_mode='cu')
  plan = dict(schema=r.SCHEMA, mode='collect', environment={'DEV': 'DO_NOT_INITIALIZE_GPU', 'PROFILE': '0'},
    sources={}, resolved_command=['mock-pipeline.py'], schedule=samples, warmup=0, protocol={'mock': True})
  verified = SimpleNamespace(plan=plan, reference=reference, selection=selection,
    blocks={'--packed-quantize': [], '--pad-prompt-to': ['2112']},
    manifest={'records': []}, cache=base, cache_index={'records': []}, paired=None)
  x = np.ones((1, 160, 32), np.float32)
  metadata = dict(camera_indices=np.array([6, 1]), absolute_timestamps=np.array([20, 10]), nav_text=['Continue straight'])
  prepared = {'tokenized_data': {'pixel_values': np.ones((8, 32), np.uint8)},
              'ego_history_xyz': np.zeros((1, 1, 20, 3), np.float32), 'ego_history_rot': np.zeros((1, 1, 20, 3, 3), np.float32),
              'actual_prompt_length': 129}
  packed = (np.ones((1, 160), np.int64), np.zeros((3, 1, 160), np.int64))
  class Adapter:
    def prepare(self, data):
      return None, None, dict(prepared)
  class Decoder:
    def __init__(self, value):
      self.model = value
    def run(self, model, settings, config, tokenizer, prepared, packed, seed, inputs):
      if fail:
        raise RuntimeError('injected pipeline failure')
      model.padded_prefill(x)
      return prediction()
  adapter_original, init_original, run_original = Adapter.prepare, Decoder.__init__, Decoder.run
  # A fake loader supplies harmless CPU modules through the actual import-hook path.
  class Loader:
    def __init__(self, file): self.file = file
    def create_module(self, spec): return None
    def exec_module(self, module):
      if self.file == 'fast-inputs.py': module.FastInputs = Adapter
      elif self.file == 'block-decode.py': module.BlockDecoder = Decoder
      elif self.file in ('quantize-packed.py', 'padded-prefill.py'): module.install = lambda model: 28
  def fake_spec(name, location, *a, **kw):
    return importlib.machinery.ModuleSpec(name, Loader(Path(location).name))
  def imported(name):
    spec = importlib.util.spec_from_file_location(name, EXP / (name + '.py'))
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module
  def pipeline_main():
    adapter = imported('fast-inputs').FastInputs()
    imported('quantize-packed').install(model)
    imported('padded-prefill').install(model)
    # Inject the completed tiny loaded-byte proof before the real attach check.
    state[0].loaded_verification = {'complete': True, 'all_bytes_equal': True, 'records': []}
    decoder = imported('block-decode').BlockDecoder(model)
    queries = []
    for sample in samples:
      _, _, data = adapter.prepare(metadata)
      result = decoder.run(model, SimpleNamespace(diffusion_steps=3), None, None, data, packed, 0, None)
      queries.append({'sample': sample, 'prediction': result})
    p.json_write(args.output, {'warmup': [], 'runs': queries, 'summary': {'profiling_enabled': False}})
  state = []
  original_hooks, original_collector = r.RunnerHooks, p.Collector
  class Observed(original_hooks):
    def __init__(self, *a, **kw): super().__init__(*a, **kw); state.append(self)
  def cpu_collector(*a, **kw):
    return original_collector(*a, capture=p.capture_numpy, **kw)
  env, argv, spec_before, path_before = dict(os.environ), sys.argv, importlib.util.spec_from_file_location, list(sys.path)
  with patch.object(r, 'RunnerHooks', Observed), patch.object(p, 'Collector', cpu_collector), \
       patch.object(r, 'local', lambda name: SimpleNamespace(main=pipeline_main)), \
       patch.object(importlib.util, 'spec_from_file_location', fake_spec):
    if fail:
      rejects(lambda: r.execute(args, verified), RuntimeError)
    else:
      r.execute(args, verified)
  require(Adapter.prepare is adapter_original and Decoder.__init__ is init_original and Decoder.run is run_original, 'Class hooks leaked')
  require(model.padded_prefill.jit is original_padded_jit, 'Padded-prefill JIT was not restored')
  require(dict(os.environ) == env and sys.argv is argv and importlib.util.spec_from_file_location is spec_before and sys.path == path_before,
          'Process hooks/environment leaked')
  side = json.loads(args.output.with_suffix('.int4model.json').read_text())
  require(side['hooks_restored'] and side['complete'] is not fail, 'Restoration/completion state incorrect')
  if not fail:
    manifest = json.loads((args.capture_dir / 'manifest.json').read_text())
    require(manifest['complete'] and len(manifest['samples']) == 32 and sum(len(s['layers']) for s in manifest['samples']) == 896,
            'Full frozen collector did not complete exactly 32x28')
    report = json.loads(args.output.read_text())
    require(report['summary']['profiling_enabled'], 'Collect timing was not marked profiling')
    controls = json.loads(r.controls_path(args.output).read_text())
    require(len(controls['queries']) == 32 and len(set(q['sample'] for q in controls['queries'])) == 32, 'Duplicate warmup/sample')
    with np.load(args.capture_dir / 'prepared-00.npz', allow_pickle=False) as archive:
      np.testing.assert_array_equal(archive['input_ids'], packed[0].astype(np.int32))
      np.testing.assert_array_equal(archive['pixel_values'], prepared['tokenized_data']['pixel_values'])
      require(int(archive['valid_rows']) == 129, 'Unpadded length not recorded')


def main():
  artifacts = EXP / 'agent-int4model-run-artifacts'
  artifacts.mkdir(exist_ok=True)
  with tempfile.TemporaryDirectory(prefix='mock-', dir=artifacts) as directory:
    root = Path(directory)
    test_guards()
    test_gate(root)
    test_verification(root)
    test_controls()
    test_loaded_constructor(root)
    test_settings()
    test_full_collection(root)
    test_full_collection(root, fail=True)
  require('tinygrad' not in sys.modules and 'torch' not in sys.modules, 'CPU runner checks imported GPU/model runtimes')
  result = dict(cpu_only=True, all_pass=True, full_cohort_mock=dict(samples=32, layers=28, captures=896),
    checks=['candidate outside packed wrapper, no duplicate FP8 quantization', 'independent unchanged non-target controls',
            'kernel gate completeness, source and mode rejection', 'bounded actual-byte callback verification and mutation rejection',
            'paired-control serialization and finite-output rejection', 'incompatible baseline precision/history/scale-policy rejection',
            'loaded-model constructor verification and original initializer restoration',
            'full 32x28 frozen collector through packed/padded import hooks with actual prepared arrays and forbidden stale JIT replay',
            'finally restoration after injected failure'],
    source_sha256=p.sha_file(EXP / 'agent-int4model-run.py'))
  p.json_write(artifacts / 'runner-cpu-check.json', result)
  print(json.dumps(result, indent=2))


if __name__ == '__main__': main()
