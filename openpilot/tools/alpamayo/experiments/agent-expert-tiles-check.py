"""CPU dispatch, composition, mutation, provenance, and offline compiler checks."""
import argparse
import copy
from functools import partial
import importlib.util
import json
import os
from pathlib import Path
import sys
from types import MethodType, SimpleNamespace

import numpy as np

HERE = Path(__file__).resolve().parent
WORK = Path('/mnt/nvme-500/tmp/alpamayo-speed-20261116')


def load(name, directory=HERE):
  spec = importlib.util.spec_from_file_location('agent_expert_tiles_check_'+name.replace('-', '_'), directory/(name+'.py'))
  module = importlib.util.module_from_spec(spec)
  sys.modules[spec.name] = module
  spec.loader.exec_module(module)
  return module


def rejects(fn):
  try: fn()
  except (AssertionError, ValueError): return
  raise AssertionError('Invalid dispatch, provenance, or comparison passed')


def dispatch_checks(H):
  from tinygrad import Tensor, TinyJit, dtypes
  from tinygrad.device import Buffer
  E = load('expert-kernels')
  for name in ('expert.expert.layers.28.mlp.fc2', 'expert.expert.layers.-1.mlp.fc2', 'expert.expert.layers.00.mlp.fc2',
               'vlm.model.language_model.layers.0.mlp.fc2', 'expert.other.layers.0.mlp.fc2',
               'expert.expert.layers.0.mlp.fc20', 'expert.expert.layers.0.self_attn.q_proj'):
    assert H.NAME.fullmatch(name) is None
  logs, quantizers, fallbacks = [], [], []
  shapes = {'self_attn.q_proj': (2048, 512), 'self_attn.k_proj': (1024, 512), 'self_attn.v_proj': (1024, 512),
    'self_attn.o_proj': (512, 2048), 'mlp.fc1': (3840, 512), 'mlp.fc2': (512, 3840)}
  tensors = {shape: Tensor.ones(*shape, device='CPU', dtype=dtypes.fp8e4m3).realize() for shape in set(shapes.values())}
  model = SimpleNamespace(fp8_wmma=True, dtype=dtypes.bfloat16, weights={}, weight_scales={})
  for layer in range(28):
    for suffix, shape in shapes.items():
      name = f'expert.expert.layers.{layer}.{suffix}'
      model.weights[name+'.weight'] = tensors[shape]
      model.weight_scales[name+'.weight'] = Tensor(.25, dtype=dtypes.float32, device='CPU').realize()
  for name in ('vlm.model.language_model.layers.0.mlp.fc2', 'vlm.model.visual.encoder.layers.3.mlp.fc2', 'expert.action_out_proj'):
    model.weights[name+'.weight'] = tensors[(512, 3840)]
    model.weight_scales[name+'.weight'] = Tensor(.25, dtype=dtypes.float32, device='CPU').realize()
  marker = object()
  def prior_linear(self, x, name, quantized_input=None, heads=None):
    fallbacks.append((x, name, quantized_input, heads))
    return marker
  def quantize(self, x):
    quantizers.append('first')
    return (x.float()/2).cast(dtypes.fp8e4m3).reshape(-1, x.shape[-1]).contiguous(), Tensor(2., dtype=dtypes.float32, device='CPU').realize()
  model.linear, model.quantize = MethodType(prior_linear, model), MethodType(quantize, model)
  model.attention, model.text_layer = lambda *a, **k: None, lambda *a, **k: None
  # Stand-in arithmetic checks argument forwarding and graph replay. WMMA
  # arithmetic is covered by the passed GPU evidence and compiled binary hashes.
  def calculate(q, w, scales, bias):
    value = q[:, :1].float()*w[:, 0].float().reshape(1, -1)
    value = value*(scales[0]*scales[1])
    if bias is not None: value = value+bias.float()
    return value.cast(dtypes.bfloat16)
  def original_gemm(q, w, scales, bias=None, **options):
    logs.append(('original', q, w, scales, bias, options))
    return calculate(q, w, scales, bias)
  def selected_gemm(q, w, scales, bias, tile, tiled):
    logs.append(('candidate', q, w, scales, bias, tile))
    return calculate(q, w, scales, bias)
  original_e, original_h = E.gemm, H.gemm
  E.gemm, H.gemm = original_gemm, selected_gemm
  try:
    E.install(model)
    expert_original = model.linear
    protected = model.quantize, model.attention, model.text_layer
    handle = H.install(model)
    assert (model.quantize, model.attention, model.text_layer) == protected
    rejects(lambda: H.install(model))
    tested = []
    for layer in range(28):
      for suffix, (_, width) in shapes.items():
        name = f'expert.expert.layers.{layer}.{suffix}'
        x = Tensor.full((1, 64, width), 4, dtype=dtypes.bfloat16, device='CPU')
        before = len(quantizers)
        result = model.linear(x, name)
        assert len(quantizers) == before+1
        assert logs[-1][0] == ('candidate' if suffix in H.TILES else 'original'), (name, logs[-1][0], x.dtype, model.weights[name+'.weight'].shape, model.weight_scales[name+'.weight'].dtype)
        assert logs[-1][2] is model.weights[name+'.weight']
        assert logs[-1][3][1] is model.weight_scales[name+'.weight']
        if suffix in H.TILES:
          assert logs[-1][-1] == H.TILES[suffix]
          assert result.shape == (1, 64, 512) and result.dtype == dtypes.bfloat16
          np.testing.assert_array_equal(result.numpy(), np.ones((1, 64, 512), np.float32))
        tested.append({'layer': layer, 'projection': suffix, 'selected': suffix in H.TILES})
    assert len(handle.info['trace_calls']) == 56
    name = 'expert.expert.layers.0.mlp.fc2'
    x = Tensor.full((1, 64, 3840), 4, dtype=dtypes.bfloat16, device='CPU')
    # Replacing the quantizer after installation is observed by this hook.
    def later_quantize(self, value):
      quantizers.append('later')
      return (value.float()/4).cast(dtypes.fp8e4m3).reshape(-1, value.shape[-1]).contiguous(), Tensor(4., dtype=dtypes.float32, device='CPU').realize()
    model.quantize = MethodType(later_quantize, model)
    before = len(quantizers)
    model.linear(x, name).realize()
    assert len(quantizers) == before+1 and quantizers[-1] == 'later'
    supplied = later_quantize(model, x)
    before = len(quantizers)
    model.linear(x, name, supplied).realize()
    assert len(quantizers) == before and logs[-1][1] is supplied[0] and logs[-1][3][0] is supplied[1]
    # The same 64 flattened rows retain their original leading dimensions.
    for shape in ((64, 3840), (2, 32, 3840), (1, 2, 32, 3840)):
      result = model.linear(x.reshape(shape), name, supplied)
      assert result.shape == (*shape[:-1], 512)
    bias = Tensor.full((512,), .5, dtype=dtypes.bfloat16, device='CPU').realize()
    model.weights[name+'.bias'] = bias
    np.testing.assert_array_equal(model.linear(x, name).numpy(), np.full((1, 64, 512), 1.5, np.float32))
    assert logs[-1][4] is bias
    # Actual CPU capture reads input, weight, scale, and bias mutations in place.
    q = Tensor.ones(64, 3840, dtype=dtypes.fp8e4m3, device='CPU').clone().realize()
    sx = Tensor(4., dtype=dtypes.float32, device='CPU').clone().realize()
    weight = model.weights[name+'.weight'].contiguous().clone().realize()
    ws = model.weight_scales[name+'.weight'].clone().realize()
    bias = bias.clone().realize()
    graph = TinyJit(lambda a, b, c, d, e: H.gemm(a, b, (c, d), e, H.TILES['mlp.fc2'], None).realize())
    for _ in range(3): output = graph(q, weight, sx, ws, bias)
    baseline = output.numpy().copy()
    from amd_gemm import pack, unpack
    original_bytes = [bytes(t.uop.buffer.as_memoryview()) for t in (q, weight, sx, ws, bias)]
    def copy_into(tensor, payload):
      destination = tensor.uop.buffer
      destination.copy_from(Buffer('CPU', len(payload), initial_value=payload))
      assert tensor.uop.buffer is destination
    copy_into(q, np.full(q.shape, 0x40, np.uint8).tobytes())  # FP8 2
    copy_into(sx, np.asarray(2., np.float32).tobytes())
    copy_into(ws, np.asarray(.5, np.float32).tobytes())
    copy_into(bias, np.full(bias.shape, 0x3f80, np.uint16).tobytes())  # BF16 1
    changed = graph(q, weight, sx, ws, bias).numpy()
    np.testing.assert_array_equal(changed, np.full((64, 512), 3., np.float32))
    assert not np.array_equal(changed, baseline)
    copy_into(weight, np.full(weight.shape, 0x40, np.uint8).tobytes())
    np.testing.assert_array_equal(graph(q, weight, sx, ws, bias).numpy(), np.full((64, 512), 5., np.float32))
    for tensor, payload in zip((q, weight, sx, ws, bias), original_bytes, strict=True): copy_into(tensor, payload)
    np.testing.assert_array_equal(graph(q, weight, sx, ws, bias).numpy(), baseline)
    # Replacing a weight object also reaches the next uncaptured dispatch.
    new_weight = unpack(pack(Tensor.full(weight.shape, 2, dtype=dtypes.fp8e4m3, device='CPU')).realize())
    model.weights[name+'.weight'] = new_weight
    model.linear(x, name, supplied)
    assert logs[-1][2] is new_weight
    model.weights[name+'.weight'] = weight
    # Invalid precision, shape, or scope delegates to the exact previous method.
    # The original expert method remains active; inspect dispatch logs where it applies.
    for other in ('vlm.model.language_model.layers.0.mlp.fc2', 'vlm.model.visual.encoder.layers.3.mlp.fc2', 'expert.action_out_proj'):
      before_logs = len(logs)
      value = model.linear(x, other, supplied)
      assert len(logs) == before_logs or logs[-1][0] == 'original'
      if not other.startswith('expert.'): assert value is marker
    for changed_x, heads in ((x[:, :1], None), (x[:, :63], None), (x[:, :, :2048], None), (x.cast(dtypes.float32), None), (x, 16)):
      before_logs = len(logs)
      model.linear(changed_x, name, supplied, heads)
      assert len(logs) == before_logs or logs[-1][0] == 'original'
    for field, replacement in (('weight', Tensor.ones(512, 3840, dtype=dtypes.bfloat16, device='CPU')),
                                ('weight', Tensor.ones(512, 2048, dtype=dtypes.fp8e4m3, device='CPU')),
                                ('bias', Tensor.ones(512, dtype=dtypes.float32, device='CPU'))):
      key = name+'.'+field
      old = model.weights[key]
      model.weights[key] = replacement
      model.linear(x, name, supplied)
      assert logs[-1][0] == 'original'
      model.weights[key] = old
    before = len(quantizers)
    rejects(lambda: model.linear(x, name, (supplied[0], Tensor.ones(64, device='CPU'))))
    assert len(quantizers) == before
    # Cleanup preserves a later wrapper and disables only this selected branch.
    installed = model.linear
    def outer(self, *args, **kwargs): return installed(*args, **kwargs)
    later = MethodType(outer, model)
    model.linear = later
    handle.close()
    assert model.linear is later and not hasattr(model, 'agent_expert_tiles')
    model.linear(x, name, supplied)
    assert logs[-1][0] == 'original'
    handle.close()
    # Direct cleanup restores the exact prior bound method.
    model.linear = expert_original
    direct = H.install(model)
    direct.close()
    assert model.linear is expert_original
    # Compose the actual later language quantization wrapper. Expert calls must
    # still pass through it and reach the latest scalar quantizer exactly once.
    composed = H.install(model)
    load('quantize-packed').install(model)
    later_methods = model.linear, model.quantize, model.text_layer, model.attention
    before = len(quantizers)
    model.linear(x, name).realize()
    assert logs[-1][0] == 'candidate' and len(quantizers) == before+1
    assert quantizers[-1] == 'later'
    composed.close()
    assert (model.linear, model.quantize, model.text_layer, model.attention) == later_methods
    model.linear(x, name, supplied).realize()
    assert logs[-1][0] == 'original'
    return {'all_28_layers_six_projections': tested, 'eligible_projections': 56, 'quantizer_once_and_later_wrapper': True,
      'supplied_pair_reused_without_quantization': True, 'scale_weight_bias_identity': True,
      'leading_shapes_preserved': True, 'precision_scope_fallback': True,
      'captured_cpu_input_weight_bias_scale_mutation_and_restore': True, 'cleanup_preserves_later_wrappers': True,
      'actual_later_quantize_packed_wrapper_preserved': True,
      'arithmetic_scope': 'CPU stand-in verifies forwarding and replay. GPU WMMA qualification comes from pinned evidence.'}
  finally:
    E.gemm, H.gemm = original_e, original_h


def route_checks(H, R):
  source = HERE/'agent-expert-tiles-quality-invocation.json'
  template = json.loads(source.read_text())
  for name, digest in template['template_provenance'].items(): assert H.sha((WORK/name).read_bytes()) == digest
  native = json.loads((WORK/'dynamic-vision-native-5hz/invocation.json').read_text())
  assert native['environment'] == template['environment']
  command = native['command'].copy()
  for flag, value in (('--output', str(WORK/'agent-expert-tiles-quality-reference/report.json')),
                      ('--invocation', str(WORK/'dynamic-vision-protocol/dynamic-vision-full-route.json')), ('--warmup', 2), ('--runs', 62)):
    R.replace_option(command, flag, value)
  R.replace_option(command, '--samples-json', WORK/'quality-samples.json')
  assert command == template['command']
  plan = R.plan(source, Path('/tmp/opencode/agent-expert-tiles-reference.json'))
  other = R.plan(source, Path('/tmp/opencode/agent-expert-tiles-candidate.json'))
  assert plan['pair_identity_sha256'] == other['pair_identity_sha256'] and len(plan['scheduled_samples']) == 64
  rejects(lambda: R.inspect_invocation(WORK/'padding2048-quality/invocation.json'))
  prediction = {'generated_token_ids': [1, 2, 3], 'reasoning': 'NOMINAL', 'xyz': [[0., 1., 2.]], 'rot': [[[1., 0.], [0., 1.]]]}
  rows = [{'sample': sample, 'prediction': copy.deepcopy(prediction)} for sample in plan['scheduled_samples']]
  reference = {'warmup': rows[:2], 'runs': rows[2:]}
  assert R.compare_reports(reference, copy.deepcopy(reference), plan)['all_exact']
  for field in ('generated_token_ids', 'reasoning', 'xyz', 'rot', 'sample'):
    changed = copy.deepcopy(reference)
    if field == 'sample': changed['runs'][0]['sample'] += '-changed'
    elif field == 'reasoning': changed['runs'][0]['prediction'][field] = 'BRAKE'
    elif field == 'generated_token_ids': changed['runs'][0]['prediction'][field][0] += 1
    elif field == 'xyz': changed['runs'][0]['prediction'][field][0][0] = -0.
    else: changed['runs'][0]['prediction'][field][0][0][0] = float(np.nextafter(1., 2.))
    rejects(lambda: R.compare_reports(reference, changed, plan))
  changed = copy.deepcopy(reference)
  changed['runs'].pop()
  rejects(lambda: R.compare_reports(reference, changed, plan))
  # Nest another import interceptor, as agent-speculation-run does, and verify
  # both success and failure restore the preexisting hook.
  original = importlib.util.spec_from_file_location
  handles, records = [], []
  expert = Path(plan['terminal_command'][0]).with_name('expert-kernels.py')
  try:
    with R.install_interceptor(H, 'current', H.REPORT, expert, handles, records):
      outer = importlib.util.spec_from_file_location
      def nested(*args, **kwargs): return outer(*args, **kwargs)
      importlib.util.spec_from_file_location = nested
      spec = nested('agent_expert_tiles_cleanup_probe', expert)
      assert spec.loader.__class__.__name__ == 'Loader'
      raise RuntimeError('CPU cleanup probe')
  except RuntimeError: pass
  assert importlib.util.spec_from_file_location is original
  return {'quality_plan': plan, 'all_64_strict_field_comparisons': True, 'signed_zero_and_one_ulp_rejected': True,
    'changed_tokens_text_sample_and_missing_sample_rejected': True, 'fixed_vision_invocation_rejected': True,
    'nested_import_hook_cleanup': True}


def compiler_checks(H):
  from tinygrad import Device, Tensor, dtypes
  from tinygrad.helpers import Target
  from tinygrad.renderer.llvmir import AMDLLVMRenderer
  from amd_gemm import pack, unpack
  from unittest.mock import patch
  E, W = load('agent-expert-gemm'), load('wgp_compiler')
  renderer = AMDLLVMRenderer(Target('AMD', 'LLVM', 'gfx1200'))
  renderer.compiler = W.WGPCompiler('gfx1200')
  measured = json.loads(H.REPORT.read_text())
  records = []
  for suffix, shape in H.SHAPES.items():
    for bias in (False, True):
      tile = 'x'.join(map(str, H.TILES[suffix]))
      q = Tensor.empty(64, shape[2], dtype=dtypes.fp8e4m3, device='CPU')
      w = unpack(pack(Tensor.ones(512, shape[2], dtype=dtypes.fp8e4m3, device='CPU')).realize())
      scales = tuple(Tensor.ones(1, dtype=dtypes.float32, device='CPU') for _ in range(2))
      b = Tensor.zeros(512, dtype=dtypes.bfloat16, device='CPU') if bias else None
      captured = []
      def capture(*args, fxn):
        captured.append((args, fxn))
        return args
      with patch.object(Tensor, 'custom_kernel', side_effect=capture):
        out = H.gemm(q, w, scales, b, H.TILES[suffix], H.load_tiled())
      inputs, fxn = captured[0]
      assert out is inputs[0] and out.shape == (64, 512) and out.dtype == dtypes.bfloat16
      assert inputs[1].uop is pack(q).uop and inputs[2].uop is pack(w).uop
      assert inputs[3] is scales[0] and inputs[4] is scales[1]
      assert len(inputs) == (6 if bias else 5)
      if bias: assert inputs[5] is b
      fn = partial(E.bind_renderer(fxn.func, renderer), **fxn.keywords)
      result = E.compiler_record(shape, bias, fn, renderer)
      saved = next(c for c in measured['cases'] if c['case'] == ('fc2' if suffix == 'mlp.fc2' else 'o') and c['bias'] == bias)['variants'][tile]
      assert result['binary_sha256'] == saved['binary_sha256'], 'Compiler output differs from passed GPU kernel'
      records.append({'projection': suffix, 'bias': bias, 'actual_hook_pack_scale_bias_arguments_exact': True,
        'binary_matches_passed_gpu': True, **result})
  assert not any(name.startswith('AMD') for name in Device._opened_devices)
  return records


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--output', type=Path, required=True)
  parser.add_argument('--compile', action='store_true')
  args = parser.parse_args()
  os.environ.update(DEV='CPU', PROFILE='0', BEAM='0', TC_OPT='2', PYTHONDONTWRITEBYTECODE='1')
  H, R = load('agent-expert-tiles'), load('agent-expert-tiles-route')
  report = {'gpu_executed': False, 'complete': False, 'evidence': H.verify_evidence()}
  frozen = WORK/'exact-tuning-source/alpamayo/experiments'
  assert H.verify_evidence(source_dir=frozen)['source_sha256'] == report['evidence']['source_sha256']
  for name in H.PINS:
    path = (HERE.parent if name == 'amd_gemm.py' else HERE)/name
    rejects(lambda: H.checked_bytes(name, path.read_bytes()+b'\n'))
  report['source_relocation_and_changed_bytes_rejection'] = True
  index_path = Path('/dev/shm/alpamayo-9060-fp8/index.json')
  index = json.loads(index_path.read_text())
  targets = []
  for layer in range(28):
    for suffix, (_, columns, width) in H.SHAPES.items():
      name = f'expert.expert.layers.{layer}.{suffix}'
      weight = next(record for record in index['records'] if record['name'] == name+'.weight' and record['kind'] == 'fp8')
      assert tuple(weight['shape']) == (columns, width)
      assert any(record['name'] == name+'.weight' and record['kind'] == 'scale' for record in index['records'])
      targets.append({'name': name, 'shape': weight['shape'], 'bias_present': any(record['name'] == name+'.bias' for record in index['records'])})
  report['actual_cache_index'] = {'path': str(index_path), 'sha256': H.sha(index_path.read_bytes()), 'targets': targets}
  report['dispatch'] = dispatch_checks(H)
  report['route'] = route_checks(H, R)
  if args.compile: report['compiler'] = compiler_checks(H)
  from tinygrad import Device
  assert not any(name.startswith('AMD') for name in Device._opened_devices)
  report['complete'] = True
  report['sources'] = {str(path): H.sha(path.read_bytes()) for path in HERE.glob('agent-expert-tiles*') if path.is_file()}
  args.output.parent.mkdir(parents=True, exist_ok=True)
  args.output.write_text(json.dumps(report, indent=2)+'\n')
  print(json.dumps({'complete': True, 'gpu_executed': False, 'compiled': len(report.get('compiler', [])), 'output': str(args.output)}))


if __name__ == '__main__': main()
