"""CPU checks for the optional installer and existing packed-quantizer protocol."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import copy

import numpy as np
from tinygrad import Tensor, dtypes
from amd_gemm import pack, unpack


def check(fusion):
  checks, calls = [], []
  original_project = fusion.project

  def spy(x, quantized, prepared, packed_type):
    rows, width = x.numel()//x.shape[-1], x.shape[-1]
    a = fusion.activation(quantized[0], rows, width, packed_type)
    assert a.shape == ((rows+15)//16*16, width)
    assert quantized[1].numel() == 1
    np.testing.assert_array_equal(prepared[1].numpy(), np.concatenate([np.full(n, s, np.float32) for n, s in zip(fusion.SIZES, (.25, .5, .125))]))
    calls.append((a, quantized[1], prepared))
    values = Tensor(np.arange(sum(prepared[3]), dtype=np.float32), device='CPU').cast(dtypes.bfloat16).reshape(1, -1).expand(rows, -1)
    return tuple(values[:, begin:begin+n].reshape(*x.shape[:-1], n) for begin, n in zip((0, 2048, 3072), prepared[3]))

  class Reset:
    def __init__(self): self.count = 0
    def reset(self): self.count += 1

  class Model:
    def __init__(self, root, width, bias='none'):
      self.dtype, self.fp8_wmma = dtypes.bfloat16, True
      self.weights, self.weight_scales = {}, {}
      self.quantized_calls, self.old_linear_calls, self.order = 0, 0, 'qkv'
      self.prefix = root+'0'
      self.prefill_jit, self.decode_jit, self.greedy_decode_jit = Reset(), Reset(), Reset()
      self.trajectory_jits = {'old': object()}
      self.padded_prefill = SimpleNamespace(jit=Reset(), shape='old')
      for i, (c, n) in enumerate(zip('qkv', fusion.SIZES)):
        key = self.prefix+'.self_attn.'+c+'_proj'
        bits = np.full((n, width), 56+i, np.uint8)
        self.weights[key+'.weight'] = unpack(pack(Tensor(bits, device='CPU').bitcast(dtypes.fp8e4m3)).realize())
        self.weight_scales[key+'.weight'] = Tensor(np.array([[.25, .5, .125][i]], np.float32), device='CPU').reshape(()).realize()
        if bias == 'all' or (bias == 'mixed' and i == 1):
          self.weights[key+'.bias'] = Tensor.full((n,), i/8, device='CPU', dtype=dtypes.bfloat16).contiguous().realize()

    def quantize(self, x):
      self.quantized_calls += 1
      return (Tensor.ones(x.numel()//x.shape[-1], x.shape[-1], device='CPU', dtype=dtypes.fp8e4m3).contiguous().realize(),
              Tensor([.125], device='CPU', dtype=dtypes.float32).realize())

    def linear(self, x, name, quantized_input=None, heads=None):
      self.old_linear_calls += 1
      n = self.weights[name+'.weight'].shape[0]
      return Tensor.zeros(*x.shape[:-1], n, device='CPU', dtype=dtypes.bfloat16)

    def attention(self, q, k, v, mask=None, causal=False): return q

    def norm(self, x, name, rms=False, eps=None): return x

    def text_layer(self, x, cos, sin, prefix, *args, **kwargs):
      quantized = self.quantize(x)
      return tuple(self.linear(x, prefix+'.self_attn.'+c+'_proj', quantized, heads=16 if c == 'q' else 8) for c in self.order)

  def rejected(fn, reason):
    try: fn()
    except (AssertionError, TypeError) as exc:
      assert reason in str(exc), str(exc)
    else: raise AssertionError('Expected rejection: '+reason)

  fusion.project = spy
  try:
    for scope, root, width, rows in (('expert', fusion.ROOTS['expert'][0], 512, 64), ('prefill', fusion.ROOTS['prefill'][0], 2048, 128)):
      for bias in ('none', 'all'):
        model = Model(root, width, bias)
        if scope == 'expert':
          spec = importlib.util.spec_from_file_location('agent_qkv_host_expert', Path(__file__).with_name('expert-kernels.py'))
          expert = importlib.util.module_from_spec(spec)
          spec.loader.exec_module(expert)
          assert expert.install(model) == 3
        packed_type = None
        packed_calls = []
        if scope == 'prefill':
          spec = importlib.util.spec_from_file_location('agent_qkv_host_packed', Path(__file__).with_name('quantize-packed.py'))
          packed = importlib.util.module_from_spec(spec)
          spec.loader.exec_module(packed)
          def quantize_cpu(x, scale=None):
            packed_calls.append(scale)
            raw = Tensor.ones(x.numel()//x.shape[-1], x.shape[-1], device='CPU', dtype=dtypes.fp8e4m3).contiguous()
            return pack(raw).realize(), scale
          packed.quantize = quantize_cpu
          packed.install(model, fixed_scales={model.prefix+'/0': .125}, capture_scales=True)
          packed_type = packed.PackedActivation
        spec = importlib.util.spec_from_file_location('agent_qkv_host_vision', Path(__file__).with_name('batched-vision.py'))
        vision = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(vision)
        vision.install(model)
        old_linear, old_layer = model.linear, model.text_layer
        x = Tensor.zeros(1, rows, width, device='CPU', dtype=dtypes.bfloat16).contiguous().realize()
        before = len(calls)
        with fusion.install(model, packed_type, scope) as stats:
          result = model.text_layer(x, None, None, model.prefix, cache_pos=0)
          assert tuple(t.shape for t in result) == tuple((1, rows, n) for n in fusion.SIZES)
          assert len(calls) == before+1 and model.old_linear_calls == 0
          assert (len(packed_calls) if scope == 'prefill' else model.quantized_calls) == 1
          if scope == 'prefill':
            assert model.prefill_activation_scales[model.prefix+'/0'] is packed_calls[0]
            assert calls[-1][1] is packed_calls[0]
          # The installer leaves non-QKV linears and non-target row counts on the old path.
          short = Tensor.zeros(1, 1, width, device='CPU', dtype=dtypes.bfloat16)
          model.text_layer(short, None, None, model.prefix, cache_pos=1)
          assert model.old_linear_calls == 3 and len(calls) == before+1
          model.order = 'kqv'
          rejected(lambda: model.text_layer(x, None, None, model.prefix, cache_pos=0), 'call order')
          model.order = 'qk'
          rejected(lambda: model.text_layer(x, None, None, model.prefix, cache_pos=0), 'Incomplete')
          model.order = 'qkv'
          model.text_layer(x, None, None, model.prefix, cache_pos=0)
          key = model.prefix+'.self_attn.k_proj.weight'
          saved = model.weight_scales[key]
          model.weight_scales[key] = Tensor([.75], device='CPU', dtype=dtypes.float32).realize()
          rejected(lambda: model.text_layer(x, None, None, model.prefix, cache_pos=0), 'Immutable')
          model.weight_scales[key] = saved
          assert stats['additional_immutable_bytes'] > 0
        assert model.linear == old_linear and model.text_layer == old_layer
        assert model.prefill_jit.count == model.padded_prefill.jit.count == 2
        assert not model.trajectory_jits and model.padded_prefill.shape is None
        checks.append({'scope': scope, 'bias': bias, 'shared_quantizer_exactly_once': True, 'restore_and_jit_reset': True,
                       'bad_order_partial_group_and_rebound_scale_rejected': True, 'decode_fallback': True,
                       'existing_expert_packed_quantizer_and_vision_hooks': True})
    # Mixed biases must retain the original per-projection absence semantics.
    model = Model(fusion.ROOTS['expert'][0], 512, 'mixed')
    rejected(lambda: fusion.prepare([model.weights[model.prefix+'.self_attn.'+c+'_proj.weight'] for c in 'qkv'],
                                   [model.weight_scales[model.prefix+'.self_attn.'+c+'_proj.weight'] for c in 'qkv'],
                                   [model.weights.get(model.prefix+'.self_attn.'+c+'_proj.bias') for c in 'qkv']), 'Mixed bias')
    rejected(lambda: fusion.activation(object(), 64, 512, None), 'Unknown quantized')
    checks.append({'mixed_bias_rejected_without_zero_fill': True, 'unknown_packed_type_rejected': True})
    spec = importlib.util.spec_from_file_location('agent_qkv_host_run', Path(__file__).with_name('agent-qkv-run.py'))
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    report = {'warmup': [], 'runs': [{'sample': 'cpu-fixture', 'input_latest_timestamp_us': 1,
               'prediction': {'generated_tokens': ['x'], 'generated_token_ids': [1], 'reasoning': 'x', 'xyz': [[-0.0, 1.0, 2.0]], 'rot': [[0.0]]}}]}
    assert runner.compare(report, copy.deepcopy(report))['paired_predictions'] == 1
    changed = copy.deepcopy(report)
    changed['runs'][0]['prediction']['xyz'][0][0] = 0.0
    rejected(lambda: runner.compare(changed, report), 'Exact xyz')
    changed = copy.deepcopy(report)
    changed['runs'][0]['prediction']['generated_token_ids'] = [2]
    rejected(lambda: runner.compare(changed, report), 'generated_token_ids')
    checks.append({'route_comparator_exact': True, 'signed_zero_and_token_changes_rejected': True})
  finally:
    fusion.project = original_project
  return checks
