"""Optional exact-arithmetic QKV concatenation for expert and language prefill.

Install after expert-kernels and quantize-packed, before JIT capture. Weights,
weight scales, and biases must remain immutable for this installation's lifetime.
GPU equality and route validation belong to the parent session.
"""
from contextlib import contextmanager
import importlib.util
from pathlib import Path
from types import MethodType

from tinygrad import Tensor, dtypes
from amd_gemm import pack

SPEC = importlib.util.spec_from_file_location('agent_qkv_tiled', Path(__file__).with_name('tiled-gemm.py'))
TILED = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(TILED)
SIZES = (2048, 1024, 1024)
ROOTS = {'expert': ('expert.expert.layers.', 512), 'prefill': ('vlm.model.language_model.layers.', 2048)}


def kernel(*args):
  # Keep the original 4x4 schedule, reduction loop, scale association, and cast.
  return TILED.kernel(*args, tile_m=4, tile_n=4, column_scale=True)


def prepare(weights, scales, biases):
  """Concatenate packed FP8 bytes and unmodified FP32 scales once, outside replay."""
  assert len(weights) == len(scales) == len(biases) == 3
  width, device = weights[0].shape[1], weights[0].device
  sizes = tuple(w.shape[0] for w in weights)
  assert all(w.dtype == dtypes.fp8e4m3 and w.shape[1] == width and w.shape[0] % 16 == 0 and w.device == device for w in weights)
  assert all(s.dtype == dtypes.float32 and s.numel() == 1 and s.device == device for s in scales)
  assert all(b is None for b in biases) or all(b is not None for b in biases), 'Mixed bias presence requires the original path'
  if biases[0] is not None:
    assert all(b.dtype == dtypes.bfloat16 and b.shape == (n,) and b.device == device for b, n in zip(biases, sizes))
  # Each projection has whole 16-row packed blocks. Concatenation is a byte copy.
  packed = Tensor.cat(*(pack(w) for w in weights), dim=0).contiguous().realize()
  columns = Tensor.cat(*(s.reshape(1).expand(n) for s, n in zip(scales, sizes)), dim=0).contiguous().realize()
  bias = None if biases[0] is None else Tensor.cat(*biases, dim=0).contiguous().realize()
  return packed, columns, bias, sizes


def activation(value, rows, width, packed_type):
  if isinstance(value, Tensor):
    assert value.dtype == dtypes.fp8e4m3 and value.shape == (rows, width)
    return pack(value)
  if packed_type is None or not isinstance(value, packed_type):
    raise TypeError('Unknown quantized activation layout; pass the installed PackedActivation class')
  assert (value.rows, value.width) == (rows, width)
  assert value.value.dtype == dtypes.fp8e4m3 and value.value.shape == (((rows+15)//16)*16, width)
  return value.value


def project(x, quantized_input, prepared, packed_type):
  value, scale = quantized_input
  rows, width = x.numel()//x.shape[-1], x.shape[-1]
  assert x.dtype == dtypes.bfloat16 and scale.dtype == dtypes.float32 and scale.numel() == 1
  a = activation(value, rows, width, packed_type)
  weight, columns, bias, sizes = prepared
  assert a.device == weight.device == columns.device == scale.device == x.device
  output = Tensor.empty(rows, sum(sizes), device=x.device, dtype=dtypes.bfloat16)
  result = Tensor.custom_kernel(output, a, weight, scale, columns, *(() if bias is None else (bias,)), fxn=kernel)[0]
  parts, cursor = [], 0
  for size in sizes:
    parts.append(result[:, cursor:cursor+size].reshape(*x.shape[:-1], size))
    cursor += size
  return tuple(parts)


def reset_jits(model):
  for key in ('prefill_jit', 'decode_jit', 'greedy_decode_jit'):
    if (jit := getattr(model, key, None)) is not None: jit.reset()
  if hasattr(model, 'trajectory_jits'): model.trajectory_jits.clear()
  if hasattr(model, 'padded_prefill'):
    model.padded_prefill.jit.reset()
    model.padded_prefill.shape = None


@contextmanager
def install(model, packed_type=None, scope='expert'):
  assert scope in ('expert', 'prefill', 'both') and model.fp8_wmma and model.dtype == dtypes.bfloat16
  scopes = tuple(ROOTS) if scope == 'both' else (scope,)
  original_linear, original_layer = model.linear, model.text_layer
  groups, snapshots, skipped = {}, {}, {}
  for key in model.weights:
    if not key.endswith('.self_attn.q_proj.weight'): continue
    selected = next((s for s in scopes if key.startswith(ROOTS[s][0])), None)
    if selected is None: continue
    prefix = key.removesuffix('.self_attn.q_proj.weight')
    names = tuple(prefix+'.self_attn.'+c+'_proj' for c in 'qkv')
    weights = [model.weights.get(n+'.weight') for n in names]
    scales = [model.weight_scales.get(n+'.weight') for n in names]
    biases = [model.weights.get(n+'.bias') for n in names]
    if any(w is None or s is None or w.dtype != dtypes.fp8e4m3 for w, s in zip(weights, scales)):
      skipped[prefix] = 'Missing FP8 weight or scalar weight scale'
      continue
    assert tuple(w.shape for w in weights) == tuple((n, ROOTS[selected][1]) for n in SIZES)
    if any(b is None for b in biases) and not all(b is None for b in biases):
      # Adding +0 to an absent bias can change signed zero. Keep the old path.
      skipped[prefix] = 'Mixed bias presence; original path preserves absent-bias signed-zero behavior'
      continue
    groups[prefix] = prepare(weights, scales, biases)
    snapshots[prefix] = (names, tuple(t.uop if t is not None else None for ts in (weights, scales, biases) for t in ts))
  assert groups, 'No supported QKV groups'
  stats = {'scope': scope, 'groups': list(groups), 'skipped': skipped, 'python_trace_build_calls': {},
           'additional_immutable_bytes': sum(sum(t.nbytes() for t in g[:3] if t is not None) for g in groups.values()),
           'weight_contract': 'Immutable FP8 weights, FP32 weight scales, and optional BF16 biases; no dequantization or requantization',
           'gpu_validated': False, 'pipeline_validated': False}
  active = []

  def linear(self, x, name, quantized_input=None, heads=None):
    if not active or active[-1] is None or name not in active[-1]['names']:
      return original_linear(x, name, quantized_input, heads)
    state = active[-1]
    index = state['names'].index(name)
    assert index == state['next'], 'Q/K/V call order changed'
    assert quantized_input is not None, 'Expected the existing shared activation quantization'
    if index == 0:
      state.update(input=x.uop, quantized=quantized_input, parts=project(x, quantized_input, groups[state['prefix']], packed_type))
      calls = stats['python_trace_build_calls']
      calls[state['prefix']] = calls.get(state['prefix'], 0)+1
    else:
      assert x.uop is state['input'] and all(a is b for a, b in zip(quantized_input, state['quantized'])), 'Q/K/V inputs or scales differ'
    state['next'] += 1
    return state['parts'][index]

  def text_layer(self, x, cos, sin, prefix, *args, **kwargs):
    rows = x.numel()//x.shape[-1]
    position = kwargs.get('cache_pos', args[2] if len(args) >= 3 else None)
    selected = (prefix in groups and x.dtype == dtypes.bfloat16 and
                ((prefix.startswith(ROOTS['expert'][0]) and rows == 64) or
                 (prefix.startswith(ROOTS['prefill'][0]) and rows >= 128 and isinstance(position, int) and position == 0)))
    state = None
    if selected:
      names, saved = snapshots[prefix]
      current = tuple(t.uop if t is not None else None for ts in
                      ([self.weights[n+'.weight'] for n in names], [self.weight_scales[n+'.weight'] for n in names],
                       [self.weights.get(n+'.bias') for n in names]) for t in ts)
      assert all(a is b for a, b in zip(current, saved)), 'Immutable QKV parameters changed; reinstall before capture'
      state = {'prefix': prefix, 'names': names, 'next': 0}
    active.append(state)
    try:
      result = original_layer(x, cos, sin, prefix, *args, **kwargs)
      assert state is None or state['next'] == 3, 'Incomplete Q/K/V group'
      return result
    finally:
      active.pop()

  reset_jits(model)
  model.linear, model.text_layer = MethodType(linear, model), MethodType(text_layer, model)
  try:
    yield stats
  finally:
    model.linear, model.text_layer = original_linear, original_layer
    reset_jits(model)
