"""Pack expert history once per trajectory, then update contiguous action slots.

The selected score, softmax, and PV kernels remain unchanged. Each matrix
kernel receives one full packed operand buffer, without split-buffer reads.
"""
from functools import partial
from types import MethodType

from tinygrad import Tensor, dtypes
from tinygrad.helpers import round_up
from amd_attention import _pack, _matmul, _softmax


def pack_view(x):
  """The exact _pack permutation, without allocating an intermediate copy."""
  batch, heads, rows, width = x.shape
  assert rows % 16 == 0 and width % 16 == 0
  return x.reshape(batch, heads, rows//16, 16, width//16, 4, 4).permute(0, 1, 2, 4, 5, 3, 6)


class PackedHistory:
  def __init__(self, oldk, oldv):
    assert oldk.shape == oldv.shape and oldk.dtype == oldv.dtype == dtypes.bfloat16
    batch, heads, capacity, width = oldk.shape
    assert capacity > 0 and capacity % 16 == 0 and width == 128
    self.capacity, self.shape = capacity, oldk.shape
    keys = capacity+64
    # Realize storage before partial writes. Never initialize with stale data.
    self.k = Tensor.empty(batch, heads, keys//16, width//16, 4, 16, 4,
      device=oldk.device, dtype=dtypes.uint16).realize()
    self.v = Tensor.empty(batch, heads, width//16, keys//16, 4, 16, 4,
      device=oldv.device, dtype=dtypes.uint16).realize()
    self.k[:, :, :capacity//16].assign(pack_view(oldk.bitcast(dtypes.uint16)))
    self.v[:, :, :, :capacity//16].assign(pack_view(oldv.bitcast(dtypes.uint16).transpose(-2, -1)))
    # Capture both history writes on every trajectory graph replay. Realization
    # materializes STORE+AFTER dependencies, without a device synchronization.
    self.k.realize(self.v)

  def update(self, k, v):
    assert k.shape == v.shape == (*self.shape[:2], 64, self.shape[-1])
    assert k.dtype == v.dtype == dtypes.bfloat16
    # K extends packed row tiles. Transposed V extends packed column tiles.
    self.k[:, :, self.capacity//16:].assign(pack_view(k.bitcast(dtypes.uint16)))
    self.v[:, :, :, self.capacity//16:].assign(pack_view(v.bitcast(dtypes.uint16).transpose(-2, -1)))
    self.k.realize(self.v)
    return self.k.bitcast(dtypes.bfloat16), self.v.bitcast(dtypes.bfloat16)


def attention(q, k, v, history, mask):
  batch, heads, queries, width = q.shape
  kv_heads, keys = k.shape[1], history.capacity+k.shape[2]
  assert queries == 64 and width == 128
  assert q.dtype == k.dtype == v.dtype == dtypes.bfloat16
  assert mask.shape == (keys,) or mask.shape == (queries, keys)
  packed_k, packed_v = history.update(k, v)
  scores = Tensor.empty(batch, heads, queries, keys, device=q.device, dtype=q.dtype)
  scores = Tensor.custom_kernel(scores, _pack(q), packed_k,
    fxn=partial(_matmul, shape=(batch, heads, kv_heads, queries, keys, width), scaled=True))[0]
  scores = (scores.float()+mask).cast(q.dtype).contiguous()
  probs = Tensor.empty(batch, heads, round_up(queries, 16), round_up(keys, 16), device=q.device, dtype=q.dtype)
  probs = Tensor.custom_kernel(probs, scores, fxn=_softmax)[0]
  output = Tensor.empty(q.shape, device=q.device, dtype=q.dtype)
  # Realize before a later action update can overwrite these shared operands.
  return Tensor.custom_kernel(output, probs, packed_v,
    fxn=partial(_matmul, shape=(batch, heads, kv_heads, queries, width, keys)))[0].realize()


def install(model):
  """Install after expert-kernels, before capturing any trajectory graph."""
  assert model.fp8_wmma and not model.trajectory_jits
  original_layer, original_trajectory = model.text_layer, model._trajectory
  state = None

  def layer(self, x, cos, sin, prefix, cache=None, mask=None, cache_pos=None):
    if state is None or not prefix.startswith('expert.') or cache is None or cache_pos is not None:
      return original_layer(x, cos, sin, prefix, cache, mask, cache_pos)
    assert x.shape[1] == 64 and mask is not None
    if prefix not in state:
      state[prefix] = PackedHistory(cache[0], cache[1])
    history = state[prefix]
    assert cache[0].shape == cache[1].shape == history.shape
    h = self.norm(x, f'{prefix}.input_layernorm', rms=True)
    quantized = self.quantize(h) if f'{prefix}.self_attn.q_proj.weight' in self.weight_scales else None
    q, k, v = (self.linear(h, f'{prefix}.self_attn.{name}_proj', quantized, heads=heads)
      .reshape(1, h.shape[1], heads, 128).transpose(1, 2) for name, heads in (('q', 16), ('k', 8), ('v', 8)))
    q, k = self.rope(q, k, cos, sin)
    h = attention(q, k, v, history, mask).transpose(1, 2).reshape(1, x.shape[1], 2048)
    x = x+self.linear(h, f'{prefix}.self_attn.o_proj')
    h = self.linear(self.norm(x, f'{prefix}.post_attention_layernorm', rms=True), f'{prefix}.mlp.fc1').relu().square()
    return (x+self.linear(h, f'{prefix}.mlp.fc2')).realize()

  def trajectory(self, *args, **kwargs):
    nonlocal state
    assert state is None
    state = {}
    try:
      return original_trajectory(*args, **kwargs)
    finally:
      # This resets tracing state, not GPU data. Every captured graph includes
      # fresh history writes and all three action updates for each layer.
      state = None

  model.text_layer = MethodType(layer, model)
  model._trajectory = MethodType(trajectory, model)
