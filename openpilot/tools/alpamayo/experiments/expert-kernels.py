"""Extend Harald's packed WMMA kernels to the 64-query trajectory expert.

Uses the pinned alpamayo2-edge-egpu amd_gemm/amd_attention implementation.
No changes to model weights, quantization, masking, or diffusion steps.
"""
from functools import partial
from types import MethodType

from tinygrad import Tensor, dtypes
from tinygrad.helpers import round_up
from amd_attention import _pack, _matmul, _softmax
from amd_gemm import pack, unpack, gemm


def attention(q, k, v, mask):
  batch, heads, queries, width = q.shape
  kv_heads, keys = k.shape[1:3]
  assert queries in (16, 32, 48, 64) and (mask.shape == (keys,) or mask.shape == (queries, keys))
  scores = Tensor.empty(batch, heads, queries, keys, device=q.device, dtype=q.dtype)
  scores = Tensor.custom_kernel(scores, _pack(q), _pack(k),
    fxn=partial(_matmul, shape=(batch, heads, kv_heads, queries, keys, width), scaled=True))[0]
  # Expert and draft masks contain zero and negative infinity. Preserve masked keys.
  scores = (scores.float() + mask).cast(q.dtype).contiguous()
  probs = Tensor.empty(batch, heads, round_up(queries, 16), round_up(keys, 16), device=q.device, dtype=q.dtype)
  probs = Tensor.custom_kernel(probs, scores, fxn=_softmax)[0]
  out = Tensor.empty(q.shape, device=q.device, dtype=q.dtype)
  return Tensor.custom_kernel(out, probs, _pack(v.transpose(-2, -1)),
    fxn=partial(_matmul, shape=(batch, heads, kv_heads, queries, width, keys)))[0]


def install(model):
  assert model.fp8_wmma, 'Expert WMMA kernels require gfx1200/gfx1201'
  original_linear, original_attention = model.linear, model.attention
  eligible = set()
  for name, weight in list(model.weights.items()):
    if name.startswith('expert.') and weight.dtype == dtypes.fp8e4m3:
      model.weights[name] = unpack(pack(weight).realize())
      eligible.add(name.removesuffix('.weight'))

  def linear(self, x, name, quantized_input=None, heads=None):
    shape = x.shape[:-1]
    if name in eligible and x.numel() // x.shape[-1] == 64:
      quantized, scale = self.quantize(x) if quantized_input is None else quantized_input
      return gemm(quantized, self.weights[name+'.weight'], (scale, self.weight_scales[name+'.weight']),
                  self.weights.get(name+'.bias'), dtype=x.dtype).reshape(*shape, self.weights[name+'.weight'].shape[0])
    return original_linear(x, name, quantized_input, heads)

  def attended(self, q, k, v, mask=None, causal=False):
    if (q.dtype == dtypes.bfloat16 and q.shape[2] == 64 and mask is not None
        and mask.numel() == k.shape[2] and not causal):
      return attention(q, k, v, mask)
    return original_attention(q, k, v, mask, causal)

  model.linear = MethodType(linear, model)
  model.attention = MethodType(attended, model)
  return len(eligible)
