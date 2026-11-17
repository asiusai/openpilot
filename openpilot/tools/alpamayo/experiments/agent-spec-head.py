"""Six independent original-order GEMVs sharing each loaded output-head weight.

This is a bitwise candidate, not validated hardware parity. Norms and dynamic
quantizers remain independent single-row calls. No WMMA is introduced.
"""
from types import MethodType

from tinygrad import Tensor, TinyJit, dtypes
from tinygrad.uop.ops import AxisType, KernelInfo, Ops, UOp
from amd_attention import _warp_reduce

ROWS = 6


def kernel(out, weight, weight_scale, *query_pairs):
  assert len(query_pairs) == 2*ROWS
  vocab, width = weight.shape
  assert out.shape == (ROWS, vocab) and vocab % 4 == width % 128 == 0
  group = UOp.range(vocab//4, 0, AxisType.GLOBAL)
  wave, lane = UOp.range(4, 1, AxisType.LOCAL), UOp.range(32, -1, AxisType.WARP)
  row, k = group*4+wave, UOp.range(width//128, 2, AxisType.LOOP)
  cols = [k*128+lane*4+j for j in range(4)]
  weights = [weight.flatten().index(row*width+c).cast(dtypes.float32) for c in cols]
  stores = []
  for query in range(ROWS):
    x, scale = query_pairs[2*query:2*query+2]
    # Copy amd_decode._gemv(packed=False)'s product association, K loop,
    # warp shuffle tree, scale product, and BF16 output boundary verbatim.
    value = sum(x.flatten().index(c).cast(dtypes.float32)*w for c, w in zip(cols, weights, strict=True)).reduce(k, arg=Ops.ADD)
    value = _warp_reduce(value)
    value = value*(scale.flatten().index(0)*weight_scale.flatten().index(0))
    stores.append(out.flatten().index((query*vocab+row).valid(lane.eq(0))).store(value.cast(out.dtype)).end(group, wave, lane))
  return UOp.sink(*stores, arg=KernelInfo(name='agent_spec_head_shared_gemv6', opts_to_apply=()))


def project_prepared(weight, weight_scale, queries, scales, out=None):
  assert len(queries) == len(scales) == ROWS and weight.dtype == dtypes.fp8e4m3
  assert weight.ndim == 2 and weight.shape[0] % 4 == weight.shape[1] % 128 == 0
  assert all(q.dtype == dtypes.fp8e4m3 and q.numel() == weight.shape[1] for q in queries)
  assert weight_scale.dtype == dtypes.float32 and weight_scale.numel() == 1
  assert all(scale.dtype == dtypes.float32 and scale.numel() == 1 for scale in scales)
  if out is None: out = Tensor.empty(ROWS, weight.shape[0], device=weight.device, dtype=dtypes.bfloat16)
  assert out.dtype == dtypes.bfloat16 and out.shape == (ROWS, weight.shape[0])
  pairs = tuple(value for pair in zip(queries, scales, strict=True) for value in pair)
  return Tensor.custom_kernel(out, weight, weight_scale, *pairs, fxn=kernel)[0]


def verification_payload(logits, mapping):
  assert logits.ndim == 2 and logits.shape[0] == ROWS and mapping.numel() == logits.shape[1]
  targets, checks = [], []
  for index in range(ROWS):
    row = logits[index:index+1]
    targets.append(mapping[row.reshape(-1).argmax()].reshape(1))
    checks.append(row.isfinite().all().cast(dtypes.int32).reshape(1))
  return Tensor.cat(*targets, *checks).contiguous().realize()


def heads(model, x, length, return_all=False):
  weight = model.weights['vlm.lm_head.weight']
  weight_scale = model.weight_scales['vlm.lm_head.weight']
  assert weight.dtype == dtypes.fp8e4m3 and 'vlm.lm_head.bias' not in model.weights
  assert model.dtype == dtypes.bfloat16 and hasattr(model, 'output_token_mapping')
  queries, scales = [], []
  for index in range(ROWS):
    hidden = model.norm(x[:, length-1+index:length+index], 'vlm.model.language_model.norm', rms=True)
    query, scale = model.quantize(hidden.reshape(-1, hidden.shape[-1]))
    # Realize one row's preparation before forming the next. This preserves
    # the original single-row reduction extent and separate dynamic scale.
    query.realize(scale)
    queries.append(query)
    scales.append(scale)
  logits = project_prepared(weight, weight_scale, queries, scales).realize()
  first, payload = logits[0:1].reshape(1, 1, weight.shape[0]), verification_payload(logits, model.output_token_mapping)
  # Returning realized views adds no kernels. The checker compares ALL logits.
  return (first, payload, *(logits[i:i+1] for i in range(ROWS))) if return_all else (first, payload)


def install(model):
  """Optional hook, only after install_prefill. Decoder and fallback stay intact."""
  prefill = model.padded_prefill
  assert type(prefill).__name__ == 'SpeculativePrefill' and hasattr(prefill, 'baseline')
  assert not hasattr(prefill, '_agent_spec_head_original_language')
  assert model.weights['vlm.lm_head.weight'].shape == (2112, 2048)
  assert model.output_token_mapping.numel() == 2112
  prefill._agent_spec_head_original_language = prefill.language
  def language(self, x, cos, sin, length):
    for i in range(model.config['vlm_config']['text_config']['num_hidden_layers']):
      x = model.text_layer(x, cos, sin, f'vlm.model.language_model.layers.{i}', model.cache[i], None, cache_pos=0)
    return heads(model, x, length)
  prefill.jit.reset()
  prefill.language = MethodType(language, prefill)
  prefill.jit = TinyJit(prefill.language)
  prefill.shape = None
  return {'candidate': 'shared-weight-original-order-gemv6', 'hardware_bitwise_validated': False,
          'dynamic_single_row_quantizers': 6, 'verified_targets': 6, 'full_rejection_fallback_preserved': True}
