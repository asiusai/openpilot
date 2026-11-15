"""Numerical and warm wall-clock checks of the expert kernel extension."""
import importlib.util
import json
from pathlib import Path
import time

import numpy as np
import torch
from tinygrad import Device, Tensor, TinyJit, dtypes
from tinygrad.device import Compiled
from amd_gemm import pack, unpack, gemm

spec = importlib.util.spec_from_file_location('expert_kernels', Path(__file__).with_name('expert-kernels.py'))
expert = importlib.util.module_from_spec(spec)
spec.loader.exec_module(expert)
Compiled.wait_timeout_ms = 300000
Tensor.manual_seed(42)


def timed(fn, *inputs):
  jit = TinyJit(fn)
  results = []
  for i in range(7):
    start = time.perf_counter()
    out = jit(*inputs).realize()
    Device[Device.DEFAULT].synchronize()
    if i >= 2:
      results.append((time.perf_counter()-start)*1000)
  return out.float().numpy(), float(np.median(results))


report = []
for rows, columns, width in [(64, 1024, 2048), (64, 8192, 2048), (64, 2048, 8192)]:
  x = (Tensor.randn(rows, width)*8).cast(dtypes.fp8e4m3).realize()
  weight = (Tensor.randn(columns, width)*8).cast(dtypes.fp8e4m3).realize()
  packed = unpack(pack(weight).realize())
  scales = (Tensor([.125]).realize(), Tensor([.125]).realize())
  reference, old_ms = timed(lambda a: (a.dot(weight.T, dtype=dtypes.float32)*scales[0]*scales[1]).cast(dtypes.bfloat16), x)
  actual, new_ms = timed(lambda a: gemm(a, packed, scales), x)
  np.testing.assert_allclose(actual, reference, atol=.01, rtol=.01)
  report.append({'kind': 'gemm', 'shape': [rows, columns, width], 'old_ms': old_ms, 'new_ms': new_ms,
                 'max_abs_error': float(np.max(np.abs(actual-reference)))})
  print(json.dumps(report[-1]), flush=True)

for keys, length in [(320, 237), (2368, 2027)]:
  q = Tensor.randn(1, 16, 64, 128).cast(dtypes.bfloat16).realize()
  k, v = [Tensor.randn(1, 8, keys, 128).cast(dtypes.bfloat16).realize() for _ in range(2)]
  mask = ((Tensor.arange(keys) < length) | (Tensor.arange(keys) >= keys-64)).where(0, -float('inf')).realize()
  def reference_attention(a, b, c):
    repeated_k, repeated_v = (t.repeat_interleave(2, dim=1) for t in (b, c))
    scores = (a.matmul(repeated_k.transpose(-2, -1))*128**-.5).float() + mask
    return scores.softmax(-1).cast(dtypes.bfloat16).matmul(repeated_v)
  reference, old_ms = timed(reference_attention, q, k, v)
  actual, new_ms = timed(lambda a, b, c: expert.attention(a, b, c, mask), q, k, v)
  tq, tk, tv = [torch.from_numpy(t.float().numpy()) for t in (q, k, v)]
  truth = ((tq @ tk.repeat_interleave(2, 1).transpose(-2, -1) * 128**-.5 + torch.from_numpy(mask.numpy()))
           .softmax(-1) @ tv.repeat_interleave(2, 1)).numpy()
  old_rmse = float(np.sqrt(np.mean((reference-truth)**2)))
  new_rmse = float(np.sqrt(np.mean((actual-truth)**2)))
  # Different reduction orders can differ in BF16. Check against independent FP32 attention.
  assert new_rmse <= old_rmse*1.1, (new_rmse, old_rmse)
  assert np.max(np.abs(actual-truth)) <= np.max(np.abs(reference-truth))*1.1
  report.append({'kind': 'attention', 'keys': keys, 'old_ms': old_ms, 'new_ms': new_ms,
                 'reference_rmse_vs_fp32': old_rmse, 'new_rmse_vs_fp32': new_rmse,
                 'max_abs_error': float(np.max(np.abs(actual-reference)))})
  print(json.dumps(report[-1]), flush=True)
Path('/tmp/opencode/alpamayo-expert-kernel-check.json').write_text(json.dumps(report, indent=2))
