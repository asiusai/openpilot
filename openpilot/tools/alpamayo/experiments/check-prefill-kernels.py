"""Compare causal prefill against independent FP32 attention and original WMMA."""
import importlib.util
import json
from pathlib import Path
import time
import numpy as np
import torch
from tinygrad import Device, Tensor, TinyJit, dtypes
from tinygrad.device import Compiled
from amd_attention import attention

spec = importlib.util.spec_from_file_location('prefill_kernels', Path(__file__).with_name('prefill-kernels.py'))
fast = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fast)
Compiled.wait_timeout_ms = 300000
Tensor.manual_seed(31)
torch.set_num_threads(4)

def timed(fn, *args):
  jit, times = TinyJit(fn), []
  for i in range(7):
    begin = time.perf_counter()
    out = jit(*args).realize()
    Device[Device.DEFAULT].synchronize()
    if i >= 2:
      times.append((time.perf_counter()-begin)*1000)
  return out.float().numpy(), float(np.median(times))

report = []
for length in (128, 2027):
  q = Tensor.randn(1, 16, length, 128).cast(dtypes.bfloat16).realize()
  k, v = [Tensor.randn(1, 8, length, 128).cast(dtypes.bfloat16).realize() for _ in range(2)]
  reference, old_ms = timed(lambda a, b, c: attention(a, b, c, True), q, k, v)
  actual, new_ms = timed(fast.attention, q, k, v)
  tq, tk, tv = [torch.from_numpy(t.float().numpy()) for t in (q, k, v)]
  truth = torch.nn.functional.scaled_dot_product_attention(tq, tk.repeat_interleave(2, 1),
                                                         tv.repeat_interleave(2, 1), is_causal=True).numpy()
  old_rmse, new_rmse = [float(np.sqrt(np.mean((a-truth)**2))) for a in (reference, actual)]
  assert np.isfinite(actual).all()
  assert new_rmse <= old_rmse*1.1, (new_rmse, old_rmse)
  report.append({'length': length, 'old_ms': old_ms, 'new_ms': new_ms,
                 'old_rmse': old_rmse, 'new_rmse': new_rmse, 'max_delta': float(np.max(np.abs(reference-actual)))})
  print(json.dumps(report[-1]), flush=True)
Path('/tmp/opencode/alpamayo-prefill-kernel-check.json').write_text(json.dumps(report, indent=2))
