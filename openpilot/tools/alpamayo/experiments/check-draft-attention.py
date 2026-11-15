"""Check small causal verification batches against an independent FP32 reference."""
import importlib.util
import json
from pathlib import Path
import statistics
import time
import numpy as np
import torch
from tinygrad import Device, Tensor, TinyJit, dtypes
from runtime import upload, download

spec = importlib.util.spec_from_file_location('masked_attention', Path(__file__).with_name('expert-kernels.py'))
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
torch.set_num_threads(4)
rng = np.random.default_rng(2026)
results = []


def ordinary(q, k, v, mask):
  k, v = (t.repeat_interleave(2, dim=1) for t in (k, v))
  scores = (q.matmul(k.transpose(-2, -1)) * (128**-.5)).float() + mask
  return scores.softmax(-1).cast(q.dtype).matmul(v)


for queries in (16, 32, 48, 64):
  keys = 1280
  arrays = [rng.normal(size=shape).astype(np.float32) for shape in ((1,16,queries,128),(1,8,keys,128),(1,8,keys,128))]
  reference_inputs = [torch.from_numpy(a).bfloat16().float() for a in arrays]
  mask = np.where(np.arange(keys)[None, :] <= 1035+np.arange(queries)[:, None], 0., -np.inf).astype(np.float32)
  q, k, v = reference_inputs
  reference = torch.nn.functional.scaled_dot_product_attention(q,k.repeat_interleave(2,1),v.repeat_interleave(2,1),
                                                               attn_mask=torch.from_numpy(mask)).numpy()
  values = [upload(Tensor(a,device='CPU'),dtypes.bfloat16) for a in arrays]
  values.append(upload(Tensor(mask,device='CPU'),dtypes.float32))
  row = {'queries':queries}
  for name, function in (('original',ordinary), ('packed',module.attention)):
    call = TinyJit(function)
    timings = []
    for iteration in range(7):
      started = time.perf_counter()
      output = call(*values).realize()
      Device[Device.DEFAULT].synchronize()
      if iteration >= 2:
        timings.append((time.perf_counter()-started)*1000)
    actual = download(output).astype(np.float32)
    assert np.isfinite(actual).all()
    row[name] = {'median_ms':statistics.median(timings),'rmse':float(np.sqrt(np.mean((actual-reference)**2))),
                 'max_abs':float(np.max(np.abs(actual-reference)))}
  assert row['packed']['rmse'] <= row['original']['rmse']*1.1+1e-6
  results.append(row)
  print(json.dumps(row),flush=True)
Path('/tmp/opencode/alpamayo-draft-attention-check.json').write_text(json.dumps(results,indent=2))
