"""Check restored-weight matrix shapes against original and FP32 arithmetic."""
import argparse
import importlib.util
import json
from pathlib import Path
import time

import numpy as np
import torch
from tinygrad import Device, Tensor, TinyJit, dtypes
from tinygrad.device import Compiled


def local(name):
  spec = importlib.util.spec_from_file_location(name.replace('-', '_'), Path(__file__).with_name(name+'.py'))
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--output', type=Path, required=True)
  args = parser.parse_args()
  torch.set_num_threads(4)
  Compiled.wait_timeout_ms = 10000
  local('prefill-kernels').fix_dispatch_packet()
  local('wgp_compiler').install(Device[Device.DEFAULT])
  candidate = local('bf16-linear')
  Tensor.manual_seed(1103)
  results = []
  for rows, columns, width in ((17, 48, 32), (720, 1152, 768), (720, 4304, 1152),
                               (364, 1152, 4304), (2880, 1152, 1152), (2912, 4304, 1152)):
    x = Tensor.randn(rows, width).cast(dtypes.bfloat16).realize()
    w = (Tensor.randn(columns, width) / width**.5).cast(dtypes.bfloat16).realize()
    packed = candidate.pack_weight(w)
    bias = Tensor.randn(columns).cast(dtypes.bfloat16).realize()
    a = torch.from_numpy(x.float().numpy())
    b = torch.from_numpy(w.float().numpy())
    c = torch.from_numpy(bias.float().numpy()).to(torch.bfloat16)
    expected = ((a @ b.T).to(torch.bfloat16) + c).float().numpy()
    baseline = None
    baseline_rmse = None
    for name in ('original', 'packed'):
      if name == 'original':
        run = TinyJit(lambda a, b, c: a.linear(b.T, c).realize())
        weight = w
      else:
        run = TinyJit(lambda a, b, c: candidate.linear(a, b, columns, c).realize())
        weight = packed
      times = []
      for iteration in range(7):
        start = time.perf_counter()
        output = run(x, weight, bias)
        Device[Device.DEFAULT].synchronize()
        if iteration >= 2:
          times.append((time.perf_counter()-start)*1000)
      actual = output.float().numpy()
      assert np.isfinite(actual).all()
      error = actual.astype(np.float64)-expected.astype(np.float64)
      rmse = float(np.sqrt(np.mean(error**2)))
      if baseline is None:
        baseline, baseline_rmse = actual.copy(), rmse
      record = dict(shape=[rows, columns, width], variant=name, median_ms=float(np.median(times)),
                    exact=bool(np.array_equal(actual, baseline)), differing_values=int(np.count_nonzero(actual != baseline)),
                    max_delta=float(np.max(np.abs(actual-baseline))), fp32_reference_rmse=rmse)
      results.append(record)
      args.output.write_text(json.dumps(results, indent=2)+'\n')
      print(json.dumps(record), flush=True)
      assert rmse <= baseline_rmse*1.1+1e-7, 'BF16 candidate error exceeds original arithmetic'


if __name__ == '__main__':
  main()
