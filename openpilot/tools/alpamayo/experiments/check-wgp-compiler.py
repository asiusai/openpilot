"""Compare compiler scheduling modes on actual matrix shapes, in separate processes."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
from tinygrad import Device, Tensor, TinyJit, dtypes
from amd_gemm import pack, unpack, gemm
import wgp_compiler

parser = argparse.ArgumentParser()
parser.add_argument('--wgp', action='store_true')
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
if args.wgp:
  wgp_compiler.install(Device[Device.DEFAULT])
Tensor.manual_seed(2026)
report = []
for rows, cols, width in [(784,4304,1152), (2027,9216,2048), (2027,2048,9216), (64,3840,512)]:
  x = Tensor.randn(rows,width).cast(dtypes.fp8e4m3).realize()
  w = Tensor.randn(cols,width).cast(dtypes.fp8e4m3).realize()
  weight = unpack(pack(w).realize())
  sx, sw = Tensor([.125]).realize(), Tensor([.25]).realize()
  jit = TinyJit(lambda a: gemm(a,weight,(sx,sw)).realize())
  timings = []
  for i in range(9):
    start = time.perf_counter()
    out = jit(x)
    Device[Device.DEFAULT].synchronize()
    if i >= 2:
      timings.append((time.perf_counter()-start)*1000)
  truth = (x.float().numpy() @ w.float().numpy().T)*.03125
  actual = out.float().numpy()
  np.testing.assert_allclose(actual, truth, atol=.04, rtol=.004)
  report.append({'shape':[rows,cols,width], 'wgp':args.wgp, 'median_ms':float(np.median(timings)),
                 'rmse':float(np.sqrt(np.mean((actual-truth)**2)))})
  print(json.dumps(report[-1]),flush=True)
args.output.write_text(json.dumps(report,indent=2))
