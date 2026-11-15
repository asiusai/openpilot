"""Check batched verification GEMM against independent FP32 accumulation."""
import importlib.util
from pathlib import Path
import json
import numpy as np
from tinygrad import Tensor, dtypes
from amd_gemm import pack, unpack

spec = importlib.util.spec_from_file_location('tiles', Path(__file__).with_name('tiled-gemm.py'))
tiles = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tiles)
Tensor.manual_seed(52)
results = []
for rows, columns, width in ((16, 2048, 2048), (32, 2112, 2048), (64, 9216, 2048)):
  x = (Tensor.randn(rows,width)*8).cast(dtypes.fp8e4m3).realize()
  w = unpack(pack((Tensor.randn(columns,width)*8).cast(dtypes.fp8e4m3)).realize())
  sx = Tensor(np.linspace(.01,.2,rows,dtype=np.float32).reshape(rows,1)).realize()
  sw = Tensor([.125]).realize()
  bias = Tensor.randn(columns).realize()
  expected = ((x.float().numpy() @ w.float().numpy().T)*sx.numpy()*.125 + bias.numpy())
  actual = tiles.gemm(x,w,(sx,sw),row_scale=True,bias=bias).realize().float().numpy()
  # Bound BF16 output rounding against an independent FP32 accumulation.
  np.testing.assert_allclose(actual, expected, rtol=.004, atol=.002)
  results.append({'shape':[rows,columns,width], 'rmse':float(np.sqrt(np.mean((actual-expected)**2)))})
Path('/tmp/opencode/alpamayo-row-scale-gemm-check.json').write_text(json.dumps(results,indent=2))
print(json.dumps(results))
