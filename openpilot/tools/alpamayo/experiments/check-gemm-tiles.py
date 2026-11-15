"""Compare warmed wall times and outputs for actual model matrix dimensions."""
import importlib.util
import json
from pathlib import Path
import time
import os
import numpy as np
from tinygrad import Device, Tensor, TinyJit, dtypes
from tinygrad.device import Compiled
from amd_gemm import pack, unpack

spec = importlib.util.spec_from_file_location('tiles', Path(__file__).with_name('tiled-gemm.py'))
tiles = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tiles)
Compiled.wait_timeout_ms = 10000
Tensor.manual_seed(42)
results = []
shapes = ((196,4304,1152),(196,1152,4304),(196,1152,1152)) if os.getenv('ALPAMAYO_SMALL_TILES') else ((2027, 9216, 2048), (2027, 2048, 9216), (784, 4304, 1152), (64, 3840, 512))
for rows, columns, width in shapes:
  x = (Tensor.randn(rows, width)*8).cast(dtypes.fp8e4m3).realize()
  w = unpack(pack((Tensor.randn(columns, width)*8).cast(dtypes.fp8e4m3)).realize())
  scales = (Tensor([.125]).realize(), Tensor([.125]).realize())
  reference = None
  for tile in ((4, 4), (4, 8), (8, 4), (8, 8), (2, 4), (4, 2), (2, 2), (1, 4), (4, 1)):
    try:
      jit = TinyJit(lambda a, b, s, t: tiles.gemm(a, b, (s, t), tile))
      timings = []
      for i in range(7):
        started = time.perf_counter()
        out = jit(x, w, *scales).realize()
        Device[Device.DEFAULT].synchronize()
        if i >= 2: timings.append((time.perf_counter()-started)*1000)
      values = out.float().numpy()
      if reference is None: reference = values
      np.testing.assert_array_equal(values, reference)
      result = {'shape': [rows, columns, width], 'tile': tile, 'median_ms': float(np.median(timings)), 'exact': True}
    except Exception as exc:
      result = {'shape': [rows, columns, width], 'tile': tile, 'error': str(exc)}
    results.append(result)
    print(json.dumps(result), flush=True)
    Path(os.getenv('ALPAMAYO_TILES_OUTPUT', '/tmp/opencode/alpamayo-gemm-tiles.json')).write_text(json.dumps(results, indent=2))
