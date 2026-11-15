"""Compare activation bytes and scales for language, vision, and expert shapes."""
import importlib.util
import json
import time
from types import SimpleNamespace
import numpy as np
from tinygrad import Device, Tensor, TinyJit, dtypes
from model import EdgeModel
from runtime import download
from pathlib import Path

spec = importlib.util.spec_from_file_location('fast_quantize', Path(__file__).with_name('fast-quantize.py'))
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
model = SimpleNamespace(cuda_kernels=False)
results = []
for rows, width in ((2027, 2048), (2027, 9216), (784, 4304), (64, 2048), (1, 9216)):
  x = Tensor(np.random.default_rng(0).normal(size=(rows, width)).astype(np.float32)).cast(dtypes.bfloat16).realize()
  old = EdgeModel.quantize(model, x)
  new = module.quantize(model, x)
  for a, b in zip(old, new):
    np.testing.assert_array_equal(download(a.bitcast(dtypes.uint8) if a.dtype == dtypes.fp8e4m3 else a),
                                  download(b.bitcast(dtypes.uint8) if b.dtype == dtypes.fp8e4m3 else b))
  times = []
  for fn in (EdgeModel.quantize, module.quantize):
    jit = TinyJit(lambda t: tuple(v.realize() for v in fn(model, t)))
    for _ in range(3): jit(x); Device[Device.DEFAULT].synchronize()
    start = time.perf_counter()
    for _ in range(30): jit(x)
    Device[Device.DEFAULT].synchronize()
    times.append((time.perf_counter()-start)*1000/30)
  results.append({'shape': [rows, width], 'exact': True, 'original_ms': times[0], 'new_ms': times[1]})
  print(json.dumps(results[-1]), flush=True)
Path('/tmp/opencode/alpamayo-quantize-check.json').write_text(json.dumps(results, indent=2))
