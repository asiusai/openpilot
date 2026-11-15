"""Check changing GPU data, sliced buffers, and USB readback chunk boundaries."""
import argparse
import importlib.util
import json
from pathlib import Path
import time

import numpy as np
from tinygrad import Device, Tensor

parser = argparse.ArgumentParser()
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
spec = importlib.util.spec_from_file_location('usb_readback', Path(__file__).with_name('usb-readback.py'))
readback = importlib.util.module_from_spec(spec)
spec.loader.exec_module(readback)
rng = np.random.default_rng(8206)
results = []
for dtype in (np.int32, np.float32):
  for size in (1, 16, 64, 128, 16385):
    for sliced in (False, True):
      count = size+6 if sliced else size
      source = (rng.integers(-2**29, 2**29, count, dtype=np.int32) if dtype == np.int32
                else rng.standard_normal(count).astype(np.float32))
      tensor = Tensor(source, device='CPU').to(Device.DEFAULT).realize()
      direct_times, normal_times = [], []
      for iteration in range(5):
        tensor = (tensor+1).realize()
        source = source+1
        value = tensor[3:-3] if sliced else tensor
        Device[Device.DEFAULT].synchronize()
        start = time.perf_counter()
        actual = readback.download(value)
        direct_times.append((time.perf_counter()-start)*1000)
        start = time.perf_counter()
        regular = value.numpy()
        normal_times.append((time.perf_counter()-start)*1000)
        expected = source[3:-3] if sliced else source
        np.testing.assert_array_equal(actual, regular)
        np.testing.assert_array_equal(actual, expected)
      result = {'dtype': np.dtype(dtype).name, 'size': size, 'sliced': sliced,
                'direct_median_ms': float(np.median(direct_times[1:])),
                'numpy_median_ms': float(np.median(normal_times[1:])), 'exact': True}
      results.append(result)
      print(result, flush=True)
args.output.write_text(json.dumps(results, indent=2)+'\n')
