"""Check every transferred tensor against Harald's original upload path."""
import importlib.util
import os
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
import benchmark

def load(name):
  spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(name+'.py'))
  mod = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(mod)
  return mod

torch.set_num_threads(4)
root = Path('/tmp/opencode/alpamayo-asius-20261114')
adapter = load('fast-inputs').FastInputs(str(root/'model'))
compact = load('compact-inputs').CompactInputs
settings = SimpleNamespace(max_tokens=256, vision_batch=2, byte_pixels=os.environ.get('ALPAMAYO_BYTE_PIXELS') == '1')
noise = np.random.default_rng(0).standard_normal((1, 64, 2))
reused = None
for second in (5, 45, 25, 5):
  config, _, prepared = adapter.prepare(benchmark.load_sample(root/f'sample-{second}.npz'))
  packed = adapter.native_inputs(prepared)
  baseline = benchmark.NativeInputs(settings, config, prepared, packed).upload(noise)
  if reused is None:
    reused = compact(settings, config, prepared, packed)
  else:
    assert reused.update(settings, config, prepared, packed)
  optimized = reused.upload(noise)
  for index, (a, b) in enumerate(zip(baseline, optimized, strict=True)):
    np.testing.assert_array_equal(a.float().numpy(), b.float().numpy(), err_msg=f'input {index}')
  print(f'{second}s: all ten GPU input tensors exactly match', flush=True)
