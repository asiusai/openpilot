"""Validate changing uploads, retained rows, and JIT replay for both pixel encodings."""
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
Compact = load('compact-inputs').CompactInputs
noise = np.random.default_rng(0).standard_normal((1,64,2))
for byte in (False, True):
  settings = SimpleNamespace(max_tokens=256, vision_batch=1, byte_pixels=byte)
  inputs = None
  retained = None
  for iteration, second in enumerate((5,45,25,5,45,25,5,45)):
    config, _, prepared = adapter.prepare(benchmark.load_sample(root/f'sample-{second}.npz'))
    packed = adapter.native_inputs(prepared)
    expected = benchmark.NativeInputs(settings,config,prepared,packed).upload(noise)
    if inputs is None:
      inputs = Compact(settings,config,prepared,packed)
    else:
      assert inputs.update(settings,config,prepared,packed)
    missing = set(range(10)) if iteration == 0 else {3,4,8,9} if iteration < 7 else set()
    inputs.selective_cache = SimpleNamespace(keys=tuple(range(10)), cache={i:True for i in range(10) if i not in missing}, verify=False)
    if os.getenv('ALPAMAYO_WARM_SELECTIONS') == '1':
      inputs.warmup_selections()
    actual = inputs.upload(noise)
    full = expected[0].float().numpy()
    if retained is None:
      retained = np.empty_like(full)
    for i in missing:
      start,end,_ = inputs.batches[i]
      retained[start:end] = full[start:end]
    np.testing.assert_array_equal(actual[0].float().numpy(),retained)
    for index,(a,b) in enumerate(zip(actual[1:],expected[1:],strict=True),1):
      np.testing.assert_array_equal(a.float().numpy(),b.float().numpy(),err_msg=f'input {index}')
    print(f'byte={byte} iteration={iteration} missing={sorted(missing)} bytes={inputs.uploaded_pixel_bytes}: exact',flush=True)
