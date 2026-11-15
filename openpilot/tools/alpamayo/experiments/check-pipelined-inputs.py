"""Check CPU-thread preparation without creating a GPU context."""
from concurrent.futures import ThreadPoolExecutor
import copy
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
import benchmark


def module(name):
  spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(name+'.py'))
  value = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(value)
  return value


torch.set_num_threads(4)
root = Path('/tmp/opencode/alpamayo-asius-20261114')
fast = module('fast-inputs').FastInputs
compact = module('compact-inputs').CompactInputs
cache = module('vision-cache').VisionCache
settings = SimpleNamespace(max_tokens=256, vision_batch=1, byte_pixels=True)
reference = fast(root/'model', pixel_scale=.5, cache_resize=True)
threaded = fast(root/'model', pixel_scale=.5, cache_resize=True)
decode_tokenizer = copy.deepcopy(threaded.tokenizer)


def prepare(adapter, second):
  config, tokenizer, prepared = adapter.prepare(benchmark.load_sample(root/f'sample-{second}.npz'))
  packed = adapter.native_inputs(prepared)
  return compact.arrays(settings, prepared, packed), cache.input_keys(prepared), packed


samples = (5, 25, 45, 5, 45, 25)
with ThreadPoolExecutor(max_workers=1) as pool:
  future = pool.submit(prepare, threaded, samples[0])
  for index, second in enumerate(samples):
    arrays, keys, packed = future.result()
    future = pool.submit(prepare, threaded, samples[index+1]) if index+1 < len(samples) else None
    # Decode concurrently using the separate tokenizer, as the GPU runner does.
    text = decode_tokenizer.decode(packed[0][0].tolist())
    assert isinstance(text, str)
    expected, expected_keys, expected_packed = prepare(reference, second)
    assert keys == expected_keys
    for actual, wanted in zip(arrays, expected, strict=True):
      np.testing.assert_array_equal(actual, wanted)
    for actual, wanted in zip(packed, expected_packed, strict=True):
      np.testing.assert_array_equal(actual, wanted)
    print(f'{second}s: threaded arrays, tokens, positions, and image hashes exactly match', flush=True)
