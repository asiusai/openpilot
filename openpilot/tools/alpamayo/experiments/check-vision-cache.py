"""Check repeated history images through graph capture and replay without a GPU."""
import importlib.util
import os
from pathlib import Path

import numpy as np

os.environ.setdefault('DEV', 'CPU')
from tinygrad import Tensor

spec = importlib.util.spec_from_file_location('vision_cache', Path(__file__).with_name('vision-cache.py'))
cache_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cache_module)


class Model:
  def __init__(self):
    self.weights = {'vlm.model.language_model.embed_tokens.weight': Tensor(np.arange(80, dtype=np.float32).reshape(20, 4))}

  def encode(self, *args):
    raise AssertionError('The test must use the cache adapter')

  def _vision(self, pixels, positions):
    return (pixels[0]*2+positions[0]).realize()


def main():
  cache = cache_module.VisionCache(Model())
  cache.stitcher_capacity = 2
  ids = Tensor(np.arange(10, dtype=np.int32).reshape(1, 10))
  spans = ((1, 3), (4, 6), (7, 9))
  batches = ((0, 2, 1), (2, 4, 1), (4, 6, 1))
  patterns = (((1, 2, 1), (1, 2, 3))*3 + ((2, 2, 2),)*3)*2
  for pattern in patterns:
    pixels = np.concatenate([np.full((2, 4), value, dtype=np.float32) for value in pattern])
    cache.keys = tuple(('grid', value) for value in pattern)
    actual = cache.encode(Tensor(pixels), Tensor(np.zeros_like(pixels)), ids, batches, spans).numpy()
    expected = np.arange(80, dtype=np.float32).reshape(20, 4)[:10].reshape(1, 10, 4)
    for value, (begin, end) in zip(pattern, spans, strict=True):
      expected[0, begin:end] = 2*value
    np.testing.assert_array_equal(actual, expected)
    assert len(cache.stitchers) <= 2
  print('Repeated and distinct image histories preserve every embedding through JIT replay')


if __name__ == '__main__':
  main()
