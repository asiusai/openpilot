"""Exact streaming vision features. Preserve each encoder's quantization group."""
from collections import OrderedDict
import hashlib

import numpy as np


class FeatureCache:
  def __init__(self, encode, clone, capacity=96):
    if capacity <= 0:
      raise ValueError('Positive feature-cache capacity required')
    self.encode, self.clone, self.capacity = encode, clone, capacity
    self.cache = OrderedDict()
    self.hits = self.misses = 0

  def __call__(self, pixels, positions, grid):
    """Cache a complete independent encoding group, with unchanged feature grids.

    CPU arrays identify exact prepared inputs. The bound encoder may upload them.
    clone must own independent output storage, including for reused JIT buffers.
    """
    arrays = [np.ascontiguousarray(a) for a in (pixels, positions, grid)]
    key = tuple((a.shape, a.dtype.str, hashlib.sha256(a).digest()) for a in arrays)
    if key in self.cache:
      self.hits += 1
      self.cache.move_to_end(key)
    else:
      self.misses += 1
      self.cache[key] = self.clone(self.encode(pixels, positions, grid))
      while len(self.cache) > self.capacity:
        self.cache.popitem(last=False)
    return self.cache[key]
