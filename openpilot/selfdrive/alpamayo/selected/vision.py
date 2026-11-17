"""Exact image features with group-two encoding and a 144-image LRU."""
from collections import OrderedDict
import hashlib
import numpy as np
from tinygrad import Tensor, TinyJit
from tinygrad.uop.ops import UOp


class VisionCache:
  capacity = 144
  group_limit = 2

  def __init__(self, model):
    self.model = model
    self.cache,self.encoders,self.stitchers = OrderedDict(),{},OrderedDict()
    self.keys = ()
    self.hits = self.misses = 0

  def reset(self):
    self.cache.clear()
    self.encoders.clear()
    self.stitchers.clear()
    self.keys = ()
    self.hits = self.misses = 0

  @staticmethod
  def input_keys(prepared):
    grids = prepared['tokenized_data']['image_grid_thw'].numpy()
    pixels = prepared['tokenized_data']['pixel_values'].numpy()
    offsets = np.concatenate(([0],np.cumsum(grids.prod(axis=1))))
    return tuple((tuple(grid),hashlib.sha256(pixels[a:b].tobytes()).digest())
                 for grid,a,b in zip(grids,offsets[:-1],offsets[1:],strict=True))

  def prepare(self, prepared):
    self.keys = self.input_keys(prepared)
    if len(self.keys) != 10 or any(np.prod(grid) != 720 for grid,_ in self.keys):
      raise ValueError('Require all ten full-resolution images')

  @staticmethod
  def image_view(tensor, begin, end):
    assert tensor.ndim == 2
    stride = tensor.shape[-1]*tensor.dtype.itemsize
    buffer = tensor._buffer().view((end-begin)*stride,begin*stride).ensure_allocated()
    return Tensor(UOp.from_buffer(buffer,tensor.dtype)).reshape(1,end-begin,tensor.shape[-1])

  def encode(self, pixels, positions, ids, batches, spans):
    assert len(batches) == len(self.keys) and all(count == 1 for _,_,count in batches)
    self.hits = self.misses = 0
    missing = {}
    for key,(begin,end,_) in zip(self.keys,batches,strict=True):
      if key not in self.cache:
        missing.setdefault(key,(begin,end))
    groups = {}
    for item in missing.items():
      groups.setdefault(item[0][0],[]).append(item)
    for group in groups.values():
      for offset in range(0,len(group),self.group_limit):
        chunk = group[offset:offset+self.group_limit]
        pairs = [self.image_view(t,a,b) for _,(a,b) in chunk for t in (pixels,positions)]
        Tensor.realize(*pairs)
        shape = tuple(t.shape for t in pairs)
        if shape not in self.encoders:
          self.encoders[shape] = TinyJit(self.model.encode_group)
        values = [t.clone() for t in self.encoders[shape](*pairs)]
        Tensor.realize(*values)
        self.cache.update(zip((key for key,_ in chunk),values,strict=True))
    self.misses = len(missing)
    features = []
    for key in self.keys:
      self.hits += int(key not in missing)
      self.cache.move_to_end(key)
      features.append(self.cache[key])
    while len(self.cache) > self.capacity:
      self.cache.popitem(last=False)
    unique,indices,layout = [],{},[]
    for key,feature in zip(self.keys,features,strict=True):
      if key not in indices:
        indices[key] = len(unique)
        unique.append(feature)
      layout.append(indices[key])
    layout = tuple(layout)
    grids = tuple(key[0] for key in self.keys)
    shape = (ids.shape,spans,layout,grids,grids)
    if shape not in self.stitchers:
      def stitch(tokens, *images):
        merged = Tensor.cat(*(images[i] for i in layout),dim=0)
        return self.model.embed(tokens,merged,spans)
      self.stitchers[shape] = TinyJit(stitch)
    self.stitchers.move_to_end(shape)
    while len(self.stitchers) > 8:
      self.stitchers.popitem(last=False)
    return self.stitchers[shape](ids,*unique)
