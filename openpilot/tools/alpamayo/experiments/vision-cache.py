"""Cache independent image features by exact preprocessed content and image grid.

Requires vision_batch=1: dynamic activation quantization otherwise couples images.
Timing must use advancing camera history, not repeated identical test snapshots.
"""
from collections import OrderedDict
import copy
import hashlib
from types import MethodType
import numpy as np
from tinygrad import Tensor, TinyJit
from tinygrad.uop.ops import UOp


class VisionCache:
  def __init__(self, model, capacity=48, verify=False, grouped=False, views=False, batch_encoder=None, group_limit=0):
    self.model, self.capacity = model, capacity
    self.cache, self.encoders, self.stitchers = OrderedDict(), {}, OrderedDict()
    self.stitcher_capacity = 8
    self.keys = ()
    self.hits = self.misses = 0
    self.original_encode = model.encode
    self.reference_model = None
    self.reference_encoders = {}
    self.reference_checks = 0
    self.verify = verify
    self.grouped = grouped
    self.views = views
    self.batch_encoder = batch_encoder
    assert group_limit >= 0
    self.group_limit = group_limit
    self.warmup_graphs = False
    model.encode = MethodType(lambda _, *args: self.encode(*args), model)

  @staticmethod
  def input_keys(prepared):
    grids = prepared['tokenized_data']['image_grid_thw'].numpy()
    pixels = prepared['tokenized_data']['pixel_values'].numpy()
    offsets = np.concatenate(([0], np.cumsum(grids.prod(axis=1))))
    return tuple((tuple(grid), hashlib.sha256(pixels[a:b].tobytes()).digest())
                 for grid, a, b in zip(grids, offsets[:-1], offsets[1:], strict=True))

  def prepare(self, prepared):
    self.keys = self.input_keys(prepared)

  def encode(self, pixels, positions, ids, batches, spans):
    assert len(batches) == len(self.keys) and all(count == 1 for _, _, count in batches)
    self.hits = self.misses = 0
    if self.grouped:
      missing = {}
      for key, (begin, end, _) in zip(self.keys, batches, strict=True):
        if key not in self.cache:
          missing.setdefault(key, (begin, end))
      # One graph submission for the new images, preserving per-image quantization.
      items = list(missing.items())
      limit = self.group_limit or max(1,len(items))
      groups = {}
      for item in items:
        # Front and surround images have different patch grids. A batched vision
        # graph requires one grid; stitching below restores camera-major order.
        grid = item[0][0] if self.batch_encoder else None
        groups.setdefault(grid, []).append(item)
      chunks = [group[offset:offset+limit] for group in groups.values() for offset in range(0,len(group),limit)]
      for chunk in chunks:
        pairs = [(self.image_view(t,a,b) if self.views else t[a:b].unsqueeze(0).clone())
                 for _, (a,b) in chunk for t in (pixels, positions)]
        Tensor.realize(*pairs)
        shape = tuple(t.shape for t in pairs)
        if shape not in self.encoders:
          self.encoders[shape] = TinyJit(self.batch_encoder or
                                        (lambda *xs: tuple(self.model._vision(p, v) for p, v in zip(xs[::2], xs[1::2]))))
        if self.warmup_graphs:
          while self.encoders[shape].cnt < 2:
            Tensor.realize(*self.encoders[shape](*pairs))
        values = [t.clone() for t in self.encoders[shape](*pairs)]
        Tensor.realize(*values)
        self.cache.update(zip((key for key,_ in chunk), values, strict=True))
      self.misses = len(missing)
    features = []
    for key, (begin, end, _) in zip(self.keys, batches, strict=True):
      if key in self.cache:
        self.hits += int(not self.grouped or key not in missing)
        self.cache.move_to_end(key)
      else:
        self.misses += 1
        p, v = (self.image_view(t, begin, end) if self.views else t[begin:end].unsqueeze(0).clone().realize()
                for t in (pixels, positions))
        shape = (p.shape, v.shape)
        if shape not in self.encoders:
          self.encoders[shape] = TinyJit(self.model._vision)
        # JIT outputs share storage across calls. Cache an independent allocation.
        self.cache[key] = self.encoders[shape](p, v).clone().realize()
        if len(self.cache) > self.capacity:
          self.cache.popitem(last=False)
      features.append(self.cache[key])
    while len(self.cache) > self.capacity:
      self.cache.popitem(last=False)
    # Identical history images share a cache tensor. TinyJit requires distinct
    # input buffers; retain repeated image positions inside the captured graph.
    unique, indices, layout = [], {}, []
    for key, feature in zip(self.keys, features, strict=True):
      if key not in indices:
        indices[key] = len(unique)
        unique.append(feature)
      layout.append(indices[key])
    layout = tuple(layout)
    shape = (ids.shape, spans, layout)
    if shape not in self.stitchers:
      def stitch(tokens, *images):
        embedding = self.model.weights['vlm.model.language_model.embed_tokens.weight'][tokens]
        merged = Tensor.cat(*(images[i] for i in layout), dim=0)
        chunks, cursor, feature_cursor = [], 0, 0
        for begin, end in spans:
          chunks.extend((embedding[:, cursor:begin], merged[feature_cursor:feature_cursor+end-begin].unsqueeze(0)))
          cursor, feature_cursor = end, feature_cursor+end-begin
        chunks.append(embedding[:, cursor:])
        return Tensor.cat(*chunks, dim=1).realize()
      self.stitchers[shape] = TinyJit(stitch)
    self.stitchers.move_to_end(shape)
    while len(self.stitchers) > self.stitcher_capacity:
      self.stitchers.popitem(last=False)
    result = self.stitchers[shape](ids, *unique)
    if self.verify:
      if self.reference_model is None:
        # Keep Harald's complete assembly of embeddings, but capture only one
        # independent image per grid. Capturing all 22 reference images duplicates
        # large graphs alongside the production model on the 8 GB GPU.
        self.reference_model = copy.copy(self.model)
        def reference_vision(p, v):
          # Slice offsets vary between cameras. Give each captured reference
          # graph independent inputs with the same layout on every call.
          p, v = p.clone().realize(), v.clone().realize()
          key = (p.shape, v.shape)
          if key not in self.reference_encoders:
            self.reference_encoders[key] = TinyJit(self.model._vision)
          return self.reference_encoders[key](p, v).clone().realize()
        self.reference_model._vision = reference_vision
      expected = self.reference_model._encode(pixels, positions, ids, batches, spans)
      actual_array, expected_array = result.float().numpy(), expected.float().numpy()
      np.testing.assert_array_equal(actual_array, expected_array)
      if not self.reference_checks:
        # Check the bounded reference against the original complete encoder once,
        # before its second call would capture the whole 22-image graph.
        original = self.original_encode(pixels, positions, ids, batches, spans)
        np.testing.assert_array_equal(expected_array, original.float().numpy())
      self.reference_checks += 1
    return result

  @staticmethod
  def image_view(tensor, begin, end):
    assert tensor.ndim == 2
    stride = tensor.shape[-1]*tensor.dtype.itemsize
    buffer = tensor._buffer().view((end-begin)*stride, begin*stride).ensure_allocated()
    return Tensor(UOp.from_buffer(buffer, tensor.dtype)).reshape(1,end-begin,tensor.shape[-1])
