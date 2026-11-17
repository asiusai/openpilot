"""Transfer BF16 inputs and reuse immutable vision positions across snapshots."""
import numpy as np
import torch
import weakref
from tinygrad import Device, Tensor, TinyJit, dtypes
from openpilot.selfdrive.alpamayo.inputs import rotary
from tinygrad.uop.ops import UOp


class CompactInputs:
  pixel_table = ((torch.arange(256, dtype=torch.float32)-127.5)/127.5).to(torch.bfloat16).view(torch.uint16).numpy()
  pixel_inverse = np.full(65536, 256, dtype=np.uint16)
  pixel_inverse[pixel_table] = np.arange(256, dtype=np.uint16)

  @staticmethod
  def arrays(args, prepared, packed):
    ids, positions, delta, vision_positions = packed
    arrays = [prepared['tokenized_data']['pixel_values'].numpy(), np.zeros(1, dtype=np.uint16),
              ids.astype(np.int32), *rotary(positions)]
    for length in (args.max_tokens, args.max_tokens+64):
      pos = np.arange(length) + ids.shape[1] + delta
      arrays.extend(rotary(np.broadcast_to(pos, (3, 1, length))))
    arrays.append(np.zeros((1, 64, 2), dtype=np.float32))
    direct_bytes = arrays[0].dtype == np.uint8
    if direct_bytes and not getattr(args, 'byte_pixels', False):
      raise ValueError('Unnormalized image patches require byte-pixel GPU expansion')
    values = [torch.from_numpy(a).to(torch.bfloat16).view(torch.uint16).numpy()
              if i not in (1, 2, len(arrays)-1) and not (i == 0 and direct_bytes) else a for i, a in enumerate(arrays)]
    if getattr(args, 'byte_pixels', False) and not direct_bytes:
      codes = CompactInputs.pixel_inverse[values[0]]
      # Reject incompatible normalization or interpolation, rather than silently
      # converting a value that cannot be represented by this exact palette.
      assert codes.max() < 256, 'Pixel palette is not lossless for this input'
      values[0] = codes.astype(np.uint8)
    return values

  @staticmethod
  def signature(args, config, prepared, packed):
    return (str(config._name_or_path), prepared['tokenized_data']['image_grid_thw'].numpy().tobytes(),
            args.max_tokens, args.vision_batch, getattr(args, 'byte_pixels', False),
            tuple(np.flatnonzero(packed[0][0] == config.vlm_config.image_token_id)))

  def update(self, args, config, prepared, packed, arrays=None):
    if self.signature(args, config, prepared, packed) != self._signature:
      return False
    arrays = self.arrays(args, prepared, packed) if arrays is None else arrays
    if any(tuple(a.shape) != old.shape for a, old in zip(arrays, self.cpu, strict=True)):
      return False
    for a, old in zip(arrays, self.cpu, strict=True):
      destination = np.frombuffer(old._buffer().as_memoryview(allow_zero_copy=True), dtype=a.dtype).reshape(a.shape)
      np.copyto(destination, a, casting='no')
    return True

  def __init__(self, args, config, prepared, packed, arrays=None, positions_cache=None):
    self.positions = {} if positions_cache is None else positions_cache
    ids, positions, delta, vision_positions = packed
    self._signature = self.signature(args, config, prepared, packed)
    self.layout_key = (str(config._name_or_path), prepared['tokenized_data']['image_grid_thw'].numpy().tobytes())
    if self.layout_key not in self.positions:
      if len(self.positions) >= 4:
        self.positions.pop(next(iter(self.positions)))
      value = torch.from_numpy(vision_positions).to(torch.bfloat16).view(torch.uint16).numpy()
      self.positions[self.layout_key] = Tensor(value, device='CPU').to(Device.DEFAULT).bitcast(dtypes.bfloat16).realize()
    fixed_positions = self.positions[self.layout_key]
    byte_pixels = getattr(args, 'byte_pixels', False)
    if byte_pixels:
      pixel_table = Tensor(self.pixel_table.copy(), device='CPU').to(Device.DEFAULT).bitcast(dtypes.bfloat16).realize()
    self.cpu = [Tensor(a, device='CPU').realize() for a in (self.arrays(args, prepared, packed) if arrays is None else arrays)]
    self.noise = np.frombuffer(self.cpu[-1]._buffer().as_memoryview(allow_zero_copy=True), dtype=np.float32).reshape(1, 64, 2)
    grids = prepared['tokenized_data']['image_grid_thw'].numpy()
    offsets = np.concatenate(([0], np.cumsum(grids.prod(axis=1))))
    batches, i = [], 0
    while i < len(grids):
      end = i+1
      while end < len(grids) and np.array_equal(grids[i], grids[end]) and end-i < args.vision_batch:
        end += 1
      batches.append((int(offsets[i]), int(offsets[end]), end-i))
      i = end
    self.batches = tuple(batches)
    self.pixel_table_gpu = pixel_table if byte_pixels else None
    self.selective_cache = None
    self.selective_jits = {}
    self.pixel_storage = None
    self.uploaded_pixel_bytes = 0
    image_indices = np.flatnonzero(ids[0] == config.vlm_config.image_token_id)
    self.spans = tuple((int(span[0]), int(span[-1])+1) for span in np.split(image_indices, np.flatnonzero(np.diff(image_indices) != 1)+1))
    self.upload_jit = TinyJit(lambda *xs: tuple(
      pixel_table[x.to(Device.DEFAULT).cast(dtypes.int32)].contiguous().realize() if i == 0 and byte_pixels else
      fixed_positions if i == 1 else x.to(Device.DEFAULT).bitcast(dtypes.bfloat16).contiguous().realize()
      if i not in (2, len(xs)-1) else x.to(Device.DEFAULT).contiguous().realize()
      for i, x in enumerate(xs)))

  def upload(self, noise):
    self.noise[:] = noise
    sources_all = self.cpu
    if self.selective_cache is not None:
      cache = self.selective_cache
      assert len(cache.keys) == len(self.batches) and all(n == 1 for _, _, n in self.batches)
      missing = tuple(i for i, key in enumerate(cache.keys) if key not in cache.cache)
      if self.pixel_storage is None:
        self.pixel_storage = Tensor.empty(self.cpu[0].shape, dtype=dtypes.bfloat16).realize()
        self.pixel_destinations = tuple(self.rows(self.pixel_storage,a,b) for a,b,_ in self.batches)
      if missing not in self.selective_jits:
        owner = weakref.proxy(self)
        self.selective_jits[missing] = TinyJit(lambda *xs: owner.upload_selected(missing, xs))
      self.uploaded_pixel_bytes = sum((self.batches[i][1]-self.batches[i][0])*self.cpu[0].shape[1]*self.cpu[0].dtype.itemsize
                                      for i in missing)
      sources = [self.rows(self.cpu[0],self.batches[i][0],self.batches[i][1]) for i in missing]
      return self.selective_jits[missing](*sources, *sources_all[1:])
    return self.upload_jit(*sources_all)

  @staticmethod
  def rows(tensor, begin, end):
    width = tensor.shape[1]
    buffer = tensor._buffer().view((end-begin)*width*tensor.dtype.itemsize, begin*width*tensor.dtype.itemsize).ensure_allocated()
    return Tensor(UOp.from_buffer(buffer, tensor.dtype)).reshape(end-begin, width)

  def upload_selected(self, missing, xs):
    for i, source in zip(missing, xs[:len(missing)], strict=True):
      source = source.to(Device.DEFAULT)
      pixels = self.pixel_table_gpu[source.cast(dtypes.int32)] if self.pixel_table_gpu is not None else source.bitcast(dtypes.bfloat16)
      self.pixel_destinations[i].assign(pixels).realize()
    rest = tuple(self.positions[self.layout_key] if i == 1 else x.to(Device.DEFAULT).bitcast(dtypes.bfloat16).contiguous().realize()
                 if i not in (2, len(self.cpu)-1) else x.to(Device.DEFAULT).contiguous().realize()
                 for i, x in enumerate(xs[len(missing):],1))
    return (self.pixel_storage, *rest)
