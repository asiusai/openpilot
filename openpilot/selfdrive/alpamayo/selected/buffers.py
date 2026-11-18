"""Transfer BF16 inputs and reuse immutable vision positions across snapshots."""
import numpy as np
import torch
from tinygrad import Device, Tensor, TinyJit, dtypes
from openpilot.selfdrive.alpamayo.inputs import rotary


class CompactInputs:
  pixel_table = ((torch.arange(256, dtype=torch.float32)-127.5)/127.5).to(torch.bfloat16).view(torch.uint16).numpy()

  @staticmethod
  def arrays(args, prepared, packed):
    ids, positions, delta, vision_positions = packed
    arrays = [prepared['tokenized_data']['pixel_values'].numpy(), np.zeros(1, dtype=np.uint16),
              ids.astype(np.int32), *rotary(positions)]
    for length in (args.max_tokens, args.max_tokens+64):
      pos = np.arange(length) + ids.shape[1] + delta
      arrays.extend(rotary(np.broadcast_to(pos, (3, 1, length))))
    arrays.append(np.zeros((1, 64, 2), dtype=np.float32))
    if arrays[0].dtype != np.uint8:
      raise ValueError('Expected unnormalized uint8 image patches')
    return [torch.from_numpy(a).to(torch.bfloat16).view(torch.uint16).numpy()
            if i not in (0, 1, 2, len(arrays)-1) else a for i, a in enumerate(arrays)]

  @staticmethod
  def signature(args, config, prepared, packed):
    return (str(config._name_or_path), prepared['tokenized_data']['image_grid_thw'].numpy().tobytes(),
            args.max_tokens, args.vision_batch,
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
    pixel_table = Tensor(self.pixel_table.copy(), device='CPU').to(Device.DEFAULT).bitcast(dtypes.bfloat16).realize()
    self.cpu = [Tensor(a, device='CPU').realize() for a in (self.arrays(args, prepared, packed) if arrays is None else arrays)]
    self.uploaded_pixel_bytes = self.cpu[0].numel()*self.cpu[0].dtype.itemsize
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
    image_indices = np.flatnonzero(ids[0] == config.vlm_config.image_token_id)
    self.spans = tuple((int(span[0]), int(span[-1])+1) for span in np.split(image_indices, np.flatnonzero(np.diff(image_indices) != 1)+1))
    self.upload_jit = TinyJit(lambda *xs: tuple(
      pixel_table[x.to(Device.DEFAULT).cast(dtypes.int32)].contiguous().realize() if i == 0 else
      fixed_positions if i == 1 else x.to(Device.DEFAULT).bitcast(dtypes.bfloat16).contiguous().realize()
      if i not in (2, len(xs)-1) else x.to(Device.DEFAULT).contiguous().realize()
      for i, x in enumerate(xs)))

  def upload(self, noise):
    self.noise[:] = noise
    return self.upload_jit(*self.cpu)
