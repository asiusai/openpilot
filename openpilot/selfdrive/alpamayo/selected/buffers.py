"""Transfer BF16 inputs and reuse immutable vision positions across snapshots."""
import numpy as np
import torch
from tinygrad import Device, Tensor, TinyJit, dtypes
from openpilot.selfdrive.alpamayo.inputs import rotary


class VisionBuffers:
  """Device inputs shared by anchor encoding and complete predictions for one checkpoint."""
  def __init__(self):
    words = ((torch.arange(256,dtype=torch.float32)-127.5)/127.5).to(torch.bfloat16).view(torch.uint16).numpy()
    self.palette = Tensor(words,device='CPU').to(Device.DEFAULT).bitcast(dtypes.bfloat16).realize()
    self.positions = {}

  def expand(self, pixels):
    return self.palette[pixels.to(Device.DEFAULT).cast(dtypes.int32)].contiguous().realize()

  def position_buffer(self, grids, positions):
    key = grids.tobytes()
    if key not in self.positions:
      if len(self.positions) >= 4:
        self.positions.pop(next(iter(self.positions)))
      words = torch.from_numpy(positions).to(torch.bfloat16).view(torch.uint16).numpy()
      self.positions[key] = Tensor(words, device='CPU').to(Device.DEFAULT).bitcast(dtypes.bfloat16).realize()
    return self.positions[key]

  @staticmethod
  def batches(grids, limit):
    offsets = np.concatenate(([0], np.cumsum(grids.prod(axis=1))))
    batches, i = [], 0
    while i < len(grids):
      end = i+1
      while end < len(grids) and np.array_equal(grids[i], grids[end]) and end-i < limit:
        end += 1
      batches.append((int(offsets[i]), int(offsets[end]), end-i))
      i = end
    return tuple(batches)


class CompactInputs:

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
    if arrays[0].shape != self.pixels.shape or any(a.shape != old.shape for a, old in zip(arrays[1:], self.cpu, strict=True)):
      return False
    self.pixels = arrays[0]
    for a, old in zip(arrays[1:], self.cpu, strict=True):
      destination = np.frombuffer(old._buffer().as_memoryview(allow_zero_copy=True), dtype=a.dtype).reshape(a.shape)
      np.copyto(destination, a, casting='no')
    return True

  def __init__(self, args, config, prepared, packed, arrays=None, vision=None):
    vision = VisionBuffers() if vision is None else vision
    ids, positions, delta, vision_positions = packed
    self._signature = self.signature(args, config, prepared, packed)
    grids = prepared['tokenized_data']['image_grid_thw'].numpy()
    fixed_positions = vision.position_buffer(grids,vision_positions)
    self.pixels, *metadata = self.arrays(args, prepared, packed) if arrays is None else arrays
    self.cpu = [Tensor(a, device='CPU').realize() for a in metadata]
    self.uploaded_pixel_bytes = 0
    self.noise = np.frombuffer(self.cpu[-1]._buffer().as_memoryview(allow_zero_copy=True), dtype=np.float32).reshape(1, 64, 2)
    self.batches = vision.batches(grids,args.vision_batch)
    image_indices = np.flatnonzero(ids[0] == config.vlm_config.image_token_id)
    self.spans = tuple((int(span[0]), int(span[-1])+1) for span in np.split(image_indices, np.flatnonzero(np.diff(image_indices) != 1)+1))
    self.uploads = {}

    def transfer(pixels, _position_placeholder, ids, *fields):
      normalized = None if pixels is None else vision.expand(pixels)
      angles = [x.to(Device.DEFAULT).bitcast(dtypes.bfloat16).contiguous().realize() for x in fields[:-1]]
      return normalized, fixed_positions, ids.to(Device.DEFAULT).contiguous().realize(), *angles, fields[-1].to(Device.DEFAULT).contiguous().realize()

    self.transfer = transfer

  def upload(self, noise, missing):
    ranges = [self.batches[i][:2] for i in missing]
    rows = sum(end-begin for begin,end in ranges)
    if rows not in self.uploads:
      buf = Tensor.empty(rows, self.pixels.shape[1], dtype=dtypes.uint8, device='CPU').realize() if rows else None
      dest = np.frombuffer(buf._buffer().as_memoryview(allow_zero_copy=True), dtype=np.uint8).reshape(buf.shape) if rows else None
      self.uploads[rows] = buf, dest, TinyJit(self.transfer)
    buf, dest, transfer = self.uploads[rows]
    offset = 0
    for begin,end in ranges:
      dest[offset:offset+end-begin] = self.pixels[begin:end]
      offset += end-begin
    self.uploaded_pixel_bytes = dest.nbytes if rows else 0
    self.noise[:] = noise
    return transfer(buf, *self.cpu)
