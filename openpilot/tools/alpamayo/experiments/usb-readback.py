"""Read small contiguous GPU results through Chestnut's existing PCIe BAR path.

The layout and synchronization follow Harald's pinned runtime.download. Keep
integer token IDs as integers and avoid constructing a separate tensor copy.
"""
import numpy as np
from tinygrad import Device, dtypes


def download(value):
  numpy_type = {dtypes.int32: np.int32, dtypes.float32: np.float32}.get(value.dtype)
  if numpy_type is None:
    raise TypeError(f'Unsupported direct readback dtype: {value.dtype}')
  dev = Device[value.device]
  if not dev.is_usb:
    raise ValueError('Direct BAR readback requires Chestnut')
  value = value.contiguous().realize()
  dev.synchronize()
  buffer = value._buffer()
  assert buffer.nbytes == value.numel()*4
  offset, remaining = buffer.offset, buffer.nbytes
  chunks = []
  for address, size in buffer.base.get_storage().meta.mapping.paddrs:
    if offset >= size:
      offset -= size
      continue
    take = min(remaining, size-offset)
    for start in range(0, take, 65536):
      count = min(take-start, 65536)
      view = dev.iface.pci_dev.map_bar(bar=dev.iface.vram_bar, off=address+offset+start, size=count)
      chunks.append(bytes(view[:]))
    remaining -= take
    offset = 0
    if not remaining:
      break
  assert remaining == 0
  return np.frombuffer(b''.join(chunks), dtype=numpy_type).reshape(value.shape)
