import ctypes
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pytest
from tinygrad import Device

from openpilot.selfdrive.modeld.helpers import tensor_from_dma_buf


@pytest.mark.parametrize('device', ['QCOM', 'QCOM:1'])
def test_dma_buf_import_realizes_tensor_without_allocating_camera_storage(device):
  pixels = np.arange(64, dtype=np.uint8)
  iface = SimpleNamespace(map=Mock(return_value=pixels))
  allocator = Mock()
  backend = SimpleNamespace(iface=iface, allocator=allocator)
  with patch.object(type(Device), '__getitem__', return_value=backend):
    tensor = tensor_from_dma_buf(pixels.ctypes.data, 9, pixels.nbytes, device)
    buffer = tensor.uop.buffer
    try:
      assert tensor.shape == (64,)
      assert tensor.device == device
      assert buffer.is_allocated()
      iface.map.assert_called_once_with(pixels.ctypes.data, pixels.nbytes, 9)
      allocator.alloc.assert_not_called()
      assert bytes(buffer.host[:]) == pixels.tobytes()
      pixels[0] = 200
      assert buffer.host[0] == 200
    finally:
      buffer.deallocate()
    allocator.free.assert_called_once()


@pytest.mark.parametrize('fd', [None, 9])
def test_cpu_pointer_import_keeps_camera_memory(fd):
  pixels = (ctypes.c_uint8 * 4)(1, 2, 3, 4)
  tensor = tensor_from_dma_buf(ctypes.addressof(pixels), fd, 4, 'CPU')
  assert tensor.numpy().tolist() == [1, 2, 3, 4]
  pixels[0] = 200
  assert tensor.numpy().tolist() == [200, 2, 3, 4]
