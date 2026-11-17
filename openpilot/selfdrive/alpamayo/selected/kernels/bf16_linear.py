"""Packed BF16 linears, retaining BF16 rounding before bias."""
from functools import partial

from tinygrad import Tensor, dtypes
from openpilot.selfdrive.alpamayo.amd_attention import _pack, _matmul


def pack_weight(weight):
  assert weight.ndim == 2 and weight.dtype == dtypes.bfloat16
  return _pack(weight.reshape(1, 1, *weight.shape)).realize()


def linear(x, packed_weight, columns, bias=None):
  assert x.ndim == 2 and x.dtype == packed_weight.dtype == dtypes.bfloat16
  rows, width = x.shape
  output = Tensor.empty(1, 1, rows, columns, device=x.device, dtype=x.dtype)
  output = Tensor.custom_kernel(output, _pack(x.reshape(1, 1, rows, width)), packed_weight,
    fxn=partial(_matmul, shape=(1, 1, 1, rows, columns, width)))[0].reshape(rows, columns)
  return output + bias if bias is not None else output
