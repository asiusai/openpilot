"""Experimental packed BF16 linears, retaining BF16 rounding before bias."""
from functools import partial
from types import MethodType

from tinygrad import Tensor, dtypes
from amd_attention import _pack, _matmul


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


def install(model):
  """Use explicit matrix boundaries only for restored, full-size vision inputs."""
  prefixes = ('vlm.model.visual.embeddings.patch_embedding.', 'vlm.model.visual.encoder.layers.')
  eligible = {name[:-7]: weight for name, weight in model.weights.items()
              if name.startswith(prefixes) and name.endswith('.weight') and weight.ndim == 2
              and weight.dtype == dtypes.bfloat16 and name not in model.weight_scales}
  assert eligible, 'Restore selected vision weights before installing packed BF16 linears'
  packed = {name: pack_weight(weight) for name, weight in eligible.items()}
  original = model.linear

  def patched(self, x, name, quantized_input=None, heads=None):
    if name not in packed or x.numel() // x.shape[-1] < 128:
      return original(x, name, quantized_input, heads)
    assert quantized_input is None and x.dtype == dtypes.bfloat16
    weight = eligible[name]
    bias = self.weights.get(name+'.bias')
    if x.ndim == 3 and x.shape[0] > 1 and x.shape[1] % 16 and weight.shape[0] % 64:
      # Preserve independent-image rounding at partial row and column tiles.
      outputs = [linear(x[index], packed[name], weight.shape[0], bias).unsqueeze(0) for index in range(x.shape[0])]
      return outputs[0].cat(*outputs[1:], dim=0)
    output = linear(x.reshape(-1, x.shape[-1]), packed[name], weight.shape[0], bias)
    return output.reshape(*x.shape[:-1], weight.shape[0])

  model.linear = MethodType(patched, model)
  return sorted(packed)
