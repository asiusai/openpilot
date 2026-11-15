"""Contiguous two-stage activation reduction; retains Harald's FP8 scale and rounding."""
from types import MethodType
from tinygrad import dtypes


def quantize(self, x):
  flat = x.float().flatten()
  block = 1024
  padded = flat.pad(((0, (-flat.numel()) % block),))
  partial = padded.reshape(-1, block).abs().max(axis=1).contiguous().realize()
  scale = (partial.max()/448).clamp(min_=1e-12).realize()
  return (x.float()/scale).clamp(-448, 448).cast(dtypes.fp8e4m3).reshape(-1, x.shape[-1]).contiguous(), scale


def install(model, materialize=False):
  original = model.quantize
  # The checked short-prompt path benefits. Full-resolution 2027-token prompts
  # regress by about 74 ms because materialization prevents useful fusion.
  def selected(self, x):
    if materialize and x.numel() // x.shape[-1] >= 1024 and x.shape[-1] in (2048, 9216):
      # Avoid recomputing a lazy normalization inside the global reduction.
      return quantize(self, x.contiguous().realize())
    if x.numel() >= 1_000_000 and x.numel() // x.shape[-1] < 1024 and x.shape[-1] in (2048, 9216):
      return quantize(self, x)
    return original(x)
  model.quantize = MethodType(selected, model)
