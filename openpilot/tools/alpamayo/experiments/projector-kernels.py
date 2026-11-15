"""Packed FP8 projectors with independent activation scales for every image."""
import importlib.util
from pathlib import Path
from tinygrad import Tensor, dtypes
from amd_gemm import pack, unpack


def install(model):
  spec = importlib.util.spec_from_file_location('projector_gemm', Path(__file__).with_name('tiled-gemm.py'))
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  names = ['vlm.model.projector.linear_fc1', 'vlm.model.projector.linear_fc2']
  assert all(name+'.weight' in model.weight_scales for name in names)
  weights = {name: unpack(pack(model.weights[name+'.weight']).realize()) for name in names}

  def project(value):
    images = value.shape[0]
    parts = []
    for i in range(images):
      x = model.norm(value[i:i+1], 'vlm.model.visual.post_layernorm')
      x = model.norm(x, 'vlm.model.projector.norm').reshape(-1, 4608)
      parts.append(x.realize())
    x = Tensor.cat(*parts, dim=0)
    for index, name in enumerate(names):
      if images == 1:
        quantized, scale = model.quantize(x)
      elif hasattr(model, 'grouped_quantize'):
        quantized, scale = model.grouped_quantize(x, images)
      else:
        grouped = x.float().reshape(images, -1, x.shape[-1])
        maximum = (grouped.abs().max(axis=(1, 2), keepdim=True)/448).clamp(min_=1e-12).realize()
        quantized = (grouped/maximum).clamp(-448,448).cast(dtypes.fp8e4m3).reshape(-1,x.shape[-1]).contiguous()
        scale = maximum.reshape(images,1).expand(images,grouped.shape[1]).reshape(-1).contiguous()
      x = module.gemm(quantized, weights[name], (scale,model.weight_scales[name+'.weight']),
                      row_scale=images>1, bias=model.weights.get(name+'.bias'))
      if index == 0:
        x = x.float().gelu(approximate='none').cast(model.dtype)
    x = x.realize().reshape(images, -1, x.shape[-1])
    return tuple(x[i].realize() for i in range(images))

  # Keep model._vision and its linears unchanged for independent verification.
  model.batched_projector = project
