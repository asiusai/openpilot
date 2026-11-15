"""Batch image compute while retaining one activation scale per image.

This differs from ordinary batch-wide FP8 quantization. Validate resulting
features against independent images before enabling it in performance tests.
"""
import importlib.util
from pathlib import Path
from types import MethodType
from tinygrad import Tensor, dtypes


def install(model, image_gemm=False):
  spec = importlib.util.spec_from_file_location('vision_row_gemm', Path(__file__).with_name('tiled-gemm.py'))
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  image_module = None
  if image_gemm:
    image_spec = importlib.util.spec_from_file_location('independent_image_gemm',Path(__file__).with_name('image-gemm.py'))
    image_module = importlib.util.module_from_spec(image_spec)
    image_spec.loader.exec_module(image_module)
  old_quantize, old_linear, old_norm = model.quantize, model.linear, model.norm
  state = {'images': 0}

  def norm(self, x, name, rms=False, eps=None):
    if not state['images'] or not name.startswith('vlm.model.visual.encoder.layers.') or getattr(model,'row_independent_norm',False):
      return old_norm(x, name, rms, eps)
    # Reduction scheduling changes with the outer batch size. Retain the
    # independent-image reduction order before batching the matrix products.
    parts = [old_norm(x[i:i+1].contiguous().realize(), name, rms, eps).realize()
             for i in range(state['images'])]
    return Tensor.cat(*parts, dim=0)

  def quantize(self, x):
    if not state['images']:
      return old_quantize(x)
    images = state['images']
    if hasattr(model, 'grouped_quantize'):
      return model.grouped_quantize(x, images)
    grouped = x.float().reshape(images, -1, x.shape[-1])
    scale = (grouped.abs().max(axis=(1, 2), keepdim=True)/448).clamp(min_=1e-12).realize()
    quantized = (grouped/scale).clamp(-448,448).cast(dtypes.fp8e4m3).reshape(-1,x.shape[-1]).contiguous()
    return quantized, scale.reshape(images,1).expand(images,grouped.shape[1]).reshape(-1).contiguous()

  def linear(self, x, name, quantized_input=None, heads=None):
    if not state['images'] or f'{name}.weight' not in self.weight_scales:
      return old_linear(x, name, quantized_input, heads)
    partial_mlp = name.endswith('.mlp.fc1') and x.shape[-2] % 16 and self.weights[f'{name}.weight'].shape[0] % 64
    if image_module is not None and partial_mlp:
      images, width = state['images'], x.shape[-1]
      rows = x.numel()//(images*width)
      quantized, scale = self.quantize(x) if quantized_input is None else quantized_input
      weight = self.weights[f'{name}.weight']
      image_scale = scale.reshape(images,rows)[:,0].contiguous()
      return image_module.gemm(quantized.reshape(images,rows,width),weight,
        (image_scale,self.weight_scales[f'{name}.weight']),bias=self.weights.get(f'{name}.bias')).reshape(*x.shape[:-1],weight.shape[0])
    if (name.startswith(('vlm.model.visual.embeddings.', 'vlm.model.projector.'))
        or partial_mlp):
      # Preserve the reference GEMM path for embeddings/projectors and partial
      # MLP tiles. Identical half-resolution FC1 inputs otherwise round differently.
      count = state['images']
      grouped = x.reshape(count, -1, x.shape[-1])
      state['images'] = 0
      try:
        parts = [old_linear(grouped[i:i+1], name, heads=heads).realize() for i in range(count)]
        return Tensor.cat(*parts, dim=0).reshape(*x.shape[:-1], parts[0].shape[-1])
      finally:
        state['images'] = count
    quantized, scale = self.quantize(x) if quantized_input is None else quantized_input
    weight = self.weights[f'{name}.weight']
    return module.gemm(quantized, weight, (scale,self.weight_scales[f'{name}.weight']),
                       row_scale=True, bias=self.weights.get(f'{name}.bias')).reshape(*x.shape[:-1],weight.shape[0])

  model.quantize = MethodType(quantize,model)
  model.linear = MethodType(linear,model)
  model.norm = MethodType(norm,model)

  def encode(*pairs):
    count = len(pairs)//2
    assert count > 0
    if pairs[0].shape[-2] < 128:
      # Harald uses another GEMM and attention path below 128 patches. Keep
      # independent images on that path; batching changed the checked features.
      assert state['images'] == 0
      return tuple(model._vision(p, v) for p, v in zip(pairs[::2], pairs[1::2], strict=True))
    state['images'] = count
    try:
      pixels, positions = Tensor.cat(*pairs[::2],dim=0), Tensor.cat(*pairs[1::2],dim=0)
      value = (model.linear(pixels, 'vlm.model.visual.embeddings.patch_embedding') + positions).realize()
      for layer in range(model.config['vlm_config']['vision_config']['num_hidden_layers']):
        value = model.vision_layer(value, layer)
      # Keep the complete projector graph per image. Splitting only its linears
      # changes fusion around normalization/GELU and introduces rounding changes.
      state['images'] = 0
      if hasattr(model, 'batched_projector'):
        return model.batched_projector(value)
      outputs = []
      for i in range(count):
        x = model.norm(value[i:i+1], 'vlm.model.visual.post_layernorm')
        x = model.norm(x, 'vlm.model.projector.norm').reshape(-1,4608)
        x = model.linear(x, 'vlm.model.projector.linear_fc1').float().gelu(approximate='none').cast(model.dtype)
        outputs.append(model.linear(x, 'vlm.model.projector.linear_fc2').realize())
      return tuple(outputs)
    finally:
      state['images'] = 0
  return encode
