"""Optional fixed vision activation scales, with complete GPU range capture."""
from types import MethodType

from tinygrad import Tensor, dtypes


def install(model, fixed_scales=None, capture=False):
  if not hasattr(model, 'grouped_quantize'):
    raise ValueError('Vision calibration requires independent per-image grouped quantization')
  if fixed_scales is not None and capture:
    raise ValueError('Capture dynamic scales separately from fixed-scale inference')
  original_layer, original_quantize = model.vision_layer, model.grouped_quantize
  fixed = {key:Tensor([float(value)], dtype=dtypes.float32).reshape(()).realize()
           for key,value in (fixed_scales or {}).items()}
  accumulated, row_scales = {}, {}
  state = dict(layer=None, index=0)
  model.vision_activation_scales = accumulated

  def layer(self, x, index):
    previous = dict(state)
    state.update(layer=index, index=0)
    try:
      return original_layer(x,index)
    finally:
      state.update(previous)

  def quantize(self, x, images):
    if state['layer'] is None:
      return original_quantize(x,images)
    key = f"vlm.model.visual.encoder.layers.{state['layer']}/{state['index']}"
    state['index'] += 1
    if fixed_scales is None:
      quantized, scales = original_quantize(x,images)
      if capture:
        if key not in accumulated:
          accumulated[key] = Tensor.zeros(1,dtype=dtypes.float32).contiguous().realize()
        # This mutation remains inside each captured encoder graph. It covers
        # every encoded image, including replays of different batch sizes.
        accumulated[key].assign(accumulated[key].maximum(scales.max())).realize()
      return quantized, scales
    if key not in fixed:
      raise ValueError(f'Missing vision activation scale: {key}')
    rows = x.numel()//x.shape[-1]
    if (key,rows) not in row_scales:
      row_scales[key,rows] = fixed[key].reshape(1).expand(rows).contiguous().realize()
    quantized = (x.float()/fixed[key]).clamp(-448,448).cast(dtypes.fp8e4m3)
    return quantized.reshape(rows,x.shape[-1]).contiguous(),row_scales[key,rows]

  model.vision_layer = MethodType(layer,model)
  model.grouped_quantize = MethodType(quantize,model)
