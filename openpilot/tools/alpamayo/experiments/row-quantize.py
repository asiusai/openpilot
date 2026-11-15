"""Experimental per-token FP8 activation scales for language prefill.

Weights and single-token decoding stay unchanged. This changes quantization,
so numerical and complete-model comparisons are required before selection.
"""
import importlib.util
from pathlib import Path
from types import MethodType
from tinygrad import dtypes


def install(model):
  spec = importlib.util.spec_from_file_location('row_quant_gemm',Path(__file__).with_name('tiled-gemm.py'))
  gemm = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(gemm)
  old_text, old_quantize, old_linear = model.text_layer, model.quantize, model.linear
  state = {'prefill':False}

  def quantize(self,x):
    if not state['prefill']:
      return old_quantize(x)
    if hasattr(self,'row_quantize'):
      return self.row_quantize(x)
    values = x.reshape(-1,x.shape[-1]).float().contiguous().realize()
    scale = (values.abs().max(axis=1,keepdim=True)/448).clamp(min_=1e-12).realize()
    return (values/scale).clamp(-448,448).cast(dtypes.fp8e4m3).contiguous(), scale.reshape(-1)

  def linear(self,x,name,quantized_input=None,heads=None):
    if not state['prefill'] or f'{name}.weight' not in self.weight_scales:
      return old_linear(x,name,quantized_input,heads)
    values,scale = self.quantize(x) if quantized_input is None else quantized_input
    weight = self.weights[f'{name}.weight']
    assert x.dtype == dtypes.bfloat16
    result = gemm.gemm(values,weight,(scale,self.weight_scales[f'{name}.weight']),
                       bias=self.weights.get(f'{name}.bias'),row_scale=True)
    return result.reshape(*x.shape[:-1],weight.shape[0])

  def text_layer(self,x,cos,sin,prefix,*args,**kwargs):
    previous = state['prefill']
    state['prefill'] = prefix.startswith('vlm.model.language_model.layers.') and x.shape[1]>=128
    try:
      return old_text(x,cos,sin,prefix,*args,**kwargs)
    finally:
      state['prefill'] = previous

  model.quantize = MethodType(quantize,model)
  model.linear = MethodType(linear,model)
  model.text_layer = MethodType(text_layer,model)
