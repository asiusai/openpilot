"""Experimental FP8 conversion directly into Harald's packed matrix layout.

The activation maximum, scale, division, clamp, and FP8 conversion are unchanged.
Only the intermediate contiguous row-major allocation is omitted.
"""
import importlib.util
from functools import partial
from pathlib import Path
from types import MethodType

from tinygrad import Tensor, dtypes
from amd_gemm import pack, _gemm

spec = importlib.util.spec_from_file_location('packed_quantize_reduction', Path(__file__).with_name('quantize-reduction.py'))
reduction = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reduction)


def quantize(x):
  assert x.dtype == dtypes.bfloat16 and x.shape[-1] % 16 == 0
  scale = (reduction.maximum(x)/448).clamp(min_=1e-12).realize()
  values = (x.float()/scale).clamp(-448,448).cast(dtypes.fp8e4m3).reshape(-1,x.shape[-1])
  return pack(values).realize(), scale


class PackedActivation:
  def __init__(self, value, rows, width):
    self.value, self.rows, self.width = value, rows, width


def install(model):
  """Preserve tensor-wide FP8 scales and restrict packed values to language prefill."""
  eligible = {name[:-7]: weight for name,weight in model.weights.items()
              if name.startswith('vlm.model.language_model.layers.') and name.endswith('.weight')
              and ('.mlp.' in name or '.self_attn.' in name) and weight.dtype == dtypes.fp8e4m3}
  assert eligible
  weights = {name:pack(weight).realize() for name,weight in eligible.items()}
  old_quantize,old_linear,old_layer = model.quantize,model.linear,model.text_layer
  state = {'prefill':False}

  def quantize_input(self,x):
    if not state['prefill'] or x.dtype != dtypes.bfloat16 or x.numel()//x.shape[-1] < 128:
      return old_quantize(x)
    value,scale = quantize(x)
    return PackedActivation(value,x.numel()//x.shape[-1],x.shape[-1]),scale

  def linear(self,x,name,quantized_input=None,heads=None):
    rows,width = x.numel()//x.shape[-1],x.shape[-1]
    if not state['prefill'] or name not in eligible or rows < 128:
      return old_linear(x,name,quantized_input,heads)
    packed,scale = quantized_input if quantized_input is not None else self.quantize(x)
    assert isinstance(packed,PackedActivation) and (packed.rows,packed.width)==(rows,width)
    weight = eligible[name]
    bias = self.weights.get(name+'.bias')
    output = Tensor.empty(rows,weight.shape[0],device=x.device,dtype=dtypes.bfloat16)
    value = Tensor.custom_kernel(output,packed.value,weights[name],scale,self.weight_scales[name+'.weight'],
                                 *(() if bias is None else (bias,)),fxn=partial(_gemm,packed_input=True))[0]
    return value.reshape(*x.shape[:-1],weight.shape[0])

  def text_layer(self,x,*args,**kwargs):
    previous = state['prefill']
    state['prefill'] = x.shape[1] >= 128
    try:
      return old_layer(x,*args,**kwargs)
    finally:
      state['prefill'] = previous

  model.quantize = MethodType(quantize_input,model)
  model.linear = MethodType(linear,model)
  model.text_layer = MethodType(text_layer,model)
  return len(weights)
