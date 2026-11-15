"""Fuse Q/K/V projections while retaining each original weight's FP8 scale."""
import importlib.util
from pathlib import Path
from types import MethodType
from tinygrad import Tensor, dtypes
from amd_gemm import pack,unpack


def install(model):
  spec=importlib.util.spec_from_file_location('qkv_gemm',Path(__file__).with_name('tiled-gemm.py'))
  kernels=importlib.util.module_from_spec(spec);spec.loader.exec_module(kernels)
  original=model.linear
  weights={}
  for name in model.weights:
    if not name.endswith('.q_proj.weight') or name not in model.weight_scales:
      continue
    prefix=name.removesuffix('q_proj.weight')
    names=[prefix+c+'_proj' for c in 'qkv']
    values=[model.weights[n+'.weight'] for n in names]
    sizes=[w.shape[0] for w in values]
    packed=unpack(pack(Tensor.cat(*values,dim=0)).realize())
    scales=Tensor.cat(*(model.weight_scales[n+'.weight'].reshape(1).expand(s) for n,s in zip(names,sizes)),dim=0).contiguous().realize()
    biases=[model.weights.get(n+'.bias') for n in names]
    bias=None if all(b is None for b in biases) else Tensor.cat(*(b if b is not None else Tensor.zeros(s,dtype=dtypes.bfloat16) for b,s in zip(biases,sizes)),dim=0).contiguous().realize()
    weights[prefix]=(packed,scales,bias,sizes)
  pending={}
  def linear(self,x,name,quantized_input=None,heads=None):
    kind=name[-6:] if name.endswith('_proj') else ''
    prefix=name[:-6]
    if kind not in ('q_proj','k_proj','v_proj') or prefix not in weights or x.numel()//x.shape[-1]<16:
      return original(x,name,quantized_input,heads)
    if kind=='q_proj':
      q,scale=self.quantize(x) if quantized_input is None else quantized_input
      weight,scales,bias,sizes=weights[prefix]
      y=kernels.gemm(q,weight,(scale,scales),bias=bias,column_scale=True)
      cursor=0;parts=[]
      for size in sizes:
        parts.append(y[:,cursor:cursor+size].reshape(*x.shape[:-1],size));cursor+=size
      pending[prefix]=(x.uop,parts)
    assert prefix in pending and pending[prefix][0] is x.uop, 'Q/K/V input order changed'
    output=pending[prefix][1]['qkv'.index(kind[0])]
    if kind=='v_proj':del pending[prefix]
    return output
  model.linear=MethodType(linear,model)
  return len(weights)
