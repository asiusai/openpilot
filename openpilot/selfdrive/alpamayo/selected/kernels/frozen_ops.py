"""PROGRAM adapters for the exact selected HIP objects."""
from functools import partial
from tinygrad import Tensor, dtypes
from tinygrad.uop.ops import KernelInfo, UOp
from ..frozen import program as frozen_program


def maximum_program(out, x, final, images=1):
  sink = UOp.sink(out.flatten()[0].store(x.flatten()[0].load().float()),
                 UOp.special(out.numel(), 'gidx0'), UOp.special(256, 'lidx0'), arg=KernelInfo(name='alpamayo_absmax'))
  return frozen_program(sink, 'absmax', (x.numel()//images, x.dtype == dtypes.bfloat16, final))


def maximum(x, images=1):
  assert x.dtype == dtypes.bfloat16 and x.numel() > 0 and images > 0 and x.numel()%images == 0
  values = x.contiguous().realize()
  partial_max = Tensor.empty((images*((x.numel()//images+4095)//4096),), device=x.device, dtype=dtypes.float32)
  partial_max = Tensor.custom_kernel(partial_max, values, fxn=partial(maximum_program, final=False, images=images))[0].realize()
  result = Tensor.empty(() if images == 1 else (images,1,1), device=x.device, dtype=dtypes.float32)
  return Tensor.custom_kernel(result, partial_max, fxn=partial(maximum_program, final=True, images=images))[0].realize()


def quantize(x):
  scale = (maximum(x)/448).clamp(min_=1e-12).realize()
  return (x.float()/scale).clamp(-448,448).cast(dtypes.fp8e4m3).reshape(-1,x.shape[-1]).contiguous(), scale


def grouped_quantize(x, images):
  # The maximum is order-free, so a fused reduction gives the same scale as the frozen absmax objects.
  # x stays lazy: producers such as GELU are recomputed in the reduction and in the packing quantizer
  # instead of being stored, and the quantizer writes the packed layout directly.
  assert images > 0 and x.numel() % (images*x.shape[-1]) == 0
  grouped = x.reshape(images,-1,x.shape[-1])
  scale = (grouped.float().abs().max(axis=(1,2),keepdim=True)/448).clamp(min_=1e-12).realize()
  quantized = (grouped.float()/scale).clamp(-448,448).cast(dtypes.fp8e4m3).reshape(-1,x.shape[-1])
  return quantized, scale.reshape(images,1).expand(images,grouped.shape[1]).reshape(-1).contiguous()


def norm_program(out, x, weight, bias, rms, eps):
  value = x.flatten()[0].load().float()+weight[0].load().float()+bias[0].load().float()
  sink = UOp.sink(out.flatten()[0].store(value.cast(out.dtype)),
                 UOp.special(x.numel()//x.shape[-1], 'gidx0'), UOp.special(256,'lidx0'), arg=KernelInfo(name='alpamayo_norm'))
  return frozen_program(sink, 'norm', (x.shape[-1],rms,eps))


def norm(x, weight, bias=None, rms=False, eps=1e-5):
  assert x.dtype == weight.dtype == dtypes.bfloat16
  out = Tensor.empty(x.shape, device=x.device, dtype=x.dtype)
  return Tensor.custom_kernel(out, x.contiguous().realize(), weight.contiguous().realize(),
                             (weight if bias is None else bias).contiguous().realize(),
                             fxn=partial(norm_program,rms=rms,eps=eps))[0].realize()


def norm_fp8_program(out, x, weight, scale):
  dependency = x.flatten()[0].load().float()+weight.flatten()[0].load().float()+scale.flatten()[0].load()
  sink = UOp.sink(out.flatten()[0].store(dependency.cast(dtypes.fp8e4m3)),
                 UOp.special(x.numel()//x.shape[-1], 'gidx0'), UOp.special(256, 'lidx0'), arg=KernelInfo(name='alpamayo_norm_fp8'))
  return frozen_program(sink, 'norm_fp8', (x.shape[-1], 1e-5))


def norm_quantize(x, weight, scale):
  """Prefill RMS norm (same arithmetic as norm) quantized with a fixed scale and written in amd_gemm.pack layout.

  The BF16 normalized value is rounded exactly as norm stores it; it is never stored itself.
  """
  assert x.dtype == weight.dtype == dtypes.bfloat16 and x.shape[-1] == 2048 and scale.dtype == dtypes.float32 and scale.numel() == 1
  rows = x.numel()//x.shape[-1]
  assert rows % 16 == 0
  out = Tensor.empty(rows, x.shape[-1], device=x.device, dtype=dtypes.fp8e4m3)
  return Tensor.custom_kernel(out, x.contiguous().realize(), weight.contiguous().realize(), scale, fxn=norm_fp8_program)[0]


def pack_values(v):
  batch,heads,keys,width = v.shape
  return (v.bitcast(dtypes.uint16).pad(((0,0),(0,0),(0,(-keys)%16),(0,(-width)%16)))
          .reshape(batch,heads,(keys+15)//16,2,8,(width+15)//16,16)
          .permute(0,1,5,2,3,6,4).contiguous().bitcast(dtypes.bfloat16))


def attention_program(out, q, k, v, *, shape, causal):
  batch,heads,rows,width = shape
  keys,kv_heads = k.shape[2],k.shape[1]
  prefill = width == 128
  if (prefill and not causal) or (not prefill and causal):
    raise ValueError('Frozen attention causal policy changed')
  name = 'agent_attention_rowsoftmax' if prefill else 'alpamayo_online'
  dependency = q.flatten()[0].load()+k.flatten()[0].load()+v.flatten()[0].load()
  if not prefill:
    dependency = dependency+0  # Preserve the original unmasked PROGRAM dependency expression.
  sink = UOp.sink(out.flatten()[0].store(dependency),
                 UOp.special(batch*heads*((rows+127)//128), 'gidx0'), UOp.special(256, 'lidx0'), arg=KernelInfo(name=name))
  return frozen_program(sink, 'prefill_attention' if prefill else 'vision_attention', (heads,kv_heads,rows,keys,width))


def attention(q, k, v, causal=False):
  assert q.dtype == k.dtype == v.dtype == dtypes.bfloat16
  assert q.shape[-1] == k.shape[-1] == v.shape[-1] and q.shape[-1] in (72,128)
  assert q.shape[0] == k.shape[0] == v.shape[0] and k.shape == v.shape
  out = Tensor.empty(q.shape, device=q.device, dtype=q.dtype)
  if q.shape[2] >= 1024:
    v = pack_values(v)
  return Tensor.custom_kernel(out,q.contiguous(),k.contiguous(),v.contiguous(),
                             fxn=partial(attention_program,shape=q.shape,causal=causal))[0]


def mlp_program(out, x, weight, xs, ws, output_scale, *, rows):
  columns,width = out.shape[-1],weight.shape[-1]
  dependency = x.flatten()[0].load().float()+weight.flatten()[0].load().float()+xs.flatten()[0].load()+ws.flatten()[0].load()+output_scale.flatten()[0].load()
  # Four waves share each workgroup. Each wave computes one 64 by 64 tile.
  sink = UOp.sink(out.flatten()[0].store(dependency.cast(dtypes.fp8e4m3)),
                 UOp.special(-(-(rows//64)//4)*(columns//64),'gidx0'), UOp.special(128,'lidx0'),
                 arg=KernelInfo(name='agent_packed_epilogue'))
  return frozen_program(sink,'mlp',(rows,columns,width))


def fused_mlp(x, weight, xs, ws, output_scale, rows):
  assert x.dtype == weight.dtype == dtypes.fp8e4m3 and x.shape[-1] == weight.shape[-1]
  assert xs.dtype == ws.dtype == output_scale.dtype == dtypes.float32
  assert xs.numel() == ws.numel() == output_scale.numel() == 1
  out = Tensor.empty((rows+15)//16*16,weight.shape[0],device=x.device,dtype=dtypes.fp8e4m3)
  return Tensor.custom_kernel(out,x,weight,xs,ws,output_scale,fxn=partial(mlp_program,rows=rows))[0]
