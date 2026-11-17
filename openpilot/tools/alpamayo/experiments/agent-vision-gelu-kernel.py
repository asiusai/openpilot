"""One experimental GELU plus partial image maximum, after the original BF16 fc1.

No model hook is installed here. The parent must establish exact GPU equality.
"""
from functools import partial


def kernel(out, partial_max, x, *, images, chunk=4096):
  from tinygrad import dtypes
  from tinygrad.uop.ops import UOp, Ops, KernelInfo
  assert x.dtype == out.dtype == dtypes.bfloat16 and partial_max.dtype == dtypes.float32
  count = x.numel()//images
  assert x.numel() == images*count and out.numel() == x.numel()
  chunks = (count+chunk-1)//chunk
  assert partial_max.numel() == images*chunks
  group, element = UOp.range(images*chunks, 0), UOp.range(chunk, 1)
  image, start = group//chunks, (group%chunks)*chunk
  index = image*count+start+element
  valid = start+element < count
  value = x.flatten().index(index.valid(valid)).load().float().gelu().cast(dtypes.bfloat16)
  saved = out.flatten().index(index.valid(valid)).store(value)
  # Match HIP fabsf for signed zero as well as ordinary finite values.
  magnitude = (value.after(saved).float().bitcast(dtypes.uint32) & 0x7fffffff).bitcast(dtypes.float32)
  maximum = magnitude.reduce(element, arg=Ops.MAX)
  reduced = partial_max.flatten().index(group).store(maximum).end(group)
  return UOp.sink(reduced, arg=KernelInfo(name='agent_vision_gelu_partialmax'))


def activation_and_maximum(x, images, reduction):
  from tinygrad import Tensor, dtypes
  assert x.dtype == dtypes.bfloat16 and x.shape == (images, 720, 4304) and images in (1, 2)
  count = 720*4304
  values = Tensor.empty(*x.shape, device=x.device, dtype=dtypes.bfloat16)
  partial_max = Tensor.empty(images*((count+4095)//4096), device=x.device, dtype=dtypes.float32)
  values, partial_max = Tensor.custom_kernel(values, partial_max, x, fxn=partial(kernel, images=images))[:2]
  Tensor.realize(values, partial_max)
  result = Tensor.empty(() if images == 1 else (images, 1, 1), device=x.device, dtype=dtypes.float32)
  maximum = Tensor.custom_kernel(result, partial_max, fxn=partial(reduction.program, final=True, images=images))[0].realize()
  return values, maximum


def quantize(x, images, reduction):
  from tinygrad import dtypes
  values, maximum = activation_and_maximum(x, images, reduction)
  scale = (maximum/448).clamp(min_=1e-12).realize()
  # This is the unchanged grouped quantizer's division/clamp/cast expression.
  result = (values.float()/scale).clamp(-448, 448).cast(dtypes.fp8e4m3).reshape(-1, x.shape[-1]).contiguous()
  return result, scale.reshape(images, 1).expand(images, 720).reshape(-1).contiguous()
