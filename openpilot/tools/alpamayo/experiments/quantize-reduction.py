"""Experimental BF16 activation maximum without a full FP32 intermediate.

The FP8 conversion remains Harald's tensor expression. Selection requires
exact scale and FP8 byte comparisons against the existing quantizer.
"""
from functools import lru_cache, partial
from types import MethodType

from tinygrad import Tensor, dtypes
from tinygrad.uop.ops import UOp, Ops, KernelInfo
from tinygrad.runtime.support.compiler_amd import HIPCompiler


@lru_cache(None)
def compiled(count, bf16, final):
  text = r'''
#pragma clang fp contract(off)
#define DEV __attribute__((device)) __attribute__((always_inline)) inline
extern "C" __attribute__((device)) unsigned long __ockl_get_local_id(unsigned);
extern "C" __attribute__((device)) unsigned long __ockl_get_group_id(unsigned);
DEV float value(INPUT x) { return CONVERT; }
template<int mask> DEV float shuffle(float x) {
  return __builtin_bit_cast(float,__builtin_amdgcn_ds_swizzle(__builtin_bit_cast(int,x),mask));
}
DEV float wave_max(float x) {
  x=__builtin_fmaxf(x,shuffle<0x401f>(x)); x=__builtin_fmaxf(x,shuffle<0x201f>(x));
  x=__builtin_fmaxf(x,shuffle<0x101f>(x)); x=__builtin_fmaxf(x,shuffle<0x081f>(x));
  return __builtin_fmaxf(x,shuffle<0x041f>(x));
}
extern "C" __attribute__((global,amdgpu_flat_work_group_size(256,256)))
void alpamayo_absmax(float *out, const INPUT *x) {
  int t=__ockl_get_local_id(0), output_group=__ockl_get_group_id(0);
  int chunks=(COUNT+4095)/4096;
  int image=FINAL?output_group:output_group/chunks, group=FINAL?0:output_group%chunks;
  __attribute__((shared,aligned(16))) float maxima[8];
  float maximum=0;
  int begin=FINAL?0:group*4096, end=FINAL?COUNT:(begin+4096<COUNT?begin+4096:COUNT);
  for(int i=begin+t;i<end;i+=256) maximum=__builtin_fmaxf(maximum,__builtin_fabsf(value(x[image*COUNT+i])));
  maximum=wave_max(maximum);
  if(t%32==0) maxima[t/32]=maximum;
  __builtin_amdgcn_fence(__ATOMIC_RELEASE,"workgroup"); __builtin_amdgcn_s_barrier();
  __builtin_amdgcn_fence(__ATOMIC_ACQUIRE,"workgroup");
  if(t==0) {
    maximum=0; for(int i=0;i<8;i++) maximum=__builtin_fmaxf(maximum,maxima[i]);
    out[output_group]=maximum;
  }
}
'''
  convert = '__builtin_bit_cast(float,(unsigned)x<<16)' if bf16 else 'x'
  text = f'#define COUNT {count}\n#define FINAL {int(final)}\n#define INPUT {"unsigned short" if bf16 else "float"}\n#define CONVERT {convert}\n'+text
  return text, HIPCompiler('gfx1200').compile_cached(text)


def program(out, x, final, images=1):
  sink = UOp.sink(out.flatten()[0].store(x.flatten()[0].load().float()),
                  UOp.special(out.numel(),'gidx0'),UOp.special(256,'lidx0'),arg=KernelInfo(name='alpamayo_absmax'))
  text, binary = compiled(x.numel()//images, x.dtype == dtypes.bfloat16, final)
  return UOp(Ops.PROGRAM,src=(sink,UOp(Ops.LINEAR,src=tuple(sink.toposort())),
                              UOp(Ops.SOURCE,arg=text),UOp(Ops.BINARY,arg=binary)))


def maximum(x, images=1):
  assert x.dtype == dtypes.bfloat16 and x.numel() > 0 and images > 0 and x.numel()%images == 0
  values = x.contiguous().realize()
  partial_max = Tensor.empty((images*((x.numel()//images+4095)//4096),),device=x.device,dtype=dtypes.float32)
  partial_max = Tensor.custom_kernel(partial_max,values,fxn=partial(program,final=False,images=images))[0].realize()
  result = Tensor.empty(() if images==1 else (images,1,1),device=x.device,dtype=dtypes.float32)
  return Tensor.custom_kernel(result,partial_max,fxn=partial(program,final=True,images=images))[0].realize()


def quantize(x):
  scale = (maximum(x)/448).clamp(min_=1e-12).realize()
  return (x.float()/scale).clamp(-448,448).cast(dtypes.fp8e4m3).reshape(-1,x.shape[-1]).contiguous(), scale


def grouped_quantize(x, images):
  assert images > 0 and x.numel() % (images*x.shape[-1]) == 0
  grouped=x.reshape(images,-1,x.shape[-1])
  scale=(maximum(grouped,images)/448).clamp(min_=1e-12).realize()
  quantized=(grouped.float()/scale).clamp(-448,448).cast(dtypes.fp8e4m3).reshape(-1,x.shape[-1]).contiguous()
  return quantized,scale.reshape(images,1).expand(images,grouped.shape[1]).reshape(-1).contiguous()


def row_quantize(x):
  assert x.dtype == dtypes.bfloat16 and x.numel() > 0
  values=x.reshape(-1,x.shape[-1]).contiguous().realize()
  maximums=Tensor.empty((values.shape[0],1),device=x.device,dtype=dtypes.float32)
  maximums=Tensor.custom_kernel(maximums,values,fxn=partial(program,final=True,images=values.shape[0]))[0].realize()
  scales=(maximums/448).clamp(min_=1e-12).realize()
  return (values.float()/scales).clamp(-448,448).cast(dtypes.fp8e4m3).contiguous(),scales.reshape(-1)


def install(model, grouped=False, row=False):
  original = model.quantize
  def selected(self,x):
    if x.dtype == dtypes.bfloat16 and x.numel() >= 131072:
      return quantize(x)
    return original(x)
  model.quantize = MethodType(selected,model)
  if grouped:
    model.grouped_quantize=MethodType(lambda self,x,images:grouped_quantize(x,images),model)
  if row:
    model.row_quantize=MethodType(lambda self,x:row_quantize(x),model)
