"""Experimental single-dispatch BF16 LayerNorm and RMSNorm on RDNA4."""
from functools import lru_cache, partial
from types import MethodType
from tinygrad import Tensor, dtypes
from tinygrad.uop.ops import UOp, Ops, KernelInfo
from tinygrad.runtime.support.compiler_amd import HIPCompiler


@lru_cache(None)
def compiled(width, rms, eps):
  text = r'''
#pragma clang fp contract(off)
#define DEV __attribute__((device)) __attribute__((always_inline)) inline
typedef unsigned short u16;
extern "C" __attribute__((device)) unsigned long __ockl_get_local_id(unsigned);
extern "C" __attribute__((device)) unsigned long __ockl_get_group_id(unsigned);
DEV float f(u16 x) { return __builtin_bit_cast(float, (unsigned)x << 16); }
DEV u16 b(float x) { unsigned v=__builtin_bit_cast(unsigned,x); return (v+0x7fff+((v>>16)&1))>>16; }
template<int mask> DEV float shuffle(float x) {
  return __builtin_bit_cast(float,__builtin_amdgcn_ds_swizzle(__builtin_bit_cast(int,x),mask));
}
DEV float sum(float x) {
  x+=shuffle<0x401f>(x); x+=shuffle<0x201f>(x); x+=shuffle<0x101f>(x);
  x+=shuffle<0x081f>(x); x+=shuffle<0x041f>(x); return x;
}
DEV void barrier() {
  __builtin_amdgcn_fence(__ATOMIC_RELEASE,"workgroup"); __builtin_amdgcn_s_barrier();
  __builtin_amdgcn_fence(__ATOMIC_ACQUIRE,"workgroup");
}
extern "C" __attribute__((global,amdgpu_flat_work_group_size(256,256)))
void alpamayo_norm(u16 *out,const u16 *x,const u16 *weight,const u16 *bias) {
  int t=__ockl_get_local_id(0), row=__ockl_get_group_id(0), wave=t/32, lane=t%32;
  __attribute__((shared,aligned(16))) float sums[8];
  float mean=0;
  if (!RMS) {
    float v=0; for(int i=t;i<WIDTH;i+=256) v+=f(x[row*WIDTH+i]);
    v=sum(v); if(lane==0) sums[wave]=v; barrier();
    for(int i=0;i<8;i++) mean+=sums[i]; mean/=WIDTH; barrier();
  }
  float squared=0;
  for(int i=t;i<WIDTH;i+=256) { float v=f(x[row*WIDTH+i])-mean; squared+=v*v; }
  squared=sum(squared); if(lane==0) sums[wave]=squared; barrier();
  float variance=0; for(int i=0;i<8;i++) variance+=sums[i]; variance/=WIDTH;
  float inv=__builtin_amdgcn_rsqf(variance+EPSILON);
  for(int i=t;i<WIDTH;i+=256) {
    float normalized=(f(x[row*WIDTH+i])-mean)*inv;
    out[row*WIDTH+i]=RMS?b(f(b(normalized))*f(weight[i])):b(normalized*f(weight[i])+f(bias[i]));
  }
}
'''
  text = f'#define WIDTH {width}\n#define RMS {int(rms)}\n#define EPSILON {eps:.10e}f\n'+text
  return text, HIPCompiler('gfx1200').compile_cached(text)


def program(out, x, weight, bias, rms, eps):
  value=x.flatten()[0].load().float()+weight[0].load().float()+bias[0].load().float()
  sink=UOp.sink(out.flatten()[0].store(value.cast(out.dtype)),
                UOp.special(x.numel()//x.shape[-1], 'gidx0'), UOp.special(256,'lidx0'), arg=KernelInfo(name='alpamayo_norm'))
  text,binary=compiled(x.shape[-1],rms,eps)
  return UOp(Ops.PROGRAM,src=(sink,UOp(Ops.LINEAR,src=tuple(sink.toposort())),UOp(Ops.SOURCE,arg=text),UOp(Ops.BINARY,arg=binary)))


def norm(x, weight, bias=None, rms=False, eps=1e-5):
  assert x.dtype == weight.dtype == dtypes.bfloat16
  out=Tensor.empty(x.shape,device=x.device,dtype=x.dtype)
  return Tensor.custom_kernel(out,x.contiguous().realize(),weight.contiguous().realize(),
                              (weight if bias is None else bias).contiguous().realize(),
                              fxn=partial(program,rms=rms,eps=eps))[0].realize()


def install(model):
  original=model.norm
  def apply(self,x,name,rms=False,eps=None):
    if x.dtype != dtypes.bfloat16 or x.numel()//x.shape[-1] < 64:
      return original(x,name,rms,eps)
    return norm(x,self.weights[f'{name}.weight'],None if rms else self.weights[f'{name}.bias'],
                rms,eps if eps is not None else 1e-5 if rms else 1e-6)
  model.norm=MethodType(apply,model)
  model.row_independent_norm=True
