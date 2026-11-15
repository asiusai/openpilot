"""Experimental 128-bit FP8 loads with two ordered K16 WMMA steps."""
from functools import lru_cache, partial
import importlib.util
from pathlib import Path

from tinygrad import Tensor, dtypes
from tinygrad.uop.ops import UOp, Ops, KernelInfo
from tinygrad.runtime.support.compiler_amd import HIPCompiler


def pack32(x):
  rows,width=x.shape
  assert x.dtype==dtypes.fp8e4m3 and width%32==0
  # Keep Harald's four-byte K grouping inside each FP8 fragment. A contiguous
  # eight-byte fragment permutes the floating-point accumulation order.
  return x.bitcast(dtypes.uint8).pad(((0,(-rows)%16),(0,0))).reshape(-1,16,width//32,2,2,2,4).permute(0,2,5,1,3,4,6).contiguous().bitcast(dtypes.fp8e4m3)


@lru_cache(None)
def compiled(rows,columns,width,tile_m,tile_n,wgp):
  text=r'''
#pragma clang fp contract(off)
#define DEV __attribute__((device)) __attribute__((always_inline)) inline
typedef int i2 __attribute__((ext_vector_type(2)));
typedef int i4 __attribute__((ext_vector_type(4)));
typedef float f8 __attribute__((ext_vector_type(8)));
extern "C" __attribute__((device)) unsigned long __ockl_get_local_id(unsigned);
extern "C" __attribute__((device)) unsigned long __ockl_get_group_id(unsigned);
DEV unsigned short bf(float x) { unsigned b=__builtin_bit_cast(unsigned,x); return (b+0x7fff+((b>>16)&1))>>16; }
extern "C" __attribute__((global,amdgpu_flat_work_group_size(32,32)))
void alpamayo_vector_gemm(unsigned short *out,const i4 *x,const i4 *weight,const float *xs,const float *ws) {
  int lane=__ockl_get_local_id(0),block=__ockl_get_group_id(0);
  int mt=block/NTILES,nt=block%NTILES;
  f8 accum[TM][TN];
  #pragma unroll
  for(int i=0;i<TM;i++) {
    #pragma unroll
    for(int j=0;j<TN;j++) accum[i][j]=(f8){0};
  }
  for(int k=0;k<K/32;k++) {
    i4 a[TM],b[TN];
    #pragma unroll
    for(int i=0;i<TM;i++) a[i]=(mt*TM+i)*16<M?x[((mt*TM+i)*(K/32)+k)*32+lane]:(i4){0};
    #pragma unroll
    for(int j=0;j<TN;j++) b[j]=(nt*TN+j)*16<N?weight[((nt*TN+j)*(K/32)+k)*32+lane]:(i4){0};
    #pragma unroll
    for(int step=0;step<2;step++) {
      #pragma unroll
      for(int i=0;i<TM;i++) {
        i2 av={a[i][step*2],a[i][step*2+1]};
        #pragma unroll
        for(int j=0;j<TN;j++) {
          i2 bv={b[j][step*2],b[j][step*2+1]};
          accum[i][j]=__builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(av,bv,accum[i][j]);
        }
      }
    }
  }
  float scale=xs[0]*ws[0];
  #pragma unroll
  for(int i=0;i<TM;i++) {
    #pragma unroll
    for(int j=0;j<TN;j++) {
      int col=(nt*TN+j)*16+lane%16;
      #pragma unroll
      for(int v=0;v<8;v++) {
        int row=(mt*TM+i)*16+(lane/16)*8+v;
        if(row<M && col<N) out[row*N+col]=bf(accum[i][j][v]*scale);
      }
    }
  }
}
'''
  text=f'#define M {rows}\n#define N {columns}\n#define K {width}\n#define TM {tile_m}\n#define TN {tile_n}\n#define NTILES {(columns+tile_n*16-1)//(tile_n*16)}\n'+text
  compiler=HIPCompiler('gfx1200')
  if wgp:
    spec=importlib.util.spec_from_file_location('vector_hip',Path(__file__).with_name('hip-compiler.py'))
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    compiler=module.WGPCompiler('gfx1200')
  return text,compiler.compile_cached(text)


def program(out,x,weight,xs,ws,*,width,tile,wgp):
  rows,columns=out.shape
  value=x.flatten()[0].load().float()*weight.flatten()[0].load().float()*xs.flatten()[0].load()*ws.flatten()[0].load()
  groups=((rows+tile[0]*16-1)//(tile[0]*16))*((columns+tile[1]*16-1)//(tile[1]*16))
  sink=UOp.sink(out.flatten()[0].store(value.cast(dtypes.bfloat16)),UOp.special(groups,'gidx0'),UOp.special(32,'lidx0'),arg=KernelInfo(name='alpamayo_vector_gemm'))
  text,binary=compiled(rows,columns,width,*tile,wgp)
  return UOp(Ops.PROGRAM,src=(sink,UOp(Ops.LINEAR,src=tuple(sink.toposort())),UOp(Ops.SOURCE,arg=text),UOp(Ops.BINARY,arg=binary)))


def gemm(x,packed_weight,columns,scales,tile=(4,4),wgp=False):
  assert x.ndim==2 and x.dtype==packed_weight.dtype==dtypes.fp8e4m3
  assert packed_weight.numel()==((columns+15)//16)*16*x.shape[1]
  out=Tensor.empty((x.shape[0],columns),device=x.device,dtype=dtypes.bfloat16)
  return Tensor.custom_kernel(out,pack32(x),packed_weight,*scales,fxn=partial(program,width=x.shape[1],tile=tile,wgp=wgp))[0]
