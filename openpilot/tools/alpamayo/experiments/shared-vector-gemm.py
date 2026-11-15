"""Experimental shared FP8 tiles with ordered K16 matrix instructions."""
from functools import lru_cache, partial
import importlib.util
from pathlib import Path

from tinygrad import Tensor, dtypes
from tinygrad.uop.ops import UOp, Ops, KernelInfo
from tinygrad.runtime.support.compiler_amd import HIPCompiler

spec=importlib.util.spec_from_file_location('shared_vector_layout',Path(__file__).with_name('vector-gemm.py'))
layout=importlib.util.module_from_spec(spec);spec.loader.exec_module(layout)


@lru_cache(None)
def compiled(rows,columns,width,group_m,group_n,prefetch,tile_m=4,tile_n=4):
  waves=group_m*group_n
  text=r'''
#pragma clang fp contract(off)
#define DEV __attribute__((device)) __attribute__((always_inline)) inline
typedef int i2 __attribute__((ext_vector_type(2)));
typedef int i4 __attribute__((ext_vector_type(4)));
typedef float f8 __attribute__((ext_vector_type(8)));
extern "C" __attribute__((device)) unsigned long __ockl_get_local_id(unsigned);
extern "C" __attribute__((device)) unsigned long __ockl_get_group_id(unsigned);
DEV unsigned short bf(float x) { unsigned b=__builtin_bit_cast(unsigned,x); return (b+0x7fff+((b>>16)&1))>>16; }
DEV void barrier() {
  __builtin_amdgcn_fence(__ATOMIC_RELEASE,"workgroup","local");
  __builtin_amdgcn_s_barrier();
  __builtin_amdgcn_fence(__ATOMIC_ACQUIRE,"workgroup","local");
}
extern "C" __attribute__((global,amdgpu_flat_work_group_size(THREADS,THREADS)))
void alpamayo_shared_vector(unsigned short *out,const i4 *x,const i4 *weight,const float *xs,const float *ws) {
  __attribute__((shared)) i4 shared_x[GM*TM*32],shared_w[GN*TN*32];
  int thread=__ockl_get_local_id(0),lane=thread%32,wave=thread/32,block=__ockl_get_group_id(0);
  int bm=block/NTILES,bn=block%NTILES,wm=wave/GN,wn=wave%GN;
  int mt=bm*GM+wm,nt=bn*GN+wn;
  f8 accum[TM][TN];
  #pragma unroll
  for(int i=0;i<TM;i++) {
    #pragma unroll
    for(int j=0;j<TN;j++) accum[i][j]=(f8){0};
  }
  i4 next_x[XL],next_w[WL];
  #pragma unroll
  for(int l=0;l<XL;l++) {
    int entry=thread+l*THREADS,row=bm*GM*TM+entry/32;
    next_x[l]=entry<GM*TM*32 && row*16<M?x[(row*(K/32))*32+entry%32]:(i4){0};
  }
  #pragma unroll
  for(int l=0;l<WL;l++) {
    int entry=thread+l*THREADS,row=bn*GN*TN+entry/32;
    next_w[l]=entry<GN*TN*32 && row*16<N?weight[(row*(K/32))*32+entry%32]:(i4){0};
  }
  for(int k=0;k<K/32;k++) {
    #pragma unroll
    for(int l=0;l<XL;l++) { int e=thread+l*THREADS; if(e<GM*TM*32) shared_x[e]=next_x[l]; }
    #pragma unroll
    for(int l=0;l<WL;l++) { int e=thread+l*THREADS; if(e<GN*TN*32) shared_w[e]=next_w[l]; }
    barrier();
    i4 a[TM],b[TN];
    #pragma unroll
    for(int i=0;i<TM;i++) a[i]=shared_x[(wm*TM+i)*32+lane];
    #pragma unroll
    for(int j=0;j<TN;j++) b[j]=shared_w[(wn*TN+j)*32+lane];
    barrier();
    if(PREFETCH && k+1<K/32) {
      #pragma unroll
      for(int l=0;l<XL;l++) {
        int entry=thread+l*THREADS,row=bm*GM*TM+entry/32;
        next_x[l]=entry<GM*TM*32 && row*16<M?x[(row*(K/32)+k+1)*32+entry%32]:(i4){0};
      }
      #pragma unroll
      for(int l=0;l<WL;l++) {
        int entry=thread+l*THREADS,row=bn*GN*TN+entry/32;
        next_w[l]=entry<GN*TN*32 && row*16<N?weight[(row*(K/32)+k+1)*32+entry%32]:(i4){0};
      }
    }
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
    if(!PREFETCH && k+1<K/32) {
      #pragma unroll
      for(int l=0;l<XL;l++) {
        int entry=thread+l*THREADS,row=bm*GM*TM+entry/32;
        next_x[l]=entry<GM*TM*32 && row*16<M?x[(row*(K/32)+k+1)*32+entry%32]:(i4){0};
      }
      #pragma unroll
      for(int l=0;l<WL;l++) {
        int entry=thread+l*THREADS,row=bn*GN*TN+entry/32;
        next_w[l]=entry<GN*TN*32 && row*16<N?weight[(row*(K/32)+k+1)*32+entry%32]:(i4){0};
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
  defines=dict(M=rows,N=columns,K=width,GM=group_m,GN=group_n,TM=tile_m,TN=tile_n,THREADS=waves*32,
               NTILES=(columns+group_n*tile_n*16-1)//(group_n*tile_n*16),XL=(group_m*tile_m*32+waves*32-1)//(waves*32),
               WL=(group_n*tile_n*32+waves*32-1)//(waves*32),PREFETCH=int(prefetch))
  text=''.join(f'#define {key} {value}\n' for key,value in defines.items())+text
  return text,HIPCompiler('gfx1200').compile_cached(text)


def program(out,x,weight,xs,ws,*,width,group,prefetch,tile):
  rows,columns=out.shape
  value=x.flatten()[0].load().float()*weight.flatten()[0].load().float()*xs.flatten()[0].load()*ws.flatten()[0].load()
  groups=((rows+group[0]*tile[0]*16-1)//(group[0]*tile[0]*16))*((columns+group[1]*tile[1]*16-1)//(group[1]*tile[1]*16))
  sink=UOp.sink(out.flatten()[0].store(value.cast(dtypes.bfloat16)),UOp.special(groups,'gidx0'),
                UOp.special(group[0]*group[1]*32,'lidx0'),arg=KernelInfo(name='alpamayo_shared_vector'))
  text,binary=compiled(rows,columns,width,*group,prefetch,*tile)
  return UOp(Ops.PROGRAM,src=(sink,UOp(Ops.LINEAR,src=tuple(sink.toposort())),UOp(Ops.SOURCE,arg=text),UOp(Ops.BINARY,arg=binary)))


def gemm(x,packed_weight,columns,scales,group=(2,2),prefetch=True,tile=(4,4)):
  assert x.ndim==2 and x.dtype==packed_weight.dtype==dtypes.fp8e4m3 and group in ((2,2),(4,2),(2,4),(4,4))
  assert tile in ((4,4),(2,4),(4,2),(2,2))
  assert packed_weight.numel()==((columns+15)//16)*16*x.shape[1]
  out=Tensor.empty((x.shape[0],columns),device=x.device,dtype=dtypes.bfloat16)
  return Tensor.custom_kernel(out,layout.pack32(x),packed_weight,*scales,
    fxn=partial(program,width=x.shape[1],group=group,prefetch=prefetch,tile=tile))[0]
