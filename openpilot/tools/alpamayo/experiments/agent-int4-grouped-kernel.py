"""Bounded CU-mode K128/K256 A4U/W4S prototype. No model hook or GPU execution.

Packed matrices retain the frozen K32 layout. Scales are FP32 [groups,rows].
The kernel keeps one group's INT32 tile and a persistent FP32 tile.
Each group computes two multiplies and a separate increasing-group FP32 add.
"""
from functools import lru_cache,partial
import importlib.util
from pathlib import Path
import numpy as np


@lru_cache(None)
def local(name):
  spec=importlib.util.spec_from_file_location(name.replace('-','_'),Path(__file__).with_name(name+'.py'))
  module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def validate(rows,columns,width,group):
  if group not in (128,256) or min(rows,columns,width)<=0 or width%group:raise ValueError('Require K128/K256 groups and divisible positive K')


def quantize_weights_cpu(master,group):
  """Already-widened original BF16 master -> immutable packed weights/scales."""
  master=np.asarray(master)
  if master.dtype!=np.float32 or master.ndim!=2 or not np.isfinite(master).all():raise ValueError('Expected finite original BF16 master widened to FP32')
  if np.any(master.view(np.uint32)&0xffff):raise ValueError('Weights must contain exact original BF16 values')
  validate(1,len(master),master.shape[1],group)
  q,s=local('agent-int4-grouped-probe').quantize(master,group,True)
  packed=local('agent-int4-scaled-kernel').pack_weights_cpu(q)
  scales=np.frombuffer(s.tobytes(),dtype=np.float32).reshape(s.shape)
  return packed,scales


@lru_cache(None)
def compiled_gemm(m,n,k,group,raw_group=-1,fp32=False):
  validate(m,n,k,group)
  if not -1<=raw_group<k//group:raise ValueError('Invalid raw group')
  base=local('agent-int4-scaled-kernel')
  text=f'#define M {m}\n#define N {n}\n#define K {k}\n#define G {group}\n#define NG {k//group}\n#define RAW_G {raw_group}\n#define OUT_F32 {int(fp32)}\n'+base.HEADER+r'''
typedef float f8 __attribute__((ext_vector_type(8)));
#if RAW_G>=0
typedef int out_t;
#elif OUT_F32
typedef float out_t;
#else
typedef unsigned short out_t;
#endif
extern "C" __attribute__((global,amdgpu_flat_work_group_size(128,128)))
void agent_int4_grouped(out_t* out,const int2* a,const int2* b,const float* xs,const float* ws) {
  int tid=__ockl_get_local_id(0),block=__ockl_get_group_id(0),lane=tid%32,wave=tid/32,cols=(N+63)/64;
  int row=(block/cols)*64+(wave/2)*32,col=(block%cols)*64+(wave%2)*32;
  f8 total[2][2]={};
  #pragma clang loop unroll(disable)
  for(int g=(RAW_G>=0?RAW_G:0);g<(RAW_G>=0?RAW_G+1:NG);g++) {
    int8 part[2][2]={};
    #pragma clang loop unroll(disable)
    for(int kk=0;kk<G/32;kk++) {
      int kt=g*(G/32)+kk;
      int2 av[2],bv[2];
      #pragma unroll
      for(int i=0;i<2;i++) {
        int ar=row/16+i,br=col/16+i;
        av[i]=ar<(M+15)/16?a[(ar*(K/32)+kt)*32+lane]:(int2){0,0};
        bv[i]=br<(N+15)/16?b[(br*(K/32)+kt)*32+lane]:(int2){0,0};
      }
      #pragma unroll
      for(int i=0;i<2;i++) {
        #pragma unroll
        for(int j=0;j<2;j++)
          part[i][j]=__builtin_amdgcn_wmma_i32_16x16x32_iu4_w32_gfx12(false,av[i],true,bv[j],part[i][j],false);
      }
    }
    #pragma unroll
    for(int j=0;j<2;j++) {
      int c=col+j*16+lane%16;
#if RAW_G<0
      float weight_scale=c<N?ws[g*N+c]:0.0f;
#endif
      #pragma unroll
      for(int i=0;i<2;i++) {
        #pragma unroll
        for(int q=0;q<8;q++) {
          int r=row+i*16+(lane/16)*8+q;
          if(r<M && c<N) {
#if RAW_G>=0
            out[r*N+c]=part[i][j][q];
#else
            float scale=xs[g*M+r]*weight_scale;
            float term=(float)part[i][j][q]*scale;
            total[i][j][q]=total[i][j][q]+term;
#endif
          }
        }
      }
    }
  }
#if RAW_G<0
  #pragma unroll
  for(int i=0;i<2;i++) {
    #pragma unroll
    for(int j=0;j<2;j++) {
      #pragma unroll
      for(int q=0;q<8;q++) {
        int r=row+i*16+(lane/16)*8+q,c=col+j*16+lane%16;
        if(r<M && c<N) {
#if OUT_F32
          out[r*N+c]=total[i][j][q];
#else
          out[r*N+c]=to_bf(total[i][j][q]);
#endif
        }
      }
    }
  }
#endif
}
'''
  return text,base.StrictHIPCompiler.compile_cached(text,wgp=False)


@lru_cache(None)
def compiled_quantize(m,k,group):
  validate(m,1,k,group)
  base=local('agent-int4-scaled-kernel')
  text=f'#define M {m}\n#define MP {((m+15)//16)*16}\n#define K {k}\n#define G {group}\n#define NG {k//group}\n#define LANES {group//8}\n'+base.HEADER+r'''
extern "C" __attribute__((global,amdgpu_flat_work_group_size(128,128)))
void agent_int4_grouped_quantize(unsigned* packed,float* scales,const unsigned short* x) {
  int tid=__ockl_get_local_id(0),sub=tid%LANES,wave_lane=tid%32;
  int job=__ockl_get_group_id(0)*(128/LANES)+tid/LANES;
  int g=job/MP,row=job%MP,k=g*G+sub*8;
  h8 values=(h8){0,0,0,0,0,0,0,0};
  if(row<M)values=*(const h8*)(x+row*K+k);
  float maximum=0.0f;
  #pragma unroll
  for(int i=0;i<8;i++)maximum=__builtin_fmaxf(maximum,from_bf(values[i]));
  #pragma unroll
  for(int shift=LANES/2;shift;shift>>=1) {
    int other_lane=(wave_lane&~(LANES-1))|(sub^shift);
    float other=__builtin_bit_cast(float,__builtin_amdgcn_ds_bpermute(other_lane*4,__builtin_bit_cast(int,maximum)));
    maximum=__builtin_fmaxf(maximum,other);
  }
  float scale=maximum==0.0f?1.0f:maximum/15.0f;
  if(sub==0 && row<M)scales[g*M+row]=scale;
  unsigned bits=0;
  #pragma unroll
  for(int i=0;i<8;i++) {
    float value=__builtin_fminf(15.0f,__builtin_fmaxf(0.0f,from_bf(values[i])/scale));
    bits|=((unsigned)__builtin_roundevenf(value))<<(4*i);
  }
  int lane_out=(k%32/16)*16+row%16,reg=k%16/8;
  packed[((row/16*(K/32)+k/32)*32+lane_out)*2+reg]=bits;
}
'''
  return text,base.StrictHIPCompiler.compile_cached(text,wgp=False)


def gemm_program(out,a,b,xs,ws,rows,columns,width,group,raw_group=-1,fp32=False):
  text,binary=compiled_gemm(rows,columns,width,group,raw_group,fp32)
  return local('agent-int4-scaled-kernel').metadata_program((out,a,b,xs,ws),(0,),((rows+63)//64)*((columns+63)//64),
    'agent_int4_grouped',text,binary)


def quantize_program(packed,scales,x,rows,width,group):
  text,binary=compiled_quantize(rows,width,group)
  blocks=((rows+15)//16*16)*(width//group)//(128//(group//8))
  return local('agent-int4-scaled-kernel').metadata_program((packed,scales,x),(0,1),blocks,
    'agent_int4_grouped_quantize',text,binary)


def quantize_pack(x,group,packed_out=None,scales_out=None):
  from tinygrad import Tensor,dtypes
  assert x.dtype==dtypes.bfloat16 and x.ndim==2
  m,k=x.shape;validate(m,1,k,group)
  x=x.contiguous();shape=((m+15)//16,k//32,32,8)
  if packed_out is None:packed_out=Tensor.empty(*shape,dtype=dtypes.uint8,device=x.device)
  if scales_out is None:scales_out=Tensor.empty(k//group,m,dtype=dtypes.float32,device=x.device)
  assert packed_out.shape==shape and packed_out.dtype==dtypes.uint8 and packed_out.device==x.device
  assert scales_out.shape==(k//group,m) and scales_out.dtype==dtypes.float32 and scales_out.device==x.device
  result=Tensor.custom_kernel(packed_out,scales_out,x,fxn=partial(quantize_program,rows=m,width=k,group=group))
  return result[0],result[1]


def gemm(a,b,xs,ws,rows,columns,group,out=None,raw_group=-1,fp32=False):
  from tinygrad import Tensor,dtypes
  assert a.dtype==b.dtype==dtypes.uint8 and a.ndim==b.ndim==4 and a.shape[1:]==b.shape[1:] and a.shape[2:]==(32,8)
  k=a.shape[1]*32;validate(rows,columns,k,group)
  assert a.shape[0]==(rows+15)//16 and b.shape[0]==(columns+15)//16
  assert xs.dtype==ws.dtype==dtypes.float32 and xs.shape==(k//group,rows) and ws.shape==(k//group,columns)
  assert all(t.device==a.device for t in (b,xs,ws))
  dtype=dtypes.int32 if raw_group>=0 else (dtypes.float32 if fp32 else dtypes.bfloat16)
  if out is None:out=Tensor.empty(rows,columns,dtype=dtype,device=a.device)
  assert out.shape==(rows,columns) and out.dtype==dtype and out.device==a.device
  return Tensor.custom_kernel(out,a,b,xs,ws,fxn=partial(gemm_program,rows=rows,columns=columns,width=k,group=group,raw_group=raw_group,fp32=fp32))[0]


def linear(x,packed_weight,weight_scales,group,out=None):
  packed,scale=quantize_pack(x,group)
  return gemm(packed,packed_weight,scale,weight_scales,x.shape[0],weight_scales.shape[1],group,out=out)
