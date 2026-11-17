"""Native RDNA4 signed INT4 matrix experiment, separate from model quantization.

Fragment mapping reference: JohnTDI-cpu/rdna4-wmma-guide (CC BY 4.0),
https://github.com/JohnTDI-cpu/rdna4-wmma-guide/tree/998f4f2f5710516b2e476597c0a1b9c289cb69ef
This kernel uses independent register tiling and coalesced 16 x 16 packed inputs.
"""
from functools import lru_cache,partial
from tinygrad import Tensor,dtypes
from tinygrad.uop.ops import UOp,Ops,KernelInfo
from tinygrad.runtime.support.compiler_amd import HIPCompiler


@lru_cache(None)
def compiled(rows,columns,width,parts):
  text=f'#define M {rows}\n#define N {columns}\n#define K {width}\n#define P {parts}\n'+r'''
#pragma clang fp contract(off)
typedef int int8 __attribute__((ext_vector_type(8)));
extern "C" __attribute__((device)) unsigned long __ockl_get_local_id(unsigned);
extern "C" __attribute__((device)) unsigned long __ockl_get_group_id(unsigned);
extern "C" __attribute__((global,amdgpu_flat_work_group_size(128,128)))
void alpamayo_int4(unsigned short* out,const int* a,const int* b) {
  int thread=__ockl_get_local_id(0),block=__ockl_get_group_id(0);
  int lane=thread%32,wave=thread/32,cols=(N+32*P-1)/(32*P);
  int row=(block/cols)*(32*P)+(wave/2)*(16*P);
  int col=(block%cols)*(32*P)+(wave%2)*(16*P);
  int8 acc[P][P]={};
  for(int k=0;k<K;k+=16) {
    int av[P],bv[P];
    #pragma unroll
    for(int i=0;i<P;i++) {
      int ar=row/16+i,br=col/16+i;
      av[i]=ar<(M+15)/16?a[(ar*(K/16)+k/16)*32+lane]:0;
      bv[i]=br<(N+15)/16?b[(br*(K/16)+k/16)*32+lane]:0;
    }
    #pragma unroll
    for(int i=0;i<P;i++) {
      #pragma unroll
      for(int j=0;j<P;j++)
        acc[i][j]=__builtin_amdgcn_wmma_i32_16x16x16_iu4_w32_gfx12(true,av[i],true,bv[j],acc[i][j],false);
    }
  }
  #pragma unroll
  for(int i=0;i<P;i++) {
    #pragma unroll
    for(int j=0;j<P;j++) {
      #pragma unroll
      for(int q=0;q<8;q++) {
        int r=row+i*16+(lane/16)*8+q,c=col+j*16+lane%16;
        if(r<M && c<N) {
          unsigned bits=__builtin_bit_cast(unsigned,float(acc[i][j][q]));
          out[r*N+c]=(bits+0x7fff+((bits>>16)&1))>>16;
        }
      }
    }
  }
}
'''
  return text,HIPCompiler('gfx1200').compile_cached(text)


def program(out,a,b,shape,parts):
  m,n,k=shape
  blocks=((m+32*parts-1)//(32*parts))*((n+32*parts-1)//(32*parts))
  sink=UOp.sink(out.flatten()[0].store((a.flatten()[0].load()+b.flatten()[0].load()).cast(out.dtype)),
    UOp.special(blocks,'gidx0'),UOp.special(128,'lidx0'),arg=KernelInfo(name='alpamayo_int4'))
  text,binary=compiled(m,n,k,parts)
  return UOp(Ops.PROGRAM,src=(sink,UOp(Ops.LINEAR,src=tuple(sink.toposort())),UOp(Ops.SOURCE,arg=text),UOp(Ops.BINARY,arg=binary)))


def gemm(a,b,rows,columns,parts=2):
  assert a.dtype==b.dtype==dtypes.int32 and a.ndim==b.ndim==3 and a.shape[1:]==b.shape[1:]
  assert a.shape[2]==32 and parts in (1,2,4)
  assert a.shape[0]==(rows+15)//16 and b.shape[0]==(columns+15)//16
  out=Tensor.empty(rows,columns,dtype=dtypes.bfloat16,device=a.device)
  return Tensor.custom_kernel(out,a,b,fxn=partial(program,shape=(rows,columns,a.shape[1]*16),parts=parts))[0]
