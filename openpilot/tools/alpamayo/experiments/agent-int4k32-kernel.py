"""Native gfx1200 signed INT4 K32 experiment. No model quantization or scales.

AMD RDNA4 ISA section 7.12.2, printed page 92, wave32:
  A[m,k]: lane={k[4],m[3:0]}, VGPR=k[3], nibble=k[2:0].
  B[k,n]: lane={k[4],n[3:0]}, VGPR=k[3], nibble=k[2:0].
  D[m,n]: lane={m[3],n[3:0]}, VGPR=m[2:0].
Both external inputs use logical [output row, K] order: output = a @ b.T.
Storage is [row tile, K32 tile, lane, register], with two adjacent int32 words.
This differs from concatenating the K16 instruction's register fragments.
"""
from functools import lru_cache, partial
import numpy as np


def pack_cpu(x):
  """Pack actual signed integers, with zero padding only on the row axis."""
  x = np.asarray(x)
  if x.ndim != 2 or x.shape[1] % 32 or x.shape[1] == 0:
    raise ValueError('Expected nonempty [rows, K], K divisible by 32')
  if not np.issubdtype(x.dtype, np.integer) or np.any(x < -8) or np.any(x > 7):
    raise ValueError('Expected signed INT4 values in [-8, 7]')
  k = x.shape[1]
  x = np.pad(x, ((0, (-x.shape[0]) % 16), (0, 0)))
  # Each lane holds K[4]*16 + register*8 + nibble for one logical row.
  tiles = x.reshape(-1, 16, k//32, 2, 2, 8).transpose(0, 2, 3, 1, 4, 5)
  tiles = tiles.reshape(-1, k//32, 32, 2, 8).astype(np.uint32) & 15
  return np.bitwise_or.reduce(tiles << np.arange(0, 32, 4, dtype=np.uint32), axis=-1).view(np.int32)


@lru_cache(None)
def compiled(rows, columns, width, output='bf16'):
  if min(rows, columns, width) <= 0 or width % 32 or output not in ('bf16', 'i32'):
    raise ValueError((rows, columns, width, output))
  from tinygrad.runtime.support.compiler_amd import HIPCompiler
  text = f'#define M {rows}\n#define N {columns}\n#define K {width}\n#define OUT_I32 {int(output == "i32")}\n' + r'''
#pragma clang fp contract(off)
typedef int int2 __attribute__((ext_vector_type(2)));
typedef int int8 __attribute__((ext_vector_type(8)));
extern "C" __attribute__((device)) unsigned long __ockl_get_local_id(unsigned);
extern "C" __attribute__((device)) unsigned long __ockl_get_group_id(unsigned);
#if OUT_I32
typedef int out_t;
#else
typedef unsigned short out_t;
#endif
extern "C" __attribute__((global,amdgpu_flat_work_group_size(128,128)))
void agent_int4k32(out_t* out, const int2* a, const int2* b) {
  int thread=__ockl_get_local_id(0), block=__ockl_get_group_id(0);
  int lane=thread%32, wave=thread/32, cols=(N+63)/64;
  int row=(block/cols)*64+(wave/2)*32;
  int col=(block%cols)*64+(wave%2)*32;
  int8 acc[2][2]={};
  for(int k=0;k<K/32;k++) {
    int2 av[2],bv[2];
    #pragma unroll
    for(int i=0;i<2;i++) {
      int ar=row/16+i, br=col/16+i;
      av[i]=ar<(M+15)/16?a[(ar*(K/32)+k)*32+lane]:(int2){0,0};
      bv[i]=br<(N+15)/16?b[(br*(K/32)+k)*32+lane]:(int2){0,0};
    }
    #pragma unroll
    for(int i=0;i<2;i++) {
      #pragma unroll
      for(int j=0;j<2;j++) {
        acc[i][j]=__builtin_amdgcn_wmma_i32_16x16x32_iu4_w32_gfx12(
          true,av[i],true,bv[j],acc[i][j],false);
      }
    }
  }
  #pragma unroll
  for(int i=0;i<2;i++) {
    #pragma unroll
    for(int j=0;j<2;j++) {
      #pragma unroll
      for(int q=0;q<8;q++) {
        int r=row+i*16+(lane/16)*8+q, c=col+j*16+lane%16;
        if(r<M && c<N) {
#if OUT_I32
          out[r*N+c]=acc[i][j][q];
#else
          unsigned bits=__builtin_bit_cast(unsigned,float(acc[i][j][q]));
          out[r*N+c]=(bits+0x7fff+((bits>>16)&1))>>16;
#endif
        }
      }
    }
  }
}
'''
  return text, HIPCompiler('gfx1200').compile_cached(text)


def program(out, a, b, shape, output):
  from tinygrad.uop.ops import UOp, Ops, KernelInfo
  m, n, k = shape
  blocks = ((m+63)//64)*((n+63)//64)
  sink = UOp.sink(out.flatten()[0].store((a.flatten()[0].load()+b.flatten()[0].load()).cast(out.dtype)),
                  UOp.special(blocks, 'gidx0'), UOp.special(128, 'lidx0'), arg=KernelInfo(name='agent_int4k32'))
  text, binary = compiled(m, n, k, output)
  return UOp(Ops.PROGRAM, src=(sink, UOp(Ops.LINEAR, src=tuple(sink.toposort())),
                              UOp(Ops.SOURCE, arg=text), UOp(Ops.BINARY, arg=binary)))


def gemm(a, b, rows, columns, output='bf16'):
  from tinygrad import Tensor, dtypes
  assert a.dtype == b.dtype == dtypes.int32 and a.ndim == b.ndim == 4
  assert a.shape[1:] == b.shape[1:] and a.shape[2:] == (32, 2)
  assert a.shape[0] == (rows+15)//16 and b.shape[0] == (columns+15)//16
  assert output in ('bf16', 'i32')
  out = Tensor.empty(rows, columns, dtype=dtypes.int32 if output == 'i32' else dtypes.bfloat16, device=a.device)
  return Tensor.custom_kernel(out, a, b, fxn=partial(program,
    shape=(rows, columns, a.shape[1]*32), output=output))[0]
