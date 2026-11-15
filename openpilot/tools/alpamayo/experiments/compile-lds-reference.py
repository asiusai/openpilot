"""Compile the reviewed AMD blog kernel for tinygrad's USB runtime.

Source is read from a pinned, separately downloaded reference directory.
This does not load or execute GPU code.
"""
import argparse
import re
from pathlib import Path
from tinygrad.runtime.support.compiler_amd import HIPCompiler


def source(root, tile, shape=None, fused=False):
  common = (root/'gemm_common.hpp').read_text()
  common = common[common.index('using fp8 ='):].replace('__hip_fp8_storage_t','unsigned char')
  common = re.sub(r'using (\w+)\s*=\s*([\w ]+)\s*\[\[clang::ext_vector_type\((\d+)\)\]\];',
                  r'typedef \2 \1 __attribute__((ext_vector_type(\3)));', common)
  kernel = (root/'kernel.hip').read_text()
  kernel = kernel[kernel.index('namespace\n{'):kernel.index('/// One instantiation')]
  kernel = kernel.replace('__global__ __launch_bounds__(wave_grid_threads(WavesM,\n                                               WavesN))', '__device__ __forceinline__')
  kernel = kernel.replace('threadIdx.x','__ockl_get_local_id(0)').replace('blockIdx.x','__ockl_get_group_id(0)')
  m,n,k,wm,wn = tile
  prefix = '''
#define __device__ __attribute__((device))
#define __forceinline__ __attribute__((always_inline)) inline
#define __shared__ __attribute__((shared, aligned(16)))
#define __syncthreads() { __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup"); __builtin_amdgcn_s_barrier(); __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup"); }
using size_t = unsigned long;
extern "C" __device__ size_t __ockl_get_local_id(unsigned int);
extern "C" __device__ size_t __ockl_get_group_id(unsigned int);
'''
  wrapper = f'''
extern "C" __attribute__((global, amdgpu_flat_work_group_size(1, {wm*wn*32})))
void alpamayo_lds(const fp8 *A,const fp8 *B,float *C,int M,int N,int K) {{
  gemm_fp8<WmmaFp8Policy,{m},{n},{k},{wm},{wn},true>(A,B,C,M,N,K);
}}
'''
  if shape:
    rows,cols,width = shape
    assert width % k == 0, 'LDS K tile must divide the reduction dimension'
    wrapper = f'''
extern "C" __attribute__((global, amdgpu_flat_work_group_size(1, {wm*wn*32})))
void alpamayo_lds(float *C,const fp8 *A,const fp8 *B) {{
  gemm_fp8<WmmaFp8Policy,{m},{n},{k},{wm},{wn},true>(A,B,C,{rows},{cols},{width});
}}
'''
  if fused:
    assert shape is not None
    # Keep the reference accumulator; write scaled BF16 directly in row-major order.
    # This removes the large intermediate FP32 matrix and its transpose kernel.
    prefix += '''
#pragma clang fp contract(off)
struct Output { unsigned short *data; const float *xs; const float *ws; const unsigned short *bias; };
__device__ __forceinline__ unsigned short bf16(float x) {
  unsigned int bits=__builtin_bit_cast(unsigned int,x);
  return (bits + 0x7fff + ((bits>>16)&1)) >> 16;
}
'''
    common=common.replace('float* C,', 'Output C,')
    common=common.replace('C[static_cast<size_t>(col) * M + row] = acc[i];',
      'C.data[static_cast<size_t>(row) * N + col] = bf16(acc[i] * (C.xs[0] * C.ws[0]) + __builtin_bit_cast(float, static_cast<unsigned int>(C.bias[col]) << 16));')
    kernel=kernel.replace('float* __restrict__ C,','Output C,')
    wrapper=f'''
extern "C" __attribute__((global, amdgpu_flat_work_group_size(1, {wm*wn*32})))
void alpamayo_lds(unsigned short *C,const fp8 *A,const fp8 *B,const float *xs,const float *ws,const unsigned short *bias) {{
  gemm_fp8<WmmaFp8Policy,{m},{n},{k},{wm},{wn},true>(A,B,Output{{C,xs,ws,bias}},{rows},{cols},{width});
}}
'''
  assert '__global__' not in kernel
  return prefix+common+kernel+wrapper


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--source',type=Path,default=Path('/tmp/opencode/rdna-fp8-reference'))
  parser.add_argument('--tile',nargs=5,type=int,default=[128,96,64,2,2])
  parser.add_argument('--output',type=Path,default=Path('/tmp/opencode/alpamayo-lds-reference.hsaco'))
  args=parser.parse_args()
  text=source(args.source,args.tile)
  args.output.with_suffix('.cpp').write_text(text)
  args.output.write_bytes(HIPCompiler('gfx1200').compile_cached(text))
  print(args.output)
