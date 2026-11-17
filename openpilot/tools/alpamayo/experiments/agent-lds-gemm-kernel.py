"""Packed-input adaptation of AMD's published single-buffer FP8 GEMM.

Reference: ROCm/rocm-blogs d908d68d5741b0f449d7c31efa733fe8d4865b51.
The downloaded reference stays outside the checkout. No license was found in
the source directory or repository root; see agent-lds-gemm.md before publishing.
Only operand addressing and the exact scale/BF16 epilogue change arithmetic dataflow.
"""
from functools import lru_cache
import ctypes
import hashlib
from pathlib import Path
import re
from types import FunctionType

from tinygrad import dtypes
from tinygrad.device import Compiler
from tinygrad.runtime.support import compiler_amd
from tinygrad.uop.ops import KernelInfo, Ops, UOp

REFERENCE = Path('/tmp/opencode/agent-lds-gemm-amd-reference')
HASHES = {'gemm_common.hpp': 'fed624bc60abfcdcb90103c27aaf7b6514993dcd107a91c68f6f186a7a66db22',
  'kernel.hip': '3361c1f48656c9881e002d56542a143ddb8d5985a463b45efbaa97f16f0358a0'}
CONFIGS = {'amd3': (256,160,128,4,2), 'amd4': (192,192,128,2,4)}


def compiler_assembly(text):
  """Observe COMGR's assembly action in a private namespace; no device access."""
  original,comgr = compiler_amd.compile_hip,compiler_amd.comgr
  captured=[]
  class Proxy:
    def __getattr__(self,name): return getattr(comgr,name)
    def amd_comgr_do_action(self,kind,action,data,result):
      if kind==comgr.AMD_COMGR_ACTION_CODEGEN_BC_TO_RELOCATABLE:
        compiler_amd.check(comgr.amd_comgr_create_data_set(ctypes.byref(extra:=comgr.amd_comgr_data_set_t())))
        try:
          compiler_amd.check(comgr.amd_comgr_do_action(comgr.AMD_COMGR_ACTION_CODEGEN_BC_TO_ASSEMBLY,action,data,extra))
          captured.append(compiler_amd._get_comgr_data(extra,comgr.AMD_COMGR_DATA_KIND_SOURCE).decode())
        finally: compiler_amd.check(comgr.amd_comgr_destroy_data_set(extra))
      return comgr.amd_comgr_do_action(kind,action,data,result)
  def options(action,value): return compiler_amd.set_options(action,value.replace(b'-mcumode',b'-mno-cumode'))
  compile_source=FunctionType(original.__code__,dict(original.__globals__,set_options=options,comgr=Proxy()),argdefs=original.__defaults__)
  binary=compile_source(text,'gfx1200')
  assert len(captured)==1
  return binary,captured[0]


def source(rows, columns, width, mode='amd3', row_scale=False, bias=False):
  bm,bn,bk,wm,wn = CONFIGS[mode]
  assert rows > 0 and columns > 0 and columns % 16 == 0 and width > 0 and width % bk == 0
  # CopyAddress uses signed byte offsets, including invalid trailing block rows.
  assert max((rows+bm-1)//bm*bm,(columns+bn-1)//bn*bn)*width < 2**31
  texts = {}
  for name, expected in HASHES.items():
    data = (REFERENCE/name).read_bytes()
    assert hashlib.sha256(data).hexdigest() == expected, f'Changed AMD reference: {name}'
    texts[name] = data.decode()
  common = texts['gemm_common.hpp']
  common = common[common.index('using fp8 ='):].replace('__hip_fp8_storage_t','unsigned char')
  common = re.sub(r'using (\w+)\s*=\s*([\w ]+)\s*\[\[clang::ext_vector_type\((\d+)\)\]\];',
    r'typedef \2 \1 __attribute__((ext_vector_type(\3)));', common)
  # Packed K16 words already stagger lane banks. There is no TN row-stride padding.
  common = common.replace('constexpr int lds_pad = 16;', 'constexpr int lds_pad = 0;')
  begin, end = common.index('template<int BlockK, int Threads, int Stride>\nstruct CopyAddress'), common.index('/// Config word')
  common = common[:begin] + r'''
template<int BlockK, int Threads, int Stride> struct CopyAddress {
  static constexpr int rows_per_chunk = Threads / BlockK * 16;
  static_assert(Threads % BlockK == 0, "whole packed row tiles per chunk");
  int row,word;
  __device__ __forceinline__ explicit CopyAddress(int tid): row(tid/BlockK*16),word(tid%BlockK*4) {}
  __device__ __forceinline__ int lds(int c) const { return (row+c*rows_per_chunk)*BlockK+word*4; }
  __device__ __forceinline__ int global(int c,int origin,int ko,int K) const {
    return (origin+row+c*rows_per_chunk)*K+ko*16+word*4;
  }
};
template<typename Policy,int BlockK,int Threads,int Stride> struct OperandReader {
  using Address=CopyAddress<BlockK,Threads,Stride>;
  int lane_offset;
  const fp8* tile;
  __amdgpu_buffer_rsrc_t resource;
  int offset,chunk_bytes;
  __device__ __forceinline__ OperandReader(int wave_offset,int lane,__amdgpu_buffer_rsrc_t r,
      const Address& a,int origin,int K): lane_offset(wave_offset*BlockK+lane*4),tile(nullptr),resource(r),
      offset(a.global(0,origin,BlockK,K)),chunk_bytes(Address::rows_per_chunk*K) {}
  __device__ __forceinline__ void read_from(const fp8* lds) { tile=lds+lane_offset; }
  __device__ __forceinline__ void advance() { offset+=BlockK*16; }
  template<int Row,int Slice,typename Fragment> __device__ __forceinline__ void read_fragment(Fragment& f) const {
    const unsigned* at=reinterpret_cast<const unsigned*>(tile+Row*16*BlockK+Slice*256);
    f=(Fragment){(int)at[0],(int)at[32]};
  }
  template<int Index> __device__ __forceinline__ void prefetch(fp8x16* into) const {
    load_chunk(into[Index],resource,offset+Index*chunk_bytes);
  }
};
''' + common[end:]
  common = common.replace('float* C,', 'Output C,')
  original = 'C[static_cast<size_t>(col) * M + row] = acc[i];'
  epilogue = '''float scale=ROW_SCALE?C.xs[row]*C.weight_scale:C.tensor_scale;
#if WITH_BIAS
                float value=__builtin_fmaf(acc[i],scale,__builtin_bit_cast(float,((unsigned)C.bias[col])<<16));
#else
                float value=acc[i]*scale;
#endif
                C.data[row*N+col]=bf(value);'''
  assert common.count(original) == 1
  common = common.replace(original, epilogue)
  kernel = texts['kernel.hip']
  kernel = kernel[kernel.index('namespace\n{'):kernel.index('/// One instantiation')]
  kernel = re.sub(r'__global__ __launch_bounds__\(wave_grid_threads\(WavesM,\s*WavesN\)\)', '__device__ __forceinline__', kernel)
  kernel = kernel.replace('threadIdx.x','__ockl_get_local_id(0)').replace('blockIdx.x','__ockl_get_group_id(0)')
  kernel = kernel.replace('float* __restrict__ C,','Output C,')
  kernel = kernel.replace('static_cast<unsigned>(M) * K * sizeof(fp8)', 'static_cast<unsigned>((M+15)/16*16) * K * sizeof(fp8)')
  # The pinned COMGR path needs explicit unrolling of constant-index array loops.
  # Otherwise the epilogue keeps accumulators in private scratch despite zero spills.
  common = common.replace('__device__ static', '__device__ __forceinline__ static')
  kernel = kernel.replace('__device__ static', '__device__ __forceinline__ static')
  common = re.sub(r'\n( +)(for\(int (?:chunk|m|n|i) =)', r'\n\1#pragma unroll\n\1\2', common)
  kernel = re.sub(r'\n( +)(for\(int (?:chunk|m|n|i) =)', r'\n\1#pragma unroll\n\1\2', kernel)
  assert '__global__' not in kernel
  prefix = f'#define ROW_SCALE {int(row_scale)}\n#define WITH_BIAS {int(bias)}\n' + r'''
// Adapted from ROCm/rocm-blogs commit d908d68d5741b0f449d7c31efa733fe8d4865b51.
#if !defined(__gfx1200__)
#error This exact packed FP8 candidate requires gfx1200
#endif
#define __device__ __attribute__((device))
#define __forceinline__ __attribute__((always_inline)) inline
#define __shared__ __attribute__((shared,aligned(16)))
#define __syncthreads() { __builtin_amdgcn_fence(__ATOMIC_RELEASE,"workgroup"); __builtin_amdgcn_s_barrier(); __builtin_amdgcn_fence(__ATOMIC_ACQUIRE,"workgroup"); }
#pragma clang fp contract(off)
using size_t=unsigned long;
extern "C" __device__ size_t __ockl_get_local_id(unsigned);
extern "C" __device__ size_t __ockl_get_group_id(unsigned);
struct Output { unsigned short* data; const float* xs; float weight_scale; float tensor_scale; const unsigned short* bias; };
__device__ __forceinline__ unsigned short bf(float value) {
  unsigned b=__builtin_bit_cast(unsigned,value),regular=b+((b>>16)&1)+0x7fffu;
  unsigned special=b|(((b&0xffffu)!=0)<<16),mask=0u-(((-b & 0x7f800000u)!=0));
  asm("" : "+v"(mask));
  return (unsigned short)(((regular&mask)|(special&~mask))>>16);
}
'''
  wrapper = f'''
extern "C" __attribute__((global,amdgpu_flat_work_group_size({wm*wn*32},{wm*wn*32})))
void agent_lds_gemm(unsigned short* __restrict out,const fp8* __restrict x,const fp8* __restrict weight,
  const float* __restrict xs,const float* __restrict ws
  {',const unsigned short* __restrict bias' if bias else ''}) {{
  float weight_scale=ws[0],tensor_scale=xs[0]*weight_scale;
  gemm_fp8<WmmaFp8Policy,{bm},{bn},{bk},{wm},{wn},true>(x,weight,Output{{out,xs,weight_scale,tensor_scale,{'bias' if bias else 'nullptr'}}},{rows},{columns},{width});
}}
'''
  return prefix+common+kernel+wrapper


@lru_cache(None)
def compiled(rows, columns, width, mode='amd3', row_scale=False, bias=False):
  original = compiler_amd.compile_hip
  def options(action, value):
    return compiler_amd.set_options(action, value.replace(b'-mcumode', b'-mno-cumode'))
  compile_source = FunctionType(original.__code__, dict(original.__globals__, set_options=options), argdefs=original.__defaults__)
  class WGPCompiler(Compiler):
    def compile(self, src): return compile_source(src, 'gfx1200')
  text = source(rows, columns, width, mode, row_scale, bias)
  return text, WGPCompiler('agent_lds_gemm_wgp_gfx1200_v2').compile_cached(text)


def program(out, x, weight, xs, ws, *bias, mode='amd3', row_scale=False):
  rows, columns = out.shape
  width = weight.shape[-1]
  assert x.shape == ((rows+15)//16*16, width) and weight.shape == (columns, width)
  assert x.dtype == weight.dtype == dtypes.fp8e4m3 and out.dtype == dtypes.bfloat16
  assert xs.dtype == ws.dtype == dtypes.float32
  assert not bias or (len(bias)==1 and bias[0].dtype==dtypes.bfloat16 and bias[0].shape==(columns,))
  assert xs.numel() == (rows if row_scale else 1) and ws.numel() == 1
  value = x.flatten()[0].load().float()+weight.flatten()[0].load().float()+xs.flatten()[0].load()+ws.flatten()[0].load()
  for b in bias: value = value+b.flatten()[0].load().float()
  bm,bn,bk,wm,wn = CONFIGS[mode]
  sink = UOp.sink(out.flatten()[0].store(value.cast(out.dtype)), UOp.special(((rows+bm-1)//bm)*((columns+bn-1)//bn),'gidx0'),
    UOp.special(wm*wn*32,'lidx0'), arg=KernelInfo(name='agent_lds_gemm'))
  text, binary = compiled(rows, columns, width, mode, row_scale, bool(bias))
  return UOp(Ops.PROGRAM, src=(sink,UOp(Ops.LINEAR,src=tuple(sink.toposort())),UOp(Ops.SOURCE,arg=text),UOp(Ops.BINARY,arg=binary)))
