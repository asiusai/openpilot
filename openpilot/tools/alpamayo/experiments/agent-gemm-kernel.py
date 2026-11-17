"""Gfx1200 FP8 GEMM with transposed WMMA fragments and contiguous BF16 stores.

The logical inputs retain amd_gemm.pack's K16 layout. Transposing both WMMA
operands assigns each lane eight adjacent output columns. This permits one
128-bit output store per fragment instead of eight 16-bit stores.

GPU correctness and performance require the separate check harness.
"""
from functools import lru_cache, partial
from tinygrad import Tensor, dtypes
from tinygrad.runtime.support.compiler_amd import HIPCompiler
from tinygrad.uop.ops import KernelInfo, Ops, UOp


MODES = ('direct', 'transpose', 'pipeline')


def source(rows, columns, width, mode='transpose', row_scale=False, bias=False, images=1, wgp=True):
  assert mode in MODES and rows > 0 and columns > 0 and width % 16 == columns % 16 == 0
  defines = dict(M=rows, N=columns, K=width, IMAGES=images, TRANSPOSE=int(mode != 'direct'),
                 PIPELINE=int(mode == 'pipeline'), ROW_SCALE=int(row_scale), BIAS=int(bias),
                 NTILES=(columns+63)//64, MTILES=(rows+63)//64, PADM=(rows+15)//16*16)
  return ''.join(f'#define {key} {value}\n' for key, value in defines.items()) + r'''
#pragma clang fp contract(off)
#define DEV __attribute__((device,always_inline)) inline
typedef unsigned u4 __attribute__((ext_vector_type(4)));
typedef unsigned short h8 __attribute__((ext_vector_type(8)));
typedef int i2 __attribute__((ext_vector_type(2)));
typedef float f8 __attribute__((ext_vector_type(8)));
extern "C" __attribute__((device)) unsigned long __ockl_get_local_id(unsigned);
extern "C" __attribute__((device)) unsigned long __ockl_get_group_id(unsigned);
// Match tinygrad's cast_float_to_bf16, including its special-value behavior.
DEV unsigned short bf(float value) {
  unsigned b=__builtin_bit_cast(unsigned,value);
  unsigned regular=b+((b>>16)&1)+0x7fffu;
  unsigned special=b|(((b&0xffffu)!=0)<<16);
  unsigned mask=0u-(((-b & 0x7f800000u)!=0));
  // Clang otherwise converts this exact select into divergent branches for
  // each element. The empty constraint hides only the mask's 0/-1 range.
  // It emits no instructions. The backend retains dependency/hazard tracking.
  asm("" : "+v"(mask));
  return (unsigned short)(((regular&mask)|(special&~mask))>>16);
}
DEV i2 fragment(const unsigned *ptr, unsigned offset) {
  return (i2){(int)ptr[offset],(int)ptr[offset+32]};
}
extern "C" __attribute__((global,amdgpu_flat_work_group_size(32,32)))
void agent_gemm(unsigned short *__restrict out,const unsigned *__restrict x,
                const unsigned *__restrict weight,const float *__restrict xs,
                const float *__restrict ws
#if BIAS
                ,const unsigned short *__restrict bias
#endif
                ) {
  unsigned lane=__ockl_get_local_id(0),block=__ockl_get_group_id(0);
  unsigned image=block/(MTILES*NTILES),within=block%(MTILES*NTILES);
  unsigned mt=within/NTILES,nt=within%NTILES;
  unsigned ao[4],bo[4];
  // Invalid whole fragments affect only discarded outputs. Clamp their addresses
  // once, retaining the zero padding within the final valid K16 fragment.
  #pragma unroll
  for(unsigned i=0;i<4;i++) {
    unsigned ar=mt*4+i,br=nt*4+i;
    ar=ar<(M+15)/16?ar:(M+15)/16-1;
    br=br<N/16?br:N/16-1;
    ao[i]=image*(PADM*K/4)+ar*(K/16)*64+lane;
    bo[i]=br*(K/16)*64+lane;
  }
  f8 accum[4][4];
  #pragma unroll
  for(unsigned i=0;i<4;i++) {
    #pragma unroll
    for(unsigned j=0;j<4;j++) accum[i][j]=(f8){0};
  }
  i2 a[4],b[4];
  #pragma unroll
  for(unsigned i=0;i<4;i++) { a[i]=fragment(x,ao[i]); b[i]=fragment(weight,bo[i]); }
  #pragma clang loop unroll(disable)
  for(unsigned k=0;k<K/16;k++) {
    i2 na[4],nb[4];
#if PIPELINE
    // Keep the next K16 loads above current arithmetic without inserting a wait.
    // The backend still tracks their dependencies and inserts required waits.
    #pragma unroll
    for(unsigned i=0;i<4;i++) { na[i]=a[i]; nb[i]=b[i]; }
    if(k+1<K/16) {
      #pragma unroll
      for(unsigned i=0;i<4;i++) { na[i]=fragment(x,ao[i]+(k+1)*64); nb[i]=fragment(weight,bo[i]+(k+1)*64); }
    }
    __builtin_amdgcn_sched_barrier(0);
#endif
    #pragma unroll
    for(unsigned i=0;i<4;i++) {
      #pragma unroll
      for(unsigned j=0;j<4;j++) {
#if TRANSPOSE
        accum[i][j]=__builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(b[j],a[i],accum[i][j]);
#else
        accum[i][j]=__builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(a[i],b[j],accum[i][j]);
#endif
      }
    }
#if PIPELINE
    __builtin_amdgcn_sched_barrier(0);
    #pragma unroll
    for(unsigned i=0;i<4;i++) { a[i]=na[i]; b[i]=nb[i]; }
#else
    if(k+1<K/16) {
      #pragma unroll
      for(unsigned i=0;i<4;i++) { a[i]=fragment(x,ao[i]+(k+1)*64); b[i]=fragment(weight,bo[i]+(k+1)*64); }
    }
#endif
  }
  float weight_scale=ws[0];
  float tensor_scale=xs[image]*weight_scale;
  #pragma unroll
  for(unsigned j=0;j<4;j++) {
#if TRANSPOSE && BIAS
    unsigned start_col=nt*64+j*16+(lane/16)*8;
    h8 bias_bits=*(const h8 *)(bias+(start_col<N?start_col:N-8));
    f8 bias_values;
    #pragma unroll
    for(unsigned v=0;v<8;v++) bias_values[v]=__builtin_bit_cast(float,((unsigned)bias_bits[v])<<16);
#endif
    #pragma unroll
    for(unsigned i=0;i<4;i++) {
#if TRANSPOSE
      unsigned row=mt*64+i*16+lane%16,col=nt*64+j*16+(lane/16)*8;
      float scale=ROW_SCALE?xs[image*M+(row<M?row:M-1)]*weight_scale:tensor_scale;
      h8 converted;
      #pragma unroll
      for(unsigned v=0;v<8;v++) {
#if BIAS
        // The existing LLVM reference contracts scaling plus bias into FMA.
        // Spell that rounding explicitly despite the surrounding contract(off).
        float value=__builtin_fmaf(accum[i][j][v],scale,bias_values[v]);
#else
        float value=accum[i][j][v]*scale;
#endif
        converted[v]=bf(value);
      }
      if(row<M && col<N) *(u4 *)(out+(image*M+row)*N+col)=__builtin_bit_cast(u4,converted);
#else
      unsigned col=nt*64+j*16+lane%16;
      #pragma unroll
      for(unsigned v=0;v<8;v++) {
        unsigned row=mt*64+i*16+(lane/16)*8+v;
        float scale=ROW_SCALE?xs[image*M+(row<M?row:M-1)]*weight_scale:tensor_scale;
#if BIAS
        unsigned bits=((unsigned)bias[col<N?col:N-1])<<16;
        float value=__builtin_fmaf(accum[i][j][v],scale,__builtin_bit_cast(float,bits));
#else
        float value=accum[i][j][v]*scale;
#endif
        if(row<M && col<N) out[(image*M+row)*N+col]=bf(value);
      }
#endif
    }
  }
}
'''


@lru_cache(None)
def compiled(rows, columns, width, mode='transpose', row_scale=False, bias=False, images=1, wgp=True):
  text = source(rows, columns, width, mode, row_scale, bias, images, wgp)
  compiler = HIPCompiler('gfx1200')
  if wgp:
    # Give this compiler a separate cache key. Change only the existing CPU
    # compiler function's options; no runtime, device, or core-file mutation.
    from types import FunctionType
    from tinygrad.device import Compiler
    from tinygrad.runtime.support import compiler_amd
    original = compiler_amd.compile_hip
    def options(action, value):
      return compiler_amd.set_options(action, value.replace(b'-mcumode', b'-mno-cumode'))
    compile_source = FunctionType(original.__code__, dict(original.__globals__, set_options=options), argdefs=original.__defaults__)
    class WGPCompiler(Compiler):
      def compile(self, src): return compile_source(src, 'gfx1200')
    compiler = WGPCompiler('agent_gemm_wgp_gfx1200')
  return text, compiler.compile_cached(text)


def program(out, x, weight, xs, ws, *bias, rows=None, images=1, mode='transpose', row_scale=False, wgp=True, **unused):
  rows = out.shape[-2] if rows is None else rows
  columns, width = out.shape[-1], weight.shape[-1]
  dependencies = x.flatten()[0].load().float()+weight.flatten()[0].load().float()+xs.flatten()[0].load()+ws.flatten()[0].load()
  for value in bias: dependencies = dependencies+value.flatten()[0].load().float()
  groups = images*((rows+63)//64)*((columns+63)//64)
  sink = UOp.sink(out.flatten()[0].store(dependencies.cast(dtypes.bfloat16)), UOp.special(groups,'gidx0'),
                  UOp.special(32,'lidx0'), arg=KernelInfo(name='agent_gemm'))
  text, binary = compiled(rows, columns, width, mode, row_scale, bool(bias), images, wgp)
  return UOp(Ops.PROGRAM, src=(sink,UOp(Ops.LINEAR,src=tuple(sink.toposort())),UOp(Ops.SOURCE,arg=text),UOp(Ops.BINARY,arg=binary)))


def gemm_packed(x, weight, scales, rows, columns, bias=None, mode='transpose', row_scale=False, images=1, wgp=True):
  assert x.dtype == weight.dtype == dtypes.fp8e4m3
  assert x.shape[-1] == weight.shape[-1] and x.shape[-1] % 16 == columns % 16 == 0
  assert x.numel() == images*((rows+15)//16*16)*x.shape[-1]
  assert scales[0].numel() == (images*rows if row_scale else images)
  out = Tensor.empty(images,rows,columns,device=x.device,dtype=dtypes.bfloat16)
  result = Tensor.custom_kernel(out,x,weight,*scales,*(() if bias is None else (bias,)),
    fxn=partial(program,rows=rows,images=images,mode=mode,row_scale=row_scale,wgp=wgp))[0]
  return result[0] if images == 1 else result
