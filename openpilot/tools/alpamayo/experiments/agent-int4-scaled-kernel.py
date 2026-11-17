"""Standalone A4 unsigned / W4 signed native K32 fc2 kernels.

Packed byte shape: [ceil(rows/16), K/32, 32, 8]. Inputs to quantize_pack
must be finite, nonnegative BF16 values after ReLU-square. No model hook.
"""
from functools import lru_cache, partial
import importlib.util
from pathlib import Path
from types import FunctionType
import numpy as np


def _frozen():
  spec = importlib.util.spec_from_file_location('scaled_frozen_k32', Path(__file__).with_name('agent-int4k32-kernel.py'))
  module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
  return module


def pack_weights_cpu(qweight):
  """Receive integer [N,K] signed weights. Return an independent read-only copy."""
  words = _frozen().pack_cpu(qweight)
  packed = np.frombuffer(words.tobytes(), dtype=np.uint8).reshape(*words.shape[:-1], 8)
  return packed


def pack_unsigned_cpu(q):
  """Reference input packer for prequantized [M,K] activation controls."""
  q = np.asarray(q)
  if q.ndim != 2 or q.shape[0] <= 0 or q.shape[1] <= 0 or q.shape[1] % 32:
    raise ValueError('Expected nonempty [M,K], K divisible by 32')
  if not np.issubdtype(q.dtype, np.integer) or np.any(q < 0) or np.any(q > 15): raise ValueError('Expected unsigned INT4 [0,15]')
  padded = np.pad(q, ((0, (-len(q)) % 16), (0, 0)))
  lanes = padded.reshape(-1, 16, q.shape[1]//32, 2, 8, 2).transpose(0, 2, 3, 1, 4, 5)
  lanes = lanes.reshape(-1, q.shape[1]//32, 32, 8, 2).astype(np.uint8)
  return lanes[..., 0] | (lanes[..., 1] << 4)


class StrictHIPCompiler:
  """Use the pinned COMGR pipeline with explicit IEEE FP32 denormals."""
  @staticmethod
  def compile_cached(text, wgp=False):
    from tinygrad.device import Compiler
    from tinygrad.runtime.support import compiler_amd
    original = compiler_amd.compile_hip
    def options(action, value):
      if b'--offload-arch=' in value:
        if wgp:value=value.replace(b'-mcumode',b'-mno-cumode')
        value += b' -fdenormal-fp-math=ieee,ieee -Xclang -fdenormal-fp-math-f32=ieee,ieee -ffp-contract=off -fno-unsafe-math-optimizations'
      return compiler_amd.set_options(action, value)
    compile_source = FunctionType(original.__code__, dict(original.__globals__, set_options=options), argdefs=original.__defaults__)
    class Pinned(Compiler):
      def compile(self, src): return compile_source(src, 'gfx1200')
    return Pinned(f'agent_int4_scaled_ieee_gfx1200_v1_{"wgp" if wgp else "cu"}').compile_cached(text)


HEADER = r'''
#pragma clang fp contract(off)
typedef int int2 __attribute__((ext_vector_type(2)));
typedef int int8 __attribute__((ext_vector_type(8)));
typedef unsigned short h8 __attribute__((ext_vector_type(8)));
extern "C" __attribute__((device)) unsigned long __ockl_get_local_id(unsigned);
extern "C" __attribute__((device)) unsigned long __ockl_get_group_id(unsigned);
__attribute__((device,always_inline)) float from_bf(unsigned short x) {
  return __builtin_bit_cast(float, ((unsigned)x)<<16);
}
__attribute__((device,always_inline)) unsigned short to_bf(float x) {
  unsigned u=__builtin_bit_cast(unsigned,x);
  return (u+0x7fff+((u>>16)&1))>>16;
}
'''


@lru_cache(None)
def compiled_gemm(rows, columns, width, bias=False, integer=False, wgp=False):
  if min(rows, columns, width) <= 0 or width % 32 or 120*width > 2**31-1: raise ValueError('Invalid shape or INT32 overflow bound')
  text = f'#define M {rows}\n#define N {columns}\n#define K {width}\n#define BIAS {int(bias)}\n#define INTEGER {int(integer)}\n' + HEADER + r'''
#if INTEGER
typedef int out_t;
#else
typedef unsigned short out_t;
#endif
extern "C" __attribute__((global,amdgpu_flat_work_group_size(128,128)))
void agent_int4_scaled(out_t* out,const int2* a,const int2* b,const float* xs,const float* ws
#if BIAS
                       ,const unsigned short* bias
#endif
) {
  int tid=__ockl_get_local_id(0), block=__ockl_get_group_id(0), lane=tid%32,wave=tid/32,cols=(N+63)/64;
  int row=(block/cols)*64+(wave/2)*32, col=(block%cols)*64+(wave%2)*32;
  int8 acc[2][2]={};
  for(int k=0;k<K/32;k++) {
    int2 av[2],bv[2];
    #pragma unroll
    for(int i=0;i<2;i++) {
      int ar=row/16+i,br=col/16+i;
      av[i]=ar<(M+15)/16?a[(ar*(K/32)+k)*32+lane]:(int2){0,0};
      bv[i]=br<(N+15)/16?b[(br*(K/32)+k)*32+lane]:(int2){0,0};
    }
    #pragma unroll
    for(int i=0;i<2;i++) {
      #pragma unroll
      for(int j=0;j<2;j++)
        acc[i][j]=__builtin_amdgcn_wmma_i32_16x16x32_iu4_w32_gfx12(false,av[i],true,bv[j],acc[i][j],false);
    }
  }
  #pragma unroll
  for(int j=0;j<2;j++) {
    int c=col+j*16+lane%16;
#if !INTEGER
    float weight_scale=c<N?ws[c]:0.0f;
#if BIAS
    float bias_value=c<N?from_bf(bias[c]):0.0f;
#endif
#endif
    #pragma unroll
    for(int i=0;i<2;i++) {
      #pragma unroll
      for(int q=0;q<8;q++) {
        int r=row+i*16+(lane/16)*8+q;
        if(r<M && c<N) {
#if INTEGER
          out[r*N+c]=acc[i][j][q];
#else
          float scale=xs[r]*weight_scale;
#if BIAS
          float value=__builtin_fmaf((float)acc[i][j][q],scale,bias_value);
#else
          float value=(float)acc[i][j][q]*scale;
#endif
          out[r*N+c]=to_bf(value);
#endif
        }
      }
    }
  }
}
'''
  return text, StrictHIPCompiler.compile_cached(text,wgp=wgp)


@lru_cache(None)
def compiled_quantize(rows, width, wgp=False):
  if rows <= 0 or width <= 0 or width % 32: raise ValueError('Invalid activation shape')
  text = f'#define M {rows}\n#define K {width}\n' + HEADER + r'''
extern "C" __attribute__((global,amdgpu_flat_work_group_size(128,128)))
void agent_int4_quantize(unsigned* packed,float* scales,const unsigned short* x) {
  int tid=__ockl_get_local_id(0),lane=tid%32,wave=tid/32;
  int row=__ockl_get_group_id(0)*4+wave;
  float maximum=0.0f;
  for(int word=lane;word<K/8;word+=32) {
    if(row<M) {
      h8 values=*(const h8*)(x+row*K+word*8);
      #pragma unroll
      for(int i=0;i<8;i++) maximum=__builtin_fmaxf(maximum,from_bf(values[i]));
    }
  }
  #pragma unroll
  for(int shift=16;shift;shift>>=1) {
    float other=__builtin_bit_cast(float,__builtin_amdgcn_ds_bpermute((lane^shift)*4,__builtin_bit_cast(int,maximum)));
    maximum=__builtin_fmaxf(maximum,other);
  }
  float scale=maximum==0.0f?1.0f:maximum/15.0f;
  if(lane==0 && row<M) scales[row]=scale;
  for(int word=lane;word<K/8;word+=32) {
    unsigned bits=0;
    if(row<M) {
      h8 values=*(const h8*)(x+row*K+word*8);
      #pragma unroll
      for(int i=0;i<8;i++) {
        float value=from_bf(values[i])/scale;
        value=__builtin_fminf(15.0f,__builtin_fmaxf(0.0f,value));
        unsigned q=(unsigned)__builtin_roundevenf(value);
        bits|=q<<(i*4);
      }
    }
    int k=word*8;
    int lane_out=((k%32)/16)*16+row%16,reg=(k%16)/8;
    packed[((row/16*(K/32)+k/32)*32+lane_out)*2+reg]=bits;
  }
}
'''
  return text, StrictHIPCompiler.compile_cached(text,wgp=wgp)


def metadata_program(params, outputs, blocks, name, text, binary):
  """Declare actual writes explicitly; shape syntax cannot hide HCQ WAW edges."""
  from tinygrad.helpers import Target
  from tinygrad.uop.ops import UOp, Ops, KernelInfo, ProgramInfo
  from tinygrad import dtypes
  buffers = tuple(p.buf_uop for p in params)
  assert all(p.op is Ops.PARAM and p.arg.slot == i for i, p in enumerate(buffers))
  reads = tuple(i for i in range(len(params)) if i not in outputs)
  # The sink is only an argument/launch declaration. ProgramInfo is authoritative.
  nodes = [p.flatten().index(0).load().cast(dtypes.float32) for i, p in enumerate(params) if i in reads]
  dependencies = sum(nodes[1:], nodes[0])
  stores = [params[i].flatten().index(0).store(dependencies.cast(params[i].dtype)) for i in outputs]
  sink = UOp.sink(*stores, UOp.special(blocks, 'gidx0'), UOp.special(128, 'lidx0'), arg=KernelInfo(name=name))
  info = ProgramInfo(global_size=(blocks, 1, 1), local_size=(128, 1, 1), params=buffers,
                     globals=tuple(range(len(params))), outs=tuple(outputs), ins=reads, target=Target('AMD', 'LLVM', 'gfx1200'))
  return UOp(Ops.PROGRAM, src=(sink, UOp(Ops.LINEAR, src=tuple(sink.toposort())), UOp(Ops.SOURCE, arg=text),
                              UOp(Ops.BINARY, arg=binary)), arg=info)


def gemm_program(out, a, b, xs, ws, *bias, rows, columns, width, integer=False, wgp=False):
  text, binary = compiled_gemm(rows, columns, width, bool(bias), integer,wgp=wgp)
  return metadata_program((out, a, b, xs, ws, *bias), (0,), ((rows+63)//64)*((columns+63)//64),
                          'agent_int4_scaled', text, binary)


def quantize_program(packed, scales, x, rows, width, wgp=False):
  text, binary = compiled_quantize(rows, width,wgp=wgp)
  return metadata_program((packed, scales, x), (0, 1), ((rows+15)//16)*4, 'agent_int4_quantize', text, binary)


def _gemm(a, b, activation_scale, weight_scale, *, rows, columns, bias=None, out=None, integer=False, wgp=False):
  from tinygrad import Tensor, dtypes
  assert a.dtype == b.dtype == dtypes.uint8 and a.ndim == b.ndim == 4
  assert a.shape[1:] == b.shape[1:] and a.shape[2:] == (32, 8)
  assert a.shape[0] == (rows+15)//16 and b.shape[0] == (columns+15)//16
  assert activation_scale.dtype == weight_scale.dtype == dtypes.float32
  assert activation_scale.shape == (rows,) and weight_scale.shape == (columns,)
  assert bias is None or (bias.dtype == dtypes.bfloat16 and bias.shape == (columns,))
  assert all(x.device == a.device for x in (b, activation_scale, weight_scale, *(() if bias is None else (bias,))))
  dtype = dtypes.int32 if integer else dtypes.bfloat16
  if out is None: out = Tensor.empty(rows, columns, dtype=dtype, device=a.device)
  assert out.dtype == dtype and out.shape == (rows, columns) and out.device == a.device
  return Tensor.custom_kernel(out, a, b, activation_scale, weight_scale, *(() if bias is None else (bias,)),
    fxn=partial(gemm_program, rows=rows, columns=columns, width=a.shape[1]*32, integer=integer,wgp=wgp))[0]


def gemm(a, b, activation_scale, weight_scale, *, rows, columns, bias=None, out=None,wgp=False):
  return _gemm(a, b, activation_scale, weight_scale, rows=rows, columns=columns, bias=bias, out=out,wgp=wgp)


def integer_gemm(a, b, *, rows, columns, out=None,wgp=False):
  """Diagnostic raw-integer output; scales are declared but unused by this variant."""
  from tinygrad import Tensor, dtypes
  xs = Tensor.ones(rows, dtype=dtypes.float32, device=a.device).realize()
  ws = Tensor.ones(columns, dtype=dtypes.float32, device=a.device).realize()
  return _gemm(a, b, xs, ws, rows=rows, columns=columns, out=out, integer=True,wgp=wgp)


def quantize_pack(x, *, packed_out=None, scales_out=None,wgp=False):
  from tinygrad import Tensor, dtypes
  assert x.dtype == dtypes.bfloat16 and x.ndim == 2 and x.shape[1] % 32 == 0
  x = x.contiguous()
  rows, width = x.shape
  shape = ((rows+15)//16, width//32, 32, 8)
  if packed_out is None: packed_out = Tensor.empty(*shape, dtype=dtypes.uint8, device=x.device)
  if scales_out is None: scales_out = Tensor.empty(rows, dtype=dtypes.float32, device=x.device)
  assert packed_out.shape == shape and packed_out.dtype == dtypes.uint8 and packed_out.device == x.device
  assert scales_out.shape == (rows,) and scales_out.dtype == dtypes.float32 and scales_out.device == x.device
  ret = Tensor.custom_kernel(packed_out, scales_out, x, fxn=partial(quantize_program, rows=rows, width=width,wgp=wgp))
  return ret[0], ret[1]


def linear(x, packed_weight, weight_scale, *, bias=None, out=None,wgp=False):
  packed, scale = quantize_pack(x,wgp=wgp)
  return gemm(packed, packed_weight, scale, weight_scale, rows=x.shape[0], columns=weight_scale.shape[0], bias=bias, out=out,wgp=wgp)
