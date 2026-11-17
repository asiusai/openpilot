"""Bounded SWMMAC format/fallback probe, not a production GEMM optimization.

One wave per16x16 output tile. Inputs retain original packed FP8 bytes/scales.
Compression checks all rows and all K4 groups of each16x32 activation tile.
No overflow pruning. Invalid tiles use two original dense K16 instructions.
"""
from functools import lru_cache, partial


def source(rows, columns, width, mode):
  assert rows % 16 == columns % 16 == width % 32 == 0 and mode in ('compress', 'dense', 'hybrid')
  defines = dict(M=rows, N=columns, K=width, COMPRESS=int(mode == 'compress'), HYBRID=int(mode == 'hybrid'))
  return ''.join(f'#define {key} {value}\n' for key, value in defines.items())+r'''
#pragma clang fp contract(off)
#define DEV __attribute__((device,always_inline)) inline
typedef int i2 __attribute__((ext_vector_type(2)));
typedef int i4 __attribute__((ext_vector_type(4)));
typedef float f8 __attribute__((ext_vector_type(8)));
extern "C" __attribute__((device)) unsigned long __ockl_get_local_id(unsigned);
extern "C" __attribute__((device)) unsigned long __ockl_get_group_id(unsigned);
DEV unsigned address(unsigned row,unsigned k) {
  return (row/16*(K/16)+k/16)*256+(k%16/4)*64+(row%16)*4+k%4;
}
DEV unsigned word(const unsigned char *p,unsigned row,unsigned k) {
  return *(const unsigned *)(p+address(row,k));
}
#if COMPRESS
extern "C" __attribute__((global,amdgpu_flat_work_group_size(32,32)))
void agent_sparse(unsigned *values,unsigned *metadata,unsigned *flags,const unsigned char *x) {
  unsigned lane=__ockl_get_local_id(0),tile=__ockl_get_group_id(0);
  unsigned mt=tile/(K/32),kt=tile%(K/32),row=mt*16+lane%16;
  unsigned regs[2]={0,0},meta=0,bad=0;
  #pragma unroll
  for(unsigned g=0;g<4;g++) {
    unsigned bits=word(x,row,kt*32+(lane/16)*16+g*4);
    unsigned codes[4],pos[2]={0,1},count=0;
    #pragma unroll
    for(unsigned e=0;e<4;e++) {
      codes[e]=(bits>>(e*8))&255;
      bad|=((codes[e]&127)==127);
      if(codes[e]&127) { if(count<2) pos[count]=e; count++; }
    }
    bad|=(count>2);
    if(count==1) { pos[1]=(pos[0]==0)?1:0; }
    if(pos[0]>pos[1]) { unsigned t=pos[0];pos[0]=pos[1];pos[1]=t; }
    // If count>2 these are placeholders; the WHOLE tile is flagged dense.
    unsigned pair=codes[pos[0]]|(codes[pos[1]]<<8);
    regs[g/2]|=pair<<((g%2)*16);
    meta|=(pos[0]|(pos[1]<<2))<<(g*4);
  }
  values[(tile*32+lane)*2]=regs[0]; values[(tile*32+lane)*2+1]=regs[1];
  metadata[tile*32+lane]=meta;
  unsigned any_bad=__builtin_amdgcn_ballot_w32(bad!=0);
  if(lane==0) flags[tile]=(any_bad==0);
}
#else
extern "C" __attribute__((global,amdgpu_flat_work_group_size(32,32)))
void agent_sparse(float *out,const unsigned char *x,const unsigned char *w,
                  const unsigned *values,const unsigned *metadata,const unsigned *flags,
                  const float *xs,const float *ws) {
  unsigned lane=__ockl_get_local_id(0),block=__ockl_get_group_id(0);
  unsigned mt=block/(N/16),nt=block%(N/16);
  f8 c=(f8){0};
  #pragma clang loop unroll(disable)
  for(unsigned kt=0;kt<K/32;kt++) {
    unsigned tile=mt*(K/32)+kt;
#if HYBRID
    if(flags[tile]) {
      i2 a=(i2){(int)values[(tile*32+lane)*2],(int)values[(tile*32+lane)*2+1]};
      unsigned col=nt*16+lane%16,base=kt*32+(lane/16)*16;
      i4 b=(i4){(int)word(w,col,base),(int)word(w,col,base+4),
                  (int)word(w,col,base+8),(int)word(w,col,base+12)};
      c=__builtin_amdgcn_swmmac_f32_16x16x32_fp8_fp8_w32(a,b,c,(int)metadata[tile*32+lane]);
    } else
#endif
    {
      #pragma unroll
      for(unsigned h=0;h<2;h++) {
        unsigned base=kt*32+h*16+(lane/16)*8;
        i2 a=(i2){(int)word(x,mt*16+lane%16,base),(int)word(x,mt*16+lane%16,base+4)};
        i2 b=(i2){(int)word(w,nt*16+lane%16,base),(int)word(w,nt*16+lane%16,base+4)};
        c=__builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(a,b,c);
      }
    }
  }
  float scale=xs[0]*ws[0];
  #pragma unroll
  for(unsigned g=0;g<8;g++) {
    unsigned row=mt*16+(lane/16)*8+g,col=nt*16+lane%16;
    out[row*N+col]=c[g]*scale;
  }
}
#endif
'''


@lru_cache(None)
def compiled(rows, columns, width, mode):
  from tinygrad.runtime.support.compiler_amd import HIPCompiler
  text = source(rows, columns, width, mode)
  return text, HIPCompiler('gfx1200').compile_cached(text)


def program(*args, rows, columns, width, mode):
  from tinygrad import dtypes
  from tinygrad.uop.ops import UOp, Ops, KernelInfo
  if mode == 'compress':
    values, metadata, flags, x = args
    # Describe every output and dynamic dependency to TinyJit/scheduler.
    seed = x.flatten()[0].load().cast(dtypes.uint32)
    stores = tuple(a.flatten()[0].store(seed) for a in (values, metadata, flags))
    groups = rows//16*(width//32)
  else:
    out, *inputs = args
    seed = inputs[0].flatten()[0].load().float()
    for value in inputs[1:]: seed = seed+value.flatten()[0].load().float()
    stores = (out.flatten()[0].store(seed),)
    groups = rows//16*(columns//16)
  sink = UOp.sink(*stores, UOp.special(groups, 'gidx0'), UOp.special(32, 'lidx0'), arg=KernelInfo(name='agent_sparse'))
  text, binary = compiled(rows, columns, width, mode)
  return UOp(Ops.PROGRAM, src=(sink, UOp(Ops.LINEAR, src=tuple(sink.toposort())),
                             UOp(Ops.SOURCE, arg=text), UOp(Ops.BINARY, arg=binary)))


def compress(x):
  from tinygrad import Tensor, dtypes
  assert x.dtype in (dtypes.uint8, dtypes.fp8e4m3) and x.ndim == 2
  rows, width = x.shape
  assert rows % 16 == width % 32 == 0
  tiles = rows//16*(width//32)
  outputs = (Tensor.empty(tiles, 32, 2, device=x.device, dtype=dtypes.uint32),
             Tensor.empty(tiles, 32, device=x.device, dtype=dtypes.uint32),
             Tensor.empty(tiles, device=x.device, dtype=dtypes.uint32))
  return tuple(Tensor.custom_kernel(*outputs, x, fxn=partial(program, rows=rows, columns=16, width=width, mode='compress'))[:3])


def multiply(x, w, scales, compressed, mode='hybrid'):
  from tinygrad import Tensor, dtypes
  assert mode in ('dense', 'hybrid') and x.ndim == w.ndim == 2 and x.shape[1] == w.shape[1]
  rows, width = x.shape
  columns = w.shape[0]
  assert x.dtype == w.dtype and x.dtype in (dtypes.uint8, dtypes.fp8e4m3)
  assert x.device == w.device and all(a.device == x.device for a in (*scales, *compressed))
  assert rows % 16 == columns % 16 == width % 32 == 0
  assert len(compressed) == 3 and all(a.numel() == 1 and a.dtype == dtypes.float32 for a in scales)
  tiles = rows//16*(width//32)
  assert tuple(a.shape for a in compressed) == ((tiles, 32, 2), (tiles, 32), (tiles,))
  assert all(a.dtype == dtypes.uint32 for a in compressed)
  out = Tensor.empty(rows, columns, device=x.device, dtype=dtypes.float32)
  return Tensor.custom_kernel(out, x, w, *compressed, *scales,
    fxn=partial(program, rows=rows, columns=columns, width=width, mode=mode))[0]
