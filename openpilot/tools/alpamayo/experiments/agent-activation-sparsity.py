"""Read-only BF16 and selected packed-FP8 fc2 statistics. No device opens at import.

Counters use complete K4 groups. Row/K32 and 16-row/K32 eligibility excludes
incomplete blocks. Scalar/group counters include every real prompt element.
"""
from functools import lru_cache, partial
import inspect
import math
from types import MethodType
import numpy as np

NAMES = ('elements', 'zeros', 'negative_zeros', 'negative_nonzero', 'nonfinite', 'subnormal',
         'at_quantizer_limit', 'above_quantizer_limit', 'division_nonfinite',
         *('group4_nnz_'+str(i) for i in range(5)), 'row_k32_blocks', 'row_k32_eligible',
         *('row_k32_bad_groups_'+str(i) for i in range(9)), 'tile16_k32_blocks', 'tile16_k32_eligible',
         *('tile16_bad_bucket_'+str(i) for i in range(9)))
F = len(NAMES)
S = F*2
BLOCKS = 32
THREADS = 128
MAGIC = 0x53504152
MAX_ELEMENTS = 2112*9216
TILE_BUCKETS = ('0', '1', '2-3', '4-7', '8-15', '16-31', '32-63', '64-127', '128')


def bf16(values):
  a = np.asarray(values, np.float32).view(np.uint32)
  return ((a + np.uint32(0x7fff) + ((a >> 16) & 1)) >> 16).astype(np.uint16)


def bf32(words):
  return (np.asarray(words, np.uint16).astype(np.uint32) << 16).view(np.float32)


def fp8_table():
  code = np.arange(127, dtype=np.uint8)
  exponent, mantissa = code >> 3, code & 7
  return np.where(exponent == 0, mantissa.astype(np.float32)/512,
                  (1+mantissa.astype(np.float32)/8)*np.exp2(exponent.astype(np.float32)-7)).astype(np.float32)


def selected_quantize_cpu(words, scale):
  """Independent E4M3 RNE oracle for the selected divide/clamp/cast sequence.

  The production observer never requantizes. It reads the original packed bytes.
  NaN payload conversion is deliberately rejected; diagnostics still count NaNs.
  """
  x = bf32(words)
  scale = np.float32(scale)
  if not np.isfinite(scale) or scale <= 0 or np.isnan(x).any(): raise ValueError('Invalid scale or NaN input')
  with np.errstate(over='ignore', invalid='ignore'):
    y = np.clip(np.divide(x, scale, dtype=np.float32), np.float32(-448), np.float32(448))
  v = np.abs(y)
  table = fp8_table()
  high = np.searchsorted(table, v).clip(0, 126)
  low = np.maximum(high-1, 0)
  dl, dh = v-table[low], table[high]-v
  code = np.where(dl < dh, low, np.where(dh < dl, high, np.where((low & 1) == 0, low, high)))
  return code.astype(np.uint8) | (np.signbit(y).astype(np.uint8) << 7)


def pack_cpu(codes):
  q = np.asarray(codes, np.uint8)
  if q.ndim != 2 or min(q.shape) <= 0 or q.shape[1] % 16: raise ValueError('Require [M,K], K multiple of 16')
  q = np.pad(q, ((0, (-q.shape[0]) % 16), (0, 0)))
  return q.reshape(-1, 16, q.shape[1]//16, 4, 4).transpose(0, 2, 3, 1, 4).copy().reshape(q.shape)


def unpack_cpu(packed, rows):
  p = np.asarray(packed, np.uint8)
  return p.reshape(-1, p.shape[1]//16, 4, 16, 4).transpose(0, 3, 1, 2, 4).reshape(p.shape)[:rows]


def oracle(words, packed, scale, valid):
  """Independent vectorized NumPy counters, separate from native tile traversal."""
  words = np.asarray(words, np.uint16)
  rows, width = words.shape
  if not 0 < valid <= rows or width % 16 or rows*width > MAX_ELEMENTS: raise ValueError('Unsupported bound')
  x, q = bf32(words[:valid]), unpack_cpu(packed, rows)[:valid]
  result = []
  with np.errstate(over='ignore', invalid='ignore'):
    divided = np.divide(x, np.float32(scale), dtype=np.float32)
  for bank, bits in enumerate((words[:valid], q)):
    magnitude = bits & (0x7fff if bank == 0 else 0x7f)
    zero = magnitude == 0
    negative = (bits & (0x8000 if bank == 0 else 0x80)) != 0
    finite = magnitude < 0x7f80 if bank == 0 else magnitude != 0x7f
    subnormal = (magnitude > 0) & (magnitude < (0x80 if bank == 0 else 8))
    sat = (np.abs(divided) == 448) if bank == 0 else magnitude == 126
    clip = np.abs(divided) > 448 if bank == 0 else np.zeros_like(zero)
    divbad = ~np.isfinite(divided) if bank == 0 else np.zeros_like(zero)
    hist = np.bincount((~zero).reshape(valid, width//4, 4).sum(-1).ravel(), minlength=5)
    full_k = width//32
    bad = ((~zero[:, :full_k*32]).reshape(valid, full_k, 8, 4).sum(-1) > 2).sum(-1)
    rowhist = np.bincount(bad.ravel(), minlength=9)
    full_r = valid//16
    tilebad = bad[:full_r*16].reshape(full_r, 16, full_k).sum(1)
    bucket = np.where(tilebad == 0, 0, np.floor(np.log2(np.maximum(tilebad, 1))).astype(np.int32)+1)
    tilehist = np.bincount(bucket.ravel(), minlength=9)
    counters = np.array([zero.size, zero.sum(), (zero & negative).sum(), (~zero & negative).sum(), (~finite).sum(),
                         subnormal.sum(), sat.sum(), clip.sum(), divbad.sum(), *hist, bad.size, (bad == 0).sum(),
                         *rowhist, tilebad.size, (tilebad == 0).sum(), *tilehist], np.uint32)
    assert counters.shape == (F,)
    result.append(counters)
  return np.stack(result)


def describe(counters, valid, width):
  c = np.asarray(counters)
  if c.shape != (2, F) or c.dtype != np.uint32: raise ValueError('Bad statistics representation')
  output = {}
  for bank, name in enumerate(('bf16_before', 'fp8_after_actual')):
    d = {key: int(value) for key, value in zip(NAMES, c[bank], strict=True)}
    hist = [d['group4_nnz_'+str(i)] for i in range(5)]
    rowhist = [d['row_k32_bad_groups_'+str(i)] for i in range(9)]
    tilehist = [d['tile16_bad_bucket_'+str(i)] for i in range(9)]
    if (d['elements'] != valid*width or sum(hist) != valid*(width//4)
        or sum(i*n for i, n in enumerate(hist)) != d['elements']-d['zeros']
        or sum(rowhist) != valid*(width//32) or rowhist[0] != d['row_k32_eligible']
        or sum(tilehist) != (valid//16)*(width//32) or tilehist[0] != d['tile16_k32_eligible']
        or any(v > MAX_ELEMENTS for v in d.values())):
      raise ValueError('Counter corruption, overflow, wrong current prompt, or stale observer')
    d.update(zero_fraction=d['zeros']/d['elements'], group4_nnz_histogram=hist,
             exact_2of4_eligible_fraction=sum(hist[:3])/sum(hist),
             row_k32_eligible_fraction=rowhist[0]/sum(rowhist) if sum(rowhist) else None,
             tile16_k32_eligible_fraction=tilehist[0]/sum(tilehist) if sum(tilehist) else None,
             row_k32_bad_group_histogram=rowhist, tile16_bad_group_histogram=dict(zip(TILE_BUCKETS, tilehist)),
             excluded_partial_row_k32_groups=valid*(width%32//4),
             excluded_tile_tail_rows=valid%16, excluded_tile_tail_k=width%32,
             uint32_counter_bound=MAX_ELEMENTS, counter_overflow=0)
    output[name] = d
  return output


COMMON = r'''
typedef unsigned int u32;
typedef unsigned short u16;
typedef unsigned char u8;
INLINE float frombf(u16 h) { union {u32 u;float f;} a; a.u=((u32)h)<<16;return a.f; }
INLINE void tile(int tile,const u16* x,const u8* p,float scale,int valid,u32* c) {
  int r0=(tile/((K+31)/32))*16,k0=(tile%((K+31)/32))*32;
  int tb[2]={0,0};
  for(int r=r0;r<r0+16 && r<valid && r<M;r++) {
    int rb[2]={0,0};
    for(int k=k0;k<k0+32 && k<K;k+=4) {
      int nnz[2]={0,0};
      for(int j=0;j<4;j++) {
        int col=k+j;
        u16 h=x[r*K+col];
        int address=(r/16*(K/16)+col/16)*256+(col%16/4)*64+(r%16)*4+col%4;
        u8 q=p[address];
        u32 mag[2]={(u32)(h&32767),(u32)(q&127)};
        int neg[2]={(h&32768)!=0,(q&128)!=0};
        float v=frombf(h)/scale;
        for(int b=0;b<2;b++) {
          u32* d=c+b*F;
          d[0]++;d[1]+=mag[b]==0;d[2]+=(mag[b]==0 && neg[b]);d[3]+=(mag[b]!=0 && neg[b]);
          d[4]+=b==0?mag[b]>=32640:mag[b]==127;
          d[5]+=(mag[b]>0 && mag[b]<(b==0?128:8));
          d[6]+=b==0?__builtin_fabsf(v)==448.0f:mag[b]==126;
          d[7]+=(b==0 && __builtin_fabsf(v)>448.0f);
          d[8]+=(b==0 && !__builtin_isfinite(v));
          nnz[b]+=mag[b]!=0;
        }
      }
      for(int b=0;b<2;b++) { c[b*F+9+nnz[b]]++;rb[b]+=nnz[b]>2; }
    }
    if(k0+32<=K) for(int b=0;b<2;b++) {
      c[b*F+14]++;c[b*F+15]+=rb[b]==0;c[b*F+16+rb[b]]++;
    }
    for(int b=0;b<2;b++) tb[b]+=rb[b];
  }
  if(r0+16<=valid && k0+32<=K) for(int b=0;b<2;b++) {
    int bucket=0,n=tb[b];if(n){bucket=1;while(n>>=1)bucket++;}
    c[b*F+25]++;c[b*F+26]+=tb[b]==0;c[b*F+27+bucket]++;
  }
}
'''


@lru_cache(None)
def compiled(rows, width, stage, cpu=False):
  if not 0 < rows*width <= MAX_ELEMENTS or width % 16 or stage not in ('partial', 'final'):
    raise ValueError('Unsupported observer shape')
  prefix = f'#define M {rows}\n#define K {width}\n#define F {F}\n#define S {S}\n#define B {BLOCKS}\n'
  if cpu:
    prefix += '#define INLINE static inline\n'
    if stage == 'partial':
      body = r'''
void agent_activation_partial(void** args) {
  u32* out=(u32*)args[0];const u16* x=(u16*)args[1];const u8* p=(u8*)args[2];
  const float* scale=(float*)args[3];const u32* control=(u32*)args[4];
  for(int b=0;b<B;b++) for(int f=0;f<S;f++)out[b*S+f]=0;
  for(int t=0;t<((M+15)/16)*((K+31)/32);t++)tile(t,x,p,scale[0],control[0],out+(t%B)*S);
}
'''
    else:
      body = r'''
void agent_activation_final(void** args) {
  u32* out=(u32*)args[0];const u32* p=(u32*)args[1];const u32* scale=(u32*)args[2];const u32* ctl=(u32*)args[3];
  for(int f=0;f<S;f++){u32 sum=0;for(int b=0;b<B;b++)sum+=p[b*S+f];out[f]=sum;}
  out[S]=ctl[0];out[S+1]=ctl[1];out[S+2]=scale[0];out[S+3]=(ctl[0]>0 && ctl[0]<=M)?0x53504152:0;
}
'''
    from tinygrad.runtime.support.compiler_cpu import ClangCompiler
    compiler = ClangCompiler(['x86_64', 'native'])
  else:
    prefix += ('#define INLINE __attribute__((device,always_inline))\n'
               'extern "C" __attribute__((device)) unsigned long __ockl_get_local_id(unsigned);\n'
               'extern "C" __attribute__((device)) unsigned long __ockl_get_group_id(unsigned);\n')
    if stage == 'partial':
      body = r'''
extern "C" __attribute__((global,amdgpu_flat_work_group_size(128,128)))
void agent_activation_partial(u32* out,const u16* x,const u8* p,const float* scale,const u32* ctl) {
  int tid=__ockl_get_local_id(0),block=__ockl_get_group_id(0);
  u32 c[S]={};
  for(int t=block*128+tid;t<((M+15)/16)*((K+31)/32);t+=B*128)tile(t,x,p,scale[0],ctl[0],c);
  __attribute__((shared)) u32 scratch[S][128];
  for(int f=0;f<S;f++)scratch[f][tid]=c[f];
  __builtin_amdgcn_s_barrier();
  if(tid<S){u32 sum=0;for(int i=0;i<128;i++)sum+=scratch[tid][i];out[block*S+tid]=sum;}
}
'''
    else:
      body = r'''
extern "C" __attribute__((global,amdgpu_flat_work_group_size(128,128)))
void agent_activation_final(u32* out,const u32* p,const u32* scale,const u32* ctl) {
  int f=__ockl_get_local_id(0);
  if(f<S){u32 sum=0;for(int b=0;b<B;b++)sum+=p[b*S+f];out[f]=sum;}
  if(f==0){out[S]=ctl[0];out[S+1]=ctl[1];out[S+2]=scale[0];out[S+3]=(ctl[0]>0 && ctl[0]<=M)?0x53504152:0;}
}
'''
    from tinygrad.runtime.support.compiler_amd import HIPCompiler
    compiler = HIPCompiler('gfx1200')
  source = prefix+COMMON+body
  return source, compiler.compile_cached(source)


def program(*params, rows, width, stage, cpu):
  from tinygrad import dtypes
  from tinygrad.helpers import Target
  from tinygrad.uop.ops import UOp, Ops, KernelInfo, ProgramInfo
  source, binary = compiled(rows, width, stage, cpu)
  buffers = tuple(p.buf_uop for p in params)
  reads = tuple(range(1, len(params)))
  nodes = [params[i].flatten().index(0).load().cast(dtypes.uint32) for i in reads]
  dummy = sum(nodes[1:], nodes[0])
  blocks = BLOCKS if stage == 'partial' else 1
  name = 'agent_activation_'+stage
  launch = () if cpu else (UOp.special(blocks, 'gidx0'), UOp.special(THREADS, 'lidx0'))
  sink = UOp.sink(params[0].flatten().index(0).store(dummy), *launch, arg=KernelInfo(name=name))
  target = Target('CPU', 'CLANG', 'x86_64,native') if cpu else Target('AMD', 'LLVM', 'gfx1200')
  info = ProgramInfo(global_size=(1 if cpu else blocks, 1, 1), local_size=(1 if cpu else THREADS, 1, 1),
    params=buffers, globals=tuple(range(len(params))), outs=(0,), ins=reads, target=target)
  return UOp(Ops.PROGRAM, src=(sink, UOp(Ops.LINEAR, src=tuple(sink.toposort())), UOp(Ops.SOURCE, arg=source),
                              UOp(Ops.BINARY, arg=binary)), arg=info)


def observe(x, packed, scale, control, scratch, output, cpu=False):
  from tinygrad import Tensor, dtypes
  rows, width = math.prod(x.shape[:-1]), x.shape[-1]
  if x.dtype != dtypes.bfloat16 or scale.dtype != dtypes.float32 or scale.numel() != 1:
    raise ValueError('Require unchanged BF16 activation and tensor-wide FP32 scale')
  if packed.dtype != dtypes.fp8e4m3 or packed.shape != ((rows+15)//16*16, width):
    raise ValueError('Require the actual selected PackedActivation.value layout')
  if control.shape != (2,) or control.dtype != dtypes.uint32: raise ValueError('Bad control buffer')
  # Raw integer views prevent CPU non-native dtype lowering from decoding or
  # re-encoding FP8 subnormals. HIP receives the identical original byte storage.
  partials = Tensor.custom_kernel(scratch, x.bitcast(dtypes.uint16), packed.bitcast(dtypes.uint8), scale, control,
    fxn=partial(program, rows=rows, width=width, stage='partial', cpu=cpu))[0]
  result = Tensor.custom_kernel(output, partials, scale, control,
    fxn=partial(program, rows=rows, width=width, stage='final', cpu=cpu))[0]
  result.realize()
  return result


class Observer:
  """Persistent explicit buffers are written inside every prefill JIT execution."""
  def __init__(self, model, *, cpu=False):
    from tinygrad import Tensor, dtypes
    self.model, self.original = model, model.linear
    self.quantize = model.quantize
    closure = inspect.getclosurevars(self.quantize.__func__).nonlocals
    self.state = closure.get('state')
    if not isinstance(self.state, dict) or set(self.state) != {'prefill', 'prefix', 'index'}:
      raise ValueError('The selected quantize-packed closure changed; reject unsupported quantizer')
    if not hasattr(model, 'padded_prefill'): raise ValueError('This initial diagnostic requires padded prefill')
    device = next(iter(model.weights.values())).device
    self.control = Tensor.zeros(2, device=device, dtype=dtypes.uint32).contiguous().realize()
    self.buffers, self.trace_counts = {}, {}
    self.epoch, self.valid = 0, None
    self.control_id = self.control.uop.buffer
    def linear(this, x, name, quantized_input=None, heads=None):
      prefix = name.removesuffix('.mlp.fc2')
      if (not name.endswith('.mlp.fc2') or not prefix.startswith('vlm.model.language_model.layers.')
          or not self.state['prefill'] or x.numel()//x.shape[-1] < 128):
        return self.original(x, name, quantized_input, heads)
      if quantized_input is not None or self.valid is None or self.state != dict(prefill=True, prefix=prefix, index=3):
        raise ValueError('fc2 boundary or current-query observer contract changed')
      quantized = self.quantize(x)
      packed, scale = quantized
      if self.state['index'] != 4 or (packed.rows, packed.width) != (math.prod(x.shape[:-1]), x.shape[-1]):
        raise ValueError('Unexpected selected fc2 PackedActivation')
      if name not in self.buffers:
        scratch = Tensor.empty(BLOCKS, S, device=x.device, dtype=dtypes.uint32).realize()
        output = Tensor.empty(S+4, device=x.device, dtype=dtypes.uint32).realize()
        self.buffers[name] = scratch, output
      scratch, output = self.buffers[name]
      self.trace_counts[name] = self.trace_counts.get(name, 0)+1
      observe(x, packed.value, scale, self.control, scratch, output, cpu=cpu)
      return self.original(x, name, quantized, heads)
    model.linear = MethodType(linear, model)

  def prepare(self, valid):
    from tinygrad import Tensor, dtypes
    if not 128 <= valid <= 2112: raise ValueError('Invalid actual prompt length')
    self.valid, self.epoch = valid, self.epoch+1
    if self.epoch >= 2**31: raise ValueError('Epoch overflow')
    self.control.assign(Tensor(np.array([valid, self.epoch], np.uint32), device='CPU').to(self.control.device)).realize()
    if self.control.uop.buffer is not self.control_id: raise ValueError('Control allocation identity changed')
    # Every output must be overwritten by the CURRENT graph, including on replay.
    for _, output in self.buffers.values(): output.assign(Tensor.full(output.shape, 0xdeadbeef, device=output.device, dtype=dtypes.uint32)).realize()

  def read(self, width=9216):
    if len(self.buffers) != 28: raise ValueError('Expected all 28 language fc2 layers')
    result = {}
    for name, (_, output) in sorted(self.buffers.items()):
      raw = output.numpy()  # Only 304 bytes/layer. Never download activation tensors.
      if raw[-4] != self.valid or raw[-3] != self.epoch or raw[-1] != MAGIC:
        raise ValueError('Missing current-input graph write, bad binding, or stale statistics')
      scale = raw[-2:-1].view(np.float32)[0]
      if not np.isfinite(scale) or scale <= 0: raise ValueError('Invalid actual scale')
      result[name] = dict(actual_scale=float(scale), actual_scale_bits=int(raw[-2]),
                          **describe(raw[:S].reshape(2, F), self.valid, width))
    return result

  def restore(self): self.model.linear = self.original
