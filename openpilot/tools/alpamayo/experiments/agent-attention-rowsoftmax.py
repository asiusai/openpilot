"""Experimental gfx1200 online attention with row-owned softmax state.

Each wave still handles 16 queries and 32 keys per iteration. Transpose FP32
WMMA scores before softmax. A lane then owns eight columns of one query.
Preserve the selected kernel's BF16 probabilities, tile order, and sum tree.
This module does not initialize a GPU until attention() is executed.
"""
from functools import lru_cache, partial
import importlib.util
from pathlib import Path
from types import MethodType

from tinygrad import Tensor, dtypes
from tinygrad.runtime.support.compiler_amd import HIPCompiler
from tinygrad.uop.ops import KernelInfo, Ops, UOp


@lru_cache(None)
def baseline_module():
  spec = importlib.util.spec_from_file_location('agent_attention_original_online', Path(__file__).with_name('online-attention.py'))
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module


def replace_once(text, old, new):
  assert text.count(old) == 1, (old, text.count(old))
  return text.replace(old, new)


def score_fma_fragments(keys, causal):
  """Pinned HIP compiler contracts only completely unmasked 16-key fragments.

  Causal predicates retain the rounded multiply. For noncausal 32-key tiles,
  the final tile determines which fragment bounds the compiler eliminates.
  The rowsoftmax check verifies this policy against actual baseline assembly.
  """
  tail = keys % 32 or 32
  return tuple(not causal and tail >= 16*(sub+1) for sub in range(2))


@lru_cache(None)
def compiled(heads, kv_heads, rows, keys, causal, width, waves=8, value_layout=False, transpose='shuffle'):
  assert width in (72, 128) and waves in (4, 8)
  assert transpose in ('shuffle', 'lds', 'stage', 'shuffle-stage', 'shuffle-fma')
  preserve_score_fma = transpose == 'shuffle-fma'
  fused_fragments = score_fma_fragments(keys, causal) if preserve_score_fma else (False, False)
  stage_values = transpose in ('stage', 'shuffle-stage')
  assert not (stage_values and value_layout)
  # Read the known source generator, without changing its module or environment.
  text, _ = baseline_module().compiled(heads, kv_heads, rows, keys, causal,
    waves=waves, width=width, block_keys=32, stage_kv=True, key_padding=4,
    kv_load=1, value_layout=value_layout, probability_mma=True)
  text = replace_once(text, 'void alpamayo_online(', 'void agent_attention_rowsoftmax(')
  if preserve_score_fma:
    # Keep one score loop body, including small two-tile controls. This also
    # makes the assembly contraction check count each fragment exactly once.
    text = replace_once(text, '  for(int start=0;start<end;start+=32) {',
      '  #pragma unroll 1\n  for(int start=0;start<end;start+=32) {')
  if stage_values:
    # Exact vector query loads. Both supported widths are multiples of eight.
    text = replace_once(text, '''  for(int d=0;d<D;d++) for(int i=0;i<8;i++) {
    int dim=d*16+(lane/16)*8+i;
    query[d][i]=qr<R && dim<W?q[(head*R+qr)*W+dim]:0;
  }''', '''  for(int d=0;d<D;d++) {
    int dim=d*16+(lane/16)*8;
    query[d]=qr<R && dim<W?__builtin_bit_cast(s8,*(const volatile u8*)(q+(head*R+qr)*W+dim)):(s8){};
  }''')
    text = replace_once(text, '''for(int i=0;i<8;i++) { shared_k[key_index(row,dim+i)]=loaded_k[i]; shared_v[index+i]=loaded_v[i]; }''',
      '''for(int i=0;i<8;i++) {
        shared_k[key_index(row,dim+i)]=loaded_k[i];
        // The next WMMA reads eight adjacent keys for one output dimension.
        int target=((dim/16)*2+row/16)*256+(row%16/8)*128+((dim+i)%16)*8+row%8;
        shared_v[target]=loaded_v[i];
      }''')
    old = '''      for(int i=0;i<8;i++) { int key=start+sub*16+(lane/16)*8+i;
        #if 1
        value[i]=shared_v[(sub*16+(lane/16)*8+i)*D*16+d*16+lane%16];
        #else
        value[i]=key<K && d*16+lane%16<W?v[(kh*K+key)*W+d*16+lane%16]:0;
        #endif
      }'''
    text = replace_once(text, old, '      value=*(const s8*)(shared_v+(d*2+sub)*256+lane*8);')
    if transpose == 'stage':
      text = replace_once(text, '''  for(int d=0;d<D;d++) for(int i=0;i<8;i++) {
    int row=tile*16+(lane/16)*8+i;
    if(row<R && d*16+lane%16<W) out[(head*R+row)*W+d*16+lane%16]=tobf(accum[d][i]/denom[i]);
  }''', '''  for(int i=0;i<8;i++) {
    int row=tile*16+(lane/16)*8+i;
    float row_denom=denom[i];
    for(int d=0;d<D;d++) if(row<R && d*16+lane%16<W)
      out[(head*R+row)*W+d*16+lane%16]=tobf(accum[d][i]/row_denom);
  }''')
      return text, HIPCompiler('gfx1200').compile_cached(text)
  text = replace_once(text, 'f8 accum[D]={}; f8 denom={}, maxima;\n  for(int i=0;i<8;i++) maxima[i]=-__builtin_inff();',
    'f8 accum[D]={}; float denom=0.0f, maxima=-__builtin_inff();')
  # The original shuffle variants transpose rounded scaled scores. shuffle-fma
  # instead transposes raw QK for fragments where baseline contracts scale-max.
  # Keep the rounded scaled value for the maximum and raw QK for explicit FMA.
  transpose_code = '''
    for(int sub=0;sub<2;sub++) scores[sub]=transpose_probability(scores[sub],lane);
'''
  if transpose == 'lds':
    text = replace_once(text, 'short *prob=probabilities+wave*16*(16+0);', '''short *prob=probabilities+wave*16*(16+0);
  __attribute__((shared,aligned(16))) float transpose_scratch[WAVES_VALUE*2*16*20];
  float *scratch=transpose_scratch+wave*2*16*20;'''.replace('WAVES_VALUE', str(waves)))
    transpose_code = '''
    for(int sub=0;sub<2;sub++) for(int i=0;i<8;i++)
      scratch[sub*16*20+((lane/16)*8+i)*20+lane%16]=scores[sub][i];
    __builtin_amdgcn_fence(__ATOMIC_RELEASE,"workgroup","local"); __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE,"workgroup","local");
    for(int sub=0;sub<2;sub++) for(int i=0;i<8;i++)
      scores[sub][i]=scratch[sub*16*20+(lane%16)*20+(lane/16)*8+i];
'''
  start = text.index('    for(int i=0;i<8;i++) {\n      int row=tile*16+(lane/16)*8+i;')
  end = text.index('    for(int sub=0;sub<32/16;sub++) {\n    #if 1', start)
  scale_scores = '''
    for(int sub=0;sub<2;sub++) for(int i=0;i<8;i++) {
      int row=tile*16+(lane/16)*8+i, col=start+sub*16+lane%16;
      scores[sub][i]=(col<K && (!CAUSAL_VALUE || col<=row))?scores[sub][i]*SCALE_VALUE:-__builtin_inff();
    }
'''.replace('CAUSAL_VALUE', str(int(causal))).replace('SCALE_VALUE', repr(width**-.5)+'f')
  after_transpose = ''
  if any(fused_fragments):
    # Only unmasked fragments are eligible. Leave their raw FP32 scores intact.
    condition = ' || '.join(f'sub=={sub}' for sub, fused in enumerate(fused_fragments) if fused)
    scale_scores = replace_once(scale_scores, '      int row=', f'      if({condition}) continue;\n      int row=')
    for sub, fused in enumerate(fused_fragments):
      if fused:
        after_transpose += f'''    f8 raw_score_{sub}=scores[{sub}];
    for(int i=0;i<8;i++) scores[{sub}][i]=raw_score_{sub}[i]*{repr(width**-.5)}f;
'''
  softmax = '''
    float localmax=-__builtin_inff();
    for(int i=0;i<8;i++) localmax=__builtin_fmaxf(localmax,__builtin_fmaxf(scores[0][i],scores[1][i]));
    // Lane pairs differing in bit 4 own the two column halves of one row.
    float next=__builtin_fmaxf(maxima,__builtin_fmaxf(localmax,shuffle<0x401f>(localmax)));
    float scale=next==-__builtin_inff()?1.0f:__builtin_exp2f((maxima-next)*1.4426950408889634f);
    f8 sums;
    for(int i=0;i<8;i++) {
      scores[0][i]=next==-__builtin_inff()?0.0f:__builtin_exp2f((scores[0][i]-next)*1.4426950408889634f);
      scores[1][i]=next==-__builtin_inff()?0.0f:__builtin_exp2f((scores[1][i]-next)*1.4426950408889634f);
      sums[i]=scores[0][i]+scores[1][i];
      // Preserve the original column XOR 8,4,2,1 tree exactly.
      sums[i]+=shuffle<0x401f>(sums[i]);
    }
    for(int i=0;i<4;i++) sums[i]+=sums[i+4];
    for(int i=0;i<2;i++) sums[i]+=sums[i+2];
    float total_sum=sums[0]+sums[1];
    denom=denom*scale+total_sum;
    maxima=next;
    for(int i=0;i<8;i++) {
      float row_scale=lane_read(scale,(lane/16)*8+i);
      for(int d=0;d<D;d++) accum[d][i]*=row_scale;
    }
'''
  for sub, fused in enumerate(fused_fragments):
    if fused:
      softmax = replace_once(softmax, f'(scores[{sub}][i]-next)',
        f'__builtin_fmaf(raw_score_{sub}[i],{repr(width**-.5)}f,-next)')
  text = text[:start] + scale_scores + transpose_code + after_transpose + softmax + text[end:]
  old = '''    s8 stored, identity={};
    for(int i=0;i<8;i++) {
      stored[i]=tobf(scores[sub][i]);
      identity[i]=lane%16==(lane/16)*8+i?tobf(1.0f):0;
    }
    f8 transposed=mma(stored,identity,(f8){});
    s8 probability;
    for(int i=0;i<8;i++) probability[i]=tobf(transposed[i]);'''
  text = replace_once(text, old, '''    s8 probability;
    for(int i=0;i<8;i++) probability[i]=tobf(scores[sub][i]);''')
  text = replace_once(text, '''  for(int d=0;d<D;d++) for(int i=0;i<8;i++) {
    int row=tile*16+(lane/16)*8+i;
    if(row<R && d*16+lane%16<W) out[(head*R+row)*W+d*16+lane%16]=tobf(accum[d][i]/denom[i]);
  }''', '''  for(int i=0;i<8;i++) {
    int row=tile*16+(lane/16)*8+i;
    float row_denom=lane_read(denom,(lane/16)*8+i);
    for(int d=0;d<D;d++) if(row<R && d*16+lane%16<W)
      out[(head*R+row)*W+d*16+lane%16]=tobf(accum[d][i]/row_denom);
  }''')
  return text, HIPCompiler('gfx1200').compile_cached(text)


def program(out, q, k, v, *, shape, causal, waves, value_layout, transpose):
  batch, heads, rows, width = shape
  kv_heads, keys = k.shape[1:3]
  sink = UOp.sink(out.flatten()[0].store(q.flatten()[0].load()+k.flatten()[0].load()+v.flatten()[0].load()),
    UOp.special(batch*heads*((rows+16*waves-1)//(16*waves)), 'gidx0'),
    UOp.special(32*waves, 'lidx0'), arg=KernelInfo(name='agent_attention_rowsoftmax'))
  text, binary = compiled(heads, kv_heads, rows, keys, causal, width, waves, value_layout, transpose)
  return UOp(Ops.PROGRAM, src=(sink, UOp(Ops.LINEAR, src=tuple(sink.toposort())), UOp(Ops.SOURCE, arg=text), UOp(Ops.BINARY, arg=binary)))


def attention(q, k, v, causal=False, *, waves=8, transpose='shuffle', value_layout=None):
  assert q.dtype == k.dtype == v.dtype == dtypes.bfloat16
  assert q.shape[-1] == k.shape[-1] == v.shape[-1] and q.shape[-1] in (72, 128)
  assert q.shape[0] == k.shape[0] == v.shape[0] and k.shape == v.shape
  assert q.shape[1] % k.shape[1] == 0
  if value_layout is None:
    value_layout = q.shape[2] >= 1024
  if transpose in ('stage', 'shuffle-stage'):
    value_layout = False
  out = Tensor.empty(q.shape, device=q.device, dtype=q.dtype)
  if value_layout:
    v = baseline_module().pack_values(v)
  return Tensor.custom_kernel(out, q.contiguous(), k.contiguous(), v.contiguous(),
    fxn=partial(program, shape=q.shape, causal=causal, waves=waves, value_layout=value_layout, transpose=transpose))[0]


def install(model, target='all', transpose='shuffle', waves=8):
  """Parent-only opt-in. Install before JIT capture, after existing attention hooks."""
  assert target in ('all', 'vision', 'prefill')
  original = model.attention
  def attend(self, q, k, v, mask=None, causal=False):
    eligible = (mask is None and q.shape[2] >= 128 and q.shape[-1] in (72, 128)
                and (target == 'all' or q.shape[-1] == (72 if target == 'vision' else 128)))
    return attention(q, k, v, causal, waves=waves, transpose=transpose) if eligible else original(q, k, v, mask, causal)
  model.attention = MethodType(attend, model)
