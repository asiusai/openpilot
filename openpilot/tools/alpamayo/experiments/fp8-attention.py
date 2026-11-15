"""Experimental FP8 attention. Changes Q/K/V and probability precision; opt-in only."""
from functools import lru_cache, partial
import importlib.util
from pathlib import Path

from tinygrad import Tensor, dtypes
from tinygrad.uop.ops import UOp, Ops, KernelInfo
from tinygrad.runtime.support.compiler_amd import HIPCompiler


@lru_cache(None)
def compiled(heads, kv_heads, rows, keys, width, causal, waves=8):
  spec = importlib.util.spec_from_file_location('fp8_base_attention', Path(__file__).with_name('online-attention.py'))
  online = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(online)
  text, _ = online.compiled(heads, kv_heads, rows, keys, causal, waves, width, 32, True, key_padding=4)
  replacements = [
    ('alpamayo_online', 'alpamayo_fp8_attention'),
    ('typedef short s8 __attribute__((ext_vector_type(8)));',
     'typedef int i2 __attribute__((ext_vector_type(2)));\n'
     'DEV void put8(i2 &x, int i, unsigned char value) { x[i/4]=(unsigned)x[i/4]|((unsigned)value<<((i%4)*8)); }\n'
     'DEV unsigned char tofp(float value) { return (unsigned char)__builtin_amdgcn_cvt_pk_fp8_f32(value,value,0,false); }'),
    ('DEV f8 mma(s8 a,s8 b,f8 c) { return __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a,b,c); }',
     'DEV f8 mma(i2 a,i2 b,f8 c) { return __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(a,b,c); }'),
    ('const u16 *q,const u16 *k,const u16 *v',
     'const unsigned char *q,const unsigned char *k,const unsigned char *v,const float *qs,const float *ks,const float *vs'),
    ('short probabilities[', 'unsigned char probabilities['),
    ('short shared_k[', 'unsigned char shared_k['),
    ('short *prob=', 'unsigned char *prob='),
    ('s8 query[D];', 'i2 query[D]={};'),
    ('query[d][i]=qr<R && dim<W?q[(head*R+qr)*W+dim]:0;',
     'put8(query[d],i,qr<R && dim<W?q[(head*R+qr)*W+dim]:0);'),
    ('s8 key;', 'i2 key={};'),
    ('key[i]=shared_k[key_index(sub*16+lane%16,dim)];',
     'put8(key,i,shared_k[key_index(sub*16+lane%16,dim)]);'),
    ('key[i]=(col<K && dim<W)?k[(kh*K+col)*W+dim]:0;',
     'put8(key,i,(col<K && dim<W)?k[(kh*K+col)*W+dim]:0);'),
    ('scores[sub][i]*', 'scores[sub][i]*qs[head*R+(row<R?row:0)]*ks[kh*K+col]*'),
    ('s8 probability;', 'i2 probability={};'),
    ('=tobf(scores[sub][i]);', '=tofp(scores[sub][i]*448.0f);'),
    ('probability[i]=prob[(lane%16)*(16+0)+(lane/16)*8+i];',
     'put8(probability,i,prob[(lane%16)*(16+0)+(lane/16)*8+i]);'),
    ('s8 value;', 'i2 value={};'),
    ('value[i]=shared_v[(sub*16+(lane/16)*8+i)*D*16+d*16+lane%16];',
     'put8(value,i,shared_v[(sub*16+(lane/16)*8+i)*D*16+d*16+lane%16]);'),
    ('value[i]=key<K && d*16+lane%16<W?v[(kh*K+key)*W+d*16+lane%16]:0;',
     'put8(value,i,key<K && d*16+lane%16<W?v[(kh*K+key)*W+d*16+lane%16]:0);'),
    ('tobf(accum[d][i]/denom[i])', 'tobf((accum[d][i]/denom[i])*vs[kh]*(1.0f/448.0f))'),
  ]
  for before, after in replacements:
    assert before in text, before
    text = text.replace(before, after)
  return text, HIPCompiler('gfx1200').compile_cached(text)


def program(out, q, k, v, qs, ks, vs, *, shape, causal, waves):
  batch, heads, rows, width = shape
  text, binary = compiled(heads, k.shape[1], rows, k.shape[2], width, causal, waves)
  sink = UOp.sink(out.flatten()[0].store((q.flatten()[0].load().cast(dtypes.float32)
    + k.flatten()[0].load().cast(dtypes.float32) + v.flatten()[0].load().cast(dtypes.float32)
    + qs.flatten()[0].load() + ks.flatten()[0].load() + vs.flatten()[0].load()).cast(dtypes.bfloat16)),
    UOp.special(batch*heads*((rows+16*waves-1)//(16*waves)), 'gidx0'), UOp.special(waves*32, 'lidx0'),
    arg=KernelInfo(name='alpamayo_fp8_attention'))
  return UOp(Ops.PROGRAM, src=(sink,UOp(Ops.LINEAR,src=tuple(sink.toposort())),
                              UOp(Ops.SOURCE,arg=text),UOp(Ops.BINARY,arg=binary)))


def quantized_inputs(q, k, v):
  result = []
  scales = []
  for value, axes in ((q, (-1,)), (k, (-1,)), (v, (-2,-1))):
    value = value.contiguous().realize()
    scale = (value.float().abs().max(axis=axes, keepdim=True)/448).clamp(min_=1e-12).contiguous().realize()
    result.append((value.float()/scale).clamp(-448,448).cast(dtypes.fp8e4m3).contiguous().realize())
    scales.append(scale)
  return (*result, *scales)


def attention(q, k, v, causal=True, waves=8):
  assert q.dtype == k.dtype == v.dtype == dtypes.bfloat16
  assert q.shape[-1] == k.shape[-1] == v.shape[-1] and q.shape[-1] in (72,128)
  assert waves in (4,8) and q.shape[1] % k.shape[1] == 0
  out = Tensor.empty(q.shape, device=q.device, dtype=q.dtype)
  return Tensor.custom_kernel(out, *quantized_inputs(q,k,v),
    fxn=partial(program,shape=q.shape,causal=causal,waves=waves))[0]
