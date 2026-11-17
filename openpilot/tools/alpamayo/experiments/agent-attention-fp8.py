"""Numerical candidate: FP8 QK, BF16 probabilities/PV, fixed head scales.

Unlike fp8-attention.py, this converts Q/K in one combined pass without any
runtime reductions. The attention kernel uses vector loads and retains BF16 P/V.
Scales must be fixed, calibrated positive powers of two, separately per head.
This is approximate attention, never an exact replacement for BF16 inputs.
"""
from functools import lru_cache, partial
import importlib.util
from pathlib import Path
from types import MethodType

import numpy as np
from tinygrad import Tensor, dtypes
from tinygrad.runtime.support.compiler_amd import HIPCompiler
from tinygrad.uop.ops import KernelInfo, Ops, UOp


@lru_cache(None)
def baseline_module():
  spec = importlib.util.spec_from_file_location('agent_attention_fp8_baseline', Path(__file__).with_name('online-attention.py'))
  result = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(result)
  return result


def replace_once(text, old, new):
  assert text.count(old) == 1, (old, text.count(old))
  return text.replace(old, new)


def validate_scales(values, heads):
  values = np.asarray(values, dtype=np.float32)
  assert values.shape == (heads,) and np.isfinite(values).all() and (values > 0).all()
  mantissa, exponent = np.frexp(values)
  assert (mantissa == .5).all() and (exponent >= -20).all() and (exponent <= 20).all(), values
  return values


@lru_cache(None)
def compiled(heads, kv_heads, rows, keys, width, causal, value_layout):
  assert width in (72, 128)
  text, _ = baseline_module().compiled(heads, kv_heads, rows, keys, causal, waves=8, width=width,
    block_keys=32, stage_kv=True, key_padding=8, kv_load=1, value_layout=value_layout, probability_mma=True)
  text = replace_once(text, 'void alpamayo_online(u16 *out,const u16 *q,const u16 *k,const u16 *v',
    'void agent_attention_fp8_qk(u16 *out,const unsigned char *q,const unsigned char *k,const u16 *v,const float *qs,const float *ks')
  marker = 'DEV int key_index(int row,int dim) {'
  text = replace_once(text, marker, '''typedef int i2 __attribute__((ext_vector_type(2)));
DEV float quant_input(u16 bits,float inverse) {
  return __builtin_fmaxf(-448.0f,__builtin_fminf(448.0f,frombf(bits)*inverse));
}
DEV i2 quant_eight(u8 values,float inverse) {
  i2 result;
  for(int word=0;word<2;word++) {
    unsigned low=__builtin_amdgcn_cvt_pk_fp8_f32(
      quant_input(values[word*4],inverse),quant_input(values[word*4+1],inverse),0,false);
    result[word]=__builtin_amdgcn_cvt_pk_fp8_f32(
      quant_input(values[word*4+2],inverse),quant_input(values[word*4+3],inverse),low,true);
  }
  return result;
}
DEV f8 mma_fp8(i2 a,i2 b,f8 c) { return __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(a,b,c); }
DEV i2 load_fp8_eight(const unsigned char *source,int head,int row,int dim) {
  if(row>=KEYS_VALUE || dim>=WIDTH_VALUE) return (i2){};
  return *(const volatile i2*)(source+(head*KEYS_VALUE+row)*WIDTH_VALUE+dim);
}
'''.replace('KEYS_VALUE', str(keys)).replace('WIDTH_VALUE', str(width)) + marker)
  text = replace_once(text, '  int qr=tile*16+lane%16;', '''  int qr=tile*16+lane%16;
  // Immutable per-head scales. No per-token scale loads occur in the key loop.
  float q_scale=qs[head], k_scale=ks[kh];
  float score_scale=(q_scale*k_scale)*SCALE_VALUE;
'''.replace('SCALE_VALUE', repr(width**-.5)+'f'))
  text = replace_once(text, '__attribute__((shared,aligned(16))) short shared_k[32*(D*16+8)],shared_v[32*D*16];',
    '__attribute__((shared,aligned(16))) unsigned char shared_k[32*(D*16+8)];\n'
    '  __attribute__((shared,aligned(16))) short shared_v[32*D*16];')
  text = replace_once(text, '''  s8 query[D];
  for(int d=0;d<D;d++) for(int i=0;i<8;i++) {
    int dim=d*16+(lane/16)*8+i;
    query[d][i]=qr<R && dim<W?q[(head*R+qr)*W+dim]:0;
  }''', '''  i2 query[D];
  for(int d=0;d<D;d++) {
    int dim=d*16+(lane/16)*8;
    query[d]=qr<R && dim<W?*(const volatile i2*)(q+(head*R+qr)*W+dim):(i2){};
  }''')
  if value_layout:
    text = replace_once(text, 'u8 loaded_k=load_eight(k,kh,start+row,dim);', 'i2 loaded_k=load_fp8_eight(k,kh,start+row,dim);')
  else:
    text = replace_once(text, 'u8 loaded_k=load_eight(k,kh,start+row,dim), loaded_v=load_eight(v,kh,start+row,dim);',
      'i2 loaded_k=load_fp8_eight(k,kh,start+row,dim); u8 loaded_v=load_eight(v,kh,start+row,dim);')
  text = replace_once(text,
    '      for(int i=0;i<8;i++) { shared_k[key_index(row,dim+i)]=loaded_k[i]; shared_v[index+i]=loaded_v[i]; }',
    '''      *(i2*)(shared_k+key_index(row,dim))=loaded_k;
      *(u8*)(shared_v+index)=loaded_v;''')
  start = text.index('      s8 key;')
  end = text.index('      scores[sub]=mma(query[d],key,scores[sub]);', start)
  text = text[:start] + '''      i2 key=*(const i2*)(shared_k+key_index(sub*16+lane%16,d*16+(lane/16)*8));
''' + text[end:]
  text = replace_once(text, 'scores[sub]=mma(query[d],key,scores[sub]);', 'scores[sub]=mma_fp8(query[d],key,scores[sub]);')
  text = replace_once(text, f'scores[sub][i]*{repr(width**-.5)}f', 'scores[sub][i]*score_scale')
  return text, HIPCompiler('gfx1200').compile_cached(text)


def program(out, q, k, v, qs, ks, *, causal, value_layout):
  batch, heads, rows, width = q.shape
  kv_heads, keys = k.shape[1:3]
  text, binary = compiled(heads, kv_heads, rows, keys, width, causal, value_layout)
  sink = UOp.sink(out.flatten()[0].store((q.flatten()[0].load()+k.flatten()[0].load()+v.flatten()[0].load()
    + qs.flatten()[0].load()+ks.flatten()[0].load()).cast(dtypes.bfloat16)),
    UOp.special(batch*heads*((rows+127)//128), 'gidx0'), UOp.special(256, 'lidx0'),
    arg=KernelInfo(name='agent_attention_fp8_qk'))
  return UOp(Ops.PROGRAM, src=(sink, UOp(Ops.LINEAR, src=tuple(sink.toposort())),
    UOp(Ops.SOURCE, arg=text), UOp(Ops.BINARY, arg=binary)))


def attention(q, k, v, qs, ks, causal=False):
  assert q.dtype == k.dtype == v.dtype == dtypes.bfloat16 and k.shape == v.shape
  assert q.shape[0] == k.shape[0] and q.shape[-1] == k.shape[-1] and q.shape[-1] in (72, 128)
  assert q.shape[1] % k.shape[1] == 0
  assert qs.shape == q.shape[:2] and ks.shape == k.shape[:2] and qs.dtype == ks.dtype == dtypes.float32
  value_layout = q.shape[2] >= 1024
  if value_layout:
    v = baseline_module().pack_values(v)
  q, k = quantized_inputs(q, k, qs, ks)
  out = Tensor.empty(q.shape, device=q.device, dtype=dtypes.bfloat16)
  return Tensor.custom_kernel(out, q.contiguous(), k.contiguous(), v.contiguous(), qs.contiguous(), ks.contiguous(),
    fxn=partial(program, causal=causal, value_layout=value_layout))[0]


def trace_layout(keys, width):
  """Bounded diagnostic storage for one query, never used during timing."""
  tiles, depth = (keys+31)//32, (width+15)//16
  cursor, layout = 0, {}
  for name, shape in [('partial', (tiles, depth, 32)), ('scaled', (tiles, 32)),
      ('probability', (tiles, 32)), ('state', (tiles, 3)), ('accumulator', (tiles, width))]:
    size = int(np.prod(shape))
    layout[name] = dict(offset=cursor, shape=shape, size=size)
    cursor += size
  return layout, cursor


@lru_cache(None)
def trace_compiled(heads, kv_heads, rows, keys, width, causal, value_layout, variant, target_head, target_row):
  """Diagnostic source taps, with the original arithmetic statements intact.

  Extra stores can affect compiler scheduling. Compare this kernel's final
  output with its untapped counterpart before interpreting any intermediate.
  This diagnostic does not change compiled(), attention(), or model hooks.
  """
  assert variant in ('bf16', 'fp8') and 0 <= target_head < heads and 0 <= target_row < rows
  if variant == 'fp8':
    source, _ = compiled(heads, kv_heads, rows, keys, width, causal, value_layout)
    name, operation = 'agent_attention_fp8_trace', 'mma_fp8'
    source = replace_once(source, 'void agent_attention_fp8_qk(', f'void {name}(')
  else:
    source, _ = baseline_module().compiled(heads, kv_heads, rows, keys, causal, waves=8, width=width,
      block_keys=32, stage_kv=True, key_padding=4, kv_load=1, value_layout=value_layout, probability_mma=True)
    name, operation = 'agent_attention_bf16_trace', 'mma'
    source = replace_once(source, 'void alpamayo_online(', f'void {name}(')
  source = replace_once(source, f'void {name}(u16 *out,', f'void {name}(u16 *out,float *trace,')
  layout, size = trace_layout(keys, width)
  condition = f'head=={target_head} && tile*16+(lane/16)*8+i=={target_row}'
  statement = f'      scores[sub]={operation}(query[d],key,scores[sub]);'
  source = replace_once(source, statement, statement + f'''
      for(int i=0;i<8;i++) if({condition})
        trace[{layout['partial']['offset']}+(start/32*D+d)*32+sub*16+lane%16]=scores[sub][i];''')
  statement = '        localmax=__builtin_fmaxf(localmax,scores[sub][i]);'
  source = replace_once(source, statement, f'''        if({condition})
          trace[{layout['scaled']['offset']}+start+sub*16+lane%16]=scores[sub][i];
''' + statement)
  statement = '        sum+=scores[sub][i];'
  source = replace_once(source, statement, f'''        if({condition})
          trace[{layout['probability']['offset']}+start+sub*16+lane%16]=scores[sub][i];
''' + statement)
  statement = '      denom[i]=denom[i]*scale+total(sum); maxima[i]=next;'
  source = replace_once(source, statement, statement + f'''
      if({condition} && lane%16==0) {{
        trace[{layout['state']['offset']}+start/32*3]=next;
        trace[{layout['state']['offset']}+start/32*3+1]=scale;
        trace[{layout['state']['offset']}+start/32*3+2]=denom[i];
      }}''')
  statement = '      accum[d]=mma(probability,value,accum[d]);'
  # The last fragment records the accumulator after both PV operations.
  source = replace_once(source, statement, statement + f'''
      if(sub==1) for(int i=0;i<8;i++) if({condition} && d*16+lane%16<W)
        trace[{layout['accumulator']['offset']}+start/32*W+d*16+lane%16]=accum[d][i];''')
  return source, HIPCompiler('gfx1200').compile_cached(source), name, layout, size


def trace_program(out, trace, q, k, v, *scales, causal, variant, target_head, target_row):
  batch, heads, rows, width = q.shape
  kv_heads, keys = k.shape[1:3]
  source, binary, name, _, _ = trace_compiled(heads, kv_heads, rows, keys, width, causal,
    rows >= 1024, variant, target_head, target_row)
  dependency = q.flatten()[0].load()+k.flatten()[0].load()+v.flatten()[0].load()
  for scale in scales:
    dependency += scale.flatten()[0].load()
  sink = UOp.sink(out.flatten()[0].store(dependency.cast(dtypes.bfloat16)),
    trace.flatten()[0].store(dependency.cast(dtypes.float32)),
    UOp.special(batch*heads*((rows+127)//128), 'gidx0'), UOp.special(256, 'lidx0'), arg=KernelInfo(name=name))
  return UOp(Ops.PROGRAM, src=(sink, UOp(Ops.LINEAR, src=tuple(sink.toposort())),
    UOp(Ops.SOURCE, arg=source), UOp(Ops.BINARY, arg=binary)))


def attention_trace(q, k, v, qs, ks, *, causal, variant, target_head, target_row):
  """Parent-only intermediates for a single query; exclude from all timing."""
  assert q.dtype == k.dtype == v.dtype == dtypes.bfloat16 and k.shape == v.shape
  assert variant in ('bf16', 'fp8')
  out = Tensor.empty(q.shape, device=q.device, dtype=dtypes.bfloat16)
  _, size = trace_layout(k.shape[2], q.shape[-1])
  trace = Tensor.full((size,), float('nan'), device=q.device, dtype=dtypes.float32).contiguous()
  if q.shape[2] >= 1024:
    v = baseline_module().pack_values(v)
  scales = ()
  if variant == 'fp8':
    q, k = quantized_inputs(q, k, qs, ks)
    scales = (qs.contiguous(), ks.contiguous())
  return Tensor.custom_kernel(out, trace, q.contiguous(), k.contiguous(), v.contiguous(), *scales,
    fxn=partial(trace_program, causal=causal, variant=variant, target_head=target_head, target_row=target_row))[:2]


@lru_cache(None)
def quant_compiled(q_shape, k_shape):
  batch, heads, rows, width = q_shape
  _, kv_heads, keys, _ = k_shape
  q_stride, k_stride = rows*width, keys*width
  q_blocks, k_blocks = (q_stride+2047)//2048, (k_stride+2047)//2048
  text, _ = compiled(1, 1, 17, 33, 72, False, False)
  text = text[:text.index('extern "C" __attribute__((global')]
  text += '''extern "C" __attribute__((global,amdgpu_flat_work_group_size(1,256)))
void agent_attention_fp8_quantize(unsigned char *outq,unsigned char *outk,
    const u16 *q,const u16 *k,const float *qs,const float *ks) {
  int group=__ockl_get_group_id(0), thread=__ockl_get_local_id(0);
  if(group<Q_GROUPS) {
    int head=group/Q_BLOCKS, index=((group%Q_BLOCKS)*256+thread)*8;
    if(index<Q_STRIDE) *(i2*)(outq+head*Q_STRIDE+index)=quant_eight(*(const u8*)(q+head*Q_STRIDE+index),1.0f/qs[head]);
  } else {
    int local=group-Q_GROUPS, head=local/K_BLOCKS, index=((local%K_BLOCKS)*256+thread)*8;
    if(index<K_STRIDE) *(i2*)(outk+head*K_STRIDE+index)=quant_eight(*(const u8*)(k+head*K_STRIDE+index),1.0f/ks[head]);
  }
}
'''
  for name, value in dict(Q_GROUPS=batch*heads*q_blocks, Q_BLOCKS=q_blocks, K_BLOCKS=k_blocks,
      Q_STRIDE=q_stride, K_STRIDE=k_stride).items():
    text = text.replace(name, str(value))
  return text, HIPCompiler('gfx1200').compile_cached(text), batch*(heads*q_blocks+kv_heads*k_blocks)


def quant_program(outq, outk, q, k, qs, ks):
  text, binary, groups = quant_compiled(q.shape, k.shape)
  sink = UOp.sink(
    outq.flatten()[0].store((q.flatten()[0].load()+qs.flatten()[0].load()).cast(dtypes.uint8)),
    outk.flatten()[0].store((k.flatten()[0].load()+ks.flatten()[0].load()).cast(dtypes.uint8)),
    UOp.special(groups, 'gidx0'), UOp.special(256, 'lidx0'), arg=KernelInfo(name='agent_attention_fp8_quantize'))
  return UOp(Ops.PROGRAM, src=(sink, UOp(Ops.LINEAR, src=tuple(sink.toposort())),
    UOp(Ops.SOURCE, arg=text), UOp(Ops.BINARY, arg=binary)))


def quantized_inputs(q, k, qs, ks):
  outq, outk = (Tensor.empty(x.shape, device=x.device, dtype=dtypes.uint8) for x in (q, k))
  return Tensor.custom_kernel(outq, outk, q.contiguous(), k.contiguous(), qs.contiguous(), ks.contiguous(), fxn=quant_program)[:2]


@lru_cache(None)
def probe_compiled(rows, width):
  text, _ = compiled(1, 1, 17, 33, 72, False, False)
  text = text[:text.index('extern "C" __attribute__((global')]
  text += '''extern "C" __attribute__((global,amdgpu_flat_work_group_size(1,256)))
void agent_attention_fp8_conversion(unsigned char *out,const u16 *x,const float *scale) {
  int index=__ockl_get_group_id(0)*256+__ockl_get_local_id(0);
  if(index<COUNT) *(i2*)(out+index*8)=quant_eight(*(const u8*)(x+index*8),1.0f/scale[0]);
}
'''.replace('COUNT', str(rows*width//8))
  return text, HIPCompiler('gfx1200').compile_cached(text)


def probe_program(out, x, scale):
  text, binary = probe_compiled(*x.shape)
  sink = UOp.sink(out.flatten()[0].store((x.flatten()[0].load()+scale.flatten()[0].load()).cast(dtypes.uint8)),
    UOp.special((x.numel()//8+255)//256, 'gidx0'), UOp.special(256, 'lidx0'),
    arg=KernelInfo(name='agent_attention_fp8_conversion'))
  return UOp(Ops.PROGRAM, src=(sink, UOp(Ops.LINEAR, src=tuple(sink.toposort())),
    UOp(Ops.SOURCE, arg=text), UOp(Ops.BINARY, arg=binary)))


def conversion_probe(x, scale):
  assert len(x.shape) == 2 and x.numel() % 8 == 0 and x.dtype == dtypes.bfloat16 and scale.shape == (1,)
  out = Tensor.empty(x.shape, device=x.device, dtype=dtypes.uint8)
  return Tensor.custom_kernel(out, x.contiguous(), scale, fxn=probe_program)[0]


def install_capture(model):
  """Parent-only baseline calibration. Return head maxima updated on JIT replay.

  Read these tensors only after the calibration run. This observer adds work;
  its timings are not candidate performance measurements.
  """
  original_attention, original_layer = model.attention, model.text_layer
  current, captured = [], {}
  def layer(self, x, cos, sin, prefix, *args, **kwargs):
    current.append(prefix)
    try:
      return original_layer(x, cos, sin, prefix, *args, **kwargs)
    finally:
      assert current.pop() == prefix
  def attend(self, q, k, v, mask=None, causal=False):
    if (current and current[-1].startswith('vlm.model.language_model.layers.') and causal and mask is None
        and q.shape[2] >= 128 and q.shape[-1] == 128):
      name = current[-1]
      if name not in captured:
        captured[name] = {label: Tensor.zeros(x.shape[1], device=x.device, dtype=dtypes.float32).contiguous().realize()
          for label, x in (('q', q), ('k', k))}
      for label, x in (('q', q), ('k', k)):
        target = captured[name][label]
        target.assign(target.maximum(x.float().abs().max(axis=(0, 2, 3)))).realize()
    return original_attention(q, k, v, mask, causal)
  model.text_layer, model.attention = MethodType(layer, model), MethodType(attend, model)
  return captured


def captured_scales(captured, margin=1.25):
  assert margin >= 1 and captured
  scales, maxima = {}, {}
  for name, values in captured.items():
    maxima[name] = {label: tensor.numpy() for label, tensor in values.items()}
    assert all(np.isfinite(x).all() and (x >= 0).all() for x in maxima[name].values())
    scales[name] = {label: np.exp2(np.ceil(np.log2(np.maximum(x*margin/448, 2.**-20)))).astype(np.float32).tolist()
      for label, x in maxima[name].items()}
  return dict(scales=scales, maxima={name: {label: x.tolist() for label, x in values.items()} for name, values in maxima.items()},
    margin=margin, calibrated=True, quality_validated=False)


def install_prefill(model, calibrated_scales):
  """Opt-in after parent kernel/route checks. Require every language layer.

  calibrated_scales maps the complete text-layer prefix to {q:16, k:8}
  positive power-of-two dequantization scales. It contains no runtime amax.
  """
  layers = model.config['vlm_config']['text_config']['num_hidden_layers']
  expected = {f'vlm.model.language_model.layers.{i}' for i in range(layers)}
  assert set(calibrated_scales) == expected
  validated = {name: (validate_scales(row['q'], 16), validate_scales(row['k'], 8))
    for name, row in calibrated_scales.items()}
  tensors = {}
  current = []
  original_attention, original_layer = model.attention, model.text_layer
  def layer(self, x, cos, sin, prefix, *args, **kwargs):
    current.append(prefix)
    try:
      return original_layer(x, cos, sin, prefix, *args, **kwargs)
    finally:
      assert current.pop() == prefix
  def attend(self, q, k, v, mask=None, causal=False):
    eligible = current and current[-1] in validated and causal and mask is None and q.shape[2] >= 128 and q.shape[-1] == 128
    if not eligible:
      return original_attention(q, k, v, mask, causal)
    name = current[-1]
    assert q.shape[0] == 1
    if (name, q.device) not in tensors:
      tensors[name, q.device] = tuple(Tensor(x[None], device=q.device).realize() for x in validated[name])
    return attention(q, k, v, *tensors[name, q.device], causal=True)
  model.text_layer, model.attention = MethodType(layer, model), MethodType(attend, model)
