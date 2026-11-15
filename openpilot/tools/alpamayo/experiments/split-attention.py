"""Experimental parallel key partitions with FP32 online-softmax merging.

This changes reduction order. Compare with independent FP32 attention and the
official model before selecting it for inference.
"""
from functools import lru_cache, partial
import importlib.util
from pathlib import Path

from tinygrad import Tensor, dtypes
from tinygrad.uop.ops import UOp, Ops, KernelInfo
from tinygrad.runtime.support.compiler_amd import HIPCompiler


@lru_cache(None)
def compiled(batch, heads, kv_heads, rows, keys, width, causal, splits, waves, key_padding):
  spec = importlib.util.spec_from_file_location('split_online_source', Path(__file__).with_name('online-attention.py'))
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  text, _ = module.compiled(heads, kv_heads, rows, keys, causal, waves=waves, width=width,
                            block_keys=32, stage_kv=True, key_padding=key_padding)
  old = 'void alpamayo_online(u16 *out,const u16 *q'
  assert old in text
  text = text.replace(old, 'void alpamayo_split(float *out,float *statistics,const u16 *q')
  old = 'int group=block%'
  assert old in text
  text = text.replace(old, f'int split=block%{splits}; block/={splits};\n  int group=block%')
  old = 'for(int start=0;start<end;start+='
  assert old in text
  chunk = ((keys+splits*32-1)//(splits*32))*32
  text = text.replace(old, f'int begin=split*{chunk};\n  end=end<begin+{chunk}?end:begin+{chunk};\n  for(int start=begin;start<end;start+=')
  old = 'if(row<R && d*16+lane%16<W) out[(head*R+row)*W+d*16+lane%16]=tobf(accum[d][i]/denom[i]);'
  assert old in text
  text = text.replace(old, f'''if(row<R && d*16+lane%16<W) {{
      out[((split*{batch*heads}+head)*R+row)*W+d*16+lane%16]=accum[d][i];
      if(d==0 && lane%16==0) {{
        statistics[((split*{batch*heads}+head)*R+row)*2]=maxima[i];
        statistics[((split*{batch*heads}+head)*R+row)*2+1]=denom[i];
      }}
    }}''')
  return text, HIPCompiler('gfx1200').compile_cached(text)


def program(out, statistics, q, k, v, *, shape, causal, splits, waves, key_padding):
  batch, heads, rows, width = shape
  sink = UOp.sink(out.flatten()[0].store(q.flatten()[0].load().float()+k.flatten()[0].load().float()+v.flatten()[0].load().float()),
                  statistics.flatten()[0].store(q.flatten()[0].load().float()),
                  UOp.special(batch*heads*((rows+16*waves-1)//(16*waves))*splits,'gidx0'),
                  UOp.special(waves*32,'lidx0'),arg=KernelInfo(name='alpamayo_split'))
  text, binary = compiled(batch,heads,k.shape[1],rows,k.shape[2],width,causal,splits,waves,key_padding)
  return UOp(Ops.PROGRAM,src=(sink,UOp(Ops.LINEAR,src=tuple(sink.toposort())),
                              UOp(Ops.SOURCE,arg=text),UOp(Ops.BINARY,arg=binary)))


@lru_cache(None)
def compiled_merge(head_rows,width,splits):
  text=r'''
extern "C" __attribute__((device)) unsigned long __ockl_get_local_id(unsigned);
extern "C" __attribute__((device)) unsigned long __ockl_get_group_id(unsigned);
extern "C" __attribute__((global,amdgpu_flat_work_group_size(256,256)))
void alpamayo_split_merge(unsigned short *out,const float *values,const float *statistics) {
  int index=__ockl_get_group_id(0)*256+__ockl_get_local_id(0);
  if(index>=HEADROWS*WIDTH) return;
  int row=index/WIDTH;
  float maximum=-__builtin_inff();
  for(int split=0;split<SPLITS;split++) maximum=__builtin_fmaxf(maximum,statistics[(split*HEADROWS+row)*2]);
  float numerator=0,denominator=0;
  for(int split=0;split<SPLITS;split++) {
    float d=statistics[(split*HEADROWS+row)*2+1];
    if(d>0) {
      float w=__builtin_exp2f((statistics[(split*HEADROWS+row)*2]-maximum)*1.4426950408889634f);
      numerator+=values[split*HEADROWS*WIDTH+index]*w;
      denominator+=d*w;
    }
  }
  float value=numerator/denominator;
  unsigned b=__builtin_bit_cast(unsigned,value);
  out[index]=(b+0x7fff+((b>>16)&1))>>16;
}
'''
  text=f'#pragma clang fp contract(off)\n#define HEADROWS {head_rows}\n#define WIDTH {width}\n#define SPLITS {splits}\n'+text
  return text,HIPCompiler('gfx1200').compile_cached(text)


def merge_program(out,values,statistics,*,splits):
  sink=UOp.sink(out.flatten()[0].store((values.flatten()[0].load()+statistics.flatten()[0].load()).cast(dtypes.bfloat16)),
    UOp.special((out.numel()+255)//256,'gidx0'),UOp.special(256,'lidx0'),arg=KernelInfo(name='alpamayo_split_merge'))
  text,binary=compiled_merge(out.numel()//out.shape[-1],out.shape[-1],splits)
  return UOp(Ops.PROGRAM,src=(sink,UOp(Ops.LINEAR,src=tuple(sink.toposort())),UOp(Ops.SOURCE,arg=text),UOp(Ops.BINARY,arg=binary)))


def attention(q,k,v,causal=True,splits=4,waves=8,key_padding=4,fused_merge=True):
  assert q.dtype == k.dtype == v.dtype == dtypes.bfloat16
  assert q.shape[-1] == k.shape[-1] == v.shape[-1] and q.shape[-1] in (72,128)
  assert splits in (1,2,4,8) and waves in (4,8) and key_padding in (0,4)
  values = Tensor.empty((splits,*q.shape),device=q.device,dtype=dtypes.float32)
  statistics = Tensor.empty((splits,*q.shape[:-1],2),device=q.device,dtype=dtypes.float32)
  values, statistics, *_ = Tensor.custom_kernel(values,statistics,q.contiguous(),k.contiguous(),v.contiguous(),
    fxn=partial(program,shape=q.shape,causal=causal,splits=splits,waves=waves,key_padding=key_padding))
  values, statistics = values.realize(), statistics.realize()
  if fused_merge:
    output=Tensor.empty(q.shape,device=q.device,dtype=q.dtype)
    return Tensor.custom_kernel(output,values,statistics,fxn=partial(merge_program,splits=splits))[0]
  maxima, denominators = statistics[...,0:1], statistics[...,1:2]
  global_maximum = maxima.max(axis=0,keepdim=True).contiguous().realize()
  # A causal partition can have no permitted keys. Its contribution is zero.
  weights = (denominators>0).where((maxima-global_maximum).exp(),0).contiguous().realize()
  denominator = (weights*denominators).sum(axis=0).contiguous().realize()
  return ((values*weights).sum(axis=0)/denominator).cast(dtypes.bfloat16)
