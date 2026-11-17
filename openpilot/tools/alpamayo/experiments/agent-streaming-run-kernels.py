"""Private cache-copy kernels and original-slot selected shuffle-FMA attention."""
from functools import lru_cache,partial
import importlib.util
from pathlib import Path
import sys
from tinygrad import Tensor,dtypes
from tinygrad.uop.ops import UOp,Ops,KernelInfo
from tinygrad.runtime.support.compiler_amd import HIPCompiler

HERE=Path(__file__).resolve().parent
def load(name,path):
  spec=importlib.util.spec_from_file_location(name,path);value=importlib.util.module_from_spec(spec)
  sys.modules[name]=value;spec.loader.exec_module(value);return value
common=load('agent_streaming_kernels_common',HERE/'agent-streaming-run-common.py')

def gather_ast(out,source,slots):
  rows,capacity,width=out.shape[-2],source.shape[-2],source.shape[-1]
  i=UOp.range(out.max_numel(),0);plane,row,col=i//(rows*width),i//width%rows,i%width
  slot=slots.flatten()[row]
  value=source.flatten().index((plane*capacity*width+slot*width+col).valid((slot>=0)&(slot<capacity)))
  return out.flatten()[i].store(value).end(i).sink(arg=KernelInfo(name='agent_streaming_gather'))

def scatter_ast(out,source,slots):
  rows,capacity,width=source.shape[-2],out.shape[-2],source.shape[-1]
  i=UOp.range(source.max_numel(),0);plane,row,col=i//(rows*width),i//width%rows,i%width
  slot=slots.flatten()[row]
  destination=out.flatten().index((plane*capacity*width+slot*width+col).valid((slot>=0)&(slot<capacity)))
  return destination.store(source.flatten()[i]).end(i).sink(arg=KernelInfo(name='agent_streaming_scatter'))

def assemble_ast(out,queries,bank,query_rows,bank_rows):
  rows,width=out.shape[-2:];qrows,brows=queries.shape[-2],bank.shape[-2]
  i=UOp.range(out.max_numel(),0);plane,row,col=i//(rows*width),i//width%rows,i%width
  qi,bi=query_rows.flatten()[row],bank_rows.flatten()[row]
  q=queries.flatten().index((plane*qrows*width+qi*width+col).valid((qi>=0)&(qi<qrows)))
  b=bank.flatten().index((plane*brows*width+bi*width+col).valid((bi>=0)&(bi<brows)))
  value=(qi>=0).where(q,(bi>=0).where(b,0))
  return out.flatten()[i].store(value).end(i).sink(arg=KernelInfo(name='agent_streaming_assemble'))

def gather(source,slots):
  out=Tensor.empty(*source.shape[:-2],slots.shape[0],source.shape[-1],device=source.device,dtype=source.dtype)
  return Tensor.custom_kernel(out,source,slots,fxn=gather_ast)[0]

def scatter(out,source,slots):
  if out.shape[:-2]!=source.shape[:-2] or source.shape[-2]!=slots.shape[0]:raise ValueError('Scatter geometry')
  return Tensor.custom_kernel(out,source,slots,fxn=scatter_ast)[0]

def assemble(queries,bank,query_rows,bank_rows):
  out=Tensor.empty(*queries.shape[:-2],query_rows.shape[0],queries.shape[-1],device=queries.device,dtype=queries.dtype)
  return Tensor.custom_kernel(out,queries,bank,query_rows,bank_rows,fxn=assemble_ast)[0]

@lru_cache(1)
def selected():return common.load('agent-attention-rowsoftmax',common.EXP)

def replace(text,old,new):
  if text.count(old)!=1:raise ValueError('Frozen selected attention source changed: '+old)
  return text.replace(old,new)

@lru_cache(32)
def compiled(heads,kv_heads,rows,keys,width=128):
  if width!=128 or rows<1:raise ValueError('Language attention only')
  text,_=selected().compiled(heads,kv_heads,rows,keys,True,width,8,True,'shuffle-fma')
  text=replace(text,'void agent_attention_rowsoftmax(','void agent_streaming_slot_attention(')
  text=replace(text,'const u16 *v\n','const u16 *v,const int *slots,const int *limits\n')
  text=replace(text,'  int limit=(group+1)*8*16;\n  int end=1?(K<limit?K:limit):K;',
    '''  int last=(group+1)*8*16;
  if(last>limits[1]) last=limits[1];
  int max_slot=last>group*8*16?slots[last-1]:-1;
  int end=max_slot+1;
  if(end>K) end=K;''')
  text=replace(text,'(col<K && (!1 || col<=row))?',
    '(row<R && slots[row]>=0 && col<K && col<limits[0] && col<=slots[row])?')
  text=replace(text,'out[(head*R+row)*W+d*16+lane%16]=tobf(accum[d][i]/row_denom);',
    'out[(head*R+row)*W+d*16+lane%16]=(slots[row]>=0 && row_denom>0.0f)?tobf(accum[d][i]/row_denom):0;')
  return text,HIPCompiler('gfx1200').compile_cached(text)

def program(out,q,k,v,slots,limits):
  batch,heads,rows,width=q.shape;kv_heads,keys=k.shape[1:3]
  witness=q.flatten()[0].load()+k.flatten()[0].load()+v.flatten()[0].load()+slots[0].load()+limits[0].load()
  sink=UOp.sink(out.flatten()[0].store(witness.cast(out.dtype)),
    UOp.special(batch*heads*((rows+127)//128),'gidx0'),UOp.special(256,'lidx0'),
    arg=KernelInfo(name='agent_streaming_slot_attention'))
  text,binary=compiled(heads,kv_heads,rows,keys,width)
  return UOp(Ops.PROGRAM,src=(sink,UOp(Ops.LINEAR,src=tuple(sink.toposort())),UOp(Ops.SOURCE,arg=text),UOp(Ops.BINARY,arg=binary)))

def attention(q,k,v,slots,limits):
  if q.dtype!=dtypes.bfloat16 or q.dtype!=k.dtype or k.dtype!=v.dtype:raise ValueError('BF16 attention required')
  if q.shape[-1]!=128 or k.shape!=v.shape or q.shape[2]!=slots.shape[0] or limits.shape!=(2,):raise ValueError('Slot attention geometry')
  out=Tensor.empty(q.shape,device=q.device,dtype=q.dtype)
  # Preserve the selected full-prefill value layout for every compact shape.
  packed=selected().baseline_module().pack_values(v)
  return Tensor.custom_kernel(out,q.contiguous(),k.contiguous(),packed.contiguous(),slots,limits,fxn=program)[0]
