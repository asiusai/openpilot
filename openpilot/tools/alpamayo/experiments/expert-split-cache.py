"""Read packed history and new action keys separately, preserving WMMA order."""
from functools import partial
import importlib.util
from pathlib import Path
from types import MethodType
from tinygrad import Device,Tensor,dtypes
from tinygrad.codegen.opt import Opt,OptOps
from tinygrad.codegen.opt.postrange import Scheduler
from tinygrad.helpers import round_up
from tinygrad.uop.ops import AxisType,KernelInfo,Ops,UOp
from amd_attention import _pack,_softmax


def matmul(out,a,old,new,shape,capacity,scaled=False):
  batch,heads,kv_heads,rows,columns,width=shape
  pr,pc,pw=(round_up(n,16) for n in (rows,columns,width))
  head=UOp.range(batch*heads,0)
  m,n,k=UOp.range(round_up(rows,64),1),UOp.range(round_up(columns,64),2),UOp.range(pw,3)
  def index(h,row,length,inner,dim):
    return ((h*(length//16)+row//16)*(inner//16)+dim//16)*256+(dim%16//4)*64+(row%16)*4+dim%4
  aa=a.flatten().index(index(head,m,pr,pw,k).valid(m<rows))
  kh=(head//heads)*kv_heads+(head%heads)//(heads//kv_heads)
  if scaled:
    previous=old.flatten().index(index(kh,n,capacity,pw,k).valid(n<capacity))
    current=new.flatten().index(index(kh,n-capacity,pc-capacity,pw,k).valid((n>=capacity)&(n<columns)))
    bb=(n<capacity).where(previous,current)
  else:
    previous=old.flatten().index(index(kh,n,pc,capacity,k).valid((n<columns)&(k<capacity)))
    current=new.flatten().index(index(kh,n,pc,pw-capacity,k-capacity).valid((n<columns)&(k>=capacity)&(k<width)))
    bb=(k<capacity).where(previous,current)
  value=(aa*bb).cast(dtypes.float32).reduce(k,arg=Ops.ADD).cast(out.dtype)
  if scaled:value=value*UOp.const(width**-.5,out.dtype)
  store=out.flatten().index(((head*rows+m)*columns+n).valid((m<rows)&(n<columns))).store(value)
  schedule=Scheduler(store.end(head,m,n).sink(arg=KernelInfo()),Device[Device.DEFAULT].renderer)
  schedule.convert_loop_to_global()
  ranges=schedule.apply_opt(Opt(OptOps.TC,0,(-1,0,1)))
  for i in (1,0):schedule.apply_opt(Opt(OptOps.SPLIT,schedule.rngs.index(ranges[i]),(4,AxisType.UPCAST)))
  return schedule.get_optimized_ast(name_override='alpamayo_split_scores' if scaled else 'alpamayo_split_values')


def attention(q,k,v,history,mask,capacity):
  batch,heads,queries,width=q.shape
  keys=capacity+k.shape[2];kv_heads=k.shape[1]
  assert queries==64 and capacity%16==0 and width%16==0
  scores=Tensor.empty(batch,heads,queries,keys,dtype=q.dtype,device=q.device)
  scores=Tensor.custom_kernel(scores,_pack(q),history[0],_pack(k),
    fxn=partial(matmul,shape=(batch,heads,kv_heads,queries,keys,width),capacity=capacity,scaled=True))[0]
  scores=(scores.float()+mask).cast(q.dtype).contiguous()
  probs=Tensor.empty(batch,heads,round_up(queries,16),round_up(keys,16),dtype=q.dtype,device=q.device)
  probs=Tensor.custom_kernel(probs,scores,fxn=_softmax)[0]
  output=Tensor.empty(q.shape,dtype=q.dtype,device=q.device)
  return Tensor.custom_kernel(output,probs,history[1],_pack(v.transpose(-2,-1)),
    fxn=partial(matmul,shape=(batch,heads,kv_heads,queries,width,keys),capacity=capacity))[0]


def install(model):
  assert not model.trajectory_jits
  original_layer,original_trajectory=model.text_layer,model._trajectory
  state={}
  def layer(self,x,cos,sin,prefix,cache=None,mask=None,cache_pos=None):
    if not prefix.startswith('expert.') or cache is None or cache_pos is not None:
      return original_layer(x,cos,sin,prefix,cache,mask,cache_pos)
    assert x.shape[1]==64 and mask is not None
    if prefix not in state:
      state[prefix]=(_pack(cache[0]).realize(),_pack(cache[1].transpose(-2,-1)).realize())
    h=self.norm(x,f'{prefix}.input_layernorm',rms=True)
    quantized=self.quantize(h) if f'{prefix}.self_attn.q_proj.weight' in self.weight_scales else None
    q,k,v=(self.linear(h,f'{prefix}.self_attn.{name}_proj',quantized,heads=heads).reshape(1,h.shape[1],heads,128).transpose(1,2)
           for name,heads in (('q',16),('k',8),('v',8)))
    q,k=self.rope(q,k,cos,sin)
    h=attention(q,k,v,state[prefix],mask,cache[0].shape[2]).transpose(1,2).reshape(1,x.shape[1],2048)
    x=x+self.linear(h,f'{prefix}.self_attn.o_proj')
    h=self.linear(self.norm(x,f'{prefix}.post_attention_layernorm',rms=True),f'{prefix}.mlp.fc1').relu().square()
    return (x+self.linear(h,f'{prefix}.mlp.fc2')).realize()
  def trajectory(self,*args,**kwargs):
    assert not state
    try:return original_trajectory(*args,**kwargs)
    finally:state.clear()
  model.text_layer=MethodType(layer,model)
  model._trajectory=MethodType(trajectory,model)
