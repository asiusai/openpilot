"""One GPU-resident pair of coordinates per serial block. Unchanged greedy decisions every six tokens."""
from functools import partial
from tinygrad import Tensor,dtypes
from tinygrad.uop.ops import UOp,KernelInfo
from openpilot.selfdrive.alpamayo import amd_attention as att
from .serial_attention import direct_scores,direct_values


def coords_kernel(out,pos,step):
  return UOp.group(out.index(0).store(pos.cast(dtypes.int32)),out.index(1).store(step.cast(dtypes.int32))).sink(arg=KernelInfo(name='serial_block_coordinates',opts_to_apply=()))


def store_kernel(cache,k,v,ctrl,*,delta):
  # Same BF16 K,V already rounded by the selected projections/RoPE. Global guards
  # on the caller's variables prove 0 <= position + delta < capacity.
  _,_,heads,capacity,width=cache.shape;n=heads*width;i=UOp.range(n,0)
  dst=((i//width)*capacity+ctrl.index(0)+delta)*width+i%width
  return UOp.group(cache.flatten().index(dst).store(k.flatten().index(i)),
                   cache.flatten().index(dst+n*capacity).store(v.flatten().index(i))).end(i).sink(arg=KernelInfo(name='serial_kv_write_from_coordinates'))


def cache_attention(q,cache,k,v,ctrl,delta,mask):
  # Both reads use AFTER(the actual write), preserving read-after-write/WAW scheduling.
  stored=Tensor.custom_kernel(cache,k.contiguous(),v.contiguous(),ctrl,fxn=partial(store_kernel,delta=delta))[0]
  s=Tensor.custom_kernel(Tensor.empty(1,16,1,2304,device=q.device,dtype=q.dtype),q.contiguous(),stored,mask,fxn=direct_scores)[0]
  p=Tensor.custom_kernel(Tensor.empty(s.shape,device=q.device,dtype=q.dtype),s,fxn=partial(att._softmax,packed=False))[0]
  partials=Tensor.custom_kernel(Tensor.empty(1,16,18,128,device=q.device,dtype=dtypes.float32),p,stored,fxn=direct_values)[0]
  return partials.sum(2,keepdim=True).cast(q.dtype)


def block_six(m,logits,cos,sin,pos,step):
  assert m.amd and not m.images and not m.verifying and not m.trajectory_active and m.prefill_prefix is None
  assert logits.shape==(1,1,2112) and cos.shape==sin.shape==(1,1,64,128) and len(m.cache)==28
  ctrl=Tensor.custom_kernel(Tensor.empty(2,device=logits.device,dtype=dtypes.int32),Tensor(pos),Tensor(step),fxn=coords_kernel)[0].realize()
  idx=ctrl[1]+Tensor.arange(6);coss=cos.reshape(64,128)[idx].contiguous().realize();sins=sin.reshape(64,128)[idx].contiguous().realize();tokens=[]
  for j in range(6):
    # Exactly the original selected head and first-on-tie argmax.
    token=m.output_token_mapping[logits.reshape(-1).argmax()].realize();tokens.append(token.float().reshape(1).realize())
    x=m.weights['vlm.model.language_model.embed_tokens.weight'][token].reshape(1,1,-1)
    c,s=(a[j].reshape(1,1,1,128) for a in (coss,sins));mask=(Tensor.arange(2304)>ctrl[0]+j).where(-float('inf'),0.).float().contiguous().realize()
    for i,cache in enumerate(m.cache):
      x=m.text_layer(x,c,s,f'vlm.model.language_model.layers.{i}',cache,mask,serial_position=(ctrl,j))
    logits=m.linear(m.norm(x,'vlm.model.language_model.norm',rms=True),'vlm.lm_head').realize()
  return logits,Tensor.cat(*tokens).realize()
