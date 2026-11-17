"""Connected, token-aligned local joint objective. No optimizer or device selection.

Targets retain h12 -> h16, post-RoPE K12/V12, and expert h12 -> h16 meanings.
There is deliberately no final-logit, velocity, or trajectory loss without its downstream graph.
"""
import math


def linear(x,weights,name):return x@weights[name+'.weight'].T
def norm(x,weights,name):return x*(x.square().mean(-1,keepdim=True)+1e-5).rsqrt()*weights[name+'.weight']


def qkv(x,cos,sin,weights,head_dim):
  h=norm(x,weights,'input_layernorm');batch,rows,_=x.shape
  q,k,v=[linear(h,weights,'self_attn.'+name+'_proj').reshape(batch,rows,weights['self_attn.'+name+'_proj.weight'].shape[0]//head_dim,head_dim).transpose(1,2)
         for name in ('q','k','v')]
  def rotate(t):return (-t[...,head_dim//2:]).cat(t[...,:head_dim//2],dim=-1)
  return q*cos+rotate(q)*sin,k*cos+rotate(k)*sin,v


def tail(x,attention,weights):
  h=attention.transpose(1,2).reshape(x.shape[0],x.shape[1],-1)
  residual=x+linear(h,weights,'self_attn.o_proj')
  return residual+linear(linear(norm(residual,weights,'post_attention_layernorm'),weights,'mlp.fc1').relu().square(),weights,'mlp.fc2')


def language(weights,x,cos,sin,*,head_dim=128,query_chunk=128):
  """Exact FP32 causal attention with bounded query tiles; no key truncation or detach."""
  from tinygrad import Tensor
  if query_chunk<1 or head_dim%2:raise ValueError('Invalid attention geometry')
  q,k,v=qkv(x,cos,sin,weights,head_dim)
  repeats=q.shape[1]//k.shape[1];key=k.repeat_interleave(repeats,dim=1);value=v.repeat_interleave(repeats,dim=1)
  chunks=[]
  for start in range(0,x.shape[1],query_chunk):
    stop=min(start+query_chunk,x.shape[1])
    causal=(Tensor.arange(x.shape[1]).to(x.device).reshape(1,-1)>Tensor.arange(start,stop).to(x.device).reshape(-1,1)).where(-math.inf,0)
    probabilities=(q[:,:,start:stop].matmul(key.transpose(-2,-1))*head_dim**-.5+causal).softmax(-1)
    chunks.append(probabilities.matmul(value))
  return tail(x,Tensor.cat(*chunks,dim=2),weights),k,v


def expert(weights,x,cos,sin,context_k,context_v,mask,*,head_dim=128):
  """The context stays differentiable. The old expert-only graph detached it."""
  q,k,v=qkv(x,cos,sin,weights,head_dim)
  repeats=q.shape[1]//k.shape[1]
  key=context_k.cat(k,dim=2).repeat_interleave(repeats,dim=1)
  value=context_v.cat(v,dim=2).repeat_interleave(repeats,dim=1)
  probabilities=(q.matmul(key.transpose(-2,-1))*head_dim**-.5+mask.detach()).softmax(-1)
  return tail(x,probabilities.matmul(value),weights)


def reconstruction(outputs,targets,start,stop):
  y,k,v=outputs;ty,tk,tv=targets
  return (y[:,start:stop]-ty[:,start:stop].detach()).square().mean()+.1*(
    (k[:,:,start:stop]-tk[:,:,start:stop].detach()).square().mean()+
    (v[:,:,start:stop]-tv[:,:,start:stop].detach()).square().mean())


def objective(language_weights,expert_weights,data,*,head_dim=128,query_chunk=128,minimum_prefill=2048):
  """Average prompt/reasoning reconstruction, plus mean of all three expert residual losses.

  One fixed teacher token sequence aligns both student modules. Every expert step uses
  the captured teacher's actual actions/time/rotary/mask and current student-produced K/V.
  This trains a local conditional objective, not the student's free-running solver.
  """
  from tinygrad import Tensor
  prompt,valid,capacity=(int(data[k]) for k in ('prompt_length','valid_length','capacity'))
  if not 0<prompt<valid<=capacity or len(data['expert'])!=3:raise ValueError('Complete teacher sequence and three actual Euler targets required')
  extent=max(minimum_prefill,((valid+query_chunk-1)//query_chunk)*query_chunk)
  extent=min(extent,capacity)
  lang=data['language']
  if lang['x'].shape[1]!=capacity:raise ValueError('Missing full captured language context')
  outputs=language(language_weights,lang['x'][:,:extent].detach(),lang['cos'][:,:,:extent].detach(),lang['sin'][:,:,:extent].detach(),
                   head_dim=head_dim,query_chunk=query_chunk)
  targets=tuple(lang[k] for k in ('y','k','v'))
  prompt_loss=reconstruction(outputs,targets,0,prompt)
  reasoning_loss=reconstruction(outputs,targets,prompt,valid)
  k,v=(value.pad(((0,0),(0,0),(0,capacity-extent),(0,0))) for value in outputs[1:])
  # Only masked, invalid tail capacity is padded; all valid KV comes from the student graph.
  residuals=[];predicted=[]
  for step in data['expert']:
    y=expert(expert_weights,step['x'].detach(),step['cos'].detach(),step['sin'].detach(),k,v,step['mask'],head_dim=head_dim)
    residuals.append((y-step['y'].detach()).square().mean());predicted.append(y)
  expert_loss=Tensor.stack(*residuals).mean()
  losses=dict(prompt=prompt_loss,reasoning=reasoning_loss,expert=expert_loss)
  losses['total']=.5*(prompt_loss+reasoning_loss)+expert_loss
  return losses,dict(language=outputs,context_k=k,context_v=v,expert=predicted,language_extent=extent)


def memory_assessment(valid_length=2049,query_chunk=128):
  """Explicit tensor accounting, not a measured allocator peak or device admission."""
  capacity=2304;extent=min(capacity,max(2048,math.ceil(valid_length/query_chunk)*query_chunk))
  language_parameters=50335744;expert_parameters=7078912;parameters=language_parameters+expert_parameters
  score_full=16*extent*extent*4;score_chunk=16*query_chunk*extent*4
  # Conservative named tensor subtotal: projections, probabilities, activations, targets and optimizer state.
  language_activations=4*(extent*(2048*10+9216*3)+16*extent*extent*3)
  expert_activations=3*4*(64*(512*10+3840*3)+16*64*(capacity+64)*3)
  state=4*parameters*4
  targets=4*(capacity*(2048*2+1024*2+128*2)+3*(64*512*2+64*128*2+capacity+64))
  return dict(valid_length=valid_length,prefill_rows=2048,language_extent=extent,context_capacity=capacity,action_rows=64,euler_steps=3,
    query_chunk=query_chunk,trainable_parameters=parameters,master_bytes=parameters*4,
    checkpoint_weights_and_two_moments_bytes=parameters*4*3+12,best_bytes=parameters*4,
    master_gradient_two_moments_bytes=state,unchunked_language_score_bytes=score_full,
    largest_chunk_score_bytes=score_chunk,expert_score_bytes=16*64*(capacity+64)*4,
    largest_parameter_bytes=9216*2048*4,named_live_tensor_subtotal_bytes=state+language_activations+expert_activations+targets,
    peak_measured=False,allocation_qualification_required=True,
    note='Chunking bounds each score buffer; retained backward intermediates still scale quadratically. Runtime scratch, copies and compiler temporaries are excluded.')
