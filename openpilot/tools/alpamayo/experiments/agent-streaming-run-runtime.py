"""Optional recurrent prefill. Install below the frozen packed-quantizer wrapper.

The original decode and expert layer paths remain passthrough. Raw image K/V is
staged separately; only a completed prediction can commit it to the tensor bank.
"""
from collections import OrderedDict
from contextlib import contextmanager
import importlib.util
from pathlib import Path
import sys
from types import MethodType
import numpy as np
from tinygrad import Tensor,TinyJit,dtypes
from tinygrad.uop.ops import UOp

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('agent_streaming_runtime_common',HERE/'agent-streaming-run-common.py')
common=importlib.util.module_from_spec(spec);sys.modules[spec.name]=common;spec.loader.exec_module(common)
kernels=common.load('agent-streaming-run-kernels')

class StreamingPrefill:
  def __init__(self,model):
    if (model.config['vlm_config']['text_config']['num_hidden_layers']!=28 or
        model.config['expert_config']['llm_config']['num_hidden_layers']!=28):raise ValueError('Original 28-layer geometry required')
    self.model=model;self.original_layer=model.text_layer;self.phase=None;self.frame_slots=None
    self.planner=common.Planner();self.plan=None;self.bank=None;self.pending_kv=None
    self.full_jit=TinyJit(self._full);self.sparse_jits=OrderedDict();self.commit_jit=TinyJit(self._commit)
    self.original_padded=None;self.last_event=None;self.force_full=False
    model.text_layer=MethodType(lambda _,*a,**kw:self.layer(*a,**kw),model)

  def install_padded(self):
    self.original_padded=self.model.padded_prefill
    self.model.padded_prefill=self.padded_prefill

  def prepare(self,meta,spans,prompt,*,force_full=False):
    self.plan=self.planner.begin(meta,spans,prompt,force_full=force_full)

  def abort(self):
    self.planner.abort();self.plan=None;self.last_event=None

  @contextmanager
  def stage(self,phase,frames,**fields):
    old={key:getattr(self,key,None) for key in ('phase','frame_slots',*fields)}
    self.phase,self.frame_slots=phase,frames
    for key,value in fields.items():setattr(self,key,value)
    try:yield
    finally:
      for key,value in old.items():setattr(self,key,value)

  def _capture(self,layer,raw):
    # pending_kv and bank are disjoint allocations. The gather's source carries
    # assemble's bank-read dependency; no bank write happens in this graph.
    value=kernels.gather(raw,self.frame_slots).realize()
    self.pending_kv[layer].assign(value).realize()

  def layer(self,x,cos,sin,prefix,cache=None,mask=None,cache_pos=None):
    if self.phase not in ('full','sparse') or not prefix.startswith('vlm.model.language_model.layers.'):
      return self.original_layer(x,cos,sin,prefix,cache,mask,cache_pos)
    layer=int(prefix.rsplit('.',1)[1]);model=self.model
    # Exact frozen EdgeModel projection/norm/MLP sequence. The surrounding
    # quantize-packed.text_layer wrapper owns prefix/index/fixed-scale state.
    h=model.norm(x,f'{prefix}.input_layernorm',rms=True)
    quantized=model.quantize(h) if f'{prefix}.self_attn.q_proj.weight' in model.weight_scales else None
    q,k,v=(model.linear(h,f'{prefix}.self_attn.{name}_proj',quantized,heads=heads)
      .reshape(1,h.shape[1],heads,128).transpose(1,2) for name,heads in (('q',16),('k',8),('v',8)))
    raw=Tensor.stack(k,v).contiguous()
    if self.phase=='full':
      self._capture(layer,raw)
      q,k=model.rope(q,k,cos,sin)
      store=cache[:,:,:,:x.shape[1]].uop.store(Tensor.stack(k,v).uop)
      assigned=Tensor(cache.uop.after(store))
      k,v=(assigned[i,:,:,:x.shape[1]] for i in range(2))
      h=model.attention(q,k,v,None,True)
    else:
      full=kernels.assemble(raw,self.bank[layer],self.query_read,self.bank_read).realize()
      self._capture(layer,full)
      q,_=model.rope(q,k,cos,sin)
      _,k=model.rope(full[0],full[0],self.full_cos,self.full_sin)
      store=cache[:,:,:,:2048].uop.store(Tensor.stack(k,full[1]).uop)
      assigned=Tensor(cache.uop.after(store))
      # Consume the returned AFTER value. Reading the old cache tensor here
      # would omit the required store-to-attention dependency.
      h=kernels.attention(q,assigned[0,:,:,:2048],assigned[1,:,:,:2048],self.query_slots,self.limits)
    h=h.transpose(1,2).reshape(1,x.shape[1],2048)
    x=x+model.linear(h,f'{prefix}.self_attn.o_proj')
    h=model.linear(model.norm(x,f'{prefix}.post_attention_layernorm',rms=True),f'{prefix}.mlp.fc1').relu().square()
    return (x+model.linear(h,f'{prefix}.mlp.fc2')).realize()

  def _full(self,x,cos,sin,frames,length):
    with self.stage('full',frames):
      for i in range(28):x=self.model.text_layer(x,cos,sin,f'vlm.model.language_model.layers.{i}',self.model.cache[i],None,cache_pos=0)
    value=self.model.norm(x[:,length-1:length],'vlm.model.language_model.norm',rms=True)
    return self.model.linear(value,'vlm.lm_head').realize()

  def _sparse(self,x,cos,sin,frames,slots,query_read,bank_read,limits,last):
    x,c,s=(kernels.gather(t,slots).realize() for t in (x,cos,sin))
    with self.stage('sparse',frames,query_slots=slots,query_read=query_read,bank_read=bank_read,
                    full_cos=cos,full_sin=sin,limits=limits):
      for i in range(28):x=self.model.text_layer(x,c,s,f'vlm.model.language_model.layers.{i}',self.model.cache[i],None,cache_pos=0)
    value=self.model.norm(x[:,last:last+1],'vlm.model.language_model.norm',rms=True)
    return self.model.linear(value,'vlm.lm_head').realize()

  def _allocate(self,x):
    model=self.model
    if x.shape!=(1,2048,2048):raise ValueError('Frozen padded input geometry required')
    if self.bank is None:
      if model.cache is not None and model.cache[0].shape!=(2,1,8,2304,128):raise ValueError('Unexpected active cache')
      if model.cache is None:
        model.cache=[Tensor.zeros(2,1,8,2304,128,device=x.device,dtype=model.dtype).realize() for _ in range(28)]
      self.bank=[Tensor.zeros(2,1,8,48*180,128,device=x.device,dtype=model.dtype).realize() for _ in range(28)]
      self.pending_kv=[Tensor.zeros(2,1,8,1800,128,device=x.device,dtype=model.dtype).realize() for _ in range(28)]
      model.prefill_jit.reset();model.decode_jit.reset();model.greedy_decode_jit.reset();model.trajectory_jits.clear()
    model.prompt_length=2048

  def padded_prefill(self,x,cos,sin,max_tokens,length):
    if self.plan is None or self.plan['prompt']!=length:raise RuntimeError('Missing current prediction metadata')
    if max_tokens!=64:raise ValueError('Frozen 64-token reasoning bound required')
    self._allocate(x);p=self.plan
    def device(value):return Tensor(np.asarray(value,dtype=np.int32),device='CPU').to(x.device).contiguous().realize()
    frames=device(p['image_slots'])
    if p['event']['reset']:
      valid=UOp.variable('streaming_actual_prompt',1,2048).bind(length)
      result=self.full_jit(x,cos,sin,frames,valid)
    else:
      rows=p['rows']
      if rows not in self.sparse_jits:
        if len(self.sparse_jits)>=8:
          _,retired=self.sparse_jits.popitem(last=False);retired.reset()
        self.sparse_jits[rows]=TinyJit(self._sparse)
      self.sparse_jits.move_to_end(rows)
      last=UOp.variable('streaming_last_query',0,rows-1).bind(p['last'])
      result=self.sparse_jits[rows](x,cos,sin,frames,device(p['query_slots']),device(p['query_read']),
        device(p['bank_read']),device([length,p['last']+1]),last)
    self.last_event=dict(p['event'],sparse_graphs=len(self.sparse_jits),bank_tensor_bytes=990904320,
      pending_tensor_bytes=206438400,active_cache_tensor_bytes=264241152)
    return result

  def _commit(self,destinations):
    # The pending tensors were materialized by _capture before decode/expert.
    # Each scatter writes a disjoint bank from its completed staging tensor.
    for bank,pending in zip(self.bank,self.pending_kv,strict=True):
      kernels.scatter(bank,pending,destinations).realize()

  def commit(self):
    if self.plan is None or self.last_event is None:raise RuntimeError('No successful prefill')
    destinations=Tensor(self.plan['destinations'],device='CPU').to(self.bank[0].device).contiguous().realize()
    self.commit_jit(destinations)
    # Failure cannot publish valid host metadata for incomplete GPU bank writes.
    self.model.synchronize();self.planner.commit();self.plan=None
    return self.last_event
