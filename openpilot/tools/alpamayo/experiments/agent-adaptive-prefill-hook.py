"""Opt-in 16-token prefill prefix verification over completed past predictions.

No new attention, GEMM, scale, or quantizer implementation. This changes the
accepted suffix arithmetic and therefore remains a numerical candidate.
"""
from contextlib import contextmanager
import copy
import functools
import importlib.util
from pathlib import Path
import time
from types import MethodType
import numpy as np


def local(name):
  spec=importlib.util.spec_from_file_location(name.replace('-','_'),Path(__file__).with_name(name+'.py'))
  value=importlib.util.module_from_spec(spec);spec.loader.exec_module(value);return value


P=local('agent-adaptive-prefill-policy')


class State:
  def __init__(self,anchor,audit_prefix_once=False):
    self.anchor=int(anchor);self.previous=None;self.active=None;self.records=[]
    self.completed=0;self.running=False;self.current=None;self.suppress_commit=0
    self.audit_prefix_once=audit_prefix_once;self.audit_done=False

  @contextmanager
  def current_input_only(self,decoder):
    """Discarded startup probes cannot become completed-past draft history."""
    if self.running: raise RuntimeError('Cannot start warm probes during a prediction')
    previous,completed=self.previous,self.completed
    history=copy.deepcopy(decoder.history);stats=copy.deepcopy(decoder.statistics)
    nominal={key:copy.deepcopy(getattr(decoder,key)) for key in ('prefill_draft_records','prefill_draft_old_history') if hasattr(decoder,key)}
    records=len(self.records);self.suppress_commit+=1
    try: yield
    finally:
      self.previous,self.completed=previous,completed;decoder.history=history;decoder.statistics=stats
      for key,value in nominal.items():setattr(decoder,key,value)
      del self.records[records:];self.suppress_commit-=1

  def enter(self,args,prepared,packed):
    if self.running: raise RuntimeError('One active prediction; no nested or concurrent state')
    ids,positions,delta,_=packed
    prompt=int(prepared['actual_prompt_length'])
    assert ids.shape==(1,2048) and positions.shape==(3,1,2048)
    draft=P.choose(self.previous,prompt,2048,args.max_tokens,self.anchor)
    # This scalar is exactly CompactInputs.arrays' decode position origin.
    self.current={'prompt':prompt,'next_position':int(ids.shape[1]+delta),
      'draft':draft,'completed_before':self.completed,'previous':self.previous}
    self.active=None;self.running=True

  def finish(self,result):
    tokens=P.complete(result['generated_token_ids'],self.anchor)
    if len(tokens)!=result['generated_tokens']: raise ValueError('Incomplete output count')
    if not np.isfinite(np.asarray(result['xyz'])).all() or not np.isfinite(np.asarray(result['rot'])).all():
      raise ValueError('Nonfinite completed trajectory cannot enter draft history')
    if not self.suppress_commit: self.previous=tokens;self.completed+=1


def make_prefill(model,baseline,state,rotary):
  from tinygrad import Tensor,TinyJit,dtypes
  from tinygrad.helpers import round_up
  from tinygrad.uop.ops import UOp

  class AdaptivePrefill:
    def __init__(self):
      self.baseline=baseline;self.jit=TinyJit(self.language);self.shape=None
      self.last_logits=None;self.last_payload=None;self.last_record=None

    def __getattr__(self,name): return getattr(self.baseline,name)

    def language(self,x,cos,sin,length):
      for i in range(model.config['vlm_config']['text_config']['num_hidden_layers']):
        x=model.text_layer(x,cos,sin,f'vlm.model.language_model.layers.{i}',model.cache[i],None,cache_pos=0)
      heads,targets,checks=[],[],[]
      for index in range(P.WIDTH+1):
        hidden=model.norm(x[:,length-1+index:length+index],'vlm.model.language_model.norm',rms=True)
        logits=model.linear(hidden,'vlm.lm_head').realize()
        heads.append(logits)
        targets.append(model.output_token_mapping[logits.reshape(-1).argmax()].reshape(1))
        checks.append(logits.isfinite().all().cast(dtypes.int32).reshape(1))
      return heads[0],Tensor.cat(*heads,dim=1).contiguous().realize(),Tensor.cat(*targets,*checks).contiguous().realize()

    def __call__(self,x,cos,sin,max_tokens,length):
      if not state.running or state.current['prompt']!=length: raise RuntimeError('Prefill outside the current owned prediction')
      current=state.current;draft=current['draft'];state.active=bool(draft)
      self.last_logits=self.last_payload=self.last_record=None
      if not draft: return self.baseline(x,cos,sin,max_tokens,length)
      assert x.shape[1]==2048 and length+P.WIDTH<=2048 and max_tokens>=P.WIDTH
      old_logits=old_prefix=None
      auditing=state.audit_prefix_once and not state.audit_done
      if auditing:
        # Diagnostic only: this entire extra prefill/readback is explicitly timed
        # and makes the paired quality run unsuitable for throughput claims.
        old_logits=self.baseline(x,cos,sin,max_tokens,length).clone().realize()
        old_prefix=[cache[:,:,:,:length].clone().realize() for cache in model.cache]
      started=time.perf_counter()
      codes=np.asarray((*draft,*([0]*(P.WIDTH-len(draft)))),dtype=np.int32)
      token_ids=Tensor(codes,device='CPU').to(x.device).realize()
      replacement=model.weights['vlm.model.language_model.embed_tokens.weight'][token_ids].reshape(1,P.WIDTH,-1)
      # END filler rows after a short complete draft stay at the original zero
      # rotary positions. They remain beyond the accepted expert key mask.
      positions=np.zeros((3,1,P.WIDTH),dtype=np.int64)
      positions[:,:,:len(draft)]=current['next_position']+np.arange(len(draft))
      rc,rs=rotary(positions)
      import torch
      # Use the exact original CPU float32->BF16 conversion before transfer.
      rotary_bits=[torch.from_numpy(a).to(torch.bfloat16).view(torch.uint16).numpy() for a in (rc,rs)]
      c,s=(Tensor(a,device='CPU').to(x.device).bitcast(cos.dtype).realize() for a in rotary_bits)
      amended_x=Tensor.cat(x[:,:length],replacement,x[:,length+P.WIDTH:],dim=1).realize()
      amended_cos=Tensor.cat(cos[:,:,:length],c,cos[:,:,length+P.WIDTH:],dim=2).realize()
      amended_sin=Tensor.cat(sin[:,:,:length],s,sin[:,:,length+P.WIDTH:],dim=2).realize()
      capacity=max(round_up(x.shape[1]+max_tokens,256),getattr(model,'minimum_cache_capacity',0))
      if model.cache is None or model.cache[0].shape[3]!=capacity:
        model.cache=[Tensor.zeros(2,1,8,capacity,128,dtype=model.dtype).realize()
          for _ in range(model.config['vlm_config']['text_config']['num_hidden_layers'])]
        model.prefill_jit.reset();model.decode_jit.reset();model.greedy_decode_jit.reset();model.trajectory_jits.clear()
        self.jit.reset();self.shape=None;self.baseline.jit.reset();self.baseline.shape=None
      shape=(x.shape,cos.shape,capacity)
      if self.shape!=shape:self.jit.reset();self.shape=shape
      model.prompt_length=x.shape[1]
      valid=UOp.variable('adaptive_prefill_length',1,x.shape[1]-P.WIDTH).bind(length)
      first,self.last_logits,self.last_payload=self.jit(amended_x,amended_cos,amended_sin,valid)
      self.last_record={'numerical_candidate':True,'draft':list(draft),'actual_prompt':length,
        'completed_before':current['completed_before'],'previous_completed':list(current['previous']),
        'width':P.WIDTH,'head_projections':P.WIDTH+1,'extra_head_projections_vs_nominal':11,
        'draft_cpu_to_gpu_bytes':int(codes.nbytes+sum(a.nbytes for a in rotary_bits)),
        'embedding_stitch_output_bytes':int(np.prod(x.shape)*2),
        'rotary_origin':current['next_position'],'draft_build_and_prefill_host_ms':(time.perf_counter()-started)*1000,
        'audit_prefix_once':auditing,'target_rows':list(range(length-1,length+P.WIDTH))}
      if auditing:
        np.testing.assert_array_equal(first.bitcast(dtypes.uint16).numpy(),old_logits.bitcast(dtypes.uint16).numpy(),
          err_msg='Adaptive padding changed original first logits')
        differences=Tensor.stack(*[(before.bitcast(dtypes.uint16)!=after[:,:,:,:length].bitcast(dtypes.uint16)).sum().cast(dtypes.int32)
          for before,after in zip(old_prefix,model.cache,strict=True)]).numpy()
        if np.any(differences):raise AssertionError(f'Adaptive padding changed original prompt KV: {differences.tolist()}')
        state.audit_done=True;self.last_record['prefix_audit']={'first_logits_exact':True,'all_layer_prefix_kv_exact':True,
          'layer_count':len(model.cache),'rows':length}
      return first

  return AdaptivePrefill()


def attach_decoder(decoder,model,state,original_generate,resume):
  old_run=decoder.run;nominal_generate=decoder.generate
  old_instance={name:vars(decoder).get(name) for name in ('run','generate')}
  had={name:name in vars(decoder) for name in old_instance}

  def generate(self,model,args,config,tokenizer,logits,dc,ds,prompt):
    if not state.active: return nominal_generate(model,args,config,tokenizer,logits,dc,ds,prompt)
    prefill=model.padded_prefill;record=prefill.last_record
    if record is None or record['actual_prompt']!=prompt:raise RuntimeError('Missing current prefill payload')
    payload=self.read_tokens(prefill.last_payload).reshape(-1)
    accepted,finite=P.prefix(payload,record['draft'],state.anchor)
    record.update(targets=payload[:P.WIDTH+1].astype(int).tolist(),finite_flags=payload[P.WIDTH+1:].astype(int).tolist(),
      all_targets_finite=finite,accepted_prefix=accepted)
    if accepted==0:
      # All rejected suffix KV remains masked and is overwritten by existing
      # lookup/serial generation. No speculative suffix is retained.
      generated=original_generate(model,args,config,tokenizer,logits,dc,ds,prompt)
    elif record['draft'][accepted-1]==state.anchor:
      generated=record['draft'][:accepted]
      self.history=[old for old in self.history if old!=generated][-63:]+[generated[:]]
      self.statistics.append({'lookup_rounds':0,'verified_tokens':0,'serial_tokens':0})
    else:
      # Row accepted is the target distribution AFTER all retained tokens.
      # It chooses the mismatch replacement or the token after a full prefix.
      continuation=self.select_logits(prefill.last_logits,accepted)
      self._adaptive_seed=record['draft'][:accepted]
      try:generated=resume(self,model,args,config,tokenizer,continuation,dc,ds,prompt)
      finally:del self._adaptive_seed
    record.update(P.contract(prompt,accepted,len(generated),model.cache[0].shape[3],args.max_tokens),generated=list(generated))
    self.statistics[-1]=dict(self.statistics[-1],adaptive_prefill=record)
    state.records.append(record)
    return generated

  def run(self,model,args,config,tokenizer,prepared,packed,seed,inputs):
    state.enter(args,prepared,packed)
    history_before=copy.deepcopy(self.history)
    try:
      result=old_run(model,args,config,tokenizer,prepared,packed,seed,inputs)
      state.finish(result)
      return result
    except BaseException:
      # A failed trajectory must not leak decoded text into the next request.
      self.history=history_before
      raise
    finally:state.running=False;state.current=None;state.active=None

  decoder.generate=MethodType(generate,decoder);decoder.run=MethodType(run,decoder)
  installed={name:getattr(decoder,name) for name in old_instance}
  def close():
    for name in old_instance:
      if getattr(decoder,name) is not installed[name]:raise RuntimeError('Adaptive decoder ownership changed: '+name)
      if had[name]:setattr(decoder,name,old_instance[name])
      else:delattr(decoder,name)
  return close


class Overlay:
  """Install before the existing speculation runner imports its implementation."""
  def __init__(self,experiment_dir,audit_prefix_once=False):
    self.experiment_dir=Path(experiment_dir).resolve();self.audit_prefix_once=audit_prefix_once
    self.states=[];self.closers=[];self.prefills=[];self.restored=False;self.ast_proof=None

  def patch(self,module):
    old_prefill,old_decoder=module.install_prefill,module.install_decoder
    owner=self
    @functools.wraps(old_prefill)
    def install_prefill(model,draft,anchor,audit=False):
      if audit:raise ValueError('Use the bounded adaptive prefix audit, not the old all-query audit')
      baseline=old_prefill(model,draft,anchor,audit=False)
      from inputs import rotary
      state=State(anchor,owner.audit_prefix_once)
      adaptive=make_prefill(model,baseline,state,rotary)
      model.padded_prefill=adaptive
      owner.states.append(state);owner.prefills.append((model,baseline,adaptive))
      return adaptive
    @functools.wraps(old_decoder)
    def install_decoder(decoder,draft,anchor,audit=False):
      original_generate=decoder.generate
      # The selected standalone recipe uses the original lookup class here.
      # Reject an unexpected startup/class wrapper instead of bypassing it.
      lookup=original_generate.__func__
      if lookup.__code__.co_filename!=str(owner.experiment_dir/'lookup-decode.py'):
        raise ValueError('Unexpected lookup generate owner; Consumer/startup integration is not installed')
      resume,owner.ast_proof=P.resume_lookup(lookup)
      result=old_decoder(decoder,draft,anchor,audit=False)
      model,_,adaptive=owner.prefills[-1]
      if decoder.model is not model or decoder.history:raise ValueError('Require one empty initial lookup owner')
      owner.closers.append(attach_decoder(decoder,model,owner.states[-1],original_generate,resume))
      return result
    module.install_prefill,module.install_decoder=install_prefill,install_decoder
    def restore():
      assert module.install_prefill is install_prefill and module.install_decoder is install_decoder
      module.install_prefill,module.install_decoder=old_prefill,old_decoder
    self.closers.insert(0,restore)

  def __enter__(self):
    self.original_spec=importlib.util.spec_from_file_location;owner=self
    def spec_hook(name,location,*args,**kwargs):
      spec=owner.original_spec(name,location,*args,**kwargs)
      if location is None or Path(location).resolve()!=owner.experiment_dir/'agent-speculation-prefill.py':return spec
      loader=spec.loader
      class Loader:
        def create_module(self,value):return loader.create_module(value)
        def exec_module(self,value):loader.exec_module(value);owner.patch(value)
      spec.loader=Loader();return spec
    self.spec_hook=spec_hook;importlib.util.spec_from_file_location=spec_hook;return self

  def __exit__(self,*error):
    # The original speculation runner leaves its private import hook installed.
    # The surrounding standalone runner restores that layer before closing us.
    if importlib.util.spec_from_file_location is not self.spec_hook:raise RuntimeError('Close the inner speculation import hook first')
    for close in reversed(self.closers):close()
    for model,baseline,adaptive in reversed(self.prefills):
      if model.padded_prefill is not adaptive:raise RuntimeError('Adaptive prefill owner changed')
      adaptive.jit.reset();model.padded_prefill=baseline
    importlib.util.spec_from_file_location=self.original_spec;self.restored=True
