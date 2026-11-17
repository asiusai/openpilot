"""Numerical candidate: verify a complete fixed draft inside padded prefill.

Prefill and ordinary decode use different numerical paths. Target verification
does not establish parity with the previous decoder or qualify route accuracy.
"""
from functools import lru_cache
import importlib.util
from pathlib import Path
from types import MethodType
import numpy as np


@lru_cache(None)
def local(name):
  spec = importlib.util.spec_from_file_location(name.replace('-','_'),Path(__file__).with_name(name+'.py'))
  value = importlib.util.module_from_spec(spec);spec.loader.exec_module(value)
  return value


def install_prefill(model, draft, anchor, audit=False):
  from tinygrad import Tensor, dtypes
  from tinygrad.helpers import round_up
  from tinygrad.uop.ops import UOp
  policy = local('agent-speculation-policy')
  draft = policy.validate_draft(draft,anchor)
  baseline_class = local('padded-prefill').PaddedPrefill
  if not hasattr(model,'output_token_mapping'):
    raise ValueError('This candidate requires the existing checked reduced vocabulary')

  class SpeculativePrefill(baseline_class):
    def __init__(self):
      super().__init__(model)
      self.baseline = baseline_class(model)
      self.last_targets = None
      self.last_prompt = None
      self.audit = audit
      self.prefix_audit = None

    def language(self,x,cos,sin,length):
      for i in range(model.config['vlm_config']['text_config']['num_hidden_layers']):
        x = model.text_layer(x,cos,sin,f'vlm.model.language_model.layers.{i}',model.cache[i],None,cache_pos=0)
      targets,checks = [],[]
      # Keep every head projection independent. In particular, row zero retains
      # the baseline's single-row norm, activation scale, and GEMV arithmetic.
      for index in range(6):
        hidden = model.norm(x[:,length-1+index:length+index],'vlm.model.language_model.norm',rms=True)
        logits = model.linear(hidden,'vlm.lm_head').realize()
        if index == 0: original_first_logits = logits
        targets.append(model.output_token_mapping[logits.reshape(-1).argmax()].reshape(1))
        checks.append(logits.isfinite().all().cast(dtypes.int32).reshape(1))
      return original_first_logits,Tensor.cat(*targets,*checks).contiguous().realize()

    def __call__(self,x,cos,sin,max_tokens,length):
      self.last_targets = None
      self.last_prompt = int(length)
      self.prefix_audit = None
      if length+6 > x.shape[1] or max_tokens < 6:
        return self.baseline(x,cos,sin,max_tokens,length)
      old_logits,old_prefix = None,None
      if self.audit:
        # Reconstruct exactly the old END padding in the six replaced slots.
        # The remaining padding and every real prefix row are unchanged.
        end_embedding = model.weights['vlm.model.language_model.embed_tokens.weight'][0].reshape(1,1,-1)
        reference_x = Tensor.cat(x[:,:length],end_embedding.expand(1,6,x.shape[-1]),x[:,length+6:],dim=1).realize()
        reference_cos = Tensor.cat(cos[:,:,:length],Tensor.ones(1,1,6,cos.shape[-1],device=cos.device,dtype=cos.dtype),cos[:,:,length+6:],dim=2).realize()
        reference_sin = Tensor.cat(sin[:,:,:length],Tensor.zeros(1,1,6,sin.shape[-1],device=sin.device,dtype=sin.dtype),sin[:,:,length+6:],dim=2).realize()
        old_logits = self.baseline(reference_x,reference_cos,reference_sin,max_tokens,length).clone().realize()
        old_prefix = [cache[:,:,:,:length].clone().realize() for cache in model.cache]
      # Match the existing PaddedPrefill allocation and graph reset contract.
      capacity = max(round_up(x.shape[1]+max_tokens,256),getattr(model,'minimum_cache_capacity',0))
      if model.cache is None or model.cache[0].shape[3] != capacity:
        model.cache = [Tensor.zeros(2,1,8,capacity,128,dtype=model.dtype).realize()
                       for _ in range(model.config['vlm_config']['text_config']['num_hidden_layers'])]
        model.prefill_jit.reset();model.decode_jit.reset();model.greedy_decode_jit.reset();model.trajectory_jits.clear()
        self.jit.reset();self.shape=None
      shape = (x.shape,cos.shape,capacity)
      if self.shape != shape: self.jit.reset();self.shape=shape
      model.prompt_length = x.shape[1]
      valid = UOp.variable('valid_prefill_draft_length',1,x.shape[1]-6).bind(length)
      first_logits,targets = self.jit(x,cos,sin,valid)
      self.last_targets = targets
      if self.audit:
        np.testing.assert_array_equal(first_logits.bitcast(dtypes.uint16).numpy(),old_logits.bitcast(dtypes.uint16).numpy(),
                                      err_msg='Speculation changed original prompt logits')
        differences = Tensor.stack(*[(old.bitcast(dtypes.uint16) != current[:,:,:,:length].bitcast(dtypes.uint16)).sum().cast(dtypes.int32)
                                     for old,current in zip(old_prefix,model.cache,strict=True)]).numpy()
        if np.any(differences): raise AssertionError(f'Speculation changed causal prefix KV: {differences.tolist()}')
        self.prefix_audit = dict(first_logits_exact=True,prefix_kv_exact=True,prefix_rows=int(length))
      return first_logits

  model.padded_prefill = SpeculativePrefill()
  return model.padded_prefill


def install_decoder(decoder, draft, anchor, audit=False):
  policy = local('agent-speculation-policy')
  draft = policy.validate_draft(draft,anchor)
  original_generate = decoder.generate
  decoder.prefill_draft_records = []
  decoder.prefill_draft_old_history = []

  def generate(self,model,args,config,tokenizer,logits,dc,ds,prompt):
    prefill = model.padded_prefill
    if prefill.last_prompt != prompt: raise AssertionError('Prefill used a stale actual prompt length')
    if prefill.last_targets is None:
      result = original_generate(model,args,config,tokenizer,logits,dc,ds,prompt)
      self.prefill_draft_records.append(dict(eligible=False,actual_prompt=prompt,generated=result))
      return result
    payload = self.read_tokens(prefill.last_targets).reshape(-1)
    if payload.shape != (12,): raise AssertionError('Incomplete target token and finiteness payload')
    targets = payload[:6].astype(np.int64)
    finite = bool(np.all(payload[6:] == 1))
    accepted = policy.accept_targets(targets,np.asarray(draft),anchor,finite)
    record = dict(eligible=True,actual_prompt=prompt,targets=targets.tolist(),draft=list(draft),
                  all_targets_finite=finite,accepted=accepted,numerical_candidate=True,
                  target_rows=policy.target_rows(prompt).tolist(),prefix_audit=prefill.prefix_audit)
    history_before = [old[:] for old in self.history]
    old_result,old_statistics = None,None
    if audit:
      # Audit is diagnostic. Preserve candidate KV before the old decoder
      # independently overwrites the complete proposed suffix.
      candidate_kv = [cache[:,:,:,prompt:prompt+6].clone().realize() for cache in model.cache]
      import copy
      shadow = copy.copy(self)
      shadow.history = [old[:] for old in self.prefill_draft_old_history]
      shadow.statistics = []
      old_result = original_generate.__func__(shadow,model,args,config,tokenizer,logits,dc,ds,prompt)
      self.prefill_draft_old_history = [old[:] for old in shadow.history]
      old_statistics = shadow.statistics
      record.update(old_decoder_tokens=old_result,old_decoder_statistics=old_statistics,
                    accepted_tokens_equal_old=(not accepted or list(draft)==old_result))
      if accepted and list(draft)==old_result:
        from tinygrad import Tensor
        maximum = Tensor.stack(*[(saved.float()-cache[:,:,:,prompt:prompt+6].float()).abs().max()
                                for saved,cache in zip(candidate_kv,model.cache,strict=True)]).numpy()
        record['accepted_kv_max_delta_by_layer'] = maximum.astype(float).tolist()
      if accepted:
        for saved,cache in zip(candidate_kv,model.cache,strict=True):
          cache[:,:,:,prompt:prompt+6].assign(saved).realize()
        self.history = history_before
    if accepted:
      generated = list(draft)
      self.history = [old for old in self.history if old != generated][-63:]+[generated[:]]
      self.statistics.append(dict(lookup_rounds=0,verified_tokens=6,serial_tokens=0,prefill_draft=record))
    else:
      # Rejection keeps no matched speculative prefix. Start from the original
      # first-token logits, original prompt position, and unchanged decode RoPE.
      generated = original_generate(model,args,config,tokenizer,logits,dc,ds,prompt)
      self.statistics[-1] = dict(self.statistics[-1],prefill_draft=record)
    record.update(generated=generated,final_cache_length=prompt+len(generated),action_rotary_offset=len(generated))
    self.prefill_draft_records.append(record)
    return generated

  decoder.generate = MethodType(generate,decoder)
  return decoder
