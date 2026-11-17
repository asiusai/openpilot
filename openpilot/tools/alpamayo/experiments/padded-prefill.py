"""Keep language graph shapes fixed while selecting the real final prompt token.

Padding follows every real token. Causal attention prevents real tokens from
reading padding. Fixed activation scales prevent padding changing tensor maxima.
"""
import numpy as np
from tinygrad import Tensor, TinyJit
from tinygrad.helpers import round_up
from tinygrad.uop.ops import UOp


def pad_inputs(prepared, packed, length):
  ids, positions, delta, vision_positions = packed
  actual = ids.shape[1]
  if not 0 < actual <= length:
    raise ValueError(f'Prompt length {actual} exceeds the configured bound {length}')
  prepared = dict(prepared, actual_prompt_length=actual)
  count = length-actual
  return prepared, (np.pad(ids,((0,0),(0,count))), np.pad(positions,((0,0),(0,0),(0,count))),
                    delta-count, vision_positions)


class PaddedPrefill:
  def __init__(self,model):
    self.model=model
    self.jit=TinyJit(self.language)
    self.shape=None

  def language(self,x,cos,sin,length):
    model=self.model
    for i in range(model.config['vlm_config']['text_config']['num_hidden_layers']):
      x=model.text_layer(x,cos,sin,f'vlm.model.language_model.layers.{i}',model.cache[i],None,cache_pos=0)
    x=model.norm(x[:,length-1:length], 'vlm.model.language_model.norm',rms=True)
    return model.linear(x,'vlm.lm_head').realize()

  def __call__(self,x,cos,sin,max_tokens,length):
    model=self.model
    capacity=max(round_up(x.shape[1]+max_tokens,256),getattr(model,'minimum_cache_capacity',0))
    if model.cache is None or model.cache[0].shape[3]!=capacity:
      model.cache=[Tensor.zeros(2,1,8,capacity,128,dtype=model.dtype).realize()
                   for _ in range(model.config['vlm_config']['text_config']['num_hidden_layers'])]
      model.prefill_jit.reset();model.decode_jit.reset();model.greedy_decode_jit.reset();model.trajectory_jits.clear()
      self.jit.reset();self.shape=None
    shape=(x.shape,cos.shape,capacity)
    if self.shape!=shape:
      self.jit.reset();self.shape=shape
    model.prompt_length=x.shape[1]
    valid=UOp.variable('valid_prefill_length',1,x.shape[1]).bind(length)
    return self.jit(x,cos,sin,valid)


def install(model):
  model.padded_prefill=PaddedPrefill(model)
