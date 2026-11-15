"""Experimental previous-text draft, checked by the target model on every input.

Verification uses per-row FP8 activation scales. Batched attention can round
differently from serial decode; full-model parity tests are required.
"""
import importlib.util
from pathlib import Path
from types import MethodType
import math
import numpy as np
from tinygrad import Tensor, TinyJit, dtypes
from tinygrad.uop.ops import UOp

spec = importlib.util.spec_from_file_location('block_decoder', Path(__file__).with_name('block-decode.py'))
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class DraftDecoder(module.BlockDecoder):
  def __init__(self, model, block=6):
    super().__init__(model, block)
    if not hasattr(model, 'output_token_mapping'):
      raise ValueError('Draft experiment currently requires the checked reduced vocabulary')
    self.previous = []
    self.history_limit = 0
    self.history_full_only = False
    self.history = []
    self.verifiers = {}
    self.statistics = []
    self.warmup_graphs = False
    self.fast_attention = False
    self.keep_verified_prefix = False
    self.tile = (4, 4)
    self.compact_io = False
    self.choose_jit = TinyJit(lambda logits: model.output_token_mapping[logits.reshape(-1).argmax()].float().realize())
    self.select_jits = {}
    spec = importlib.util.spec_from_file_location('draft_gemm', Path(__file__).with_name('tiled-gemm.py'))
    self.gemm = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(self.gemm)
    spec = importlib.util.spec_from_file_location('draft_attention', Path(__file__).with_name('expert-kernels.py'))
    self.attention = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(self.attention)

  def first_token(self, logits):
    if self.compact_io:
      return int(self.read_tokens(self.choose_jit(logits)).item())
    return int(self.readback(self.model.output_token_mapping[logits.reshape(-1).argmax()].realize()).item())

  def read_tokens(self, tokens):
    # Direct integer reads avoid a separate GPU conversion and submission.
    return tokens.numpy() if self.compact_io and not self.direct_readback else self.readback(tokens)

  def select_logits(self, logits, index):
    if not self.compact_io:
      return logits[:,index:index+1].clone().realize()
    count = logits.shape[1]
    if count not in self.select_jits:
      self.select_jits[count] = TinyJit(lambda values, row: values[:,row:row+1].clone().realize())
    row = UOp.variable('selected_draft_row',0,count-1).bind(index)
    return self.select_jits[count](logits,row)

  def remember(self, generated):
    self.previous = generated[:]
    if self.history_limit:
      self.history = ([old for old in self.history if old != generated] + [generated[:]])[-self.history_limit:]

  def verify(self, token_ids, cos, sin, position, step):
    model = self.model
    count = token_ids.shape[0]
    x = model.weights['vlm.model.language_model.embed_tokens.weight'][token_ids].reshape(1, count, -1)
    c, s = cos[:, :, step:step+count], sin[:, :, step:step+count]
    mask = (Tensor.arange(model.cache[0].shape[3]).reshape(1, -1) >
            (Tensor.arange(count)+position).reshape(-1, 1)).where(-math.inf, 0)
    original = model.quantize
    original_linear = model.linear
    original_attention = model.attention
    if self.fast_attention:
      model.attention = MethodType(lambda _, q, k, v, mask=None, causal=False: self.attention.attention(q, k, v, mask), model)
    def per_row(self, value):
      value = value.reshape(-1, value.shape[-1]).float()
      scale = (value.abs().max(axis=1, keepdim=True)/448).clamp(min_=1e-12)
      return (value/scale).clamp(-448,448).cast(dtypes.fp8e4m3).contiguous(), scale
    model.quantize = MethodType(per_row, model)
    gemm = self.gemm.gemm
    def linear(self, value, name, quantized_input=None, heads=None):
      if (scale := self.weight_scales.get(name+'.weight')) is None:
        return original_linear(value, name, quantized_input, heads)
      shape = value.shape[:-1]
      quantized, activation_scale = per_row(self, value) if quantized_input is None else quantized_input
      return gemm(quantized, self.weights[name+'.weight'], (activation_scale,scale), tile=tile,
                  row_scale=True, bias=self.weights.get(name+'.bias')).reshape(*shape,-1)
    tile = self.tile
    model.linear = MethodType(linear, model)
    try:
      for i in range(model.config['vlm_config']['text_config']['num_hidden_layers']):
        x = model.text_layer(x, c, s, f'vlm.model.language_model.layers.{i}', model.cache[i], mask, cache_pos=position)
      logits = model.linear(model.norm(x, 'vlm.model.language_model.norm', rms=True), 'vlm.lm_head').realize()
    finally:
      model.quantize = original
      model.linear = original_linear
      model.attention = original_attention
    tokens = model.output_token_mapping[logits.argmax(axis=-1)].realize()
    return logits, tokens

  def warm_verifiers(self, proposed, logits, dc, ds, prompt):
    if not self.warmup_graphs:
      return
    model = self.model
    if self.compact_io:
      while self.choose_jit.cnt < 2:
        self.choose_jit(logits).realize()
    for size in (16,32,48,64):
      warm_tokens = Tensor(np.array((proposed*((size+len(proposed)-1)//len(proposed)))[:size],dtype=np.int32),device='CPU').to(logits.device).realize()
      warm_pos = UOp.variable('draft_position',0,model.cache[0].shape[3]-size).bind(prompt)
      warm_step = UOp.variable('draft_step',0,dc.shape[2]-size).bind(0)
      warm_key = (size,dc.shape,model.cache[0].shape[3])
      if warm_key not in self.verifiers:
        self.verifiers[warm_key] = TinyJit(self.verify)
      while self.verifiers[warm_key].cnt < 2:
        Tensor.realize(*self.verifiers[warm_key](warm_tokens,dc,ds,warm_pos,warm_step))
      if self.compact_io and (size not in self.select_jits or self.select_jits[size].cnt < 2):
        verified, _ = self.verifiers[warm_key](warm_tokens,dc,ds,warm_pos,warm_step)
        while size not in self.select_jits or self.select_jits[size].cnt < 2:
          self.select_logits(verified,0)

  def generate(self, model, args, config, tokenizer, logits, dc, ds, prompt):
    generated = []
    proposed = self.previous[:min(64,args.max_tokens)]
    if proposed:
      first = self.first_token(logits)
      historical = False
      if first != proposed[0] and self.history_limit:
        for old in reversed(self.history):
          if old[0] == first:
            proposed = old[:min(64,args.max_tokens)]
            historical = True
            break
      if first != proposed[0]:
        self.statistics.append({'proposed':len(proposed), 'accepted':0, 'first_token_mismatch':True})
        generated = super().generate(model,args,config,tokenizer,logits,dc,ds,prompt)
        self.remember(generated)
        return generated
      count = ((len(proposed)+15)//16)*16
      self.warm_verifiers(proposed,logits,dc,ds,prompt)
      tokens = Tensor(np.array(proposed+[proposed[-1]]*(count-len(proposed)), dtype=np.int32), device='CPU').to(logits.device).realize()
      pos = UOp.variable('draft_position', 0, model.cache[0].shape[3]-count).bind(prompt)
      step = UOp.variable('draft_step', 0, dc.shape[2]-count).bind(0)
      key = (count, dc.shape, model.cache[0].shape[3])
      if key not in self.verifiers:
        self.verifiers[key] = TinyJit(self.verify)
      calls = self.verifiers[key].cnt
      phase = 'compile' if calls == 0 else 'capture' if calls == 1 else 'replay'
      verified_logits, predicted = self.verifiers[key](tokens, dc, ds, pos, step)
      predictions = [first, *self.read_tokens(predicted).reshape(-1).astype(int).tolist()]
      for i, candidate in enumerate(proposed):
        if candidate != predictions[i]:
          break
        generated.append(candidate)
        if candidate == config.traj_ids['future_start']:
          self.statistics.append({'proposed': len(proposed), 'accepted':len(generated), 'verification_phase':phase, 'historical_draft':historical})
          self.remember(generated)
          return generated
      accepted = len(generated)
      self.statistics.append({'proposed':len(proposed), 'accepted':accepted, 'verification_phase':phase, 'historical_draft':historical})
      if self.keep_verified_prefix and accepted and not (historical and self.history_full_only):
        # Only matched tokens remain. Each serial step masks all rejected K/V
        # rows and overwrites its own position before the next step reads it.
        # Batched verification has different rounding; this is experimental.
        logits = self.select_logits(verified_logits,accepted-1)
        for offset in range(accepted, args.max_tokens, self.block):
          logits, tokens = self.decode(logits, dc, ds, prompt+offset, offset,
                                       min(self.block, args.max_tokens-offset))
          for token in self.read_tokens(tokens).reshape(-1):
            generated.append(int(token))
            if int(token) == config.traj_ids['future_start']:
              self.remember(generated)
              self.statistics[-1]['continued_verified_prefix'] = True
              return generated
        raise RuntimeError(f'No trajectory anchor after {args.max_tokens} tokens: {tokenizer.decode(generated)}')
      if historical and self.history_full_only:
        self.statistics[-1]['historical_partial_discarded'] = True
      # Restart from the original prompt logits on rejection. This overwrites
      # every proposed K/V entry, preserving serial behavior on changed text.
      generated = super().generate(model,args,config,tokenizer,logits,dc,ds,prompt)
      self.remember(generated)
      return generated
    for step in range(len(generated), args.max_tokens):
      logits, token = model.greedy_decode(logits, dc, ds, prompt+step, step)
      token = int(self.read_tokens(token).item())
      generated.append(token)
      if token == config.traj_ids['future_start']:
        self.remember(generated)
        return generated
    raise RuntimeError(f'No trajectory anchor after {args.max_tokens} tokens: {tokenizer.decode(generated)}')
