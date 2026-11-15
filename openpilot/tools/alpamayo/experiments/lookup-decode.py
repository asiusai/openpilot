"""Experimental target-verified drafts from a bounded history of generated text.

Every accepted token must match current target logits. Verification arithmetic
can differ from serial arithmetic, as in DraftDecoder; compare full outputs.
"""
import importlib.util
from pathlib import Path
import numpy as np
from tinygrad import Tensor, TinyJit
from tinygrad.uop.ops import UOp

spec = importlib.util.spec_from_file_location('lookup_draft', Path(__file__).with_name('draft-decode.py'))
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class LookupDecoder(module.DraftDecoder):
  def __init__(self, model):
    super().__init__(model)
    self.history = []
    self.fallback_block = self.block
    self.carry_verified_token = False

  def continuation(self, context, limit):
    # Prefer the longest matching suffix, then the most recent observation.
    for width in range(min(8, len(context)), 0, -1):
      suffix = context[-width:]
      for previous in reversed(self.history):
        for end in range(len(previous)-1, width-1, -1):
          if previous[end-width:end] == suffix:
            return previous[end:end+limit]
    return []

  def generate(self, model, args, config, tokenizer, logits, dc, ds, prompt):
    generated = []
    stats = {'lookup_rounds': 0, 'verified_tokens': 0, 'serial_tokens': 0}
    anchor = config.traj_ids['future_start']
    next_token, pending_logits = None, None
    if self.history:
      self.warm_verifiers(self.history[-1],logits,dc,ds,prompt)
    while len(generated) < args.max_tokens:
      offset = len(generated)
      remaining = args.max_tokens-offset
      first = self.first_token(logits) if next_token is None else next_token
      next_token = None
      proposed = [first] + self.continuation(generated+[first], min(64,remaining)-1)
      count = ((len(proposed)+15)//16)*16
      if len(proposed) <= 1 or count > remaining:
        if pending_logits is not None:
          # Clone before another verifier can overwrite its reusable output.
          logits = self.select_logits(*pending_logits)
          pending_logits = None
        logits, tokens = self.decode(logits, dc, ds, prompt+offset, offset, min(self.fallback_block, remaining))
        accepted = self.read_tokens(tokens).reshape(-1).astype(int).tolist()
        stats['serial_tokens'] += len(accepted)
      else:
        tokens = Tensor(np.array(proposed+[proposed[-1]]*(count-len(proposed)),dtype=np.int32),device='CPU').to(logits.device).realize()
        pos = UOp.variable('draft_position',0,model.cache[0].shape[3]-count).bind(prompt+offset)
        step = UOp.variable('draft_step',0,dc.shape[2]-count).bind(offset)
        key = (count,dc.shape,model.cache[0].shape[3])
        if key not in self.verifiers:
          self.verifiers[key] = TinyJit(self.verify)
        if self.warmup_graphs:
          while self.verifiers[key].cnt < 2:
            Tensor.realize(*self.verifiers[key](tokens,dc,ds,pos,step))
        verified, predicted = self.verifiers[key](tokens,dc,ds,pos,step)
        choices = [first,*self.read_tokens(predicted).reshape(-1).astype(int).tolist()]
        accepted = []
        for candidate, choice in zip(proposed, choices):
          if candidate != choice:
            break
          accepted.append(candidate)
          if candidate == anchor:
            break
        assert accepted  # First token came directly from current target logits.
        if self.carry_verified_token and accepted[-1] != anchor:
          # These are the same current-target logits used by first_token.
          # Keep the CPU argmax and defer copying its row until serial fallback.
          next_token = choices[len(accepted)]
          pending_logits = (verified,len(accepted)-1)
        elif not self.compact_io or accepted[-1] != anchor:
          logits = self.select_logits(verified,len(accepted)-1)
        stats['lookup_rounds'] += 1
        stats['verified_tokens'] += len(accepted)
      for token in accepted:
        generated.append(token)
        if token == anchor:
          self.history = [old for old in self.history if old != generated][-63:] + [generated[:]]
          self.statistics.append(stats)
          return generated
    raise RuntimeError(f'No trajectory anchor after {args.max_tokens} tokens: {tokenizer.decode(generated)}')
