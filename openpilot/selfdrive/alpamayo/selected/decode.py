"""Target-verified nominal/lookup decoding with block-six serial fallback."""
import math
import numpy as np
from tinygrad import Tensor, TinyJit
from tinygrad.uop.ops import UOp
from .policy import accept_targets, validate_draft
from .transport import download


class Decoder:
  def __init__(self, model, draft, anchor):
    self.model = model
    self.draft,self.anchor = validate_draft(draft,anchor),anchor
    self.history,self.statistics = [],{}
    self.jits,self.verifiers,self.select_jits = {},{},{}
    self.choose_jit = TinyJit(lambda logits: model.output_token_mapping[logits.reshape(-1).argmax()].float().realize())

  def reset(self):
    self.history.clear()
    self.statistics = {}
    self.jits.clear()
    self.verifiers.clear()
    self.select_jits.clear()
    self.choose_jit.reset()

  def first_token(self, logits):
    return int(download(self.choose_jit(logits)).item())

  def select_logits(self, logits, index):
    count = logits.shape[1]
    if count not in self.select_jits:
      self.select_jits[count] = TinyJit(lambda values,row: values[:,row:row+1].clone().realize())
    row = UOp.variable('selected_draft_row',0,count-1).bind(index)
    return self.select_jits[count](logits,row)

  def decode(self, logits, cos, sin, position, step, count):
    key = (cos.shape,count,self.model.cache[0].shape[3])
    if key not in self.jits:
      def run(logits, c, s, pos, offset):
        tokens = []
        for i in range(count):
          logits,token = self.model._greedy_decode(logits,c,s,pos+i,offset+i)
          tokens.append(token.reshape(1))
        return logits,Tensor.cat(*tokens).realize()
      self.jits[key] = TinyJit(run)
    position = UOp.variable('block_position',0,self.model.cache[0].shape[3]-count).bind(position)
    step = UOp.variable('block_step',0,cos.shape[2]-count).bind(step)
    return self.jits[key](logits,cos,sin,position,step)

  def verify(self, token_ids, cos, sin, position, step):
    model = self.model
    count = token_ids.shape[0]
    x = model.weights['vlm.model.language_model.embed_tokens.weight'][token_ids].reshape(1,count,-1)
    c,s = cos[:,:,step:step+count],sin[:,:,step:step+count]
    mask = (Tensor.arange(model.cache[0].shape[3]).reshape(1,-1) > (Tensor.arange(count)+position).reshape(-1,1)).where(-math.inf,0)
    with model.verification():
      for i in range(model.config['vlm_config']['text_config']['num_hidden_layers']):
        x = model.text_layer(x,c,s,f'vlm.model.language_model.layers.{i}',model.cache[i],mask,cache_pos=position)
      logits = model.linear(model.norm(x,'vlm.model.language_model.norm',rms=True),'vlm.lm_head').realize()
    tokens = model.output_token_mapping[logits.argmax(axis=-1)].realize()
    return logits,tokens

  def continuation(self, context, limit):
    for width in range(min(8,len(context)),0,-1):
      suffix = context[-width:]
      for previous in reversed(self.history):
        for end in range(len(previous)-1,width-1,-1):
          if previous[end-width:end] == suffix:
            return previous[end:end+limit]
    return []

  def generate_lookup(self, logits, dc, ds, prompt, max_tokens):
    model = self.model
    generated = []
    stats = {'lookup_rounds':0,'verified_tokens':0,'serial_tokens':0}
    next_token,pending_logits = None,None
    while len(generated) < max_tokens:
      offset = len(generated)
      remaining = max_tokens-offset
      first = self.first_token(logits) if next_token is None else next_token
      next_token = None
      proposed = [first]+self.continuation(generated+[first],min(64,remaining)-1)
      count = ((len(proposed)+15)//16)*16
      if len(proposed) <= 1 or count > remaining:
        if pending_logits is not None:
          logits = self.select_logits(*pending_logits)
          pending_logits = None
        logits,tokens = self.decode(logits,dc,ds,prompt+offset,offset,min(6,remaining))
        accepted = download(tokens).reshape(-1).astype(int).tolist()
        stats['serial_tokens'] += len(accepted)
      else:
        tokens = Tensor(np.array(proposed+[proposed[-1]]*(count-len(proposed)),dtype=np.int32),device='CPU').to(logits.device).realize()
        pos = UOp.variable('draft_position',0,model.cache[0].shape[3]-count).bind(prompt+offset)
        step = UOp.variable('draft_step',0,dc.shape[2]-count).bind(offset)
        key = (count,dc.shape,model.cache[0].shape[3])
        if key not in self.verifiers:
          self.verifiers[key] = TinyJit(self.verify)
        verified,predicted = self.verifiers[key](tokens,dc,ds,pos,step)
        choices = [first,*download(predicted).reshape(-1).astype(int).tolist()]
        accepted = []
        for candidate,choice in zip(proposed,choices,strict=False):
          if candidate != choice:
            break
          accepted.append(candidate)
          if candidate == self.anchor:
            break
        assert accepted
        if accepted[-1] != self.anchor:
          next_token = choices[len(accepted)]
          pending_logits = (verified,len(accepted)-1)
        stats['lookup_rounds'] += 1
        stats['verified_tokens'] += len(accepted)
      for token in accepted:
        generated.append(token)
        if token == self.anchor:
          self.history = [old for old in self.history if old != generated][-63:]+[generated[:]]
          self.statistics = stats
          return generated
    raise RuntimeError(f'No trajectory anchor after {max_tokens} tokens')

  def generate(self, prefill, logits, dc, ds, prompt, max_tokens=64):
    if prefill.last_prompt != prompt:
      raise ValueError('Prefill used a stale actual prompt length')
    record = {'eligible':prefill.last_targets is not None,'accepted':False,'actual_prompt':prompt}
    if prefill.last_targets is not None:
      payload = download(prefill.last_targets).reshape(-1)
      if payload.shape != (12,):
        raise ValueError('Incomplete target token and finiteness payload')
      targets = payload[:6].astype(np.int64)
      finite = bool(np.all(payload[6:] == 1))
      accepted = accept_targets(targets,np.asarray(self.draft),self.anchor,finite)
      record.update(targets=targets.tolist(),draft=list(self.draft),all_targets_finite=finite,accepted=accepted)
      if accepted:
        generated = list(self.draft)
        self.history = [old for old in self.history if old != generated][-63:]+[generated[:]]
        self.statistics = {'lookup_rounds':0,'verified_tokens':6,'serial_tokens':0,'prefill_draft':record}
        return generated
    generated = self.generate_lookup(logits,dc,ds,prompt,max_tokens)
    self.statistics['prefill_draft'] = record
    return generated
