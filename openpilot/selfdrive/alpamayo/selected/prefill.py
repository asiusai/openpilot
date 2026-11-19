"""Padded prefill (2176 rows) with draft verification heads."""
from tinygrad import Tensor, TinyJit, dtypes
from tinygrad.helpers import round_up
from tinygrad.uop.ops import UOp


class PaddedPrefill:
  def __init__(self, model):
    self.model = model
    self.jit = TinyJit(self.language)
    self.shape = None

  def language(self, x, cos, sin, length):
    model = self.model
    for i in range(model.config['vlm_config']['text_config']['num_hidden_layers']):
      x = model.text_layer(x,cos,sin,f'vlm.model.language_model.layers.{i}',model.cache[i],None,cache_pos=0)
    x = model.norm(x[:,length-1:length],'vlm.model.language_model.norm',rms=True)
    return model.linear(x,'vlm.lm_head').realize()

  def allocate(self, x, cos, max_tokens):
    model = self.model
    capacity = round_up(x.shape[1]+max_tokens,256)
    if capacity != 2304:
      raise ValueError('Selected KV allocation must contain 2304 rows')
    if model.cache is None or model.cache[0].shape[3] != capacity:
      model.cache = [Tensor.zeros(2,1,8,capacity,128,dtype=model.dtype).realize()
                     for _ in range(model.config['vlm_config']['text_config']['num_hidden_layers'])]
      model.prefill_jit.reset()
      model.decode_jit.reset()
      model.trajectory_jits.clear()
      self.jit.reset()
      self.shape = None
    shape = (x.shape,cos.shape,capacity)
    if self.shape != shape:
      self.jit.reset()
      self.shape = shape
    model.prompt_length = x.shape[1]

  def __call__(self, x, cos, sin, max_tokens, length):
    self.allocate(x,cos,max_tokens)
    valid = UOp.variable('valid_prefill_length',1,x.shape[1]).bind(length)
    return self.jit(x,cos,sin,valid)

  def reset(self):
    self.jit.reset()
    self.shape = None


class SpeculativePrefill(PaddedPrefill):
  """Prefill that also verifies a draft placed in the padded rows after the prompt.

  With H heads, H-1 draft tokens are placed and head i holds the target after draft token i-1 (head 0 follows the
  prompt). The decoder accepts the matching draft prefix, whose KV rows this prefill wrote, and continues from the
  first differing head. Prompts up to 2145 tokens fit 32 heads (31 draft tokens); the frontend caps navigation text.
  """
  HEADS = 32

  def __init__(self, model):
    super().__init__(model)
    self.jits = {}
    self.last_targets = self.last_prompt = self.last_logits = None
    self.heads = 0

  def language(self, x, cos, sin, length, heads):
    model = self.model
    for i in range(model.config['vlm_config']['text_config']['num_hidden_layers']):
      x = model.text_layer(x,cos,sin,f'vlm.model.language_model.layers.{i}',model.cache[i],None,cache_pos=0)
    rows,targets,checks = [],[],[]
    for index in range(heads):
      hidden = model.norm(x[:,length-1+index:length+index],'vlm.model.language_model.norm',rms=True)
      logits = model.linear(hidden,'vlm.lm_head').realize()
      rows.append(logits)
      targets.append(model.output_token_mapping[logits.reshape(-1).argmax()].reshape(1))
      checks.append(logits.isfinite().all().cast(dtypes.int32).reshape(1))
    return rows[0],Tensor.cat(*rows,dim=1).contiguous().realize(),Tensor.cat(*targets,*checks).contiguous().realize()

  def __call__(self, x, cos, sin, max_tokens, length):
    from .policy import PromptTooLong
    self.last_targets = self.last_logits = None
    self.last_prompt = int(length)
    heads = self.HEADS
    if length+heads-1 > x.shape[1] or max_tokens < heads:
      raise PromptTooLong(f'Prompt uses {length} tokens; the draft prefill holds {x.shape[1]-heads+1}')
    self.allocate(x,cos,max_tokens)
    key = (heads,self.shape)
    if key not in self.jits:
      from .graph_cache import CachedJit
      self.jits[key] = CachedJit(lambda x,c,s,n: self.language(x,c,s,n,heads),f'prefill-{heads}-{self.shape}')
    valid = UOp.variable(f'valid_prefill_heads{heads}_length',1,x.shape[1]-heads+1).bind(length)
    first_logits,self.last_logits,self.last_targets = self.jits[key](x,cos,sin,valid)
    self.heads = heads
    return first_logits

  def reset(self):
    super().reset()
    self.jits.clear()
    self.last_targets = self.last_prompt = self.last_logits = None
