"""2048-row padded prefill, with six independent nominal verification heads."""
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
  def __init__(self, model):
    super().__init__(model)
    self.baseline = PaddedPrefill(model)
    self.last_targets = self.last_prompt = None

  def language(self, x, cos, sin, length):
    model = self.model
    for i in range(model.config['vlm_config']['text_config']['num_hidden_layers']):
      x = model.text_layer(x,cos,sin,f'vlm.model.language_model.layers.{i}',model.cache[i],None,cache_pos=0)
    targets,checks = [],[]
    for index in range(6):
      hidden = model.norm(x[:,length-1+index:length+index],'vlm.model.language_model.norm',rms=True)
      logits = model.linear(hidden,'vlm.lm_head').realize()
      if index == 0:
        original_first_logits = logits
      targets.append(model.output_token_mapping[logits.reshape(-1).argmax()].reshape(1))
      checks.append(logits.isfinite().all().cast(dtypes.int32).reshape(1))
    return original_first_logits,Tensor.cat(*targets,*checks).contiguous().realize()

  def __call__(self, x, cos, sin, max_tokens, length):
    self.last_targets = None
    self.last_prompt = int(length)
    if length+6 > x.shape[1] or max_tokens < 6:
      return self.baseline(x,cos,sin,max_tokens,length)
    self.allocate(x,cos,max_tokens)
    valid = UOp.variable('valid_prefill_draft_length',1,x.shape[1]-6).bind(length)
    first_logits,self.last_targets = self.jit(x,cos,sin,valid)
    return first_logits

  def reset(self):
    super().reset()
    self.baseline.reset()
    self.last_targets = self.last_prompt = None
