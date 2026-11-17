"""Scoped optional installation through the actual frozen packed quantizer.

The original text_layer and installer closures remain installed and execute unchanged.
"""
from contextlib import contextmanager
import hashlib
import inspect
import math
from pathlib import Path
from types import FunctionType, MethodType

PREFIX = 'vlm.model.language_model.layers.'
PINS = {'model.py': 'fce24a07fdd48bd24c05e67518c986bcb462d09dfb75b47daf4140dc4e67f5fb',
        'fused-norm.py': '4271693fc1f3210f160eac0a4ddb69dad168612c712d4f77a755ef2eda178b7c',
        'quantize-packed.py': 'c57368db69b04d2c05c2140712b3a4f337c04717a165004d94693aedccbc141d'}


def require(path, name):
  if hashlib.sha256(Path(path).read_bytes()).hexdigest() != PINS[name]: raise ValueError('Unreviewed '+name)


def find(method, filename, name):
  queue, seen, found = [method], set(), []
  while queue:
    item = queue.pop()
    fn = item.__func__ if isinstance(item, MethodType) else item
    if not isinstance(fn, FunctionType) or id(fn) in seen: continue
    seen.add(id(fn))
    if len(seen) > 128: raise ValueError('Unexpected installer closure graph')
    if Path(fn.__code__.co_filename).name == filename and fn.__name__ == name: found.append(item)
    queue.extend(cell.cell_contents for cell in fn.__closure__ or () if isinstance(cell.cell_contents, (MethodType, FunctionType)))
  if len(found) != 1: raise ValueError(f'Expected one actual {filename}/{name}')
  require(found[0].__func__.__code__.co_filename, filename)
  return found[0]


class PackedNorm:
  """Single-use metadata token. Frozen linear/quantizer functions read only these fields."""
  def __init__(self, owner, x, packed, scale, prefix):
    self.owner, self.shape, self.dtype, self.device = owner, x.shape, x.dtype, x.device
    self.packed, self.scale, self.prefix, self.consumed = packed, scale, prefix, False
  def numel(self): return math.prod(self.shape)
  def __getattr__(self, key): raise TypeError('Unexpected normalized-token operation: '+key)


class Session:
  def __init__(self, model, kernel):
    from tinygrad import dtypes
    if getattr(model, '_agent_residual_norm_active', False) or getattr(model, '_agent_fused_mlp_active', False):
      raise ValueError('Another fusion is active')
    pq = find(model.quantize, 'quantize-packed.py', 'quantize_input')
    pl = find(model.linear, 'quantize-packed.py', 'linear')
    pt = find(model.text_layer, 'quantize-packed.py', 'text_layer')
    pn = find(model.norm, 'fused-norm.py', 'apply')
    if any(method.__self__ is not model for method in (pq,pl,pt,pn)): raise ValueError('Installer belongs to another model')
    q, l, t = (inspect.getclosurevars(m.__func__).nonlocals for m in (pq, pl, pt))
    if q['state'] is not l['state'] or q['state'] is not t['state']: raise ValueError('Split quantizer state')
    require(t['old_layer'].__func__.__code__.co_filename, 'model.py')
    if q['capture_scales'] or not q['fixed']: raise ValueError('Fixed language scales required; capture mode is unsupported')
    assert model.dtype == dtypes.bfloat16 and model.fp8_wmma and not model.cuda_kernels
    assert model.config['vlm_config']['text_config']['num_hidden_layers'] == 28
    expected = {f'{PREFIX}{i}/{slot}' for i in range(28) for slot in range(4)}
    if set(q['fixed']) != expected: raise ValueError('Incomplete fixed calibration')
    if any(value.dtype!=dtypes.float32 or value.numel()!=1 for value in q['fixed'].values()):
      raise ValueError('Fixed scales must be original scalar FP32 tensors')
    self.model, self.kernel, self.state, self.fixed = model, kernel, q['state'], q['fixed']
    self.jits = [getattr(model, 'prefill_jit', None)]
    padded = getattr(model, 'padded_prefill', None)
    self.jits += [getattr(padded, 'jit', None), getattr(getattr(padded, 'baseline', None), 'jit', None)]
    self.jits = [jit for jit in self.jits if jit is not None]
    if any(jit.cnt != 0 or jit.captured is not None for jit in self.jits): raise ValueError('Install before prefill tracing')
    self.global_dict = pq.__func__.__globals__
    self.old_quantize = self.global_dict['quantize']
    require(self.old_quantize.__code__.co_filename,'quantize-packed.py')
    self.original_norm, self.original_layer = model.norm, model.text_layer
    self.norm_slot = ('norm' in vars(model), vars(model).get('norm'))
    self.layer_slot = ('text_layer' in vars(model), vars(model).get('text_layer'))
    self.pending, self.events, self.closed = None, [], False

    def norm(this, x, name, rms=False, eps=None):
      if not (name.startswith(PREFIX) and name.endswith('.post_attention_layernorm') and
              self.state['prefill'] and x.shape == (1,2048,2048)):
        return self.original_norm(x, name, rms=rms, eps=eps)
      from tinygrad import Tensor
      from tinygrad.uop.ops import Ops
      prefix = name.removesuffix('.post_attention_layernorm')
      if not rms or eps not in (None, 1e-5) or self.state != dict(prefill=True,prefix=prefix,index=2):
        raise ValueError('Unexpected post-attention quantizer state')
      if self.pending is not None: raise ValueError('Unconsumed normalized token')
      if x.uop.op is not Ops.ADD or x.uop.dtype != dtypes.bfloat16 or len(x.uop.src) != 2:
        raise ValueError('Residual is not the frozen two-input BF16 ADD')
      a, other = (Tensor(value) for value in x.uop.src)
      if a.shape != x.shape or other.shape != x.shape or a.dtype != other.dtype or a.dtype != dtypes.bfloat16:
        raise ValueError('Residual operands changed shape or dtype')
      scale = self.fixed[prefix+'/2']
      residual, packed = self.kernel.fused(a, other, this.weights[name+'.weight'], scale)
      # x is the fresh local Tensor created by frozen text_layer's residual ADD.
      # Both kernel operands were extracted before replacement, preventing a cycle.
      x.replace(residual)
      token = PackedNorm(self, x, packed, scale, prefix)
      self.pending = token
      return token

    def quantize(x, scale=None):
      if not isinstance(x, PackedNorm): return self.old_quantize(x, scale)
      if self.closed or x.owner is not self or self.pending is not x or x.consumed:
        raise ValueError('Escaped or reused normalized token')
      if self.state != dict(prefill=True,prefix=x.prefix,index=3) or scale is not x.scale:
        raise ValueError('Frozen quantizer did not consume exactly slot 2')
      x.consumed, self.pending = True, None
      self.events.append(dict(prefix=x.prefix, slot_before=2, slot_after=3, rows=2048))
      return x.packed, scale

    def layer(this, *args, **kwargs):
      try:
        result = self.original_layer(*args, **kwargs)
        if self.pending is not None: raise ValueError('FC1 did not consume normalized token')
        return result
      finally: self.pending = None

    self.global_dict['quantize'] = quantize
    model.norm, model.text_layer = MethodType(norm, model), MethodType(layer, model)
    model._agent_residual_norm_active = True

  def close(self):
    if self.closed: return
    self.closed = True
    self.global_dict['quantize'] = self.old_quantize
    for name, (own, old) in (('norm', self.norm_slot), ('text_layer', self.layer_slot)):
      if own: setattr(self.model, name, old)
      else: delattr(self.model, name)
    delattr(self.model, '_agent_residual_norm_active')
    for jit in self.jits: jit.reset()
    self.pending = None


@contextmanager
def installed(model, kernel):
  session = Session(model, kernel)
  try: yield session
  finally: session.close()
