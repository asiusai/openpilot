"""Optional exact language-prefill FC1/activation/quantization fusion.

Keep the original text layer, quantizer slot transitions, PackedActivation class,
and FC2 matmul. M2048 requires explicit rows from a passed microcheck manifest.
The parent owns GPU validation, serialization, and candidate acceptance.
"""
from contextlib import contextmanager
import hashlib
import inspect
import math
from pathlib import Path
from types import FunctionType, MethodType

PREFIX = 'vlm.model.language_model.layers.'
ROWS = (2027, 2112)
SUPPORTED_ROWS = (2027, 2048, 2112)
PINS = {'model.py':'fce24a07fdd48bd24c05e67518c986bcb462d09dfb75b47daf4140dc4e67f5fb',
        'quantize-packed.py':'c57368db69b04d2c05c2140712b3a4f337c04717a165004d94693aedccbc141d',
        'agent-packed-epilogue.py':'8e2b5d1aab0ae2c66099beb306f67475e5cc613d87cb32dd18971c004fb83f1a',
        'amd_gemm.py':'a52cc1cd5331be686ef8fcd1516bba2cfbc1092ca1d64251176491add99041b2'}


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require_source(path, name):
  assert sha(path) == PINS[name], f'Unreviewed source for {name}: {path}'


def bound_in_chain(method, module, name):
  """Find the actual installed closure through ordinary installer wrappers."""
  pending, seen, found = [method], set(), []
  while pending:
    value = pending.pop()
    fn = value.__func__ if isinstance(value, MethodType) else value
    if not isinstance(fn, FunctionType) or id(fn) in seen: continue
    seen.add(id(fn))
    assert len(seen) <= 128, 'Unexpected installer closure graph'
    if fn.__globals__ is vars(module) and fn.__name__ == name: found.append(value)
    for cell in fn.__closure__ or ():
      item = cell.cell_contents
      if isinstance(item,(FunctionType,MethodType)): pending.append(item)
  assert len(found) == 1 and isinstance(found[0],MethodType), f'Expected one installed packed {name}'
  return found[0]


def prefill_jits(model):
  values = [('prefill_jit',getattr(model,'prefill_jit',None))]
  padded = getattr(model,'padded_prefill',None)
  if padded is not None:
    values.append(('padded_prefill.jit',getattr(padded,'jit',None)))
    base = getattr(padded,'baseline',None)
    if base is not None: values.append(('padded_prefill.baseline.jit',getattr(base,'jit',None)))
  return [(name,jit) for name,jit in values if jit is not None]


class FusedResult:
  """A single-use protocol token, never a materialized BF16 Tensor."""
  __slots__ = ('owner','frame','packed','scale','_shape','_dtype','_device','phase','consumer')

  def __init__(self, owner, frame, packed, scale, x):
    self.owner,self.frame,self.packed,self.scale = owner,frame,packed,scale
    self._shape,self._dtype,self._device = (*x.shape[:-1],9216),x.dtype,x.device
    self.phase,self.consumer = 'fc1',False

  def valid(self):
    if self.owner.closed or self.owner.frame is not self.frame or not self.frame['active']:
      raise RuntimeError('Fused result escaped its text-layer scope')

  @property
  def shape(self): self.valid(); return self._shape
  @property
  def dtype(self): self.valid(); return self._dtype
  @property
  def device(self): self.valid(); return self._device
  def numel(self): return math.prod(self.shape)

  def relu(self):
    self.valid()
    assert self.phase == 'fc1', 'Expected exactly one ReLU immediately after FC1'
    self.phase = 'relu'
    return self

  def square(self):
    self.valid()
    assert self.phase == 'relu', 'Expected exactly one square immediately after ReLU'
    self.phase = 'square'
    return self

  def __getattr__(self, name): raise TypeError(f'Unsupported fused-result operation: {name}')
  def __array__(self, *args, **kwargs): raise TypeError('Fused results cannot be converted to arrays')
  def __bool__(self): raise TypeError('Fused results cannot be used as conditions')


class Session:
  def __init__(self, model, packed_module, kernel, device, fixed_scales, *, dispatch=None, qualified_rows=ROWS):
    from tinygrad import dtypes
    require_source(packed_module.__file__,'quantize-packed.py')
    require_source(kernel.__file__,'agent-packed-epilogue.py')
    require_source(packed_module._gemm.__code__.co_filename,'amd_gemm.py')
    assert device.arch == 'gfx1200' and model.fp8_wmma and not model.cuda_kernels and model.dtype == dtypes.bfloat16
    assert not getattr(model,'_agent_fused_mlp_active',False), 'Nested fusion installation'
    self.rows = tuple(qualified_rows)
    assert self.rows and len(set(self.rows)) == len(self.rows) and set(self.rows) <= set(SUPPORTED_ROWS)
    self.jits = prefill_jits(model)
    assert all(jit.cnt == 0 and jit.captured is None for _,jit in self.jits), 'Install before any language-prefill tracing'
    self.model,self.module,self.kernel,self.device = model,packed_module,kernel,device
    self.original = {name:getattr(model,name) for name in ('linear','quantize','text_layer')}
    self.old_slots = {name:(name in vars(model),vars(model).get(name)) for name in self.original}
    self.pq = bound_in_chain(model.quantize,packed_module,'quantize_input')
    self.pl = bound_in_chain(model.linear,packed_module,'linear')
    self.pt = bound_in_chain(model.text_layer,packed_module,'text_layer')
    assert all(method.__self__ is model for method in (self.pq,self.pl,self.pt))
    assert tuple(inspect.signature(kernel.packed).parameters) == ('x','weight','xs','ws','output_scale','rows','variant','out')
    qc,lc,tc = (inspect.getclosurevars(m.__func__).nonlocals for m in (self.pq,self.pl,self.pt))
    assert qc['state'] is lc['state'] is tc['state']
    assert qc['capture_scales'] is False, 'Capture-scales mode is unsupported; collect calibration without fusion'
    assert tuple(inspect.signature(self.pq).parameters) == ('x',)
    assert tuple(inspect.signature(self.pl).parameters) == ('x','name','quantized_input','heads')
    original_text = tc['old_layer']
    require_source(original_text.__func__.__code__.co_filename,'model.py')
    assert original_text.__func__.__name__ == 'text_layer'
    self.state,self.fixed,self.weights,self.eligible = qc['state'],qc['fixed'],lc['weights'],lc['eligible']
    layers = model.config['vlm_config']['text_config']['num_hidden_layers']
    assert layers == 28
    self.prefixes = tuple(PREFIX+str(i) for i in range(layers))
    expected = {f'{p}/{i}' for p in self.prefixes for i in range(4)}
    assert set(self.fixed) == set(fixed_scales) == expected, 'A complete fixed four-slot language calibration is required'
    assert all(math.isfinite(float(v)) and float(v) > 0 for v in fixed_scales.values())
    assert all(t.dtype == dtypes.float32 and t.numel() == 1 and t.device == device.device for t in self.fixed.values())
    self.weight_identity = {}
    for prefix in self.prefixes:
      for suffix,shape in (('fc1',(9216,2048)),('fc2',(2048,9216))):
        name = f'{prefix}.mlp.{suffix}'
        assert name in self.eligible and self.eligible[name] is model.weights[name+'.weight']
        assert self.eligible[name].shape == shape and self.weights[name].shape == shape
        assert self.weights[name].dtype == dtypes.fp8e4m3 and self.weights[name].device == device.device
        assert model.weights.get(name+'.bias') is None, 'Language MLP fusion requires no bias'
        scale = model.weight_scales[name+'.weight']
        assert scale.dtype == dtypes.float32 and scale.numel() == 1 and scale.device == device.device
        self.weight_identity[name] = (model.weights[name+'.weight'],self.weights[name],scale)
    self.frame,self.closed = None,False
    self.dispatch = kernel.packed if dispatch is None else dispatch
    self.stats = {'variant':'transpose4x4','allowed_rows':list(self.rows),'python_trace_fc1':0,'python_trace_fc2':0,
                  'python_trace_fused_quantize':0,'layer_hits':{},'shape_hits':{},'layer_completions':0,
                  'passthrough_text_layers':0,'failed_text_layers':0,'methods_restored':False,
                  'prefill_jits_reset_on_exit':[],
                  'counter_scope':'Python eager/tracing calls; cached JIT replays do not increment these counters',
                  'capture_scales_supported':False,'kernel_gpu_evidence_required':True}
    # Reuse the original quantizer code and closure. Replace only its private
    # global dispatch for our token. Its index/key/capture handling stays intact.
    qfn = self.pq.__func__
    cloned = FunctionType(qfn.__code__,dict(qfn.__globals__,quantize=self.consume),qfn.__name__,qfn.__defaults__,qfn.__closure__)
    cloned.__kwdefaults__ = qfn.__kwdefaults__
    self.token_quantize = MethodType(cloned,model)

  def consume(self, token, scale=None):
    assert type(token) is FusedResult and token.owner is self
    token.valid()
    assert token.consumer and token.phase == 'square'
    assert self.state == {'prefill':True,'prefix':token.frame['prefix'],'index':4}, 'Unexpected packed quantizer slot transition'
    assert scale is token.scale is self.fixed[token.frame['prefix']+'/3'], 'Output scale identity changed'
    token.phase = 'quantized'
    self.stats['python_trace_fused_quantize'] += 1
    return token.packed,scale

  def quantize(self, x):
    if type(x) is not FusedResult: return self.original['quantize'](x)
    x.valid()
    assert x.owner is self and x.consumer and x.phase == 'square', 'Fused quantization is reserved for the original FC2 call'
    assert self.state == {'prefill':True,'prefix':x.frame['prefix'],'index':3}
    result = self.token_quantize(x)
    assert type(result[0]) is self.module.PackedActivation
    assert result[0].value is x.packed and result[1] is x.scale
    return result

  def linear(self, x, name, quantized_input=None, heads=None):
    from tinygrad import dtypes
    if type(x) is FusedResult:
      x.valid()
      assert x.owner is self and name == x.frame['prefix']+'.mlp.fc2'
      assert quantized_input is None and heads is None and x.phase == 'square' and not x.consumer
      assert self.state == {'prefill':True,'prefix':x.frame['prefix'],'index':3}
      self.check_weights(name)
      x.consumer = True
      try: result = self.original['linear'](x,name,quantized_input,heads)
      finally: x.consumer = False
      assert x.phase == 'quantized' and self.state['index'] == 4
      x.phase,x.frame['consumed'] = 'done',True
      self.stats['python_trace_fc2'] += 1
      return result
    frame = self.frame
    if frame is None or name != frame['prefix']+'.mlp.fc1': return self.original['linear'](x,name,quantized_input,heads)
    assert frame['active'] and frame['token'] is None and quantized_input is None and heads is None
    assert x.shape == (1,frame['rows'],2048) and x.dtype == dtypes.bfloat16 and x.device == self.device.device
    assert self.state == {'prefill':True,'prefix':frame['prefix'],'index':2}, 'FC1 must consume packed slot 2'
    self.check_weights(name)
    packed,scale = self.model.quantize(x)
    assert type(packed) is self.module.PackedActivation
    assert (packed.rows,packed.width) == (frame['rows'],2048)
    assert packed.value.shape == ((frame['rows']+15)//16*16,2048) and packed.value.dtype == dtypes.fp8e4m3
    assert packed.value.device == x.device and scale is self.fixed[frame['prefix']+'/2']
    assert self.state['index'] == 3
    output_scale = self.fixed[frame['prefix']+'/3']
    output = self.dispatch(packed.value,self.weights[name],scale,self.model.weight_scales[name+'.weight'],
                           output_scale,frame['rows'],variant='transpose4x4')
    assert output.shape == ((frame['rows']+15)//16*16,9216) and output.dtype == dtypes.fp8e4m3 and output.device == x.device
    token = frame['token'] = FusedResult(self,frame,output,output_scale,x)
    self.stats['python_trace_fc1'] += 1
    self.stats['layer_hits'][frame['prefix']] = self.stats['layer_hits'].get(frame['prefix'],0)+1
    key = str(frame['rows'])
    self.stats['shape_hits'][key] = self.stats['shape_hits'].get(key,0)+1
    return token

  def check_weights(self, name):
    old,packed,scale = self.weight_identity[name]
    assert self.model.weights[name+'.weight'] is old and self.weights[name] is packed
    assert self.model.weight_scales[name+'.weight'] is scale and self.model.weights.get(name+'.bias') is None

  def text_layer(self, x, *args, **kwargs):
    from tinygrad import dtypes
    prefix = args[2] if len(args) >= 3 else kwargs['prefix']
    past = args[5] if len(args) >= 6 else kwargs.get('cache_pos')
    eligible = prefix in self.prefixes and x.shape in ((1,r,2048) for r in self.rows) and x.dtype == dtypes.bfloat16 and type(past) is int and past == 0
    previous = self.frame
    frame = {'prefix':prefix,'rows':x.shape[1],'active':True,'token':None,'consumed':False} if eligible else None
    self.frame = frame
    try:
      if frame is None: self.stats['passthrough_text_layers'] += 1
      result = self.original['text_layer'](x,*args,**kwargs)
      if frame is not None:
        assert frame['token'] is not None and frame['consumed'], 'Eligible text layer did not complete the original FC1/ReLU/square/FC2 sequence'
        self.stats['layer_completions'] += 1
      return result
    except BaseException:
      self.stats['failed_text_layers'] += 1
      raise
    finally:
      if frame is not None: frame['active'] = False
      self.frame = previous


@contextmanager
def install(model, packed_module, kernel, device, fixed_scales, *, dispatch=None, qualified_rows=ROWS):
  session = Session(model,packed_module,kernel,device,fixed_scales,dispatch=dispatch,qualified_rows=qualified_rows)
  installed = {}
  marker = ('_agent_fused_mlp_active' in vars(model),vars(model).get('_agent_fused_mlp_active'))
  try:
    model._agent_fused_mlp_active = True
    for name in session.original:
      method = getattr(session,name)
      installed[name] = MethodType(lambda self,*a,_method=method,**kw:_method(*a,**kw),model)
      setattr(model,name,installed[name])
    yield session
  finally:
    session.closed = True
    if session.frame is not None: session.frame['active'] = False
    conflict = [name for name,value in installed.items() if getattr(model,name) is not value]
    for name,(existed,value) in session.old_slots.items():
      if name in installed and getattr(model,name) is installed[name]:
        if existed: setattr(model,name,value)
        else: delattr(model,name)
    if marker[0]: model._agent_fused_mlp_active = marker[1]
    else: delattr(model,'_agent_fused_mlp_active')
    session.stats['methods_restored'] = not conflict
    # The captured graph embeds fused kernels. Restoring Python methods alone
    # cannot remove them. These prefill JITs were cold before this installation.
    if session.stats['python_trace_fc1']:
      for name,jit in session.jits:
        jit.reset()
        session.stats['prefill_jits_reset_on_exit'].append(name)
    if conflict: raise RuntimeError(f'Other code replaced installed methods; preserved those changes: {conflict}')
