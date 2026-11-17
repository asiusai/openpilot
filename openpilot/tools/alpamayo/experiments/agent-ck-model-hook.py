"""Instance-owned exact-shape hook. Delegates every original text_layer operation."""
from types import MethodType
import inspect

QSHAPE, KVSHAPE = (1,16,2048,128), (1,8,2048,128)

def eligible(context, q, k, v, mask, causal):
  from tinygrad import dtypes
  return (context is not None and mask is None and causal is True and q.shape == QSHAPE
    and k.shape == v.shape == KVSHAPE and q.dtype == k.dtype == v.dtype == dtypes.bfloat16
    and q.device == k.device == v.device == 'AMD' and context['cache_pos'] == 0
    and context['cache'].shape == (2,1,8,2304,128))

class Hook:
  """Install after quantize-packed and selected shuffle-FMA; close before releasing the model."""
  def __init__(self, model, callback):
    assert not hasattr(model,'_agent_ck_model_hook')
    assert model.config['vlm_config']['text_config']['num_hidden_layers'] == 28
    self.model, self.callback, self.context = model, callback, None
    self.saved = {name:(name in vars(model),vars(model).get(name),getattr(model,name)) for name in ('attention','text_layer')}
    self.closed = False; self.calls = []; self.owners = []
    original_layer, original_attention = model.text_layer, model.attention
    def layer(obj,x,*args,**kwargs):
      assert not self.closed and self.context is None
      # Bind through the original core signature, regardless of wrapping signature.
      bound = inspect.signature(type(obj).text_layer).bind(obj,x,*args,**kwargs)
      bound.apply_defaults(); values = bound.arguments
      prefix, cache, pos = values['prefix'], values['cache'], values['cache_pos']
      index = None
      if prefix.startswith('vlm.model.language_model.layers.'):
        tail = prefix.removeprefix('vlm.model.language_model.layers.')
        if tail.isdigit() and 0 <= int(tail) < 28: index = int(tail)
      if index is not None and isinstance(pos,int) and pos == 0 and x.shape == (1,2048,2048):
        assert obj.cache[index] is cache, 'Unexpected language cache owner'
        self.context = {'layer':index,'prefix':prefix,'cache':cache,'cache_pos':pos}
      try: return original_layer(x,*args,**kwargs)
      finally: self.context = None
    def attention(obj,q,k,v,mask=None,causal=False):
      assert not self.closed
      if not eligible(self.context,q,k,v,mask,causal): return original_attention(q,k,v,mask,causal)
      self.calls.append(self.context['layer'])
      return callback(self,self.context,q,k,v,original_attention)
    self.wrappers = {'attention':MethodType(attention,model),'text_layer':MethodType(layer,model)}
    for name, value in self.wrappers.items(): setattr(model,name,value)
    model._agent_ck_model_hook = self

  def close(self):
    assert not self.closed and self.context is None
    for name, value in self.wrappers.items(): assert getattr(self.model,name) is value, 'Concurrent hook replacement'
    for name, (owned,value,_) in self.saved.items():
      if owned: setattr(self.model,name,value)
      else: delattr(self.model,name)
    assert self.model._agent_ck_model_hook is self
    del self.model._agent_ck_model_hook
    self.owners.clear(); self.closed = True

def candidate_outputs(common,q,k,v,ostore,lstore):
  """Optional integration primitive; caller retains both owners through the final fence.

  Poisoning and all packing stay inside the caller's boundary. The qualified ELF writes O and LSE.
  This function does not authorize a route run or supply a passed live-layout attestation.
  """
  helper = common.load('agent-ck-output-common')
  out,lse = helper.outputs('candidate',True,q,k,v,ostore,lstore)
  out.realize(lse)
  return out,lse
