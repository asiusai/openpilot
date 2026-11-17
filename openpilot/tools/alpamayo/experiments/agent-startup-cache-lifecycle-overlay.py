"""Candidate-only KV lifecycle repair over the immutable persistence installer."""
import ast
from contextlib import contextmanager, ExitStack
import hashlib
import importlib.util
from pathlib import Path
from unittest.mock import patch

class LazyClosedOwners:
  """Guard immutable owners immediately; bind original full guard at real first use."""
  def __init__(self,m,model):
    self.model=model;self.full=None;self.bindings=0
    self.closed=m.load(m.EXP/'agent-plan-family-reload-diagnostic.py').ClosedOwners
    class ImmutableView:
      cache=()
      def __getattr__(self,name): return getattr(model,name)
    self.immutable=self.closed(ImmutableView())
    if model.cache is not None: self.bind_ready()

  def layout(self):
    from tinygrad import dtypes
    cache=self.model.cache
    if not isinstance(cache,list) or len(cache)!=28: raise ValueError('Real prefill must create all28 KV owners before serial decode')
    if any(tuple(t.shape)!=(2,1,8,2304,128) or t.dtype is not dtypes.bfloat16 or t.device!='AMD' for t in cache):
      raise ValueError('KV layout/dtype/device differs from qualified serial plans')
    if len({t.uop.base for t in cache})!=28: raise ValueError('Distinct original KV allocations required')

  def bind_ready(self):
    self.immutable.guard();self.layout()
    if self.full is None:
      self.full=self.closed(self.model);self.bindings+=1
    self.full.guard()

  def guard(self):
    self.immutable.guard()
    if self.full is None: raise ValueError('Full KV guard was never bound by actual serial startup')
    self.layout();self.full.guard()

def specialize(base,m,state):
  source=(m.EXP/'agent-startup-persist-install.py').read_text()
  tree=ast.parse(source);cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='SerialPlans')
  method=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='__init__')
  original=ast.get_source_segment(source,method)
  before="self.owners=m.load(m.EXP/'agent-plan-family-reload-diagnostic.py').ClosedOwners(components.model) if builder is None else None"
  after='self.owners=_lazy_closed_owners(m,components.model) if builder is None else None'
  if original.count(before)!=1: raise ValueError('Frozen constructor anchor changed')
  adapted=original.replace(before,after)
  if adapted.replace(after,before)!=original: raise AssertionError('Non-lifecycle constructor edit')
  def owners(m,model):
    value=LazyClosedOwners(m,model);state['owners'].append(value);return value
  namespace=dict(vars(base),_lazy_closed_owners=owners)
  exec(compile(adapted,str(Path(__file__))+':constructor','exec'),namespace)
  parent=base.SerialPlans
  class SerialPlans(parent):
    __init__=namespace['__init__']
    def decode(self,*args,**kwargs):
      if self.owners is not None: self.owners.bind_ready()
      return super().decode(*args,**kwargs)
  state['derivation']=dict(frozen_file_sha256=hashlib.sha256(source.encode()).hexdigest(),
    original_constructor_sha256=hashlib.sha256(original.encode()).hexdigest(),
    adapted_constructor_sha256=hashlib.sha256(adapted.encode()).hexdigest(),before=before,after=after,
    unchanged_methods=['build','seal','installing'],unchanged_functions=['initialize','signatures'],
    decode_change='Require bound real KV owners before delegating every argument to the unchanged decode method')
  if any(getattr(SerialPlans,k) is not getattr(parent,k) for k in ('build','seal','installing')):
    raise AssertionError('Unrelated persistence implementation changed')
  return SerialPlans

@contextmanager
def imports(m):
  original=importlib.util.spec_from_file_location;state={'owners':[],'modules':0};stack=ExitStack()
  def spec(name,location,*args,**kwargs):
    result=original(name,location,*args,**kwargs)
    if result is None or location is None or Path(location).resolve()!=m.EXP/'agent-startup-persist-install.py': return result
    loader=result.loader
    class Loader:
      def create_module(self,spec): return loader.create_module(spec)
      def exec_module(self,module):
        loader.exec_module(module);before=module.SerialPlans;after=specialize(module,m,state)
        module.SerialPlans=after;state['modules']+=1
        def restore():
          if module.SerialPlans is not after: raise RuntimeError('Lifecycle overlay ownership changed')
          module.SerialPlans=before
        stack.callback(restore)
    result.loader=Loader();return result
  try:
    with patch.object(importlib.util,'spec_from_file_location',spec): yield state
  finally: stack.close()

def summary(state):
  for value in state['owners']: value.guard()
  return dict(imported_installers=state['modules'],production_owner_guards=len(state['owners']),
    KV_bindings=[v.bindings for v in state['owners']],derivation=state.get('derivation'),
    immutable_owners_guarded_before_prefill=True,full_original_owner_guard_retained=True)
