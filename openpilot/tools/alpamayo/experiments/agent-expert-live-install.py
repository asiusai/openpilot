"""Only qualified expert PV and explicit storage bindings around frozen live hooks."""
import copy
import functools
import sys
from unittest.mock import patch


def factory(N):
  m,manifest = N.live()
  original = N.load(N.EXP/'agent-exact-live-install.py','expert_live_frozen_install')
  base_class = original.composition_class(m)

  class PVComposition(base_class):
    TARGETS = base_class.TARGETS | {'expert-kernels.py'}

    def __init__(self,*args,**kwargs):
      super().__init__(*args,**kwargs)
      self.pv_handle = None

    def after_load(self,module,path):
      if path.name != 'expert-kernels.py': return super().after_load(module,path)
      self.require_path(path)
      install = module.install
      @functools.wraps(install)
      def wrapped(model,*values,**options):
        assert self.pv_handle is None and self.packed is None and self.session is None
        result = install(model,*values,**options)
        P,evidence = N.pv()
        hook = N.load(N.PV_ROOT/'agent-expert-pv-qualified-hook.py','expert_live_frozen_pv_hook')
        hook.common = lambda:P
        before = tuple(getattr(model,n) for n in ('linear','quantize','text_layer','attention'))
        search = sys.path[:]
        try: self.pv_handle = hook.install(model,module,candidate=self.candidate,evidence=evidence)
        finally: sys.path[:] = search
        assert before == tuple(getattr(model,n) for n in ('linear','quantize','text_layer','attention'))
        self.stack.callback(self.pv_handle.close)
        self.stack.enter_context(self.pv_handle.observe_programs())
        self.events.append('qualified_pv_before_later_quantizer_and_precision_wrappers')
        return result
      self.patched(module,'install',wrapped)

  class Hooks(original.Hooks):
    def __init__(self,mode,**kwargs):
      with patch.object(original,'composition_class',lambda common:PVComposition):
        super().__init__(m,manifest,mode,**kwargs)
      self.storage_bindings = []

    def replay(self,module):
      super().replay(module)
      normalize = module.normalize
      def storage(*args,**kwargs):
        normalized,parsed,protocol = normalize(*args,**kwargs)
        # This is the recorded recipe. The actual storage bindings are explicit below.
        m.equal(protocol,N.read(N.REFERENCE)['protocol'],'saved_two_query_protocol')
        before = copy.deepcopy(normalized)
        normalized = copy.deepcopy(normalized)
        normalized['environment'].update(N.storage())
        assert before['command'] == normalized['command']
        assert all(normalized['environment'][k] == v for k,v in before['environment'].items() if k not in N.storage())
        self.storage_bindings.append({'recorded_environment':{k:before['environment'].get(k) for k in N.storage()},
          'actual_environment':N.storage(),'identical_weight_cache':True,'model_arguments_unchanged':True,
          'numeric_environment_unchanged':True})
        return normalized,parsed,protocol
      self.patch(module,'normalize',storage)

    def coverage(self):
      result = super().coverage()
      assert self.composition.pv_handle is not None
      assert self.composition.session.frame is None and not self.composition.session.state['prefill']
      assert self.storage_bindings and all(x['actual_environment'] == N.storage() for x in self.storage_bindings)
      return result

  return Hooks
