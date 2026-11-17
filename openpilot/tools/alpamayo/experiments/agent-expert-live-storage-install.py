"""Bind storage before transformed main copies globals; observe actual factory I/O."""
import copy
from contextlib import contextmanager
import functools
import os
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch


def factory(S,*,pv_candidate=True):
  N=S.N;m,manifest=N.live()
  original=N.load(N.EXP/'agent-exact-live-install.py','storage_exact_live_installer')
  base=original.composition_class(m)

  class Composition(base):
    TARGETS=base.TARGETS|{'expert-kernels.py','benchmark-fast.py'}
    def __init__(self,*a,**kw):
      super().__init__(*a,**kw);self.pv_handle=None;self.storage_events=[];self.constructor_restored=False

    def environment(self,event):
      actual={k:os.environ.get(k) for k in N.storage()}
      assert actual==N.storage(),f'Actual storage differs at {event}'
      value=dict(event=event,actual_environment=actual)
      self.storage_events.append(value);return value

    def after_load(self,module,path):
      if path.name=='benchmark-fast.py':
        self.require_path(path);owner=self;real_load=module.load;real_path=module.Path;real_tempfile=module.tempfile
        assert module.EdgeModel.__init__ is real_load
        def restore_constructor():
          assert module.EdgeModel.__init__ is real_load
          module.EdgeModel.__init__=module.original_init
          owner.constructor_restored=module.EdgeModel.__init__ is module.original_init
        self.stack.callback(restore_constructor)
        class ObservedPath(type(Path())):
          def open(path,*a,**kw):
            if path.parent==N.WEIGHTS:
              assert path.name in ('index.json','weights.bin')
              owner.storage_events.append(dict(event='cache_open',path=str(path.resolve()),mode=a[0] if a else kw.get('mode','r')))
            return super().open(*a,**kw)
        @contextmanager
        def temporary(*a,**kw):
          with real_tempfile.TemporaryDirectory(*a,**kw) as directory:
            assert Path(directory).parent==Path(N.storage()['TMPDIR'])
            owner.storage_events.append(dict(event='factory_temporary_directory',path=directory))
            yield directory
        @functools.wraps(real_load)
        def load(model,*a,**kw):
          from tinygrad.helpers import CACHEDB
          value=owner.environment('factory_enter');value['cachedb_constant']=CACHEDB
          assert CACHEDB==N.storage()['CACHEDB']
          with patch.object(module,'Path',ObservedPath),patch.object(module,'tempfile',SimpleNamespace(TemporaryDirectory=temporary)):
            result=real_load(model,*a,**kw)
          owner.environment('factory_complete');return result
        self.patched(module,'load',load);self.patched(module.EdgeModel,'__init__',load)
        return
      if path.name!='expert-kernels.py':return super().after_load(module,path)
      self.require_path(path);install=module.install
      @functools.wraps(install)
      def wrapped(model,*a,**kw):
        assert self.pv_handle is None and self.packed is None and self.session is None
        result=install(model,*a,**kw);P,evidence=N.pv()
        hook=N.load(N.PV_ROOT/'agent-expert-pv-qualified-hook.py','storage_qualified_pv_hook');hook.common=lambda:P
        before=tuple(getattr(model,n) for n in ('linear','quantize','text_layer','attention'));search=sys.path[:]
        try:self.pv_handle=hook.install(model,module,candidate=pv_candidate,evidence=evidence)
        finally:sys.path[:]=search
        assert before==tuple(getattr(model,n) for n in ('linear','quantize','text_layer','attention'))
        self.stack.callback(self.pv_handle.close);self.stack.enter_context(self.pv_handle.observe_programs())
        self.events.append('qualified_pv_before_later_quantizer_and_precision_wrappers');return result
      self.patched(module,'install',wrapped)

  class Hooks(original.Hooks):
    def __init__(self,mode,**kw):
      assert mode in ('consumer','paced')
      with patch.object(original,'composition_class',lambda common:Composition):super().__init__(m,manifest,mode,**kw)
      self.storage_events=self.composition.storage_events

    def replay(self,module):
      # Install before original.replay builds a main with dict(module.__dict__).
      original_normalize=module.normalize
      def normalize(*a,**kw):
        normalized,args,protocol=original_normalize(*a,**kw)
        m.equal(protocol,manifest['original_protocol'],'unaltered_full_original_recipe')
        before=normalized;normalized=copy.deepcopy(normalized);normalized['environment'].update(N.storage())
        assert before['command']==normalized['command']
        assert all(normalized['environment'][k]==v for k,v in before['environment'].items() if k not in N.storage())
        self.storage_events.append(dict(event='normalize',actual_environment={k:normalized['environment'][k] for k in N.storage()},
          recorded_environment={k:before['environment'].get(k) for k in N.storage()},numeric_environment_unchanged=True))
        return normalized,args,protocol
      self.patch(module,'normalize',normalize)
      super().replay(module)
      assert module.main.__globals__ is not vars(module)
      assert module.main.__globals__['normalize'] is module.normalize,'Detached normalize binding'

    def bridge(self,module):
      super().bridge(module)
      if self.mode!='paced':return
      original_init=module.Preparer.__init__
      def initialize(preparer,*a,**kw):
        result=original_init(preparer,*a,**kw)
        value=self.composition.environment('paced_preparer')
        value['sampler_artifacts']=str(m.HERE/'agent-exact-live-artifacts')
        assert self.sampler is not None
        return result
      self.patch(module.Preparer,'__init__',initialize)

    def coverage(self):
      result=super().coverage()
      assert self.composition.pv_handle is not None
      assert self.composition.session.frame is None and not self.composition.session.state['prefill']
      S.check_storage(self.storage_events,paced=self.mode=='paced')
      return result
  return Hooks
