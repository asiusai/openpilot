"""Reuse the full selected live installer; attach one immutable CPU hook after binding."""
import copy
from functools import wraps
import time
from types import MethodType


def factory(C):
  base=C.load(C.OLD/'agent-expert-live-storage-install.py','cpu_integrated_qualified_install').factory(C.S,pv_candidate=True)
  cpu=C.load(C.HOOK,'cpu_integrated_exact_content_guard')
  class Hooks(base):
    def __init__(self,*a,**kw):
      super().__init__(*a,**kw)
      self.cpu_handle=None
      self.cpu_info=dict(consumer_bindings=0,hook_source_sha256=C.HOOK_SHA,calls=[],cleanup_exact=False)

    def runtime(self,module):
      super().runtime(module)
      original=module.bind_components
      owner=self
      @wraps(original)
      def bind(*a,**kw):
        consumer=original(*a,**kw)
        assert owner.cpu_handle is None and consumer._pending is None and not consumer._closed
        handle=cpu.install(consumer);owner.cpu_handle=handle
        assert handle.guard.shadow is None
        owner.cpu_info.update(consumer_bindings=1,provenance=handle.provenance,
          placement='after canonical Consumer construction; before any live query',initial_shadow_bytes=0)
        def close():
          owner.cpu_info['guard_before_cleanup']=handle.guard.state()
          handle.close()
          owner.cpu_info.update(cleanup_exact=consumer._bind_inputs==handle.original,
            shadow_after_cleanup=handle.guard.state()['shadow_bytes'])
        owner.stack.callback(close)
        installed=consumer._bind_inputs
        def timed(this,prepared,packed,upload):
          assert this is consumer and this._pending is not None,'CPU hook must run inside the actual Consumer query'
          before=handle.guard.state();start=time.perf_counter_ns()
          record=dict(query_ns=this._pending['query_ns'],inside_consumer_predict=True,
            shadow_before=before['shadow_bytes'],completed=False)
          try:
            result=installed(prepared,packed,upload);record['completed']=True;return result
          finally:
            after=handle.guard.state()
            record.update(elapsed_ms=(time.perf_counter_ns()-start)/1e6,shadow_after=after['shadow_bytes'],
              hits_delta=after['counts'].get('full_content_hits',0)-before['counts'].get('full_content_hits',0),
              misses_delta=after['counts'].get('full_content_misses',0)-before['counts'].get('full_content_misses',0))
            owner.cpu_info['calls'].append(record)
        measured=MethodType(timed,consumer);consumer._bind_inputs=measured
        def restore_measurement():
          assert consumer._bind_inputs is measured
          consumer._bind_inputs=installed
        owner.stack.callback(restore_measurement)
        return consumer
      self.patch(module,'bind_components',bind)

    def coverage(self):
      result=super().coverage()
      assert self.cpu_handle is not None and self.cpu_info['consumer_bindings']==1
      assert self.cpu_handle.guard.shadow is not None
      return result
  return Hooks
