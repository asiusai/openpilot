"""Install CK before genuine current-input startup; retain original arrival ownership."""
from contextlib import ExitStack
import copy

def factory(S):
  A,C,N=S.selected()
  base=A.original_load(S.EXP/'agent-arrival-warp-ownership-install.py','ck_selected_arrival_owner').factory(A)
  H=S.hook(True)
  class Hooks(base):
    def __init__(self,mode):
      super().__init__(mode)
      self.ck=None
      self.ck_report=dict(gpu_executed=False,completed=False,readback_bytes=0,prefills=[],phase='before_startup',
        no_measured_graph_diagnostics=True,quality_reference_sha256=S.QUALITY_SHA,constructor_bound=False,cleanup_exact=False)

    def __enter__(self):
      result=super().__enter__()
      original=self.warm.initialize
      def initialize(components,*args,**kwargs):
        assert self.ck is None and components.model is self.composition.model
        assert components.decoder.history==[] and components.inputs is None
        assert type(components.model.padded_prefill).__name__=='SpeculativePrefill'
        # PV observation was installed by the completed model factory. Nest CK after it and restore in reverse order.
        self.stack.enter_context(H.native_scope(S.C,self.ck_report))
        self.ck=H.Integration(S.C,components.model,components.decoder,'candidate',self.ck_report)
        # Startup and live calls share this exact owner. No fixed64 index assumption in continuous operation.
        owner=self.ck
        class Proxy:
          def __getattr__(self,name):return getattr(owner.original_jit,name)
          def reset(self):owner.original_jit.reset()
          def __call__(self,*values,**options):
            assert not options
            before=owner.original_jit.cnt;start=len(owner.hook.calls)
            ret=owner.original_jit(*values)
            assert owner.hook.calls[start:]==(list(range(28)) if before<2 else [])
            owner.last_args=values;owner.after_prefill(f'actual_prefill_{len(owner.report["prefills"])}',before)
            owner.control_done=True
            return ret
        owner.proxy=Proxy();components.model.padded_prefill.jit=owner.proxy
        self.ck_report.update(gpu_executed=True,constructor_bound=True)
        def close():
          owner.close();self.ck_report['cleanup_exact']=owner.closed
        self.stack.callback(close)
        result=original(components,*args,**kwargs)
        for key in ('ready','full_kv_bits_restored','actual_first_logits_bits_restored','target_payload_bits_restored','warmed_owner_retained'):
          assert result[key] is True,key
        assert components.model.padded_prefill.jit is owner.proxy
        self.ck_report['startup_prefills']=len(self.ck_report['prefills'])
        self.ck_report['startup_ready']=True
        return result
      self.patch(self.warm,'initialize',initialize)
      return result

    def runtime(self,module):
      super().runtime(module);original=module.bind_components
      def bind(components,*args,**kwargs):
        assert self.ck is not None and self.ck_report.get('startup_ready') is True
        assert components.model is self.ck.model and components.model.padded_prefill.jit is self.ck.proxy
        value=original(components,*args,**kwargs)
        self.ck_report['consumer_owns_warmed_model']=value.model is self.ck.model
        assert self.ck_report['consumer_owns_warmed_model']
        return value
      self.patch(module,'bind_components',bind)

    def coverage(self):
      result=super().coverage()
      assert self.ck is not None and self.ck_report['startup_ready'] and self.ck_report['consumer_owns_warmed_model']
      self.ck.device.synchronize()
      rows=self.ck_report['prefills'];assert len(rows)>self.ck_report['startup_prefills']
      assert all(r['captured_native_calls']==28 for r in rows[1:])
      assert all(r['all28_audit_counts'] is None for r in rows[2:])
      # Final O/LSE finite/guard check occurs after the paced observation. Decoder may have changed KV suffix slots.
      from tinygrad import Tensor
      flags=[H.audit(row['stores'][j],row[name]) for row in self.ck.layers.values() for j,name in enumerate(('out','lse'))]
      counts=self.ck.download(Tensor.cat(*[v.flatten() for v in flags]).contiguous().realize())
      assert not counts.any(),'Final selected O/LSE guard or finite failure'
      self.ck_report.update(completed=True,final_output_counts=counts.tolist(),final_fence=True)
      result['ck_actual_prefills']=len(rows);result['ck_all28_capture_replay']=True
      return result
  return Hooks
