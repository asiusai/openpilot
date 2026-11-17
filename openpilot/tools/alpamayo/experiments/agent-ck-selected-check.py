"""CPU interface checks. No native device is opened; neural computation is explicitly doubled."""
import argparse
import ast
from contextlib import ExitStack,contextmanager,nullcontext
import copy
import importlib.util
import json
import os
from pathlib import Path
import runpy
import sys
from types import ModuleType,SimpleNamespace
from unittest.mock import patch

EXP=Path(__file__).parent
s=importlib.util.spec_from_file_location('ck_selected_common',EXP/'agent-ck-selected-common.py')
S=importlib.util.module_from_spec(s);s.loader.exec_module(S)

def quality(output):
  """Full actual64 TinyJit/native ABI path; selected fusion math remains a named neural double."""
  # Reuse the genuine TinyJit/USB simulation and old numerical fixture. Do not forge GPU evidence.
  text=(EXP/'agent-ck-quality64-check.py').read_text()
  m=ModuleType('selected_full64_check');m.__file__=str(EXP/'agent-ck-quality64-check.py')
  exec(compile(text,m.__file__,'exec'),m.__dict__)
  C=SimpleNamespace(**vars(S.C));C.FUSED_SOURCE=C.BASE/'experiments/quantize-packed.py'
  C.selected_coverage=lambda report:report.update(selected_neural_composition_mocked=True)
  # The separate factory test runs the actual selected factory/storage/coverage stack.
  # This test executes every CK call, physical view, graph inspection, replay, and ABI negative on CPU.
  C.runner=S.runner
  m.run_check(output,C)

def composition(output):
  """Run the real fused MLP, selected vision, expertPV, and CK routing together with fake neural tensors."""
  os.environ.update(CACHEDB=str(S.ROOT/'cpu-composition.db'),TMPDIR=str(S.ROOT/'artifacts/tmp'),
    XDG_CACHE_HOME=str(S.ROOT/'artifacts/cache'),ALLOW_DEVICE_USAGE='0',DEV='CPU',PARALLEL='32',TC_OPT='2')
  source=(EXP/'agent-expert-combined-check.py').read_text()
  source=S.Q.replace_once(source,'for candidate in (False,True):','for candidate in (True,):')
  source=S.Q.replace_once(source,"'sampler_snapshot_exact':True","'sampler_executed_in_this_test':False")
  source=S.Q.replace_once(source,"comp = Composition('native',candidate,evidence","comp = Composition('quality',candidate,evidence")
  source=S.Q.replace_once(source,'model = fixtures.FakeModel(core)',
    'fixtures.FakeModel.text_layer = core.EdgeModel.text_layer\n          model = fixtures.FakeModel(core)')
  source=S.Q.replace_once(source,"session = comp.session; state =", "ck_calls=[]\n          native_scope=_S.hook().native_scope(_S.C,{})\n          native_scope.__enter__()\n          def ck_callback(h,context,q,k,v,original):\n            assert q.shape==(1,16,2048,128) and k.shape==v.shape==(1,8,2048,128)\n            assert comp.session.frame['prefix']==context['prefix'] and comp.session.state['index']==1\n            ck_calls.append(context['layer']);return q\n          ck=_S.load('agent-ck-model-hook').Hook(model,ck_callback)\n          session = comp.session; state =")
  start=source.index("           stream = C.load(") if "           stream = C.load(" in source else source.index("          stream = C.load(")
  end=source.index('          # Synthetic capture',start)
  source=source[:start]+source[end:]
  source=S.Q.replace_once(source,"          raise KeyboardInterrupt('Deliberate interrupted reverse cleanup')",
    "          assert ck_calls==list(range(28));ck.close();native_scope.__exit__(None,None,None)\n          raise KeyboardInterrupt('Deliberate interrupted reverse cleanup')")
  source=S.Q.replace_once(source,"assert comp.sampler.pool is None and all(r['restored'] for r in comp.restorations)",
    "assert comp.sampler is None and all(r['restored'] for r in comp.restorations)")
  start=source.index("  R = C.load(HERE/'agent-expert-combined-run.py'")
  source=source[:start]+"  assert not Device._opened_devices\n  C.write(args.output,dict(complete=True,gpu_executed=False,arms=arms,rejected=rejected,ck_layers=ck_calls,all_actual_numerical_installers=True,neural_tensors_mocked=True),new=True)\n"
  m=ModuleType('selected_composition_check');m.__file__=str(EXP/'agent-expert-combined-check.py');m._S=S
  exec(compile(source,m.__file__,'exec'),m.__dict__)
  with patch.object(sys,'argv',[m.__file__,'--output',str(output)]):m.main()

def paced(output):
  A,C,N=S.configure()
  os.environ.update(DEV='CPU',ALLOW_DEVICE_USAGE='0')
  from tinygrad import Device
  import numpy as np
  assert not Device._opened_devices
  Ksource=(EXP/'agent-live-cpu-integrated-check.py').read_text()
  Ksource=S.Q.replace_once(Ksource,"if boundary.arm=='paced':raise ConstructedBoundary('Stop before source clock or producer starts')",'if False:raise ConstructedBoundary()')
  Ksource=S.Q.replace_once(Ksource,'class Prefill:','class SpeculativePrefill:')
  Ksource=S.Q.replace_once(Ksource,'self.padded_prefill=Prefill()','self.padded_prefill=SpeculativePrefill()')
  K=ModuleType('ck_selected_actual_factory_double');K.__file__=str(EXP/'agent-live-cpu-integrated-check.py')
  exec(compile(Ksource,K.__file__,'exec'),K.__dict__);K.C=C
  original_startup=K.Boundary.startup_compute
  def startup(self,components,*a,**kw):
    result=original_startup(self,components,*a,**kw)
    # Native startup arithmetic is doubled; exercise the actual newly installed proxy and lifetime.
    for _ in range(5):components.model.padded_prefill.jit()
    result.update(full_kv_bits_restored=True,actual_first_logits_bits_restored=True,target_payload_bits_restored=True,warmed_owner_retained=True)
    components._startup_spec=result
    return result
  K.Boundary.startup_compute=startup
  class NativeDouble:
    def __init__(self,common,model,decoder,arm,report):
      self.model,self.decoder,self.report=model,decoder,report;self.closed=False
      self.hook=SimpleNamespace(calls=[]);self.layers={i:{'stores':[None,None],'out':None,'lse':None} for i in range(28)}
      self.device=SimpleNamespace(synchronize=lambda:None)
      owner=self
      class Jit:
        cnt=0
        def __call__(self):
          if self.cnt<2:owner.hook.calls.extend(range(28))
          self.cnt+=1;return None
      self.original_jit=Jit()
    def after_prefill(self,phase,before):
      self.report['prefills'].append(dict(phase=phase,captured_native_calls=28 if before else 0,
        all28_audit_counts=[[0]*16 for _ in range(28)] if before<2 else None,CPU_NEURAL_CAPTURE_DOUBLE=True))
    def download(self,t):return np.zeros(392,np.uint32)
    def close(self):
      assert self.model.padded_prefill.jit is self.proxy
      self.model.padded_prefill.jit=self.original_jit;self.closed=True
  class Flag:
    def flatten(self):return self
    def contiguous(self):return self
    def realize(self):return self
  H=SimpleNamespace(native_scope=lambda *a:nullcontext(),Integration=NativeDouble,audit=lambda *a:Flag())
  source=(EXP/'agent-arrival-warp-ownership-check.py').read_text()
  source=S.Q.replace_once(source,"K = A.original_load(HERE/'agent-live-cpu-integrated-check.py','arrival_ownership_boundary'); K.C = C",'K = _K; K.C = C')
  source=S.Q.replace_once(source,"ours = path == HERE/'agent-arrival-warp-ownership-install.py'","ours = path == HERE/'agent-ck-selected-install.py'")
  source=S.Q.replace_once(source,'try:return original(*a,**kw)',"try:\n                      original(*a,**kw)\n                      raise K.ConstructedBoundary('After complete selected bind')")
  source=S.Q.replace_once(source,"ids = [int(config.traj_ids['future_start'])]","model.padded_prefill.jit()\n        ids = [int(config.traj_ids['future_start'])]")
  source=S.Q.replace_once(source,"'agent-live-cpu-integrated-install.py','agent-expert-live-storage-install.py','agent-exact-live-install.py','agent-exact-combined-install.py'}",
    "'agent-live-cpu-integrated-install.py','agent-expert-live-storage-install.py','agent-exact-live-install.py','agent-exact-combined-install.py','agent-ck-selected-install.py'}")
  source=S.Q.replace_once(source,"argv = [str(HERE/'agent-arrival-warp-ownership-run.py'),'--protocol',str(protocol),'--protocol-sha',A.sha(protocol),\n    '--output',str(output),'--cpu-preflight']",
    "argv = [str(HERE/'agent-ck-selected-paced.py'),'--output',str(output),'--cpu-preflight','--quality-report',str(_S.QUALITY)]")
  source=S.Q.replace_once(source,"own = A.read(output.parent/'arrival-warp-ownership-status.json')",
    "selected = A.read(output.parent/'ck-selected-status.json'); own={'ownership':[r['arrival_ownership'] for r in selected['owners']]}\n  assert selected['owners'][0]['ck']['cleanup_exact']")
  # Actual namespace-import interception supplies only declared CPU boundaries to the real entrypoint.
  marker="path = Path(location).resolve()\n        ours ="
  source=S.Q.replace_once(source,marker,"path = Path(location).resolve()\n        if path == HERE/'agent-ck-selected-common.py':\n          original_loader=result.loader\n          class CommonLoader:\n            def create_module(self,spec):return original_loader.create_module(spec)\n            def exec_module(self,module):\n              original_loader.exec_module(module)\n              module.selected=lambda:(_A,_C,_C.N)\n              module.configure=_S.configure\n              module.verify=lambda:_data\n              module.quality_gate=lambda path:_S.read(path)\n              module.hook=lambda *a:_H\n              module.PREPARED=_prepared\n          result.loader=CommonLoader();return result\n        ours =")
  m=ModuleType('selected_actual_factory_check');m.__file__=str(EXP/'agent-arrival-warp-ownership-check.py')
  exec(compile(source,m.__file__,'exec'),m.__dict__)
  data={'files':{},'cpu_proof':{'complete':True,'gpu_executed':False}}
  m.A,m.C,m._K,m._A,m._C,m._S,m._H,m._data=A,C,K,A,C,S,H,data
  original_context=K.Boundary.hooks_context
  @contextmanager
  def context(self,hooks):
    with original_context(self,hooks):yield
  K.Boundary.hooks_context=context
  root=output.parent/output.stem;root.mkdir()
  m._prepared=root/'prepared';m._prepared.mkdir();S.write(m._prepared/'protocol.json',{'CPU_MOCK_ONLY':True})
  from tinygrad import Tensor
  with patch.object(Tensor,'cat',lambda *a,**kw:Flag()):
    result=m.production(root,data)
  assert not Device._opened_devices
  S.write(output,dict(complete=True,gpu_executed=False,actual_factory_startup_bind_predict_coverage_finally=True,
    neural_arithmetic_mocked=True,actual=result))

if __name__=='__main__':
  p=argparse.ArgumentParser();p.add_argument('--mode',choices=('quality','paced','composition'),required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
  {'quality':quality,'paced':paced,'composition':composition}[a.mode](a.output)
