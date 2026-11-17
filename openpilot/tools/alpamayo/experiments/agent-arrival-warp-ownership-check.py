"""Actual runpy/factory/Consumer and every original coverage method, with neural work doubled."""
import argparse
from contextlib import ExitStack
import copy
import importlib.util
import json
import os
from pathlib import Path
import runpy
import socket
import sys
from types import FunctionType, MethodType, SimpleNamespace
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('arrival_ownership_check_common', HERE/'agent-arrival-warp-ownership-common.py')
A = importlib.util.module_from_spec(spec); spec.loader.exec_module(A)
C = A.C


def blocked(*args, **kwargs): raise AssertionError('No device/network in CPU ownership preflight')


def neural_factory(module, stack, boundary, K):
  """Execute original full cache IO/hash/temporary lifecycle with CPU-only neural buffers.

  FP8 conversion is a declared neural double returning existing cache bytes.
  Bulk upload/readback is a bounded host chunk, not a GPU correctness test.
  No storage event, coverage function, source guard, or IO check is fabricated.
  """
  import numpy as np
  print('CPU preflight: neural conversion/upload/capture doubles; actual original cache IO and hash checks follow', flush=True)
  index = A.read(C.N.WEIGHTS/'index.json')
  name = 'vlm.model.language_model.layers.0.mlp.fc1.weight'
  expected = {}
  with (C.N.WEIGHTS/'weights.bin').open('rb') as stream:
    for kind in ('scale','fp8'):
      record = next(r for r in index['records'] if r['name'] == name and r['kind'] == kind)
      stream.seek(record['offset']); expected[kind] = stream.read(record['nbytes'])
  class NeuralTensor:
    def __init__(self, value=None, kind='fp8'): self.kind = kind
    def __getattr__(self, name):
      if name in ('float','abs','clamp','realize','cast','reshape','bitcast'): return lambda *a,**kw:self
      raise AttributeError(name)
    def max(self): return NeuralTensor(kind='scale')
    def __truediv__(self, other): return self
    def numpy(self): return np.frombuffer(expected[self.kind], dtype=np.uint8)
  class HostChunk:
    def __init__(self, device, size, opaque=None, *, owner=None, offset=0):
      self.device,self.size,self.opaque,self.owner,self.offset = device,size,opaque,owner or self,offset
    def ensure_allocated(self): return self
    def view(self, size, offset): return HostChunk(self.device,size,owner=self.owner,offset=self.offset+offset)
    def copy_from(self, source):
      assert source.device == 'NPY' and len(source.opaque) == self.size
      self.owner.last = (self.offset, source.opaque)
    def as_memoryview(self):
      offset, raw = self.owner.last
      assert offset == self.offset and len(raw) == self.size
      return raw
  for field, value in dict(safe_load=lambda p:{name:NeuralTensor()}, upload=lambda value,dtype:value,
      Tensor=NeuralTensor, Buffer=HostChunk, UOp=SimpleNamespace(from_buffer=lambda *a:None),
      pack=lambda v:v, unpack=lambda v:v).items(): stack.enter_context(patch.object(module,field,value))
  dev = module.Device[module.Device.DEFAULT]
  dev.synchronize = lambda:None


def finish_factory(module, boundary, K):
  # Run after all canonical factory installers have checked constructor ownership.
  original = module.EdgeModel.__init__
  def complete(model, *args, **kwargs):
    try:
      original(model, *args, **kwargs)
      boundary.factory_seen = True
      boundary.events.append('actual_factory_full_index_blob_hash_temporary_complete_neural_double')
    finally:
      assert module.EdgeModel.__init__ is complete
      module.EdgeModel.__init__ = original
    raise K.FactoryBoundary()
  module.EdgeModel.__init__ = complete


def graph(names):
  # Descriptors stand only for mocked GPU neural captures. Actual coverage parses them.
  call = SimpleNamespace(arg=SimpleNamespace(aux=SimpleNamespace(kernels=[(None,name) for name in names])))
  call.without_after = call
  value = SimpleNamespace(captured=SimpleNamespace(_linear=SimpleNamespace(src=[call])))
  value.reset = lambda:setattr(value,'captured',None)
  return value


def neural_metadata(hooks):
  baseline = A.read(C.output('paced'))['agent_live_cpu_integrated']['coverage']
  comp = hooks.composition
  comp.arm_calls = 1
  comp.model.prefill_jit = graph(['agent_packed_epilogue_cpu_neural_double']*28)
  for row in baseline['vision_graphs']:
    key = tuple(tuple(s) for s in row['shape'])
    comp.cache.encoders[key] = graph(['agent_vision_tiles_3x3_cpu_neural_double']*25+['agent_vision_tiles_3x6_cpu_neural_double']*25)
  comp.session = SimpleNamespace(stats=copy.deepcopy(baseline['language_stats']), frame=None, state={'prefill':False})
  comp.vision_info = copy.deepcopy(baseline['vision_hook'])
  comp.pv_handle = SimpleNamespace(info={'CPU_NEURAL_CAPTURE_DOUBLE':True})


def coverage_checks(hooks):
  original = importlib.util.spec_from_file_location
  calls = []
  def trace(frame,event,arg):
    if event == 'call' and frame.f_code.co_name == 'coverage': calls.append(frame.f_code.co_filename)
  previous = sys.getprofile(); sys.setprofile(trace)
  try: result = hooks.coverage()
  finally: sys.setprofile(previous)
  assert importlib.util.spec_from_file_location is original
  expected = {'agent-arrival-warp-ownership-install.py','agent-arrival-warp-integrated-install.py',
    'agent-live-cpu-integrated-install.py','agent-expert-live-storage-install.py','agent-exact-live-install.py','agent-exact-combined-install.py'}
  assert {Path(p).name for p in calls} == expected and len(calls) == len(expected)
  negatives = []
  def rejected(label, operation, wanted=(RuntimeError,ValueError,AssertionError)):
    try: operation()
    except wanted: negatives.append(label)
    else: raise AssertionError('Accepted invalid coverage state: '+label)
  foreign = lambda *a,**kw:None
  with patch.object(importlib.util,'spec_from_file_location',foreign):
    rejected('foreign outer hook',hooks.coverage)
    assert importlib.util.spec_from_file_location is foreign
  clone = FunctionType(original.__code__,original.__globals__,closure=original.__closure__)
  with patch.object(importlib.util,'spec_from_file_location',clone): rejected('same-code different outer identity',hooks.coverage)
  with patch.object(hooks,'spec',MethodType(lambda self,*a,**kw:None,hooks)): rejected('bound method rebinding',hooks.coverage)
  cell = hooks._ownership_cells['previous']; old = cell.cell_contents
  try:
    cell.cell_contents = MethodType(old.__func__,object())
    rejected('closure previous owner replaced',hooks.coverage)
  finally: cell.cell_contents = old
  comp = hooks.composition
  with patch.object(comp,'arm_calls',0): rejected('original factory coverage',hooks.coverage)
  layers = comp.session.stats['layer_hits']; key = next(iter(layers)); value = layers.pop(key)
  try: rejected('original incomplete numerical layer coverage',hooks.coverage)
  finally: layers[key] = value
  with patch.object(comp.model,'prefill_jit',graph([])): rejected('original missing numerical capture',hooks.coverage)
  with patch.object(hooks,'binds',0): rejected('original Consumer owner count',hooks.coverage)
  shadow = hooks.cpu_handle.guard.shadow
  try:
    hooks.cpu_handle.guard.shadow = None
    rejected('original CPU input ownership',hooks.coverage)
  finally: hooks.cpu_handle.guard.shadow = shadow
  def rebind(frame,event,arg):
    if event == 'call' and frame.f_code is comp.coverage.__func__.__code__:
      importlib.util.spec_from_file_location = foreign
  previous = sys.getprofile(); sys.setprofile(rebind)
  try: rejected('foreign rebinding inside original numerical coverage',hooks.coverage)
  finally: sys.setprofile(previous)
  assert importlib.util.spec_from_file_location is original
  hooks.check_chain()
  assert hooks.ownership['scoped_restorations'] == hooks.ownership['checks']
  assert hooks.coverage()['language_graphs'] == result['language_graphs']
  return dict(all_actual_super_coverage_executed=True, coverage_sources=calls, negatives=negatives,
    actual_numerical_coverage_functions_unchanged=True, neural_capture_metadata_mocked=True,
    original_bound_owner_guard_executed=True, scoped_restoration_after_failure=True)


def production(root,data):
  import numpy as np
  from tinygrad import Device
  K = A.original_load(HERE/'agent-live-cpu-integrated-check.py','arrival_ownership_boundary'); K.C = C
  class Boundary(K.Boundary):
    def components_after_factory(self,*args,**kwargs):
      value = super().components_after_factory(*args,**kwargs)
      def neural_run(decoder, model, settings, config, tokenizer, prepared, packed, **kwargs):
        ids = [int(config.traj_ids['future_start'])]
        decoder.prefill_draft_records.append(dict(eligible=bool(prepared['prefill_draft_eligible']), accepted=False,
          actual_prompt=int(prepared['actual_prompt_length']), generated=ids))
        return dict(xyz=np.zeros((1,64,3),np.float32).tolist(),
          rot=np.broadcast_to(np.eye(3,dtype=np.float32),(1,64,3,3)).tolist(), generated_token_ids=ids,
          CPU_NEURAL_EXECUTION_DOUBLE=True)
      value.decoder.run = MethodType(neural_run,value.decoder); value.runner = value.decoder.run
      return value

    def observe(self,previous,stack):
      inner = super().observe(previous,stack); boundary = self
      def spec(name,location,*args,**kwargs):
        result = inner(name,location,*args,**kwargs)
        if result is None or location is None:return result
        path = Path(location).resolve()
        ours = path == HERE/'agent-arrival-warp-ownership-install.py'
        factory = path == C.N.LIVE/'experiments/benchmark-fast.py'
        pipeline = path == C.N.LIVE/'experiments/benchmark-pipeline.py'
        if not (ours or factory or pipeline):return result
        loader = result.loader
        class Loader:
          def create_module(self,spec):return loader.create_module(spec)
          def exec_module(self,module):
            loader.exec_module(module)
            if factory:
              neural_factory(module,stack,boundary,K)
              return
            if pipeline:
              original_local = module.local_module
              def local(name):
                value = original_local(name)
                if name == 'benchmark-fast': finish_factory(value,boundary,K)
                return value
              stack.enter_context(patch.object(module,'local_module',local))
              return
            real_factory = module.factory
            def extended(common):
              Base = real_factory(common)
              class CPUHooks(Base):
                def runtime(self,module):
                  super().runtime(module); original = module.bind_components
                  def bind(*a,**kw):
                    try:return original(*a,**kw)
                    except K.ConstructedBoundary:
                      consumer = boundary.consumer; producer = self.arrival.producer
                      bridge = self.arrival.cache.bridge
                      runner = bridge.Bridge(producer.store,self.arrival.preparer,hz=10,clock=producer.clock,
                        require_small_model=False,navigation=producer.trace.nav(producer.trace.start))
                      row = runner.run_once(consumer)
                      assert row is not None and consumer._pending is None and consumer.records[-1]['completed']
                      assert self.cpu_handle.guard.shadow is not None
                      neural_metadata(self)
                      boundary.neural_complete = True
                      raise
                  self.patch(module,'bind_components',bind)
                def __exit__(self,*error):
                  try:
                    if error[0] is K.ConstructedBoundary and boundary.neural_complete:
                      # Canonical launcher's finally has now restored its temporary
                      # import hooks, exactly as at production post-run coverage.
                      boundary.coverage_result = coverage_checks(self)
                      boundary.coverage_result['actual_Consumer_predict_synchronize'] = True
                      boundary.coverage_result['actual_factory_storage'] = C.S.check_storage(self.storage_events,paced=True)
                  finally:
                    result = super().__exit__(*error)
                  return result
              return CPUHooks
            module.factory = extended
        result.loader = Loader();return result
      return spec
  boundary = Boundary('paced'); previous = importlib.util.spec_from_file_location
  protocol = root/'cpu-protocol.json'; A.write(protocol,dict(data,cpu_only=True),new=True)
  output = root/'entry/report.json'
  argv = [str(HERE/'agent-arrival-warp-ownership-run.py'),'--protocol',str(protocol),'--protocol-sha',A.sha(protocol),
    '--output',str(output),'--cpu-preflight']
  saved = {name:sys.modules.get(name) for name in ('model','benchmark','inputs','native_warp','stream_inputs')}
  error = None
  try:
    with ExitStack() as stack:
      stack.enter_context(patch.dict(os.environ));stack.enter_context(patch.object(sys,'path',sys.path[:]))
      stack.enter_context(patch.object(sys,'argv',argv))
      stack.enter_context(patch.object(type(Device),'__getitem__',blocked));stack.enter_context(patch.object(socket.socket,'connect',blocked))
      stack.enter_context(patch.object(importlib.util,'spec_from_file_location',boundary.observe(previous,stack)))
      try:runpy.run_path(argv[0],run_name='__main__')
      except K.ConstructedBoundary as exc:error = exc
  finally:
    for name,old in saved.items():
      if old is None:sys.modules.pop(name,None)
      else:sys.modules[name] = old
  assert error is not None and importlib.util.spec_from_file_location is previous and not Device._opened_devices
  status = A.read(output.parent/'arrival-warp-integrated-status.json')
  own = A.read(output.parent/'arrival-warp-ownership-status.json')
  assert status['hooks_restored'] and status['constructor_restored'] and not status['passed'] and not status['gpu_executed']
  assert status['arrival']['producer_binding_restored'] and status['arrival']['cache_cleanup_bytes'] == 0
  assert status['cpu_hook']['cleanup_exact'] and status['cpu_hook']['shadow_after_cleanup'] == 0
  assert own['ownership'][0]['finally_restored'] and boundary.hooks.sampler.pool is None
  assert not boundary.hooks.composition.cache.encoders
  assert boundary.consumer._closed and not boundary.consumer._lock.locked()
  return dict(boundary.coverage_result, complete=True, hooks_restored=True, actual_entrypoint=True,
    ownership=own['ownership'], owner_rejections=boundary.owner_errors, source_clock_started=False,
    neural_factory_bytes_streamed_and_hashed=A.read(C.N.WEIGHTS/'index.json')['nbytes'])


def main():
  p = argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
  args = p.parse_args(); A.configure()
  assert not args.output.exists() and args.output.resolve().is_relative_to(A.WORK)
  root = A.ARTIFACTS/args.output.stem;root.mkdir()
  files = A.files();data = dict(files=files,foundation=A.foundation(),cpu_only=True,window=[35,43],hz=10,timeout_seconds=900)
  from tinygrad import Device
  from tinygrad.helpers import Context
  with Context(ALLOW_DEVICE_USAGE=0):result = production(root,data)
  assert not Device._opened_devices and A.files() == files
  A.write(args.output,dict(complete=True,gpu_executed=False,devices=[],production=result,files=files),new=True)
  print(dict(complete=True,gpu_executed=False,all_actual_super_coverage_executed=True),flush=True)


if __name__ == '__main__':main()
