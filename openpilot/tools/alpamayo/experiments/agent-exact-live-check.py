"""CPU controls for live hooks, received-floor preparation, ownership, and restoration."""
import argparse
import copy
from dataclasses import replace
from contextlib import ExitStack
import importlib.util
import json
from pathlib import Path
import socket
import sys
from types import MethodType, SimpleNamespace
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('exact_live_check_common',HERE/'agent-exact-live-common.py')
m = importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
sys.path[:0] = [str(m.c.TINYGRAD),str(m.ROOT/'experiments'),str(m.ROOT),
  '/tmp/opencode/openpilot-alpamayo','/home/john/asius/openpilot/opendbc_repo']


def rejected(fn):
  try: fn()
  except (ValueError,RuntimeError,AssertionError): return
  raise AssertionError('Expected explicit rejection')


def forbidden(*args,**kwargs): raise AssertionError('GPU and network access forbidden')


def fake_factory(hooks,doubles):
  from tinygrad import dtypes
  fixtures=m.load(m.c.MLP/'experiments/agent-fused-mlp-check.py','exact_live_qualified_cpu_fixtures')
  Fake=fixtures.FakeTensor
  doubles.enter_context(patch.object(Fake,'ndim',property(lambda x:len(x.shape)),create=True))
  core=m.load(m.ROOT/'model.py','exact_live_cpu_core')
  doubles.enter_context(patch.object(core,'Tensor',Fake))
  model=fixtures.FakeModel(core)
  model.norm=MethodType(lambda self,x,name,rms=False,eps=None:x,model)
  model.encode_jits={};model.encode=lambda *a:None
  for layer in range(27):
    for fc,shape in (('fc1',(4304,1152)),('fc2',(1152,4304))):
      name=f'vlm.model.visual.encoder.layers.{layer}.mlp.{fc}'
      model.weights[name+'.weight']=Fake(shape=shape,dtype=dtypes.bfloat16 if layer<2 else dtypes.fp8e4m3)
      model.weights[name+'.bias']=Fake(shape=(shape[0],),dtype=dtypes.bfloat16)
      if layer>=2:model.weight_scales[name+'.weight']=Fake([.125],dtype=dtypes.float32)
  model.weights['vlm.model.visual.embeddings.patch_embedding.weight']=Fake(shape=(1152,1176),dtype=dtypes.bfloat16)
  model.grouped_quantize=lambda x,images:(Fake(shape=(images*720,x.shape[-1]),dtype=dtypes.fp8e4m3),Fake(shape=(images*720,),dtype=dtypes.float32))
  dispatches=[]
  def dispatch(*args,**kwargs):
    dispatches.append(True)
    return Fake(shape=(2048,9216),dtype=dtypes.fp8e4m3)
  hooks.composition.device_factory=lambda:SimpleNamespace(arch='gfx1200',device='AMD')
  hooks.composition.mlp_dispatch=dispatch
  doubles.enter_context(patch.object(hooks.composition.vision,'Tensor',Fake))
  doubles.enter_context(patch.object(hooks.composition.vision,'gemm',lambda x,w,*a,**kw:Fake(shape=(*x.shape[:-1],w.shape[0]),dtype=dtypes.bfloat16)))
  packed=m.load(m.ROOT/'experiments/quantize-packed.py','exact_live_cpu_packed')
  doubles.enter_context(patch.object(packed,'Tensor',Fake))
  doubles.enter_context(patch.object(packed,'pack',lambda x:Fake(shape=x.shape,dtype=x.dtype)))
  doubles.enter_context(patch.object(packed,'quantize',lambda x,scale=None:(Fake(shape=(x.numel()//x.shape[-1],x.shape[-1]),dtype=dtypes.fp8e4m3),scale)))
  fixed={f'vlm.model.language_model.layers.{i}/{j}':.0037*(1+4*i+j) for i in range(28) for j in range(4)}
  packed.install(model,fixed_scales=fixed)
  m.load(m.ROOT/'experiments/batched-vision.py','exact_live_cpu_batched').install(model)
  cache=m.load(m.ROOT/'experiments/vision-cache.py','exact_live_cpu_cache').VisionCache(model,grouped=True,group_limit=2)
  m.load(m.ROOT/'experiments/block-decode.py','exact_live_cpu_block').BlockDecoder(model)
  for layer in range(28):
    model.text_layer(Fake(shape=(1,2048,2048),dtype=dtypes.bfloat16),None,None,
      f'vlm.model.language_model.layers.{layer}',Fake(shape=(2,1,8,2304,128),dtype=dtypes.bfloat16),None,0)
  import inspect
  state=inspect.getclosurevars(hooks.composition.session.original['quantize'].__func__).nonlocals['state']
  for images in (1,2):
    state['images']=images
    for layer in range(2,27):
      for fc,width in (('fc1',1152),('fc2',4304)):
        model.linear(Fake(shape=(images,720,width),dtype=dtypes.bfloat16),f'vlm.model.visual.encoder.layers.{layer}.mlp.{fc}')
    state['images']=0
  assert len(dispatches)==28
  cache.encoders[((2,720,1152),)]=type(model.prefill_jit)()
  hooks.composition.names=lambda jit:['agent_packed_epilogue_cpu']*28 if jit is model.prefill_jit else (
    ['agent_vision_tiles_3x3_b2']*25+['agent_vision_tiles_3x6_b2']*25)
  return model


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--output',type=Path,required=True)
  args = parser.parse_args()
  from tinygrad import Device
  import numpy as np
  controls = []
  with patch.object(type(Device),'__getitem__',forbidden),patch.object(socket.socket,'connect',forbidden),ExitStack() as doubles:
    plan = m.load(HERE/'agent-exact-live-plan.py','exact_live_cpu_plan')
    manifest = plan.prepare(HERE/'agent-exact-live-unused-plan',write=False)
    installer = m.load(HERE/'agent-exact-live-install.py','exact_live_cpu_installer')
    hooks = installer.Hooks(m,manifest,'paced')
    original = importlib.util.spec_from_file_location
    try:
      with hooks:
        rejected(lambda:installer.Hooks(m,manifest,'consumer').__enter__())
        controls.append('nested_composition_rejected')
        replay = m.load(m.ROOT/'experiments/agent-live-replay.py','exact_live_cpu_replay')
        dataset = SimpleNamespace(samples=manifest['selected_samples'],provenance=manifest['selected_dataset'])
        pipeline = m.load(m.ROOT/'experiments/benchmark-pipeline.py','exact_live_cpu_pipeline')
        normalized,effective,protocol = replay.normalize(m.INVOCATION,dataset,pipeline,HERE/'agent-exact-live-no-output.json')
        assert effective.samples == manifest['selected_samples'] and (effective.warmup,effective.runs)==(0,2)
        assert not effective.pipeline_inputs and not effective.freeze_warmup_gc
        assert effective.pad_prompt_to == 2048 and effective.vision_calibration is None
        controls.append('original_normalizer_and_explicit_two_query_selection')
        wrong = SimpleNamespace(samples=dataset.samples[::-1],provenance=dataset.provenance)
        rejected(lambda:replay.normalize(m.INVOCATION,wrong,pipeline,HERE/'agent-exact-live-no-output.json'))
        controls.append('changed_query_order_rejected')
        model=fake_factory(hooks,doubles)
        controls.append('qualified_28_language_and_50_vision_routes_on_identical_live_foundations')
        paced = m.load(m.ROOT/'experiments/agent-paced-replay.py','exact_live_cpu_paced')
        assert set(hooks.transforms)=={'replay','paced','parity_main'}
        # Execute both transformed startup insertion sites with explicit CPU doubles.
        before = replay.execute_existing_launcher.__globals__['_startup_before_audit']
        components = SimpleNamespace(_startup_spec={'ready':True})
        seen = []
        with patch.object(hooks.warm,'initialize',lambda owner,settings,arrays,**kw:(seen.append((owner,arrays)) or {'ready':True})):
          before(components,None,SimpleNamespace(rows=[object()],load=lambda row:{'current':True}),'consumer')
        assert seen == [(components,{'current':True})]
        controls.append('original_full_startup_insertion_invokes_current_input_once')
        before_pace = paced.model_run.__globals__['_startup_before_binding']
        rejected(lambda:before_pace(None,None,None,SimpleNamespace(thread=None),SimpleNamespace(anchor=1)))
        controls.append('startup_after_source_clock_rejected')
        trace = paced.Trace(paced.SOURCE/'manifest.json',paced.LOGS)
        assert len(trace.frames)==450
        clock = paced.Clock(trace.start)
        store = paced.bridge.Store(frame_limit=64,byte_limit=384*1024**2)
        producer = paced.Producer(trace,store,paced.RGB(m.RGB,trace.frames),clock)
        producer.prime_history()
        selection = store.select(clock(),0,{},250_000_000,False)
        assert all(f.received_ns <= selection.selected_ns and f.capture_ns <= selection.query_ns for f in selection.frames)
        import native_warp
        candidate = paced.bridge.Preparer(trace.manifest['source_camera_geometry'],trace.manifest['target_camera_geometry'],
          warp=native_warp.calibrated_image,clock=clock)
        reference = object.__new__(paced.bridge.Preparer)
        paced.bridge.Preparer.__init__.__wrapped__(reference,trace.manifest['source_camera_geometry'],
          trace.manifest['target_camera_geometry'],warp=native_warp.calibrated_image,clock=clock)
        a,am = reference.prepare(selection); b,bm = candidate.prepare(selection)
        for key in a:
          assert a[key].dtype == b[key].dtype and a[key].shape == b[key].shape and a[key].tobytes()==b[key].tobytes(),key
        assert am['frames']==bm['frames'] and am['vehicle_history_audit']==bm['vehicle_history_audit']
        assert hooks.sampler_calls==10 and clock.anchor is None and producer.thread is None
        controls.append('real_received_floor_full_input_and_history_bits_exact')
        candidate.prepare(selection)
        assert hooks.sampler_calls==10
        controls.append('original_preparer_cache_ownership_retained')
        changed=selection.frames[0].pixels.copy()^1
        changed.flags.writeable=False
        mutation=replace(selection,frames=(replace(selection.frames[0],pixels=changed),*selection.frames[1:]))
        aa,_=reference.prepare(mutation);bb,_=candidate.prepare(mutation)
        assert aa['image_frames'].tobytes()==bb['image_frames'].tobytes()!=a['image_frames'].tobytes()
        assert hooks.sampler_calls==11
        controls.append('new_source_object_pixel_mutation_recomputes_exactly')
        rejected(lambda:paced.bridge.Preparer(trace.manifest['source_camera_geometry'],trace.manifest['target_camera_geometry'],warp=native_warp.calibrated_image))
        controls.append('second_preparer_rejected')
        producer.close()
        # Wrapper guards remain independent from the original Consumer implementation tests.
        owner = object()
        closed = []
        binder = SimpleNamespace(bind_components=lambda c,*a,**kw:SimpleNamespace(inputs=c.inputs,close=lambda:closed.append(True)))
        hooks.runtime(binder)
        component = SimpleNamespace(model=model,inputs=owner,_startup_spec={'ready':True},
          vision_cache=hooks.composition.cache,decoder=SimpleNamespace(model=model,history=[],statistics=[],prefill_draft_records=[]))
        wrong = copy.copy(component);wrong.model=object()
        rejected(lambda:binder.bind_components(wrong,None))
        component.decoder.history=[[1]]
        rejected(lambda:binder.bind_components(component,None))
        component.decoder.history=[]
        assert binder.bind_components(component,None).inputs is owner
        rejected(lambda:binder.bind_components(component,None))
        controls.append('same_warmed_owner_empty_history_and_single_binding_guards')
        proof=hooks.coverage()
        assert proof['language_graphs'][0]['fused_fc1']==28 and proof['live_sampler_calls']==11
        saved=hooks.composition.vision_info['trace_calls'].pop('vlm.model.visual.encoder.layers.26.mlp.fc2/batch2')
        rejected(hooks.coverage)
        hooks.composition.vision_info['trace_calls']['vlm.model.visual.encoder.layers.26.mlp.fc2/batch2']=saved
        controls.append('missing_live_projection_coverage_rejected_cpu_graph_double')
        raise KeyboardInterrupt('Deliberate live restoration control')
    except KeyboardInterrupt: pass
    assert importlib.util.spec_from_file_location is original and hooks.sampler.pool is None and closed
    assert hooks.composition.session.stats['methods_restored'] and not hasattr(model,'agent_vision_tiles')
    controls.append('interrupted_hooks_workers_and_consumer_cleanup')
    # Compare a fresh two-row schema without importing any old 64-row result as its qualification.
    old = m.read(m.WORK/'agent-startup-spec-reference/report.json')
    new = m.read(m.WORK/'agent-startup-spec-candidate/report.json')
    for report in (old,new): report['rows']=[report['rows'][i] for i in m.INDICES]
    m.compare(old,new)
    for field in ('history_before','packed','cpu_buffers','speculation','prediction'):
      bad = copy.deepcopy(new)
      if field == 'prediction':
        values=bad['rows'][0][field]['xyz']
        while isinstance(values[0],list): values=values[0]
        values[0] += 1e-7
      else: bad['rows'][0][field] = {'changed':True}
      rejected(lambda:m.compare(old,bad))
    controls.append('input_token_history_speculation_and_float_bit_mutations_rejected')
    class Buffer:
      def __init__(self):
        self.data=np.arange(2*8*32*8,dtype=np.uint16).reshape(2,1,8,32,8)
        self.shape=self.data.shape;self.dtype=SimpleNamespace(itemsize=2)
      def _buffer(self): return self
      def as_memoryview(self,allow_zero_copy=False): return memoryview(self.data).cast('B')
    values=[Buffer() for _ in range(28)]
    before=installer.kv_hashes(values,20)
    values[-1].data[1,0,7,19,7] ^= 1
    assert installer.kv_hashes(values,20) != before
    rejected(lambda:installer.kv_hashes(values,33))
    controls.append('all_28_defined_kv_prefix_bits_and_invalid_extent')
    # Reuse original startup checks against frozen helpers and original Consumer fixtures.
    old_checks=m.load(HERE/'agent-startup-spec-check.py','exact_live_original_startup_checks')
    old_checks.warm=m.load(m.STARTUP/'agent-startup-spec-warm.py','exact_live_cpu_original_warm')
    old_checks.local=lambda name:m.load(m.ROOT/'experiments'/(name+'.py'),'exact_live_original_'+name.replace('-','_'))
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(old_checks.Proofs))
    if not result.wasSuccessful(): raise AssertionError('Original startup CPU guards failed')
    lifecycle=old_checks.live_tests()
    controls.append('original_startup_restore_and_real_consumer_lifecycle_checks')
    schedule=m.load(m.ROOT/'experiments/agent-paced-replay-check.py','exact_live_original_paced_checks')
    schedule.ARGS=SimpleNamespace(manifest=Path('/mnt/nvme-256/footage/f9ad9a96/exports/0000008e--321b2ddc03/alpamayo-4hz-calibrated-two-six/manifest.json'),
      logs=Path('/mnt/nvme-256/footage/f9ad9a96/realdata'),raw_frames=m.RGB)
    names=[name for name in unittest.defaultTestLoader.getTestCaseNames(schedule.Checks)
      if name not in ('test_missing_rgb_cache_is_explicit_and_not_replaced','test_prepared_arrays_match_canonical_helpers_at_selected_queries')]
    scheduled=unittest.TextTestRunner(verbosity=2).run(unittest.TestSuite(schedule.Checks(name) for name in names))
    if not scheduled.wasSuccessful(): raise AssertionError('Original paced scheduling guards failed')
    controls.append('original_received_20hz_clock_one_active_noqueue_and_budget_checks')
  assert not Device._opened_devices,Device._opened_devices
  storage=old_checks.cpu_storage()
  assert set(Device._opened_devices) <= {'CPU','NPY'}
  report=dict(passed=True,gpu_executed=False,controls=controls,original_startup_tests=result.testsRun,
    original_paced_tests=scheduled.testsRun,
    lifecycle=lifecycle,cpu_storage=storage,transforms=hooks.transforms,
    source_sha256={str(p):m.sha(p) for p in HERE.glob('agent-exact-live-*.py')},
    real_received_floor_snapshot_exact=True,actual_gpu_parity=False,queued_paced_replay=False)
  m.write_new(args.output,report)
  print(json.dumps(dict(passed=True,controls=len(controls),gpu_executed=False)))


if __name__ == '__main__': main()
