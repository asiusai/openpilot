"""CPU composition, ordering, cleanup, source, and comparison rejection controls."""
import argparse
import copy
from contextlib import ExitStack, contextmanager
import importlib.util
import inspect
import json
from pathlib import Path
import sys
from types import MethodType, SimpleNamespace
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('combined_check_common', HERE/'agent-exact-combined-common.py')
c = importlib.util.module_from_spec(spec)
spec.loader.exec_module(c)
sys.path[:0] = [str(c.TINYGRAD), str(c.MLP/'experiments'), str(c.MLP)]


def rejected(fn, kind=(ValueError, AssertionError, RuntimeError)):
  try: fn()
  except kind: return
  raise AssertionError('Expected rejection')


@contextmanager
def interrupted(composition):
  try:
    with composition: yield
  except KeyboardInterrupt: return
  raise AssertionError('Expected the deliberate cleanup interruption')


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--output', type=Path, required=True)
  args = parser.parse_args()
  from tinygrad import Device, dtypes
  assert not Device._opened_devices
  installer = c.load(HERE/'agent-exact-combined-install.py', 'combined_cpu_installer')
  fixtures = c.load(c.MLP/'experiments/agent-fused-mlp-check.py', 'combined_original_cpu_fixtures')
  Fake = fixtures.FakeTensor
  pins = {str(c.MLP/'experiments'/name):c.sha(c.MLP/'experiments'/name) for name in installer.Composition.TARGETS}
  manifest = dict(files=pins)
  controls, dispatches, vision_dispatches = [], [], []
  fixed = {f'vlm.model.language_model.layers.{i}/{j}':.0037*(1+i*4+j) for i in range(28) for j in range(4)}
  def dispatch(x, w, xs, ws, output_scale, rows, variant):
    assert rows == 2048 and variant == 'transpose4x4'
    dispatches.append((xs, ws, output_scale))
    return Fake(shape=(2048,9216), dtype=dtypes.fp8e4m3)
  def pack(x): return Fake(shape=x.shape, dtype=x.dtype, tag='packed:'+x.tag)
  def quantize(x, scale=None):
    assert type(x) is Fake
    rows, width = x.numel()//x.shape[-1], x.shape[-1]
    return Fake(shape=((rows+15)//16*16,width), dtype=dtypes.fp8e4m3), scale
  def vision_gemm(x, w, scales, bias, tile):
    vision_dispatches.append((x.shape, scales, bias, tile))
    return Fake(shape=(*x.shape[:-1],w.shape[0]), dtype=dtypes.bfloat16)
  original_spec = importlib.util.spec_from_file_location
  normal = installer.Composition(c, manifest, 'quality', False)
  holder = SimpleNamespace(value=object())
  before = holder.value
  with normal: normal.patched(holder, 'value', object())
  assert holder.value is before and importlib.util.spec_from_file_location is original_spec
  controls.append('normal_exit_restores_patches')
  composition = installer.Composition(c, manifest, 'native', True,
    device_factory=lambda:SimpleNamespace(arch='gfx1200',device='AMD'), mlp_dispatch=dispatch)
  with ExitStack() as mocks:
    mocks.enter_context(patch.object(Fake, 'ndim', property(lambda self:len(self.shape)), create=True))
    mocks.enter_context(patch.object(composition.vision, 'Tensor', Fake))
    mocks.enter_context(patch.object(composition.vision, 'gemm', vision_gemm))
    with interrupted(composition):
      rejected(lambda:installer.Composition(c, manifest, 'native', True).__enter__())
      controls.append('nested_install_rejected')
      core = c.load(c.MLP/'model.py', 'combined_cpu_core')
      mocks.enter_context(patch.object(core, 'Tensor', Fake))
      model = fixtures.FakeModel(core)
      model.norm = MethodType(lambda self,x,name,rms=False,eps=None:x,model)
      model.encode_jits = {}
      model.encode = lambda *args:None
      for i in range(27):
        for fc, shape in (('fc1',(4304,1152)),('fc2',(1152,4304))):
          name = f'vlm.model.visual.encoder.layers.{i}.mlp.{fc}'
          dtype = dtypes.bfloat16 if i < 2 else dtypes.fp8e4m3
          model.weights[name+'.weight'] = Fake(shape=shape, dtype=dtype, tag=name)
          model.weights[name+'.bias'] = Fake(shape=(shape[0],), dtype=dtypes.bfloat16)
          if i >= 2: model.weight_scales[name+'.weight'] = Fake([.125],dtype=dtypes.float32)
      model.weights['vlm.model.visual.embeddings.patch_embedding.weight'] = Fake(shape=(1152,1176),dtype=dtypes.bfloat16)
      def grouped(x, images):
        return Fake(shape=(images*720,x.shape[-1]),dtype=dtypes.fp8e4m3), Fake(shape=(images*720,),dtype=dtypes.float32)
      model.grouped_quantize = grouped
      batched = c.load(c.MLP/'experiments/batched-vision.py', 'combined_cpu_batched')
      rejected(lambda:batched.install(model))
      rejected(lambda:composition.arm(model))
      controls.append('vision_and_language_before_packed_rejected')
      packed = c.load(c.MLP/'experiments/quantize-packed.py', 'combined_cpu_packed')
      mocks.enter_context(patch.object(packed, 'Tensor', Fake))
      mocks.enter_context(patch.object(packed, 'pack', pack))
      mocks.enter_context(patch.object(packed, 'quantize', quantize))
      packed.install(model, fixed_scales=fixed)
      rejected(lambda:packed.install(model, fixed_scales=fixed))
      controls.append('duplicate_packed_install_rejected')
      model.prefill_jit.cnt = 1
      rejected(lambda:batched.install(model))
      model.prefill_jit.cnt = 0
      controls.append('late_vision_install_rejected')
      batched.install(model)
      vision_linear = model.linear
      cache_module = c.load(c.MLP/'experiments/vision-cache.py', 'combined_cpu_vision_cache')
      cache = cache_module.VisionCache(model, grouped=True, group_limit=2)
      decoder = c.load(c.MLP/'experiments/block-decode.py', 'combined_cpu_block_decode')
      decoder.BlockDecoder(model)
      controls.append('actual_decoder_constructor_arms_once')
      assert composition.events[:6] == ['before_factory_and_input_imports','packed_quantizer','batched_vision',
                                        'vision_tiles','vision_cache','language_fusion']
      controls.append('canonical_install_order')
      session = composition.session
      before_state = {'prefill':False,'prefix':'outer','index':19}
      session.state.update(before_state)
      scales_object = model.prefill_activation_scales
      for i in range(28):
        x = Fake(shape=(1,2048,2048),dtype=dtypes.bfloat16)
        kv = Fake(shape=(2,1,8,2304,128),dtype=dtypes.bfloat16)
        model.text_layer(x,None,None,f'vlm.model.language_model.layers.{i}',kv,None,0)
        assert session.state == before_state and session.frame is None
        assert dispatches[-1][0] is session.fixed[f'vlm.model.language_model.layers.{i}/2']
        assert dispatches[-1][2] is session.fixed[f'vlm.model.language_model.layers.{i}/3']
      assert len(dispatches) == 28 and model.prefill_activation_scales is scales_object
      controls.append('all_28_original_text_layers_fixed_slots_scales_and_state')
      # Reach the actual batched quantizer closure beneath both optional wrappers.
      q = session.original['quantize']
      image_state = inspect.getclosurevars(q.__func__).nonlocals['state']
      for images in (1,2):
        image_state['images'] = images
        for i in range(2,27):
          for fc, width in (('fc1',1152),('fc2',4304)):
            model.linear(Fake(shape=(images,720,width),dtype=dtypes.bfloat16),
                         f'vlm.model.visual.encoder.layers.{i}.mlp.{fc}')
        image_state['images'] = 0
        assert session.state == before_state
      assert len(vision_dispatches) == 100 and len(dispatches) == 28
      controls.append('all_50_vision_projections_batch1_and_batch2_with_grouped_quantizer')
      count = len(dispatches)
      for rows, past in ((64,5),(2048,1),(128,0)):
        model.text_layer(Fake(shape=(1,rows,2048),dtype=dtypes.bfloat16),None,None,
          'vlm.model.language_model.layers.0',Fake(shape=(2,1,8,2304,128),dtype=dtypes.bfloat16),None,past)
      assert len(dispatches) == count and session.state == before_state
      controls.append('decode_nonzero_cache_and_other_rows_fallback')
      key = 'vlm.model.language_model.layers.0.mlp.fc1.weight'
      scale = model.weight_scales[key]
      model.weight_scales[key] = Fake([.25],dtype=dtypes.float32)
      rejected(lambda:model.text_layer(Fake(shape=(1,2048,2048),dtype=dtypes.bfloat16),None,None,
        'vlm.model.language_model.layers.0',Fake(shape=(2,1,8,2304,128),dtype=dtypes.bfloat16),None,0))
      model.weight_scales[key] = scale
      assert session.state == before_state and session.frame is None
      controls.append('scale_replacement_exception_restores_slot_state')
      session.stats['failed_text_layers'] = 0  # The deliberate CPU rejection is not a route failure.
      # Exercise real native preparation through the process-local stream hook.
      stream = c.load(c.MLP/'experiments/stream-inputs.py', 'combined_cpu_stream')
      inv = c.read(c.MLP.parent/'native-pipeline-invocation.json')['command']
      frames = stream.NativeFrames(*(c.option(inv,k) for k in ('native-manifest','native-calibrations','native-frame-cache')))
      row = c.read(c.option(inv,'native-manifest'))['schedule'][0]
      import numpy as np
      actual, _ = frames.prepare(row['sample'])
      np.testing.assert_array_equal(actual,np.load(Path(row['sample'])/'image_frames.npy',allow_pickle=False))
      assert composition.sampler_calls == 10
      controls.append('source_bound_native_sampler_exact_saved_snapshot')
      # Coverage failures are checked independently from real device graph evidence.
      original_names = composition.names
      composition.names = lambda jit: ['agent_packed_epilogue_cpu']*28 if jit is model.prefill_jit else (
        ['agent_vision_tiles_3x3_b2']*25+['agent_vision_tiles_3x6_b2']*25)
      fake_jit = type(model.prefill_jit)()
      cache.encoders[((2,720,1152),)] = fake_jit
      proof = composition.coverage()
      assert proof['language_graphs'][0]['fused_fc1'] == 28
      missing = composition.vision_info['trace_calls'].pop('vlm.model.visual.encoder.layers.26.mlp.fc2/batch2')
      rejected(composition.coverage)
      composition.vision_info['trace_calls']['vlm.model.visual.encoder.layers.26.mlp.fc2/batch2'] = missing
      composition.names = lambda jit: []
      rejected(composition.coverage)
      composition.names = original_names
      controls.append('missing_projection_and_missing_captured_graph_rejected')
      cache.cache['test'] = object()
      model.prefill_jit.cnt, model.prefill_jit.captured = 2, object()
      runner = c.load(HERE/'agent-exact-combined-run.py', 'combined_cpu_runner')
      runner.hook_chain(composition)
      with patch.object(importlib.util, 'spec_from_file_location', lambda *a,**k:None):
        rejected(lambda:runner.hook_chain(composition))
      controls.append('foreign_outer_import_hook_rejected')
      raise KeyboardInterrupt('Deliberate CPU cleanup test')
    assert importlib.util.spec_from_file_location is original_spec
    assert session.stats['methods_restored'] and model.prefill_jit.resets == 1 and model.decode_jit.resets == 0
    assert not hasattr(model,'agent_vision_tiles') and not hasattr(model,'_agent_fused_mlp_active')
    assert not cache.encoders and not cache.cache and fake_jit.resets == 1
    assert composition.sampler.pool is None
    controls.append('interrupted_reverse_cleanup_methods_graphs_cache_and_native_workers')
  # Compare real saved pairs, then mutate each protected semantic field.
  old = c.read(c.WORK/'agent-fused-mlp-2048-native-reference/report.json')
  new = c.read(c.WORK/'agent-fused-mlp-2048-native-candidate/report.json')
  c.compare(old,new,'native')
  for label, mutate in (
      ('pixel_upload',lambda r:r['runs'][0].__setitem__('uploaded_pixel_bytes',-1)),
      ('vision_cache',lambda r:r['runs'][0].__setitem__('vision_hits',-1)),
      ('history',lambda r:r.__setitem__('history_pixel_scale',.5)),
      ('speculation',lambda r:r['runs'][0]['draft'].__setitem__('verified_tokens',-1)),
      ('float_bits',lambda r:r['runs'][0]['prediction']['xyz'][0][0][0].__setitem__(0,-0.0)),
      ('missing_row',lambda r:r['runs'].pop()),
      ('unknown_field',lambda r:r['runs'][0].__setitem__('new_unreviewed_field',True))):
    changed = copy.deepcopy(new)
    if label == 'float_bits':
      # Locate a float without assuming trajectory nesting depth.
      values = changed['runs'][0]['prediction']['xyz']
      while isinstance(values[0],list): values=values[0]
      values[0] += .000001
    else: mutate(changed)
    rejected(lambda:c.compare(old,changed,'native'))
    controls.append(label+'_comparison_rejected')
  rejected(lambda:c.equal(0.0,-0.0))
  controls.append('signed_zero_bits_rejected')
  bad = copy.deepcopy(manifest)
  bad['files'][str(c.MLP/'experiments/quantize-packed.py')] = '0'*64
  other = installer.Composition(c,bad,'quality',False)
  rejected(lambda:other.require_path(c.MLP/'experiments/quantize-packed.py'))
  controls.append('source_identity_mismatch_rejected')
  with patch.dict(sys.modules,{'native_warp':object()}): rejected(lambda:other.__enter__())
  controls.append('input_import_before_install_rejected')
  assert not Device._opened_devices, Device._opened_devices
  evidence = c.evidence()
  c.write_new(args.output, dict(passed=True,gpu_executed=False,controls=controls,
    original_micro_cases=len(evidence['vision']['completed_cases']),fused_layers=28,vision_projections_per_batch=50,
    real_gpu_graph_validation=False, note='CPU fake dispatch tests routing and restoration; parent route graphs remain required',
    source_sha256={str(p):c.sha(p) for p in HERE.glob('agent-exact-combined-*.py')}))
  print(json.dumps(dict(passed=True,controls=len(controls),gpu_executed=False)))


if __name__ == '__main__': main()
