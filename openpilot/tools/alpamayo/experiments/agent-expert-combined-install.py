"""Private wrapper around the unchanged qualified combined and PV installers."""
import functools
import importlib.util
from pathlib import Path
import sys


def factory(C):
  c, manifest = C.base()
  original = C.load(C.EXP/'agent-exact-combined-install.py', 'expert_combined_base_install')

  class Composition(original.Composition):
    TARGETS = original.Composition.TARGETS | {'expert-kernels.py','lookup-decode.py'}

    def __init__(self, stage, candidate, evidence, **kwargs):
      # Both arms retain all three already-qualified combined optimizations.
      super().__init__(c, manifest, stage, True, **kwargs)
      self.pv_candidate, self.pv_evidence = candidate, evidence
      self.pv_handle = None
      self.lookup_initial = []
      self.pv_hook = C.load(C.PV_ROOT/'agent-expert-pv-qualified-hook.py', 'expert_combined_frozen_pv_hook')
      self.pv_hook.common = lambda:C.pv()[0]

    def after_load(self, module, path):
      self.require_path(path)
      if path.name == 'expert-kernels.py':
        original_install = module.install
        @functools.wraps(original_install)
        def install(model, *values, **options):
          assert self.pv_handle is None and self.packed is None and self.session is None
          result = original_install(model, *values, **options)
          before = tuple(getattr(model,n) for n in ('linear','quantize','text_layer','attention'))
          search = sys.path[:]
          try: handle = self.pv_hook.install(model,module,candidate=self.pv_candidate,evidence=self.pv_evidence)
          finally: sys.path[:] = search
          assert before == tuple(getattr(model,n) for n in ('linear','quantize','text_layer','attention'))
          self.pv_handle = handle
          self.stack.callback(handle.close)
          self.stack.enter_context(handle.observe_programs())
          self.events.append('qualified_pv_before_packed_quantizer')
          return result
        self.patched(module,'install',install)
      elif path.name == 'lookup-decode.py':
        original_init = module.LookupDecoder.__init__
        @functools.wraps(original_init)
        def initialize(decoder, model, *values, **options):
          original_init(decoder,model,*values,**options)
          assert self.model is model and self.arm_calls == 1 and decoder.history == []
          assert not self.lookup_initial, 'Multiple lookup decoders rejected'
          self.lookup_initial.append({'history':[], 'vision_cache_empty':not self.cache.cache,
            'vision_encoders_empty':not self.cache.encoders, 'vision_stitchers_empty':not self.cache.stitchers})
          assert all(self.lookup_initial[0][k] for k in ('vision_cache_empty','vision_encoders_empty','vision_stitchers_empty'))
        self.patched(module.LookupDecoder,'__init__',initialize)
      else:
        if path.name == 'quantize-packed.py': assert self.pv_handle is not None, 'PV must precede later quantization/precision wrappers'
        super().after_load(module,path)

    def coverage(self):
      # The four declared anchors and native fixture exercise batch 2. Batch 1 remains
      # independently qualified by the original 16 micro cases and full quality64 pair.
      assert self.arm_calls == 1 and len(self.lookup_initial) == 1 and self.pv_handle is not None
      language = [{'jit':n,'fused_fc1':sum(v.startswith('agent_packed_epilogue') for v in self.names(j))}
                  for n,j in self.hook.prefill_jits(self.model)]
      vision = []
      for shape,jit in self.cache.encoders.items():
        names = self.names(jit)
        vision.append({'shape':[list(s) for s in shape], 'fc1':sum(n.startswith('agent_vision_tiles_3x3') for n in names),
          'fc2':sum(n.startswith('agent_vision_tiles_3x6') for n in names)})
      stats = self.session.stats
      assert set(stats['layer_hits']) == {self.hook.PREFIX+str(i) for i in range(28)}
      assert len(set(stats['layer_hits'].values())) == 1
      assert stats['python_trace_fc1'] == stats['python_trace_fc2'] == stats['python_trace_fused_quantize'] == stats['layer_completions'] > 0
      assert not stats['failed_text_layers'] and set(stats['shape_hits']) == {'2048'}
      assert any(r['fused_fc1'] == 28 for r in language)
      traces = self.vision_info['trace_calls']
      required = {f'vlm.model.visual.encoder.layers.{i}.mlp.fc{fc}/batch2' for i in range(2,27) for fc in (1,2)}
      assert {n for n,v in traces.items() if v > 0} == required
      assert vision and all(r['fc1'] == r['fc2'] == 25 for r in vision)
      expected_shape = [[1,720,768],[1,720,1152]]*2
      assert all(r['shape'] == expected_shape for r in vision), vision
      assert self.session.frame is None and not self.session.state['prefill']
      if self.stage == 'native': assert self.sampler is not None and self.sampler_calls > 0 and self.sampler.misses > 0
      else: assert self.sampler_calls == 0 and self.sampler is None
      return {'language_graphs':language, 'vision_graphs':vision,'language_stats':stats,'vision_hook':self.vision_info,
        'events':self.events,'sampler_calls':self.sampler_calls,'sampler':None if self.sampler is None else self.sampler.state(),
        'lookup_initial':self.lookup_initial,'vision_batches_exercised':[2],
        'batch1_evidence':'Original full quality64 and all 16 original micro cases; not exercised by this four-query pair',
        'quantizer_frame_empty':True,'all_28_mlp_and_50_vision_projections':True}

  return Composition
