"""CPU lifecycle checks for current/full history policies and input-worker isolation."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

HERE = Path(__file__).resolve().parent
sys.dont_write_bytecode = True


def local(name):
  spec = importlib.util.spec_from_file_location('agent_token_current_hook_check_'+name.replace('-', '_'), HERE/(name+'.py'))
  module = importlib.util.module_from_spec(spec)
  sys.modules[spec.name] = module
  spec.loader.exec_module(module)
  return module


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--output', type=Path, default=HERE/'agent-token-current-hook-report.json')
  args = parser.parse_args()
  if os.environ.get('DEV') != 'CPU': parser.error('Set DEV=CPU explicitly')
  import torch
  from tinygrad import Device
  assert not Device._opened_devices
  route, factory = local('agent-token-current-route'), local('agent-token-current-runtime')
  policy = factory.policy_module
  query = 3_000_000_000
  raw = [(camera, query-offset) for camera in (1, 6)
         for offset in (2_000_000_000, 1_000_000_000, 500_000_000, 100_000_000, 2_000_000 if camera == 1 else 0)]
  data = dict(camera_indices=torch.tensor([c for c, _ in raw], dtype=torch.int64),
              absolute_timestamps=torch.tensor([t//1000 for _, t in raw], dtype=torch.int64))
  rows = {'sample-3000000':dict(query_ns=query, frames=[dict(view=c, capture_ns=t) for c, t in raw]), 'future':object()}
  ids = np.zeros((1, 2112), dtype=np.int32)
  spans, cursor = [], 11
  for _ in range(10): spans.append((cursor, cursor+180)); ids[0, cursor:cursor+180] = 999; cursor += 187
  results = []
  for name, ratio in policy.POLICIES.items():
    order, restored = [], []
    class FakeModel:
      def __init__(self): order.append('original_constructor')
    class Owner:
      def __init__(self, model, require_packed):
        assert require_packed
        self.model, self.events = model, []
        self.decoders = []
        self.metadata = None
        model._agent_token_pruner = self
        order.append('private_pruner_below_packed_wrapper')
      def restore(self): restored.append(True)
      def register_decoder(self, decoder): self.decoders.append(decoder)
      def install_padded(self): self.model.padded_prefill = 'inherited_current_prefill'
      def prepare(self, prompt, frames):
        policy.validate_bound_frames(prompt, frames, require_full_resolution=True)
        self.metadata = prompt, frames
    class Inputs:
      def prepare(self, values):
        return SimpleNamespace(vlm_config=SimpleNamespace(image_token_id=999)), None, {
          'tokenized_data':{'image_grid_thw':np.tile([1,20,36], (10,1))}}
    class Native:
      def __init__(self): self.rows = rows
      def prepare(self, sample): return 'unchanged-native-images', {}
    class Decoder:
      def __init__(self, model): self.model = model
      def run(self, model, args, config, tokenizer, prepared, packed, seed, inputs):
        owner = model._agent_token_pruner
        assert owner.metadata[0] == 2027
        plan = policy.select(np.ones(2027), 2027, 2304, owner.metadata[1], name)
        owner.events.append(dict(policy=name, older_image_retention=ratio, latest_image_retention=1.,
                                 frame_retention=policy.frame_counts(plan), prompt_prefix_invariance_pass=False))
        return dict(generated_token_ids=[7,19,101,119])
    def protected_initializer(): raise AssertionError('Initializer must not be invoked by this hook fixture')
    def packed_install(model): order.append('packed_wrapper_after_controller')
    def padded_install(model): model.padded_prefill = 'original-padded'
    modules = {'benchmark-fast':SimpleNamespace(), 'fast-inputs':SimpleNamespace(FastInputs=Inputs),
      'stream-inputs':SimpleNamespace(NativeFrames=Native), 'padded-prefill':SimpleNamespace(install=padded_install),
      'quantize-packed':SimpleNamespace(install=packed_install), 'lookup-decode':SimpleNamespace(LookupDecoder=Decoder)}
    original_loader = lambda key:modules[key]
    pipeline = SimpleNamespace(local_module=original_loader, ThreadPoolExecutor=ThreadPoolExecutor, initialize=protected_initializer)
    benchmark = SimpleNamespace(EdgeModel=FakeModel)
    original_prepare, original_native = Inputs.prepare, Native.prepare
    runtime = SimpleNamespace(NumericalPruner=Owner, current_observations=policy.current_observations,
      prepared_observations=policy.prepared_observations, frame_spans=policy.frame_spans)
    installed = []
    with patch.dict(sys.modules, {'benchmark':benchmark}), ExitStack() as stack:
      assert route.install_pipeline_hooks(pipeline, runtime, stack, installed) is protected_initializer
      pipeline.local_module('benchmark-fast')
      model = benchmark.EdgeModel()
      pipeline.local_module('quantize-packed').install(model)
      pipeline.local_module('padded-prefill').install(model)
      adapter = pipeline.local_module('fast-inputs').FastInputs()
      native = pipeline.local_module('stream-inputs').NativeFrames()
      decoder = pipeline.local_module('lookup-decode').LookupDecoder(model)
      def prepare_input(sample):
        if sample == 'sample-3000000':
          assert native.prepare(sample)[0] == 'unchanged-native-images'
          return adapter.prepare(data)
        future = {key:value.clone() for key, value in data.items()}
        future['absolute_timestamps'] += 100_000
        return adapter.prepare(future)
      with pipeline.ThreadPoolExecutor(max_workers=2) as pool:
        current = pool.submit(prepare_input, 'sample-3000000')
        future = pool.submit(prepare_input, 'sample-3100000')
        config, tokenizer, prepared = current.result()
        future_prepared = future.result()[2]
      assert model._agent_token_pruner.metadata is None  # Workers must not prepare the controller.
      assert prepared['_agent_token_prune_observations'][0] == query
      assert future_prepared['_agent_token_prune_observations'][0] == query+100_000_000
      prepared['actual_prompt_length'] = 2027
      value = decoder.run(model, None, config, tokenizer, prepared, (ids,None,None,None), 0, SimpleNamespace(spans=spans))
      assert value['generated_token_ids'] == [7,19,101,119]
      assert value['token_prune_policy']['query_ns'] == query
      assert model.padded_prefill == 'inherited_current_prefill'
      assert pipeline.initialize is protected_initializer
      results.append(dict(policy=name, event=value['token_prune_policy'], future_worker_did_not_change_current=True,
                          protected_initializer_identity_preserved=True, inherited_decoder_output_preserved=True))
    assert restored == [True] and benchmark.EdgeModel is FakeModel
    assert pipeline.initialize is protected_initializer and pipeline.local_module is original_loader
    assert pipeline.ThreadPoolExecutor is ThreadPoolExecutor and Inputs.prepare is original_prepare and Native.prepare is original_native
    assert order == ['original_constructor', 'private_pruner_below_packed_wrapper', 'packed_wrapper_after_controller']
  assert not Device._opened_devices
  report = dict(cpu_only=True, opened_devices=[], policies=list(policy.POLICIES), results=results, hooks_restored=True,
                source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in HERE.glob('agent-token-current*.py')})
  args.output.write_text(json.dumps(report, indent=2)+'\n')
  print(json.dumps(dict(output=str(args.output), passed=True, opened_devices=[]), indent=2))


if __name__ == '__main__': main()
