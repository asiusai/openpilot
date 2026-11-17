"""CPU lifecycle and input-worker checks for the standalone pipeline import hook."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

HERE = Path(__file__).resolve().parent


def local(name):
  spec = importlib.util.spec_from_file_location(name.replace('-', '_'), HERE/(name+'.py'))
  result = importlib.util.module_from_spec(spec); sys.modules[spec.name] = result; spec.loader.exec_module(result)
  return result


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--output', type=Path, default=HERE/'agent-token-prune-hook-report.json')
  args = parser.parse_args()
  route, runtime = local('agent-token-prune-route'), local('agent-token-prune-runtime')
  from tinygrad import Device
  assert not Device._opened_devices
  order, restored = [], []
  class FakeModel:
    def __init__(self): order.append('model_loaded')
  class FakeOwner:
    def __init__(self, model, require_packed):
      assert require_packed
      self.model, self.events, self.decoders = model, [], []
      model._agent_token_pruner = self
      order.append('pruner_below_packed_wrapper')
    def restore(self): restored.append(True)
    def register_decoder(self, decoder): self.decoders.append(decoder)
    def prepare(self, prompt, frames): self.metadata = prompt, frames
    def install_padded(self): self.model.padded_prefill = 'numerical_padded_prefill'
  class FakeInputs:
    def prepare(self, data):
      return SimpleNamespace(vlm_config=SimpleNamespace(image_token_id=999)), None, {
        'tokenized_data': {'image_grid_thw': np.tile([1, 20, 36], (10, 1))}}
  class FakeNative:
    def __init__(self, rows): self.rows = rows
    def prepare(self, sample): return 'unchanged_images', {'unchanged_timing': 1}
  def decoder_type():
    class Decoder:
      def __init__(self, model): self.model = model
      def run(self, model, args, config, tokenizer, prepared, packed, seed, inputs):
        owner = self.model._agent_token_pruner
        assert owner.metadata[0] == 2027
        owner.events.append({'retention': 0.5, 'prompt_prefix_invariance_pass': False})
        return {'generated_token_ids': [7, 19, 101, 119]}
    return Decoder
  block, draft, lookup = decoder_type(), decoder_type(), decoder_type()
  def packed_install(model): order.append('packed_wrapper_after_pruner'); return 112
  def padded_install(model): model.padded_prefill = 'original_padded_prefill'
  modules = {
    'benchmark-fast': SimpleNamespace(), 'stream-inputs': SimpleNamespace(NativeFrames=FakeNative),
    'fast-inputs': SimpleNamespace(FastInputs=FakeInputs), 'padded-prefill': SimpleNamespace(install=padded_install),
    'quantize-packed': SimpleNamespace(install=packed_install),
    'block-decode': SimpleNamespace(BlockDecoder=block), 'draft-decode': SimpleNamespace(DraftDecoder=draft),
    'lookup-decode': SimpleNamespace(LookupDecoder=lookup),
  }
  pipeline = SimpleNamespace(local_module=lambda name: modules[name], ThreadPoolExecutor=ThreadPoolExecutor)
  benchmark = SimpleNamespace(EdgeModel=FakeModel)
  installed = []
  source_runtime = SimpleNamespace(NumericalPruner=FakeOwner, current_observations=runtime.current_observations,
                                   frame_spans=runtime.frame_spans)
  query = 3_000_000_000
  frames = [{'view': camera, 'capture_ns': query-offset} for camera in (1, 6)
            for offset in (2_000_000_000, 1_000_000_000, 500_000_000, 100_000_000, 0)]
  data = {'camera_indices': np.array([f['view'] for f in frames]),
          'absolute_timestamps': np.array([f['capture_ns']//1000 for f in frames])}
  native_rows = {'sample-3000000': {'query_ns': query, 'frames': frames}, 'future': object()}
  spans, cursor = [], 11
  ids = np.zeros((1, 2112), dtype=np.int32)
  for _ in range(10): spans.append((cursor, cursor+180)); ids[0, cursor:cursor+180] = 999; cursor += 187
  original_loader = pipeline.local_module
  original_input_prepare, original_native_prepare = FakeInputs.prepare, FakeNative.prepare
  results = []
  with patch.dict(sys.modules, {'benchmark': benchmark}), ExitStack() as stack:
    route.install_pipeline_hooks(pipeline, source_runtime, stack, installed)
    pipeline.local_module('benchmark-fast')
    model = benchmark.EdgeModel()
    pipeline.local_module('quantize-packed').install(model)
    pipeline.local_module('padded-prefill').install(model)
    assert model.padded_prefill == 'numerical_padded_prefill'
    Native = pipeline.local_module('stream-inputs').NativeFrames
    Inputs = pipeline.local_module('fast-inputs').FastInputs
    native, adapter = Native(native_rows), Inputs()
    decoder = pipeline.local_module('lookup-decode').LookupDecoder(model)
    def prepare_input(sample):
      if str(sample).endswith('3000000'):
        assert native.prepare(sample) == ('unchanged_images', {'unchanged_timing': 1})
        return adapter.prepare(data)
      future_data = {key: value.copy() for key, value in data.items()}
      future_data['absolute_timestamps'] += 100000
      return adapter.prepare(future_data)
    with pipeline.ThreadPoolExecutor(max_workers=2) as pool:
      current = pool.submit(prepare_input, 'sample-3000000')
      future = pool.submit(prepare_input, 'sample-3100000')
      config, tokenizer, prepared = current.result()
      future_prepared = future.result()[2]
    assert prepared['_agent_token_prune_observations'][0] == query
    assert future_prepared['_agent_token_prune_observations'][0] == query+100_000_000
    assert prepared['_agent_token_prune_availability_source'].startswith('native')
    assert future_prepared['_agent_token_prune_availability_source'].startswith('explicit')
    prepared['actual_prompt_length'] = 2027
    packed = (ids, None, None, None)
    result = decoder.run(model, None, config, tokenizer, prepared, packed, 0, SimpleNamespace(spans=spans))
    assert result['generated_token_ids'] == [7, 19, 101, 119]
    assert result['token_prune_policy']['query_ns'] == query
    assert model._agent_token_pruner.metadata[1][0].offset_us == -2000000
    results.append(result['token_prune_policy'])
    # Snapshot timestamps are checked against the path's explicit current query.
    bad = {key: value.copy() for key, value in data.items()}
    bad['absolute_timestamps'][-1] += 1
    try: route.snapshot_observations(bad, 'sample-3000000', source_runtime)
    except ValueError: pass
    else: raise AssertionError('Future snapshot frame accepted')
  assert pipeline.local_module is original_loader and pipeline.ThreadPoolExecutor is ThreadPoolExecutor
  assert benchmark.EdgeModel is FakeModel and restored == [True]
  assert FakeInputs.prepare is original_input_prepare and FakeNative.prepare is original_native_prepare
  assert order == ['model_loaded', 'pruner_below_packed_wrapper', 'packed_wrapper_after_pruner']
  saved = {'command': ['python', str(HERE/'benchmark-pipeline.py'), '--packed-quantize', '--prefill-calibration', 'fixed.json',
                       '--pipeline-inputs', '--compact-inputs', '--decode-block', '6', '--output', 'old.json']}
  resolved = route.qualify_command(saved, 'new.json', 2, 2)
  assert resolved[resolved.index('--output')+1] == 'new.json'
  assert '--native-manifest' not in resolved
  assert not Device._opened_devices
  report = {'cpu_only': True, 'opened_devices': [], 'hook_order': order, 'scope_restored': True,
            'native_and_snapshot_query_sources': True, 'two_input_workers_do_not_replace_current_metadata': True,
            'future_snapshot_frame_rejected': True, 'original_generated_tokens_preserved': True,
            'both_native_and_64_window_invocations_supported': True, 'prediction_events': results,
            'source_sha256': {name: hashlib.sha256((HERE/name).read_bytes()).hexdigest() for name in
                              ('agent-token-prune-runtime.py', 'agent-token-prune-route.py', 'agent-token-prune-hook-check.py')}}
  args.output.write_text(json.dumps(report, indent=2)+'\n')
  print(json.dumps(report, indent=2))


if __name__ == '__main__': main()
