"""CPU actual-contract qualification of the numerical whole-prompt policy.

Uses frozen EdgeModel/DraftDecoder method bodies with deterministic small fake
projection operators, not checkpoint weights. No GPU or route-quality claim.
The old prompt-prefix counterexample remains false and is never relabeled.
"""
import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import time
from types import MethodType, SimpleNamespace

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))


def local(name):
  spec = importlib.util.spec_from_file_location(name.replace('-', '_'), HERE/(name+'.py'))
  result = importlib.util.module_from_spec(spec)
  sys.modules[spec.name] = result
  spec.loader.exec_module(result)
  return result


def fake_model():
  from tinygrad import Tensor, TinyJit, dtypes
  from model import EdgeModel
  model = EdgeModel.__new__(EdgeModel)
  model.config = {'vlm_config': {'text_config': {'num_hidden_layers': 28, 'eos_token_id': 127}},
                  'expert_config': {'llm_config': {'num_hidden_layers': 28}},
                  'traj_ids': {'history_id0': 110, 'future_id0': 118}, 'history_vocab_size': 4, 'future_vocab_size': 4}
  model.dtype, model.fp8_wmma, model.cuda_kernels = dtypes.float32, True, False
  model.weights, model.weight_scales = {}, {}
  for prefix, hidden in (('vlm.model.language_model', 2048), ('expert.expert', 512)):
    one = Tensor.ones(hidden, device='CPU', dtype=dtypes.float32).contiguous().realize()
    for layer in range(28):
      for name in ('input_layernorm', 'post_attention_layernorm'):
        model.weights[f'{prefix}.layers.{layer}.{name}.weight'] = one
    model.weights[f'{prefix}.norm.weight'] = one
  rng = np.random.default_rng(192761)
  model.weights['vlm.model.language_model.embed_tokens.weight'] = Tensor(
    rng.normal(size=(128, 2048)).astype(np.float32)*0.1, device='CPU').realize()
  model.output_token_mapping = Tensor(np.arange(128, dtype=np.int32), device='CPU').realize()
  model.cache, model.prompt_length, model.minimum_cache_capacity = None, None, 0
  model.prefill_jit, model.decode_jit = TinyJit(model._language), TinyJit(model._language)
  model.greedy_decode_jit, model.greedy_shape, model.trajectory_jits = TinyJit(model._greedy_decode), None, {}
  model.projection_calls = []
  def linear(self, x, name, quantized_input=None, heads=None):
    self.projection_calls.append((name, x.shape[1] if x.ndim == 3 else x.shape[0]))
    hidden = 512 if name.startswith('expert.') else 2048
    if name.endswith('q_proj'):
      return x.repeat((1, 1, 2048//x.shape[-1]))*0.13
    if name.endswith('k_proj'): return x.repeat((1, 1, max(1, 1024//x.shape[-1])))[..., :1024]*0.17
    if name.endswith('v_proj'): return x.repeat((1, 1, max(1, 1024//x.shape[-1])))[..., :1024]*0.11
    if name.endswith('o_proj'): return x[..., :hidden]*0.07
    if name.endswith('fc1'): return x[..., :16]*0.09
    if name.endswith('fc2'): return x.repeat((1, 1, hidden//x.shape[-1]))*0.03
    if name == 'vlm.lm_head': return x[..., :128]*0.2
    if name == 'expert.action_out_proj': return x[..., :2]*0.12
    raise ValueError(name)
  def action_projection(self, actions, timestep):
    return actions.repeat((1, 1, 256))*0.02+timestep.reshape(1, 1, 1)*0.001
  model.linear, model.action_projection = MethodType(linear, model), MethodType(action_projection, model)
  return model


def availability_checks(runtime):
  query = 3_000_000_000
  frames = [{'view': camera, 'capture_ns': query-offset} for camera in (1, 6)
            for offset in (2_000_000_000, 1_000_000_000, 500_000_000, 100_000_000, 0)]
  class CurrentOnly(dict):
    def __getitem__(self, key):
      if key != 'current': raise AssertionError('Unavailable future row read')
      return super().__getitem__(key)
  rows = CurrentOnly(current={'query_ns': query, 'frames': frames,
                              'recorded_motion_future_xyz': object(), 'small_model': object()})
  original = runtime.current_observations(rows, 'current')
  rows['future'] = {'query_ns': query+100_000_000, 'frames': object(), 'recorded_motion_future_xyz': object()}
  assert runtime.current_observations(rows, 'current') == original
  rows['current']['recorded_motion_future_xyz'] = object()
  assert runtime.current_observations(rows, 'current') == original
  bad = {'current': {'query_ns': query, 'frames': [*frames[:-1], {'view': 6, 'capture_ns': query+1}]}}
  try: runtime.current_observations(bad, 'current')
  except ValueError: pass
  else: raise AssertionError('Future camera input accepted')
  return {'unavailable_future_records_not_read': True, 'future_ground_truth_fields_not_read': True,
          'future_frame_in_current_input_rejected': True, 'current_observations': original}


def packed_producer_check():
  from tinygrad import Tensor, dtypes
  packed = local('quantize-packed')
  prefix = 'vlm.model.language_model.layers.2'
  dummy = SimpleNamespace(weights={prefix+'.self_attn.q_proj.weight': Tensor.ones(16, 16, device='CPU').cast(dtypes.fp8e4m3).realize()})
  dummy.quantize = lambda x: (x, None)
  dummy.linear = lambda *a, **kw: None
  dummy.text_layer = lambda x, *a, **kw: dummy.quantize(x)
  count = packed.install(dummy, fixed_scales={prefix+'/0': 2.0})
  result, scale = dummy.text_layer(Tensor.ones(128, 16, device='CPU').cast(dtypes.bfloat16).reshape(1, 128, 16).realize(),
                                   None, None, prefix)
  assert isinstance(result, packed.PackedActivation) and (result.rows, result.width) == (128, 16)
  assert scale.item() == 2.0 and result.value.dtype == dtypes.fp8e4m3 and count == 1
  return {'actual_quantize_packed_wrapper': True, 'PackedActivation_preserved': True, 'fixed_scale_preserved': 2.0,
          'scope': 'Producer interface test only; actual GPU FP8 model arithmetic remains unvalidated.'}


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--output', type=Path, default=HERE/'agent-token-prune-numerical-policy-report.json')
  args = parser.parse_args()
  from tinygrad import Device, Tensor, TinyJit, dtypes
  from tinygrad.uop.ops import UOp
  assert Device.DEFAULT == 'CPU' and os.environ.get('DEV') == 'CPU'
  started = time.perf_counter()
  runtime = local('agent-token-prune-runtime')
  ref = runtime.reference
  availability = availability_checks(runtime)
  packed_contract = packed_producer_check()
  model = fake_model()
  owner = runtime.NumericalPruner(model, require_packed=False)
  # Instrument the outer packed-wrapper boundary without changing arithmetic.
  # This proves copied prefill calls remain beneath the layer wrapper.
  layer = model.text_layer
  traced = []
  def outer_layer(self, x, *values, **kwargs):
    prefix = values[2]
    traced.append((prefix, owner.phase, x.shape[1]))
    return layer(x, *values, **kwargs)
  model.text_layer = MethodType(outer_layer, model)
  prompt, full_rows, budget = 24, 64, 32
  frames = (ref.FrameSpan(2, 6, 1, -100000), ref.FrameSpan(9, 13, 6, 0))
  rng = np.random.default_rng(729341)
  a = rng.normal(size=(1, full_rows, 2048)).astype(np.float32)*0.1
  b = a.copy(); b[:, :prompt] += rng.normal(size=(1, prompt, 2048)).astype(np.float32)*0.3
  angles = np.arange(full_rows, dtype=np.float32)[:, None]*np.linspace(0.001, 0.08, 64)[None]
  cos = Tensor(np.cos(np.concatenate((angles, angles), axis=1))[None, None].astype(np.float32), device='CPU').realize()
  sin = Tensor(np.sin(np.concatenate((angles, angles), axis=1))[None, None].astype(np.float32), device='CPU').realize()
  events, logits_a, plan_a = [], None, None
  for iteration, data in enumerate((a, b, a, a)):
    owner.prepare(prompt, frames)
    if iteration == 3:
      for cache in model.cache:
        raw = cache.numpy().copy(); raw[..., prompt:, :] = rng.normal(size=raw[..., prompt:, :].shape).astype(np.float32)*1e3
        cache.assign(Tensor(raw, device='CPU')).realize()
      data = data.copy()
      data[:, prompt:] = rng.normal(size=data[:, prompt:].shape).astype(np.float32)*1e3
    x = Tensor(data.copy(), device='CPU').realize()
    logits = owner.padded_prefill(x, cos, sin, budget, prompt).numpy().copy()
    event = owner.events[-1]
    assert event['fallback'] is None and event['retained_prompt_tokens'] == 20
    assert event['original_prompt_tokens'] == prompt and event['retention'] == 0.5
    assert event['prompt_prefix_invariance_pass'] is False
    if iteration == 0:
      logits_a, plan_a = logits.copy(), owner.last_plan.slots.copy()
    elif iteration >= 2:
      np.testing.assert_array_equal(logits, logits_a)
      np.testing.assert_array_equal(owner.last_plan.slots, plan_a)
    events.append(event)
  assert owner.early_jit.cnt == 4 and next(iter(owner.late_jits.values())).cnt == 4
  assert events[0]['selected_slots_sha256'] != events[1]['selected_slots_sha256'], 'Changed selection must exercise mask replay'
  # Same padded shape, different actual prompt: both symbolic row selectors move.
  owner.prepare(prompt+1, frames)
  changed_prompt = owner.padded_prefill(Tensor(a.copy(), device='CPU').realize(), cos, sin, budget, prompt+1).numpy().copy()
  changed_plan = owner.last_plan.slots.copy()
  fresh_model = fake_model()
  fresh_owner = runtime.NumericalPruner(fresh_model, require_packed=False)
  fresh_owner.prepare(prompt+1, frames)
  fresh_prompt = fresh_owner.padded_prefill(Tensor(a.copy(), device='CPU').realize(), cos, sin, budget, prompt+1).numpy()
  np.testing.assert_array_equal(changed_prompt, fresh_prompt)
  np.testing.assert_array_equal(changed_plan, fresh_owner.last_plan.slots)
  assert changed_plan[-1] == prompt and len(changed_plan) == 21
  fresh_owner.restore()
  owner.prepare(prompt, frames)
  restored_prompt = owner.padded_prefill(Tensor(a.copy(), device='CPU').realize(), cos, sin, budget, prompt).numpy()
  np.testing.assert_array_equal(restored_prompt, logits_a)
  # Compare both full early-layer KV caches against original frozen method calls.
  baseline = fake_model()
  capacity = model.cache[0].shape[3]
  baseline.cache = [Tensor.zeros(2, 1, 8, capacity, 128, device='CPU', dtype=dtypes.float32).realize() for _ in range(28)]
  hidden = Tensor(a.copy(), device='CPU').realize()
  for i in range(2):
    if i == 1:
      normalized = baseline.norm(hidden, 'vlm.model.language_model.layers.1.input_layernorm', rms=True)
      q, k = (baseline.linear(normalized, 'vlm.model.language_model.layers.1.self_attn.'+name+'_proj', heads=heads)
              .reshape(1, full_rows, heads, 128).transpose(1, 2) for name, heads in (('q', 16), ('k', 8)))
      q, k = baseline.rope(q, k, cos, sin)
      independent_scores = ref.last_prompt_saliency(q.numpy()[0, :, :prompt], k.numpy()[0, :, :prompt], prompt)
    hidden = baseline.text_layer(hidden, cos, sin, f'vlm.model.language_model.layers.{i}', baseline.cache[i], None, cache_pos=0)
    np.testing.assert_array_equal(model.cache[i].numpy()[..., :prompt, :], baseline.cache[i].numpy()[..., :prompt, :])
  assert all(any(p == f'vlm.model.language_model.layers.{i}' and stage in ('early', 'late') for p, stage, _ in traced) for i in range(28))
  _, scores = owner.early_jit(Tensor(a.copy(), device='CPU').realize(), cos, sin,
                             UOp.variable('agent_prune_actual_prompt', 1, full_rows).bind(prompt))
  np.testing.assert_allclose(scores.numpy()[:prompt], independent_scores, rtol=2e-6, atol=1e-7)
  np.testing.assert_array_equal(scores.numpy()[prompt:], np.zeros(full_rows-prompt))
  allowed = owner.allowed.numpy().copy()
  expected = np.zeros(capacity, bool); expected[plan_a] = True; expected[prompt:] = True
  np.testing.assert_array_equal(allowed, expected)
  c = cos[:, :, :1].clone().realize(); s = sin[:, :, :1].clone().realize()
  embedding = model.weights['vlm.model.language_model.embed_tokens.weight'][7].reshape(1, 1, 2048).clone().realize()
  serial = model.decode(embedding, c, s, prompt).numpy().copy()
  # Real DraftDecoder.verify body, with empty FP8 weight scales and CPU projections.
  draft_module = local('draft-decode')
  decoder = draft_module.DraftDecoder.__new__(draft_module.DraftDecoder)
  decoder.model, decoder.fast_attention, decoder.tile = model, False, (4, 4)
  decoder.gemm = SimpleNamespace(gemm=None)
  owner.register_decoder(decoder)
  verify = TinyJit(decoder.verify)
  proposals = np.arange(16, dtype=np.int32)%90
  proposals[0] = 7
  dc = Tensor(np.tile(cos.numpy()[:, :, :16], (1, 1, 2, 1)), device='CPU').realize()
  ds = Tensor(np.tile(sin.numpy()[:, :, :16], (1, 1, 2, 1)), device='CPU').realize()
  position = UOp.variable('draft_position', 0, capacity-16).bind(prompt)
  step = UOp.variable('draft_step', 0, dc.shape[2]-16).bind(0)
  verified_rows = None
  for iteration in range(4):
    ids = proposals.copy()
    if iteration % 2: ids[3:] = 80-ids[3:]
    values, predictions = verify(Tensor(ids, device='CPU').realize(), dc, ds, position, step)
    values = values.numpy().copy()
    if verified_rows is None:
      verified_rows = values[:, :3].copy()
      np.testing.assert_allclose(values[:, :1], serial, rtol=1e-5, atol=1e-6)
    else: np.testing.assert_array_equal(values[:, :3], verified_rows)
    np.testing.assert_array_equal(predictions.numpy(), values.argmax(axis=-1))
  # Expert uses the actual frozen trajectory/expert/text-layer bodies.
  actions_array = rng.normal(size=(1, 64, 2)).astype(np.float32)*0.1
  ac = Tensor(np.tile(cos.numpy(), (1, 1, 2, 1)), device='CPU').realize()
  ass = Tensor(np.tile(sin.numpy(), (1, 1, 2, 1)), device='CPU').realize()
  trajectories = []
  for iteration in range(3):
    if iteration:
      for i, cache in enumerate(model.cache):
        values = cache.numpy().copy()
        invalid = (np.zeros(capacity, dtype=bool) if i < 2 else ~allowed) | (np.arange(capacity) >= prompt+3)
        values[..., invalid, :] = rng.normal(size=values[..., invalid, :].shape).astype(np.float32)*1e3
        cache.assign(Tensor(values, device='CPU')).realize()
    trajectory = model.trajectory(Tensor(actions_array.copy(), device='CPU').realize(), ac, ass, prompt+3, 1, 3).numpy().copy()
    if trajectories: np.testing.assert_array_equal(trajectory, trajectories[0])
    trajectories.append(trajectory)
  # Metadata failure invokes a genuine full 28-layer recomputation.
  owner.metadata = None
  fallback = owner.padded_prefill(Tensor(a.copy(), device='CPU').realize(), cos, sin, budget, prompt).numpy().copy()
  assert owner.events[-1]['full_prefill_recomputed'] and owner.events[-1]['retained_prompt_tokens'] == prompt
  hidden = Tensor(a.copy(), device='CPU').realize()
  for i in range(28): hidden = baseline.text_layer(hidden, cos, sin, f'vlm.model.language_model.layers.{i}', baseline.cache[i], None, cache_pos=0)
  expected_full = baseline.linear(baseline.norm(hidden[:, prompt-1:prompt], 'vlm.model.language_model.norm', rms=True), 'vlm.lm_head').numpy()
  np.testing.assert_array_equal(fallback, expected_full)
  try: owner.prepare(prompt, (ref.FrameSpan(2, 6, 1, 1),))
  except ValueError: pass
  else: raise AssertionError('Future camera metadata must not trigger inference fallback')
  # A larger caller reasoning budget changes allocation, never text positions.
  owner.prepare(prompt, frames)
  grown = owner.padded_prefill(Tensor(a.copy(), device='CPU').realize(), cos, sin, 513, prompt).numpy()
  assert model.cache[0].shape[3] == 768 and owner.events[-1]['original_prompt_tokens'] == prompt
  np.testing.assert_array_equal(grown, logits_a)
  assert model.decode_jit.cnt == 0 and not model.trajectory_jits
  owner.restore()
  assert not owner.enabled and not hasattr(model, '_agent_token_pruner')
  restored = model.prefill(Tensor(a[:, :prompt].copy(), device='CPU').realize(),
                           cos[:, :, :prompt].clone().realize(), sin[:, :, :prompt].clone().realize(), budget).numpy()
  np.testing.assert_allclose(restored, expected_full, rtol=1e-5, atol=1e-6)
  assert all(d.split(':')[0] in ('CPU', 'NPY', 'PYTHON') for d in Device._opened_devices), Device._opened_devices
  old_report = json.loads((HERE/'agent-token-prune-cpu-report.json').read_text())
  assert old_report['future_mutation']['mandatory_prefix_causality_gate_pass'] is False
  report = {'classification': 'NUMERICAL NEW MODEL; whole-prompt-conditioned', 'retention': 0.5,
            'prompt_prefix_invariance_pass': False, 'generation_causality_policy_eligible': True,
            'gpu_quality_or_speed_validated': False, 'unmodified_old_report_sha256': hashlib.sha256((HERE/'agent-token-prune-cpu-report.json').read_bytes()).hexdigest(),
            'availability': availability, 'actual_contract': {'edge_model_method_bodies': True, 'draft_verify_method_body': True,
              'fake_cpu_projection_weights': True, 'prefill_compile_capture_replay_events': events,
              'early_full_cache_exact_against_original': True, 'outer_layer_wrapper_preserved': True,
              'actual_layer_one_saliency_matches_independent_fp64': True, 'packed_producer_contract': packed_contract,
              'selection_and_mask_change_exercised_on_replay': True, 'padded_noninput_rows_cannot_influence_selection': True,
              'changed_actual_prompt_same_padded_shape_matches_fresh_model': True,
              'generated_suffix_poison_cannot_change_prefill_or_selection': True, 'serial_decode_checked': True,
              'target_verification_suffix_mutations': 4, 'verification_target': 'changed model, not original model',
              'expert_compile_capture_replay_poison_cases': 3, 'all_layer_validity_original_slots': True,
              'full_fallback_matches_full_recompute': True, 'future_metadata_rejected_before_fallback': True,
              'caller_reasoning_budget_513_preserves_original_positions': True,
              'cache_growth_resets_dependent_graphs': True, 'restore_preserves_outer_layer_wrappers': True,
              'restored_full_prefill_checked': True},
            'frozen_contract_sources': {name: hashlib.sha256((HERE.parent/name).read_bytes()).hexdigest() for name in
              ('model.py', 'experiments/benchmark-pipeline.py', 'experiments/draft-decode.py', 'experiments/block-decode.py',
               'experiments/lookup-decode.py', 'experiments/quantize-packed.py', 'experiments/padded-prefill.py',
               'experiments/fast-inputs.py', 'experiments/stream-inputs.py')},
            'cpu_devices': sorted(Device._opened_devices), 'wall_seconds': time.perf_counter()-started,
            'source_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in HERE.glob('agent-token-prune*.py')}}
  args.output.write_text(json.dumps(report, indent=2)+'\n')
  print(json.dumps({'output': str(args.output), 'numerical_policy_eligible': True, 'prompt_prefix_invariance_pass': False,
                    'cpu_devices': report['cpu_devices'], 'wall_seconds': report['wall_seconds']}, indent=2))


if __name__ == '__main__': main()
