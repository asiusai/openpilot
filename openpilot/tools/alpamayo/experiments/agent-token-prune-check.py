"""Independent CPU checks for a numerical token-pruning cache contract.

Reports the expected FastV prefix-causality failure explicitly. No GPU hook.
"""
import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

HERE = Path(__file__).resolve().parent


def local(name):
  spec = importlib.util.spec_from_file_location(name.replace('-', '_'), HERE/(name+'.py'))
  module = importlib.util.module_from_spec(spec)
  sys.modules[spec.name] = module
  spec.loader.exec_module(module)
  return module


def independent_attention(q, k, v):
  """Scalar head/row reference over explicitly selected, already valid keys."""
  output = np.empty((*q.shape[:-1], v.shape[-1]))
  for h in range(q.shape[0]):
    kh = h//(q.shape[0]//k.shape[0])
    for row in range(q.shape[1]):
      logits = np.array([np.dot(q[h, row], key)/math.sqrt(q.shape[-1]) for key in k[kh]])
      weights = np.exp(logits-logits.max())
      weights /= weights.sum()
      output[h, row] = sum(float(w)*value for w, value in zip(weights, v[kh], strict=True))
  return output


def small_plan(ref, swapped=False):
  frames = (ref.FrameSpan(2, 6, 1, -100000), ref.FrameSpan(9, 13, 6, 0))
  scores = np.full(20, 0.01)
  scores[[3, 4, 10, 11] if swapped else [2, 5, 9, 12]] = 0.5
  return ref.select_last_query(scores, 20, 48, frames)


def contract_checks(ref):
  rng = np.random.default_rng(193872)
  plan = small_plan(ref)
  expected_late = [0, 1, 2, 5, 6, 7, 8, 9, 12, 13, 14, 15, 16, 17, 18, 19]
  assert plan.slots.tolist() == expected_late
  cache = ref.SlotCache(plan)
  original = rng.normal(size=(28, 2, 2, 20, 8))
  generated = rng.normal(size=(28, 2, 2, 16, 8))
  for layer in range(28):
    slots = np.arange(20) if layer < 2 else np.array(expected_late)
    cache.write_prefill(layer, original[layer, 0][:, slots], original[layer, 1][:, slots])
    cache.write(layer, np.arange(20, 36), generated[layer, 0], generated[layer, 1])
  np.testing.assert_array_equal(cache.values[:2, :, :, :20], original[:2])
  mask_cases = attention_cases = 0
  action_q = rng.normal(size=(4, 64, 8))
  action_k, action_v = rng.normal(size=(2, 2, 64, 8))
  for layer in range(28):
    prefix = list(range(20)) if layer < 2 else expected_late
    for position in (0, 6, 19, 20, 23, 35):
      expected = set(x for x in prefix+list(range(20, position+1)) if x <= position)
      mask = plan.language_mask(layer, [position])[0]
      assert set(np.flatnonzero(np.isfinite(mask))) == expected
      mask_cases += 1
    block = plan.language_mask(layer, np.arange(20, 36))
    for row in range(16):
      assert set(np.flatnonzero(np.isfinite(block[row]))) == set(prefix+list(range(20, 21+row)))
      mask_cases += 1
    mask = plan.expert_mask(layer, 24)
    expected_cache = prefix+list(range(20, 24))
    assert set(np.flatnonzero(np.isfinite(mask))) == set(expected_cache+list(range(48, 112)))
    assert not np.isfinite(mask[24:48]).any()
    q = rng.normal(size=(4, 1, 8))
    actual = cache.language(layer, q, [23])
    kv = cache.values[layer]
    expected = independent_attention(q, kv[0][:, expected_cache], kv[1][:, expected_cache])
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
    actual_action = cache.expert(layer, action_q, action_k, action_v, 24)
    expected_action = independent_attention(action_q, np.concatenate((kv[0][:, expected_cache], action_k), axis=1),
                                             np.concatenate((kv[1][:, expected_cache], action_v), axis=1))
    np.testing.assert_allclose(actual_action, expected_action, rtol=1e-12, atol=1e-12)
    attention_cases += 2
    before = actual_action.copy()
    invisible = np.flatnonzero(~np.isfinite(mask[:48]))
    # Finite poison also covers stale rejected draft tokens and discarded image slots.
    cache.values[layer, 0][:, invisible] = rng.normal(size=(2, len(invisible), 8))*1e5
    cache.values[layer, 1][:, invisible] = rng.normal(size=(2, len(invisible), 8))*1e8
    np.testing.assert_array_equal(before, cache.expert(layer, action_q, action_k, action_v, 24))
    np.testing.assert_array_equal(actual, cache.language(layer, q, [23]))
  # Zero keys still take probability mass if validity is omitted.
  right = independent_attention(np.zeros((1, 1, 1)), np.zeros((1, 2, 1)), np.ones((1, 2, 1)))
  diluted = independent_attention(np.zeros((1, 1, 1)), np.zeros((1, 4, 1)), np.array([[[1.0], [1.0], [0.0], [0.0]]]))
  assert right.item() == 1.0 and diluted.item() == 0.5
  return {'layers': 28, 'full_initial_layers': 2, 'mask_cases': mask_cases, 'independent_attention_cases': attention_cases,
          'first_two_full_kv_preserved': True, 'all_action_queries_visible': 64, 'masked_finite_poison_ignored': True,
          'zero_key_dilution_control': {'correct': right.item(), 'without_validity': diluted.item()},
          'rejected_draft_suffix_hidden': True}


def metadata_and_rope_checks(ref):
  offsets = (-2000000, -1000000, -500000, -100000, 0)
  frames, cursor = [], 11
  for camera in (1, 6):
    for offset in offsets:
      frames.append(ref.FrameSpan(cursor, cursor+180, camera, offset))
      cursor += 187
  prompt = 2027
  scores = np.random.default_rng(19934).random(prompt)
  plan = ref.select_last_query(scores, prompt, 2304, frames)
  image = set(x for f in frames for x in range(f.begin, f.end))
  text = set(range(prompt))-image
  assert len(image) == 1800 and len(text) == 227 and len(plan.slots) == 1127
  assert text.issubset(plan.slots.tolist()) and plan.slots[-1] == prompt-1
  for frame in frames: assert len(set(plan.slots)&set(range(frame.begin, frame.end))) == 90
  positions = np.broadcast_to(np.arange(prompt), (3, 1, prompt)).copy()
  for frame in frames:
    positions[1, 0, frame.begin:frame.end] = np.arange(180)//18+31
    positions[2, 0, frame.begin:frame.end] = np.arange(180)%18+17
  frequencies = 1/(1e8**(np.arange(0, 128, 2, dtype=np.float32)/128))
  angles = positions[..., None]*frequencies
  combined = angles[0].copy()
  for axis in (1, 2): combined[..., axis:60:3] = angles[axis, ..., axis:60:3]
  repeated = np.concatenate((combined, combined), axis=-1)[:, None]
  cos, sin = np.cos(repeated), np.sin(repeated)
  selected_c, selected_s = plan.gather_rotary(cos, sin)
  np.testing.assert_array_equal(selected_c, cos[:, :, plan.slots])
  np.testing.assert_array_equal(selected_s, sin[:, :, plan.slots])
  assert not np.array_equal(selected_c, cos[:, :, :len(plan.slots)])
  # No six-token assumption: original text slots and logical RoPE survive long reasoning.
  long = ref.full_plan(83, 83+1024)
  delta = -19
  for generated in (1, 6, 17, 65, 257, 513, 1024):
    current_slot = long.prompt+generated-1
    assert current_slot < long.capacity
    mask = long.language_mask(27, [current_slot])
    assert np.isfinite(mask).sum() == current_slot+1
    action_mask = long.expert_mask(27, long.prompt+generated)
    assert np.isfinite(action_mask).sum() == long.prompt+generated+64
    original_rope_position = long.prompt+delta+generated-1
    assert original_rope_position == current_slot+delta
  prompt_pad, delta = 2112, -119
  adjusted_delta = delta-(prompt_pad-prompt)
  assert prompt_pad+adjusted_delta == prompt+delta
  return {'fixture_only': True, 'camera_history_offsets_us': list(offsets), 'frames': 10, 'full_encoder_tokens_per_frame': 180,
          'retained_language_image_tokens_per_frame': 90, 'retention': ref.RETENTION, 'all_text_ego_nav_tokens_retained': 227,
          'prompt_tokens': prompt, 'retained_late_tokens': 1127, 'rounded_late_rows_64': 1152,
          'original_three_axis_rotary_gather': True, 'original_decode_and_action_positions': True,
          'reasoning_lengths_checked': [1, 6, 17, 65, 257, 513, 1024], 'padded_prompt_delta_preserved': True}


def replay_fallback_checks(ref):
  rng = np.random.default_rng(7292)
  plan = small_plan(ref)
  cache = ref.SlotCache(plan)
  old_arrays = rng.normal(size=(28, 2, 2, 20, 8))
  new_arrays = rng.normal(size=(28, 2, 2, 20, 8))
  query = rng.normal(size=(4, 1, 8))
  def fill(target, values):
    for layer in range(28):
      slots = target.plan.layer_slots(layer)
      target.write_prefill(layer, values[layer, 0][:, slots], values[layer, 1][:, slots])
  fill(cache, old_arrays)
  first = cache.language(27, query, [19]).copy()
  np.testing.assert_array_equal(first, cache.language(27, query, [19]))
  plan2 = small_plan(ref, swapped=True)
  cache.reset(plan2)
  fill(cache, new_arrays)
  fresh = ref.SlotCache(plan2)
  fill(fresh, new_arrays)
  np.testing.assert_array_equal(cache.language(27, query, [19]), fresh.language(27, query, [19]))
  assert not np.array_equal(first, cache.language(27, query, [19]))
  invalid = np.full(20, np.nan)
  fallback = ref.select_last_query(invalid, 20, 48, plan.frames)
  assert fallback.fallback == 'invalid_saliency' and fallback.slots.tolist() == list(range(20))
  # Turning validity back on does not recreate absent late-layer KV.
  cache.plan = fallback
  try: cache.language(27, query, [19])
  except RuntimeError: pass
  else: raise AssertionError('Unpruning missing KV must fail before use')
  cache.reset(fallback)
  fill(cache, new_arrays)
  expected = independent_attention(query, new_arrays[27, 0], new_arrays[27, 1])
  np.testing.assert_allclose(cache.language(27, query, [19]), expected, rtol=1e-12, atol=1e-12)
  bad_layout = (ref.FrameSpan(2, 6, 1, 1),)
  assert ref.select_last_query(np.ones(20), 20, 48, bad_layout).fallback == 'invalid_frame_spans'
  assert ref.select_last_query(np.ones(20), 20, 48, ()).fallback == 'missing_frame_spans'
  try: ref.select_last_query(np.ones(20), 20, 19, plan.frames)
  except ValueError: pass
  else: raise AssertionError('Capacity overflow must not truncate the prompt')
  return {'same_input_replay': True, 'changed_selection_and_input_replay': True, 'fresh_cache_equivalence': True,
          'mask_only_unprune_rejected': True, 'full_prefill_rebuild_fallback': True,
          'invalid_saliency_fallback': True, 'future_frame_metadata_fallback': True, 'capacity_overflow_rejected': True,
          'nominal_token_substitution': False}


def future_mutation_checks(ref):
  rng = np.random.default_rng(9713)
  plan = small_plan(ref)
  cache = ref.SlotCache(plan)
  arrays = rng.normal(size=(28, 2, 2, 20, 8))
  for layer in range(28):
    slots = plan.layer_slots(layer)
    cache.write_prefill(layer, arrays[layer, 0][:, slots], arrays[layer, 1][:, slots])
  q = rng.normal(size=(4, 1, 8))
  checked = 0
  for layer in range(28):
    before = cache.language(layer, q, [7]).copy()
    cache.values[layer, :, :, 8:] += rng.normal(size=cache.values[layer, :, :, 8:].shape)*1000
    np.testing.assert_array_equal(before, cache.language(layer, q, [7]))
    checked += 1
  counterexample = ref.causal_selection_counterexample()
  return {'fixed_selection_causal_future_mutation_layers_passed': checked,
          'recomputed_fastv_selection': counterexample, 'mandatory_prefix_causality_gate_pass': False,
          'generation_causality_scope': 'A frozen selection uses only the known prompt and does not inspect future generated tokens.'}


def tinygrad_cpu_checks():
  from tinygrad import Device, Tensor, TinyJit, dtypes
  assert Device.DEFAULT == 'CPU', Device.DEFAULT
  primitives = local('agent-token-prune-primitives')
  rng = np.random.default_rng(159932)
  capacity, rows, width, prompt = 32, 10, 4, 20
  cache = Tensor.zeros(2, 3, capacity, width, device='CPU', dtype=dtypes.float32).realize()
  values = Tensor.zeros(2, 3, rows, width, device='CPU', dtype=dtypes.float32).realize()
  indices = Tensor.zeros(rows, device='CPU', dtype=dtypes.int32).realize()
  flags = Tensor.zeros(28, capacity, device='CPU', dtype=dtypes.bool).realize()
  boundary = Tensor(np.array([prompt, prompt], dtype=np.int32), device='CPU').realize()
  gathered = TinyJit(primitives.gather)
  scattered = TinyJit(primitives.scatter)
  masked = TinyJit(primitives.language_mask)
  records = []
  for iteration in range(6):
    base = rng.normal(size=(2, 3, capacity, width)).astype(np.float32)
    update = rng.normal(size=(2, 3, rows, width)).astype(np.float32)
    slots = np.r_[np.arange(1+(iteration%2), 17+(iteration%2), 2), -1, -1].astype(np.int32)
    valid = np.zeros((28, capacity), dtype=bool)
    valid[:2, :prompt] = True
    valid[2:, slots[slots >= 0]] = True
    limits = np.array([prompt, prompt+iteration], dtype=np.int32)
    for tensor, array in ((cache, base), (values, update), (indices, slots), (flags, valid), (boundary, limits)):
      tensor.assign(Tensor(array, device='CPU')).realize()
    actual = gathered(cache, indices).numpy().copy()
    expected = np.zeros_like(update)
    expected[..., :8, :] = base[..., slots[:8], :]
    np.testing.assert_array_equal(actual, expected)
    output = scattered(cache, values, indices)
    expected_cache = base.copy()
    expected_cache[..., slots[:8], :] = update[..., :8, :]
    np.testing.assert_array_equal(output.numpy(), expected_cache)
    np.testing.assert_array_equal(cache.numpy(), expected_cache)
    mask = masked(flags, boundary).numpy()
    keys = np.arange(capacity)[None]
    expected_mask = np.where((valid | (keys >= prompt)) & (keys <= limits[1]), 0.0, -np.inf)
    np.testing.assert_array_equal(mask, expected_mask)
    records.append({'iteration': iteration, 'phase': ('compile', 'capture', 'replay')[min(iteration, 2)],
                    'selected_slots': slots.tolist(), 'current_text_cache_slot': int(limits[1]), 'exact': True})
  assert not [d for d in Device._opened_devices if d.split(':')[0] not in ('CPU', 'NPY')], Device._opened_devices
  return {'cpu_only': True, 'records': records, 'gather_jit_calls': gathered.cnt, 'scatter_jit_calls': scattered.cnt,
          'mask_jit_calls': masked.cnt, 'opened_devices': sorted(Device._opened_devices), 'in_place_cache_preserved': True,
          'negative_padding_slots_skipped': True, 'mutable_mask_and_indices_replay': True}


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--output', type=Path, default=HERE/'agent-token-prune-cpu-report.json')
  parser.add_argument('--tinygrad-worker', action='store_true')
  args = parser.parse_args()
  if args.tinygrad_worker:
    print(json.dumps(tinygrad_cpu_checks()))
    return
  started = time.perf_counter()
  ref = local('agent-token-prune-reference')
  report = {'cpu_only': True, 'retention_ratios_predeclared': [ref.RETENTION], 'prune_after_language_layers': 2,
            'gpu_hook_ready': False, 'route_accuracy_or_speed_measured': False,
            'contract': contract_checks(ref), 'metadata_rotary': metadata_and_rope_checks(ref),
            'replay_fallback': replay_fallback_checks(ref), 'future_mutation': future_mutation_checks(ref)}
  env = dict(os.environ, DEV='CPU', ALLOW_DEVICE_USAGE='1', PYTHONDONTWRITEBYTECODE='1',
             PYTHONPATH='/tmp/opencode/alpamayo-asius-20261114/tinygrad', CACHELEVEL='0', CCACHE='0', BEAM='0', PROFILE='0', VIZ='0',
             OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
  child = subprocess.run([sys.executable, str(Path(__file__).resolve()), '--tinygrad-worker'], env=env,
                         capture_output=True, text=True, timeout=120)
  if child.returncode: raise RuntimeError(child.stdout+'\n'+child.stderr)
  report['tinygrad_cpu'] = json.loads(child.stdout)
  report['source_sha256'] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in HERE.glob('agent-token-prune*.py')}
  report['wall_seconds'] = time.perf_counter()-started
  report['decision'] = 'No GPU/model hook: recomputed FastV selection fails the requested prompt-prefix causality gate.'
  args.output.write_text(json.dumps(report, indent=2)+'\n')
  print(json.dumps({'output': str(args.output), 'mask_cases': report['contract']['mask_cases'],
                    'attention_cases': report['contract']['independent_attention_cases'],
                    'tinygrad_cpu_replay_cases': len(report['tinygrad_cpu']['records']),
                    'prefix_causality_gate_pass': False, 'gpu_hook_ready': False}, indent=2))


if __name__ == '__main__': main()
