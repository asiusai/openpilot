"""Independent CPU qualification of the two current-full/history policies.

All64 original quality inputs use their actual CPU preprocessing/tokenization.
Injected saliency checks quotas and masks; it is not checkpoint inference.
Separate small CPU projections exercise actual frozen controller/model methods.
"""
import argparse
from contextlib import ExitStack
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

HERE = Path(__file__).resolve().parent
sys.dont_write_bytecode = True
sys.path.insert(0, str(HERE.parent))


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def local(name, suffix=''):
  spec = importlib.util.spec_from_file_location('agent_token_current_check_'+name.replace('-', '_')+suffix, HERE/(name+'.py'))
  value = importlib.util.module_from_spec(spec)
  sys.modules[spec.name] = value
  spec.loader.exec_module(value)
  return value


def rejects(operation):
  try: operation()
  except (ValueError, RuntimeError): return
  raise AssertionError('Invalid input was accepted')


def addressed(policy, data, sample):
  query = int(Path(sample).name.removeprefix('sample-'))*1000
  row = dict(query_ns=query, frames=[dict(view=int(camera), capture_ns=int(stamp)*1000)
          for camera, stamp in zip(data['camera_indices'].reshape(-1).tolist(), data['absolute_timestamps'].reshape(-1).tolist(), strict=True)])
  class CurrentRow(dict):
    def __getitem__(self, key):
      if key not in ('query_ns', 'frames'): raise AssertionError('GT/output field accessed')
      return super().__getitem__(key)
  class CurrentOnly(dict):
    def __getitem__(self, key):
      if key != 'current': raise AssertionError('Unavailable future record accessed')
      return super().__getitem__(key)
  rows = CurrentOnly(current=CurrentRow(row), future=object())
  observations = policy.current_observations(rows, 'current')
  rows['future'] = {'frames': object(), 'reference_trajectory': object(), 'generated_candidate_tokens': object()}
  rows['current']['recorded_motion_future_xyz'] = object()
  assert policy.current_observations(rows, 'current') == observations
  return policy.prepared_observations(data, observations)


def independent_selection(scores, prompt, frames, ratio):
  latest = {camera:max(frame.capture_ns for frame in frames if frame.camera == camera) for camera in (1, 6)}
  visual = {slot for frame in frames for slot in range(frame.begin, frame.end)}
  chosen = [slot for slot in range(prompt) if slot not in visual]
  for frame in frames:
    indices = list(range(frame.begin, frame.end))
    if frame.capture_ns != latest[frame.camera]:
      indices = sorted(indices, key=lambda slot:(-float(scores[slot]), slot))[:math.ceil(len(indices)*ratio)]
    chosen.extend(indices)
  return np.array(sorted(chosen), dtype=np.int64), latest, visual


def independent_masks(plan, queries):
  capacity, prompt = plan.capacity, plan.prompt
  for layer in range(28):
    prefix = set(range(prompt)) if layer < 2 else set(map(int, plan.slots))
    expected = np.zeros((len(queries), capacity), dtype=bool)
    for row, position in enumerate(queries):
      keys = prefix | set(range(prompt, int(position)+1))
      expected[row, list(keys)] = True
    np.testing.assert_array_equal(np.isfinite(plan.language_mask(layer, queries)), expected)
    actual_length = prompt+6
    expert = np.zeros(capacity+64, dtype=bool)
    expert[list(prefix | set(range(prompt, actual_length)))] = True
    expert[capacity:] = True
    np.testing.assert_array_equal(np.isfinite(plan.expert_mask(layer, actual_length)), expert)
  return 56


def actual_inputs(policy, route, invocation):
  import torch
  torch.set_num_threads(2)
  saved = json.loads(invocation.read_text())
  command = route.qualify(saved, HERE/'agent-token-current-unused.json')
  start, end = command.index('--samples')+1, command.index('--initial-noise')
  paths = [Path(path) for path in command[start:end]]
  assert len(paths) == 64 and len(set(paths)) == 64
  fast = local('fast-inputs')
  frozen_route = local('agent-token-prune-route')
  adapter = fast.FastInputs(frozen_route.value(command, '--checkpoint'), camera_profile='nvidia',
                           allow_missing_cameras=True, cache_resize=True, direct_byte_inputs=True)
  padding = local('padded-prefill')
  rng = np.random.default_rng(171126)
  records, mask_cases, timestamp_cases, shuffle_cases = [], 0, 0, 0
  for index, path in enumerate(paths):
    data = {p.stem:[str(np.load(p, allow_pickle=False).item())] if p.stem == 'nav_text'
            else torch.from_numpy(np.load(p, allow_pickle=False, mmap_mode='c')) for p in path.glob('*.npy')}
    observations = addressed(policy, data, path)
    config, _, prepared = adapter.prepare(data)
    packed = adapter.native_inputs(prepared)
    prompt = packed[0].shape[1]
    before_ids, before_positions = packed[0].copy(), packed[1].copy()
    original_delta = packed[2]
    prepared, packed = padding.pad_inputs(prepared, packed, 2112)
    image_indices = np.flatnonzero(packed[0][0, :prompt] == config.vlm_config.image_token_id)
    splits = np.split(image_indices, np.flatnonzero(np.diff(image_indices) != 1)+1)
    spans = tuple((int(span[0]), int(span[-1])+1) for span in splits)
    grids = prepared['tokenized_data']['image_grid_thw'].tolist()
    frames = policy.frame_spans(packed[0], config.vlm_config.image_token_id, prompt, observations, spans, grids)
    assert len(frames) == 10 and all(frame.end-frame.begin == 180 for frame in frames)
    assert data['image_frames'].shape[0] == 10 and prepared['tokenized_data']['pixel_values'].shape[0] == 7200
    assert 2112+packed[2] == prompt+original_delta
    scores = rng.random(prompt)+.01
    # The latest views must survive even when their saliency is deliberately worst.
    latest = {camera:max(stamp for view, stamp in observations.frames if view == camera) for camera in (1, 6)}
    for frame in frames:
      if frame.capture_ns == latest[frame.camera]: scores[frame.begin:frame.end] = 0.0
    scores /= scores.sum()
    selected_records = {}
    for name, ratio in policy.POLICIES.items():
      plan = policy.select(scores, prompt, 2304, frames, name)
      expected, independent_latest, visual = independent_selection(scores, prompt, frames, ratio)
      np.testing.assert_array_equal(plan.slots, expected)
      assert plan.fallback is None and plan.slots[-1] == prompt-1
      assert len(plan.slots) == prompt-1800+360+8*math.ceil(180*ratio)
      assert set(range(prompt))-visual <= set(plan.slots)
      counts = policy.frame_counts(plan)
      assert sum(frame['latest'] for frame in counts) == 2
      for frame in counts:
        assert frame['latest'] == (frame['capture_ns'] == independent_latest[frame['camera']])
        assert frame['retained'] == (180 if frame['latest'] else math.ceil(180*ratio))
      np.testing.assert_array_equal(packed[1][..., plan.slots], before_positions[..., plan.slots])
      np.testing.assert_array_equal(packed[0][:, plan.slots], before_ids[:, plan.slots])
      queries = prompt+np.array([0, 1, 6, 17, 65])
      assert queries[-1] < plan.capacity
      mask_cases += independent_masks(plan, queries)
      # Stable ties are an independent control, not another policy or ratio.
      tied = policy.select(np.ones(prompt), prompt, 2304, frames, name)
      tie_expected = independent_selection(np.ones(prompt), prompt, frames, ratio)[0]
      np.testing.assert_array_equal(tied.slots, tie_expected)
      invalid = scores.copy(); invalid[0] = np.nan
      fallback = policy.select(invalid, prompt, 2304, frames, name)
      assert fallback.fallback == 'invalid_saliency'
      np.testing.assert_array_equal(fallback.slots, np.arange(prompt))
      selected_records[name] = dict(retained_prompt_tokens=len(plan.slots), late_rows=math.ceil(len(plan.slots)/64)*64,
                                   selected_slots_sha256=hashlib.sha256(plan.slots.astype('<i4').tobytes()).hexdigest(), frames=counts)
    # Joint camera/timestamp/pixel shuffle must canonicalize to identical real CPU inputs.
    permutation = torch.tensor(rng.permutation(10), dtype=torch.int64)
    shuffled = dict(data, **{key:data[key][permutation].contiguous() for key in ('image_frames', 'camera_indices', 'absolute_timestamps')})
    shuffled_observations = addressed(policy, shuffled, path)
    c2, _, p2 = adapter.prepare(shuffled)
    packed2 = adapter.native_inputs(p2)
    for a, b in zip((before_ids, before_positions, original_delta), packed2[:3], strict=True): np.testing.assert_array_equal(a, b)
    for key in ('input_ids', 'image_grid_thw', 'pixel_values', 'mm_token_type_ids'):
      assert torch.equal(prepared['tokenized_data'][key], p2['tokenized_data'][key]), key
    frames2 = policy.frame_spans(packed2[0], c2.vlm_config.image_token_id, prompt, shuffled_observations, spans, grids)
    assert frames2 == frames
    for name in policy.POLICIES:
      np.testing.assert_array_equal(policy.select(scores, prompt, 2304, frames2, name).slots,
                                    policy.select(scores, prompt, 2304, frames, name).slots)
    shuffle_cases += 1
    # Padded/proposed suffix IDs cannot create visual spans or change selection.
    poisoned_ids = packed[0].copy(); poisoned_ids[:, prompt:] = config.vlm_config.image_token_id
    assert policy.frame_spans(poisoned_ids, config.vlm_config.image_token_id, prompt, observations, spans, grids) == frames
    # Reject timestamp future/duplication, wrong camera IDs, pixel-order mismatches, and guessed role changes.
    raw = list(observations.source_frames)
    for kind in ('future', 'duplicate', 'wrong_camera'):
      changed = raw.copy()
      if kind == 'future': changed[-1] = (changed[-1][0], observations.query_ns+1)
      elif kind == 'duplicate': changed[1] = changed[0]
      else: changed[0] = (0, changed[0][1])
      rejects(lambda:policy.canonical_records(observations.query_ns, changed))
      timestamp_cases += 1
    mismatched = dict(data, camera_indices=data['camera_indices'].flip(0))
    rejects(lambda:policy.prepared_observations(mismatched, observations))
    rejects(lambda:policy.frame_spans(packed[0], config.vlm_config.image_token_id, prompt, observations, spans[::-1], grids))
    # Actual timestamps, not offset0, identify both latest views. Wide often lags the query.
    records.append(dict(sample=str(path), query_ns=observations.query_ns, actual_prompt_tokens=prompt,
                        metadata_sha256={name:sha(path/(name+'.npy')) for name in ('camera_indices', 'absolute_timestamps', 'ego_history_xyz', 'ego_history_rot', 'nav_text')},
                        original_frames=10, original_image_tokens=1800, policy_results=selected_records,
                        shuffled_full_cpu_preprocessing_identical=True, poisoned_suffix_ids_ignored=True))
    print(json.dumps(dict(cpu_input=index+1, total=64, prompt=prompt,
                          late_rows={name:value['late_rows'] for name, value in selected_records.items()})), flush=True)
  return dict(records=records, actual_cpu_inputs_checked=64, full_cpu_camera_shuffle_cases=shuffle_cases,
              independent_layer_mask_cases=mask_cases, invalid_timestamp_camera_cases=timestamp_cases,
              saliency='Deterministic injected scores for independent structural tests; no checkpoint language inference',
              used_fields='Current images, camera IDs/timestamps, ego history, navigation, and real tokenized prompt only',
              invocation_sha256=sha(invocation), sample_paths_sha256=hashlib.sha256('\n'.join(map(str, paths)).encode()).hexdigest())


def small_layout(policy):
  query = 10_000_000_000
  stamps = [query-offset for camera in (1, 6) for offset in (2_000_000_000, 1_000_000_000, 500_000_000, 100_000_000, 2_000_000 if camera == 1 else 0)]
  frames = tuple(policy.ObservedFrame(5+i*5, 9+i*5, camera, (stamp-query)//1000, stamp, query)
                 for i, (camera, stamp) in enumerate(zip((1,)*5+(6,)*5, stamps, strict=True)))
  return 56, frames


def actual_contract(factory, policy):
  from tinygrad import Tensor, TinyJit, dtypes
  from tinygrad.uop.ops import UOp
  frozen = local('agent-token-prune-policy-check')
  packed_contract = frozen.packed_producer_check()
  results = {}
  prompt, frames = small_layout(policy)
  rng = np.random.default_rng(171117)
  source = rng.normal(size=(1, 64, 2048)).astype(np.float32)*.1
  changed = source.copy(); changed[:, :prompt] += rng.normal(size=(1, prompt, 2048)).astype(np.float32)*.25
  angle = np.arange(64, dtype=np.float32)[:, None]*np.linspace(.001, .08, 64)[None]
  co = np.cos(np.concatenate((angle, angle), axis=1))[None, None].astype(np.float32)
  si = np.sin(np.concatenate((angle, angle), axis=1))[None, None].astype(np.float32)
  for name, ratio in policy.POLICIES.items():
    runtime = factory.make_runtime(name)
    # Runtime and tests must share their exact ObservedFrame class.
    runtime_frames = tuple(factory.policy_module.ObservedFrame(**frame.__dict__) for frame in frames)
    model = frozen.fake_model()  # Uses __new__; never invokes or replaces the protected initializer.
    owner = runtime.NumericalPruner(model, require_packed=False)
    cos, sin = (Tensor(values.copy(), device='CPU').realize() for values in (co, si))
    first, first_slots = None, None
    events = []
    for iteration, data in enumerate((source, changed, source, source)):
      owner.prepare(prompt, runtime_frames)
      if iteration == 3:
        for cache in model.cache:
          array = cache.numpy().copy(); array[..., prompt:, :] = rng.normal(size=array[..., prompt:, :].shape).astype(np.float32)*1e3
          cache.assign(Tensor(array, device='CPU')).realize()
        data = data.copy(); data[:, prompt:] = rng.normal(size=data[:, prompt:].shape).astype(np.float32)*1e3
      values = owner.padded_prefill(Tensor(data.copy(), device='CPU').realize(), cos, sin, 32, prompt).numpy().copy()
      event = owner.events[-1]
      assert event['fallback'] is None and event['policy'] == name
      assert event['latest_image_retention'] == 1.0 and event['older_image_retention'] == ratio
      assert event['retained_prompt_tokens'] == prompt-40+8+8*math.ceil(4*ratio)
      assert event['prompt_prefix_invariance_pass'] is False and 'retention' not in event
      expected = policy.select(np.ones(prompt), prompt, owner.last_plan.capacity, frames, name)
      assert len(expected.slots) == len(owner.last_plan.slots)
      for row in event['frame_retention']: assert row['retained'] == (4 if row['latest'] else math.ceil(4*ratio))
      if first is None: first, first_slots = values, owner.last_plan.slots.copy()
      elif iteration >= 2:
        np.testing.assert_array_equal(values, first)
        np.testing.assert_array_equal(owner.last_plan.slots, first_slots)
      events.append(dict(event))
    assert events[0]['selected_slots_sha256'] != events[1]['selected_slots_sha256']
    assert owner.early_jit.cnt == 4 and next(iter(owner.late_jits.values())).cnt == 4
    # Move the actual prompt within the same padded graph, then compare fresh caches.
    owner.prepare(prompt+1, runtime_frames)
    moved = owner.padded_prefill(Tensor(source.copy(), device='CPU').realize(), cos, sin, 32, prompt+1).numpy().copy()
    moved_slots = owner.last_plan.slots.copy()
    fresh_model = frozen.fake_model()
    fresh_owner = runtime.NumericalPruner(fresh_model, require_packed=False)
    fresh_owner.prepare(prompt+1, runtime_frames)
    fresh = fresh_owner.padded_prefill(Tensor(source.copy(), device='CPU').realize(), cos, sin, 32, prompt+1).numpy()
    np.testing.assert_array_equal(moved, fresh)
    np.testing.assert_array_equal(moved_slots, fresh_owner.last_plan.slots)
    assert moved_slots[-1] == prompt and len(moved_slots) == len(first_slots)+1
    fresh_owner.restore()
    owner.prepare(prompt, runtime_frames)
    restored_prompt = owner.padded_prefill(Tensor(source.copy(), device='CPU').realize(), cos, sin, 32, prompt).numpy()
    np.testing.assert_array_equal(restored_prompt, first)
    # Compare full initial-layer cache values against original frozen method calls.
    baseline = frozen.fake_model()
    capacity = model.cache[0].shape[3]
    baseline.cache = [Tensor.zeros(2, 1, 8, capacity, 128, device='CPU', dtype=dtypes.float32).realize() for _ in range(28)]
    hidden = Tensor(source.copy(), device='CPU').realize()
    for layer in range(2):
      hidden = baseline.text_layer(hidden, cos, sin, f'vlm.model.language_model.layers.{layer}', baseline.cache[layer], None, cache_pos=0)
      np.testing.assert_array_equal(model.cache[layer].numpy()[..., :prompt, :], baseline.cache[layer].numpy()[..., :prompt, :])
    allowed = owner.allowed.numpy().copy()
    expected_allowed = np.zeros(capacity, bool); expected_allowed[first_slots] = True; expected_allowed[prompt:] = True
    np.testing.assert_array_equal(allowed, expected_allowed)
    # Generation and target verification use real original text/cache positions.
    c, s = (value[:, :, :1].clone().realize() for value in (cos, sin))
    embedding = model.weights['vlm.model.language_model.embed_tokens.weight'][7].reshape(1, 1, 2048).clone().realize()
    serial = model.decode(embedding, c, s, prompt).numpy().copy()
    draft = local('draft-decode', name)
    decoder = draft.DraftDecoder.__new__(draft.DraftDecoder)
    decoder.model, decoder.fast_attention, decoder.tile = model, False, (4, 4)
    decoder.gemm = SimpleNamespace(gemm=None)
    owner.register_decoder(decoder)
    verifier = TinyJit(decoder.verify)
    proposals = np.arange(16, dtype=np.int32)%90; proposals[0] = 7
    dc, ds = (Tensor(np.tile(array[:, :, :16], (1, 1, 2, 1)), device='CPU').realize() for array in (co, si))
    position = UOp.variable('draft_position', 0, capacity-16).bind(prompt)
    step = UOp.variable('draft_step', 0, 16).bind(0)
    verified = None
    for iteration in range(4):
      ids = proposals.copy()
      if iteration % 2: ids[3:] = 80-ids[3:]
      values, predictions = verifier(Tensor(ids, device='CPU').realize(), dc, ds, position, step)
      values = values.numpy().copy()
      if verified is None:
        verified = values[:, :3].copy()
        np.testing.assert_allclose(values[:, :1], serial, rtol=1e-5, atol=1e-6)
      else: np.testing.assert_array_equal(values[:, :3], verified)
      np.testing.assert_array_equal(predictions.numpy(), values.argmax(-1))
      np.testing.assert_array_equal(owner.last_plan.slots, first_slots)
    # Expert ignores dropped prefix slots and every rejected future-generated slot.
    action = rng.normal(size=(1, 64, 2)).astype(np.float32)*.1
    ac, ass = (Tensor(np.tile(array, (1, 1, 2, 1)), device='CPU').realize() for array in (co, si))
    trajectory = None
    for iteration in range(3):
      if iteration:
        for layer, cache in enumerate(model.cache):
          array = cache.numpy().copy()
          invalid = (np.zeros(capacity, bool) if layer < 2 else ~allowed) | (np.arange(capacity) >= prompt+3)
          array[..., invalid, :] = rng.normal(size=array[..., invalid, :].shape).astype(np.float32)*1e3
          cache.assign(Tensor(array, device='CPU')).realize()
      value = model.trajectory(Tensor(action.copy(), device='CPU').realize(), ac, ass, prompt+3, 1, 3).numpy().copy()
      if trajectory is None: trajectory = value
      else: np.testing.assert_array_equal(value, trajectory)
    # Genuine full fallback must rebuild full caches and original target logits.
    owner.metadata = None
    fallback = owner.padded_prefill(Tensor(source.copy(), device='CPU').realize(), cos, sin, 32, prompt).numpy()
    assert owner.events[-1]['full_prefill_recomputed'] and owner.events[-1]['retained_prompt_tokens'] == prompt
    hidden = Tensor(source.copy(), device='CPU').realize()
    for layer in range(28): hidden = baseline.text_layer(hidden, cos, sin, f'vlm.model.language_model.layers.{layer}', baseline.cache[layer], None, cache_pos=0)
    full = baseline.linear(baseline.norm(hidden[:, prompt-1:prompt], 'vlm.model.language_model.norm', rms=True), 'vlm.lm_head').numpy()
    np.testing.assert_array_equal(fallback, full)
    # Caller budgets remain arbitrary; capacity growth must preserve original positions.
    owner.prepare(prompt, runtime_frames)
    grown = owner.padded_prefill(Tensor(source.copy(), device='CPU').realize(), cos, sin, 513, prompt).numpy()
    np.testing.assert_array_equal(grown, first)
    assert model.cache[0].shape[3] == 768 and model.decode_jit.cnt == 0 and not model.trajectory_jits
    # A future observation fails before fallback/inference and clears stale metadata.
    bad = list(runtime_frames)
    bad[-1] = factory.policy_module.ObservedFrame(**dict(bad[-1].__dict__, capture_ns=bad[-1].query_ns+1, offset_us=0))
    rejects(lambda:owner.prepare(prompt, bad))
    assert owner.metadata is None
    owner.restore()
    assert not hasattr(model, '_agent_token_pruner')
    results[name] = dict(events=events, initial_two_full_cache_parity=True,
      compile_capture_and_mutated_replay=True, original_text_position_serial_decode=True,
      changed_actual_prompt_same_padded_shape_matches_fresh_cache=True,
      target_verification_suffix_poison_passes=4, target='changed numerical model', expert_poison_passes=3,
      padded_and_generated_suffix_poison_ignored=True, genuine_full_fallback=True,
      arbitrary_reasoning_budget_513=True, future_camera_rejected_before_inference=True)
    print(json.dumps(dict(cpu_contract=name, passed=True)), flush=True)
  return dict(packed_producer=packed_contract, policies=results)


def prefix_counterexamples(policy):
  prompt, frames = small_layout(policy)
  q = np.zeros((1, prompt, 2)); k = np.zeros_like(q); v = np.zeros((1, prompt, 1))
  k[0, 5, 0], k[0, 6, 0] = 1, -1
  v[0, 5, 0], v[0, 6, 0] = 10, -10
  result = {}
  for name in policy.POLICIES:
    values = []
    for direction in (6., -6.):
      current = q.copy(); current[0, -1, 0] = direction
      scores = policy.reference.last_prompt_saliency(current, k, prompt)
      plan = policy.select(scores, prompt, 256, frames, name)
      selected = plan.slots
      mask = selected[None] <= 9
      value = policy.reference.probabilities(current[:, 9:10], k[:, selected], mask) @ v[:, selected]
      values.append(float(value.item()))
    assert values[0] != values[1]
    result[name] = dict(prompt_prefix_invariance_pass=False, changed_available_prompt_query=prompt-1,
                       earlier_query=9, earlier_outputs=values, unavailable_future_camera_used=False)
  return dict(original_counterexample=policy.reference.causal_selection_counterexample(), current_policies=result)


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--invocation', type=Path, default=Path('/mnt/nvme-500/tmp/alpamayo-speed-20261116/full-padded-quality/invocation.json'))
  parser.add_argument('--output', type=Path, default=HERE/'agent-token-current-cpu-report.json')
  args = parser.parse_args()
  if os.environ.get('DEV') != 'CPU': parser.error('Set DEV=CPU explicitly')
  os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
  from tinygrad import Device
  assert Device.DEFAULT == 'CPU'
  started = time.perf_counter()
  protected = json.loads((HERE/'agent-token-current-protected-before.json').read_text())
  for path, digest in protected.items(): assert sha(path) == digest, path
  factory, route = local('agent-token-current-runtime'), local('agent-token-current-route')
  policy = factory.policy_module
  counters = prefix_counterexamples(policy)
  actual = actual_inputs(policy, route, args.invocation)
  contract = actual_contract(factory, policy)
  for path, digest in protected.items(): assert sha(path) == digest, path
  assert all(name.split(':')[0] in ('CPU', 'NPY', 'PYTHON') for name in Device._opened_devices), Device._opened_devices
  contracts = ('model.py', 'experiments/benchmark-pipeline.py', 'experiments/benchmark-fast.py', 'experiments/fast-inputs.py',
               'experiments/padded-prefill.py', 'experiments/quantize-packed.py', 'experiments/block-decode.py',
               'experiments/draft-decode.py', 'experiments/lookup-decode.py', 'experiments/stream-inputs.py')
  report = dict(classification='NUMERICAL NEW MODEL', policies=list(policy.POLICIES), cpu_qualified=True,
    prompt_prefix_invariance_pass=False, prefix_counterexamples=counters,
    actual_cpu_inputs_checked=actual['actual_cpu_inputs_checked'], actual_inputs=actual,
    actual_frozen_method_contract=contract, gpu_quality_or_speed_validated=False,
    source_sha256={p.name:sha(p) for p in HERE.glob('agent-token-current*.py')}, protected_source_sha256=protected,
    contract_source_sha256={str(HERE.parent/name):sha(HERE.parent/name) for name in contracts},
    cpu_devices=sorted(Device._opened_devices), wall_seconds=time.perf_counter()-started,
    prior_uniform50_rejection=dict(ADE=[2.71346,2.84233], FDE=[6.81482,7.17402], right_turn_ADE=[4.086,4.420],
                                   failed_windows=3, unchanged_per_window_gate_percent=5, adopted=False),
    measurement='CPU preprocessing and structural/contract qualification only; injected scores and small fake projections do not establish model accuracy')
  args.output.write_text(json.dumps(report, indent=2)+'\n')
  print(json.dumps(dict(output=str(args.output), cpu_qualified=True, gpu_validated=False,
                       wall_seconds=report['wall_seconds'], devices=report['cpu_devices']), indent=2))


if __name__ == '__main__': main()
