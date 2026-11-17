"""CPU contract audit for post-layer-two visual-token pruning.

Numerical research only. No model hook, device access, or route qualification.
The FastV last-query selector fails prompt-prefix causal invariance.
"""
from dataclasses import dataclass
import math
import numpy as np

LAYERS = 28
FULL_LAYERS = 2
RETENTION = 0.5  # Only experimental retention ratio; all-token fallback is a control.


@dataclass(frozen=True)
class FrameSpan:
  begin: int
  end: int
  camera: int
  offset_us: int


@dataclass(frozen=True)
class Plan:
  prompt: int
  capacity: int
  slots: np.ndarray
  image_slots: np.ndarray
  frames: tuple[FrameSpan, ...]
  fallback: str | None = None

  def __post_init__(self):
    assert 0 < self.prompt <= self.capacity
    assert np.array_equal(self.slots, np.unique(self.slots))
    assert len(self.slots) and 0 <= self.slots[0] and self.slots[-1] < self.prompt
    assert self.prompt-1 in self.slots, 'Never prune the actual final prompt token'
    self.slots.flags.writeable = False
    self.image_slots.flags.writeable = False

  def prefix_valid(self):
    valid = np.zeros((LAYERS, self.capacity), dtype=bool)
    valid[:FULL_LAYERS, :self.prompt] = True
    valid[FULL_LAYERS:, self.slots] = True
    return valid

  def layer_slots(self, layer):
    assert 0 <= layer < LAYERS
    return np.arange(self.prompt) if layer < FULL_LAYERS else self.slots

  def language_mask(self, layer, query_slots):
    query_slots = np.asarray(query_slots, dtype=np.int64)
    assert query_slots.ndim == 1 and np.all((query_slots >= 0) & (query_slots < self.capacity))
    keys = np.arange(self.capacity)
    valid = self.prefix_valid()[layer] | (keys >= self.prompt)
    return np.where(valid[None] & (keys[None] <= query_slots[:, None]), 0.0, -np.inf)

  def expert_mask(self, layer, actual_length, action_queries=64):
    assert self.prompt <= actual_length <= self.capacity
    assert action_queries == 64
    keys = np.arange(self.capacity)
    valid = self.prefix_valid()[layer] | ((keys >= self.prompt) & (keys < actual_length))
    # Every action key is visible to every action query, as in EdgeModel._trajectory.
    return np.concatenate((np.where(valid, 0.0, -np.inf), np.zeros(action_queries)))

  def gather_rotary(self, cos, sin):
    assert cos.shape == sin.shape and cos.shape[-2] >= self.prompt
    return cos[..., self.slots, :].copy(), sin[..., self.slots, :].copy()


def full_plan(prompt, capacity, reason=None):
  return Plan(prompt, capacity, np.arange(prompt), np.empty(0, dtype=np.int64), (), reason)


def select_last_query(scores, prompt, capacity, frames):
  """FastV-style per-frame ranking, not a causally qualified inference policy.

  Scores must come from actual layer-one attention on the current input only.
  Per-frame quotas preserve every camera/history view; they do not prove quality.
  """
  if not 0 < prompt <= capacity: raise ValueError('Prompt exceeds cache capacity')
  scores = np.asarray(scores)
  frames = tuple(frames)
  if (scores.shape != (prompt,) or not np.isfinite(scores).all()
      or np.any(scores < 0) or not scores.sum() > 0): return full_plan(prompt, capacity, 'invalid_saliency')
  if not frames: return full_plan(prompt, capacity, 'missing_frame_spans')
  image = np.zeros(prompt, dtype=bool)
  selected = []
  seen_frames = set()
  for frame in frames:
    if (not 0 <= frame.begin < frame.end < prompt or frame.offset_us > 0
        or image[frame.begin:frame.end].any() or (frame.camera, frame.offset_us) in seen_frames):
      return full_plan(prompt, capacity, 'invalid_frame_spans')
    seen_frames.add((frame.camera, frame.offset_us))
    image[frame.begin:frame.end] = True
    indices = np.arange(frame.begin, frame.end)
    count = math.ceil(RETENTION * len(indices))
    # Stable ties prefer the original slot. No previous predictions or labels enter selection.
    selected.extend(indices[np.lexsort((indices, -scores[indices]))[:count]].tolist())
  text = np.flatnonzero(~image)
  keep = np.sort(np.concatenate((text, np.asarray(selected, dtype=np.int64))))
  return Plan(prompt, capacity, keep, np.flatnonzero(image), frames)


def probabilities(q, k, allowed):
  """Float64 GQA attention reference with explicit key validity."""
  q, k = np.asarray(q, dtype=np.float64), np.asarray(k, dtype=np.float64)
  assert q.ndim == k.ndim == 3 and q.shape[-1] == k.shape[-1] and q.shape[0] % k.shape[0] == 0
  keys = np.repeat(k, q.shape[0]//k.shape[0], axis=0)
  scores = q @ keys.transpose(0, 2, 1) / math.sqrt(q.shape[-1])
  scores = np.where(allowed, scores, -np.inf)
  if not np.isfinite(scores).any(axis=-1).all(): raise ValueError('All-masked query is invalid')
  exp = np.exp(scores-scores.max(axis=-1, keepdims=True))
  return exp/exp.sum(axis=-1, keepdims=True)


def attend(q, k, v, mask):
  assert np.isfinite(v).all(), 'Masked NaN values can still contaminate a matrix product'
  probs = probabilities(q, k, np.isfinite(mask))
  return probs @ np.repeat(v, q.shape[0]//v.shape[0], axis=0)


def last_prompt_saliency(q, k, prompt):
  assert q.shape[1] == k.shape[1] == prompt
  # The final real prompt query can see all real prompt keys, and nothing generated.
  return probabilities(q[:, -1:], k, np.ones((1, prompt), dtype=bool)).mean(axis=0)[0]


class SlotCache:
  """Original-slot cache reference. Values can remain stale only behind invalid masks."""
  def __init__(self, plan, heads=2, width=8):
    self.values = np.zeros((LAYERS, 2, heads, plan.capacity, width), dtype=np.float64)
    self.reset(plan)

  def reset(self, plan):
    assert plan.capacity == self.values.shape[-2]
    self.plan = plan
    self.written = np.zeros((LAYERS, plan.capacity), dtype=bool)
    self.epoch = getattr(self, 'epoch', -1)+1

  def write(self, layer, slots, k, v):
    slots = np.asarray(slots, dtype=np.int64)
    assert np.array_equal(slots, np.unique(slots)) and np.all((slots >= 0) & (slots < self.plan.capacity))
    assert k.shape == v.shape == (self.values.shape[2], len(slots), self.values.shape[-1])
    assert np.isfinite(k).all() and np.isfinite(v).all()
    self.values[layer, 0][:, slots, :] = k
    self.values[layer, 1][:, slots, :] = v
    self.written[layer, slots] = True

  def write_prefill(self, layer, k, v):
    self.write(layer, self.plan.layer_slots(layer), k, v)

  def check_visible(self, layer, mask):
    visible = np.isfinite(mask[..., :self.plan.capacity]).reshape(-1, self.plan.capacity).any(axis=0)
    if np.any(visible & ~self.written[layer]): raise RuntimeError('Fallback requires rebuilding missing layer KV entries')

  def language(self, layer, q, query_slots):
    mask = self.plan.language_mask(layer, query_slots)
    self.check_visible(layer, mask)
    return attend(q, self.values[layer, 0], self.values[layer, 1], mask)

  def expert(self, layer, q, action_k, action_v, actual_length):
    mask = self.plan.expert_mask(layer, actual_length)
    self.check_visible(layer, mask)
    k = np.concatenate((self.values[layer, 0], action_k), axis=1)
    v = np.concatenate((self.values[layer, 1], action_v), axis=1)
    return attend(q, k, v, mask)


def causal_selection_counterexample():
  """A later prompt query changes an earlier retained text row via token membership."""
  prompt, earlier_query = 7, 3
  frames = (FrameSpan(1, 3, 1, 0),)
  q = np.zeros((1, prompt, 2))
  k = np.zeros((1, prompt, 2))
  v = np.zeros((1, prompt, 1))
  k[0, 1, 0], k[0, 2, 0] = 1.0, -1.0
  v[0, 1, 0], v[0, 2, 0] = 10.0, -10.0
  rows = []
  plans = []
  full_mask = np.arange(prompt)[None] <= earlier_query
  full_before = probabilities(q[:, earlier_query:earlier_query+1], k, full_mask) @ v
  for future_value in (6.0, -6.0):
    changed = q.copy()
    changed[0, -1, 0] = future_value
    scores = last_prompt_saliency(changed, k, prompt)
    plan = select_last_query(scores, prompt, prompt+4, frames)
    selected = plan.slots
    mask = selected[None] <= earlier_query
    result = probabilities(changed[:, earlier_query:earlier_query+1], k[:, selected], mask) @ v[:, selected]
    rows.append(float(result.item()))
    plans.append(plan.slots.tolist())
  assert rows[0] != rows[1] and plans[0] != plans[1]
  return {'changed_input_slot': prompt-1, 'retained_earlier_query_slot': earlier_query,
          'full_causal_output_both': float(full_before.item()), 'pruned_outputs': rows, 'selected_slots': plans,
          'prefix_causality_pass': False, 'unavailable_future_footage_used': False,
          'explanation': 'Known later prompt content changes earlier image membership; triangular masks alone do not prevent this dependency.'}
