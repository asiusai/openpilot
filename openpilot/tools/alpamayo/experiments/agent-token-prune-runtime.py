"""Numerical whole-prompt FastV policy for the frozen EdgeModel contract.

50% per-frame image retention after two complete language layers.
Prompt-prefix invariance is false; generation causality is qualified separately.
No installation occurs on import. No GPU work is performed by this file alone.
"""
from collections import OrderedDict
from contextlib import contextmanager
from dataclasses import replace
import hashlib
import importlib.util
import math
from pathlib import Path
import sys
from types import MethodType

import numpy as np
from tinygrad import Tensor, TinyJit, dtypes
from tinygrad.helpers import round_up
from tinygrad.uop.ops import UOp

HERE = Path(__file__).resolve().parent


def local(name):
  spec = importlib.util.spec_from_file_location(name.replace('-', '_'), HERE/(name+'.py'))
  module = importlib.util.module_from_spec(spec)
  sys.modules[spec.name] = module
  spec.loader.exec_module(module)
  return module


reference = local('agent-token-prune-reference')
primitives = local('agent-token-prune-primitives')


def gather_ast(*args):
  ast = primitives._gather(*args)
  return ast.replace(arg=replace(ast.arg, opts_to_apply=None))


def scatter_ast(*args):
  ast = primitives._scatter(*args)
  return ast.replace(arg=replace(ast.arg, opts_to_apply=None))


def indexed_gather(source, slots):
  output = Tensor.empty(*source.shape[:-2], slots.shape[0], source.shape[-1], device=source.device, dtype=source.dtype)
  return Tensor.custom_kernel(output, source, slots, fxn=gather_ast)[0]


def indexed_scatter(cache, values, slots):
  return Tensor.custom_kernel(cache, values, slots, fxn=scatter_ast)[0]


def current_observations(rows, sample_key):
  """Read only the addressed row and permitted frame fields, never future rows or GT."""
  row = rows[sample_key]
  query_ns = int(row['query_ns'])
  frames = tuple((int(frame['view']), int(frame['capture_ns'])) for frame in row['frames'])
  if any(capture > query_ns for _, capture in frames):
    raise ValueError('Unavailable future camera frame in the current input')
  if tuple(camera for camera, _ in frames) != (1,)*5+(6,)*5:
    raise ValueError('Numerical policy requires both complete front-camera histories')
  if any(a[1] >= b[1] for group in (frames[:5], frames[5:]) for a, b in zip(group, group[1:])):
    raise ValueError('Camera history must be strictly chronological')
  return query_ns, frames


def frame_spans(ids, image_id, prompt, observations, spans, grids, *, strict=True):
  query_ns, frames = observations
  if any(stamp > query_ns for _, stamp in frames): raise ValueError('Unavailable camera input')
  if len(frames) != len(spans) or len(grids) != len(spans): raise ValueError('Frame/span/grid mismatch')
  ids = np.asarray(ids)[0, :prompt]
  image_mask = ids == int(image_id)
  covered = np.zeros(prompt, dtype=bool)
  result = []
  for (camera, stamp), (begin, end), grid in zip(frames, spans, grids, strict=True):
    begin, end = int(begin), int(end)
    if not 0 <= begin < end < prompt or covered[begin:end].any(): raise ValueError('Invalid image span')
    count = int(np.prod(grid))//4
    if count != end-begin or (strict and count != 180): raise ValueError('Visual feature count changed')
    covered[begin:end] = True
    result.append(reference.FrameSpan(begin, end, camera, (stamp-query_ns)//1000))
  if not np.array_equal(covered, image_mask): raise ValueError('Image IDs do not match frame spans')
  return tuple(result)


class NumericalPruner:
  """Install below the existing per-layer packed-quantizer wrapper.

  Construct immediately after model loading, before pipeline kernel wrappers.
  Prepare immutable metadata on the prediction thread, after CPU preprocessing.
  """
  def __init__(self, model, *, allow_full_fallback=True, require_packed=True, late_graph_budget=8):
    assert model.config['vlm_config']['text_config']['num_hidden_layers'] == 28
    assert model.config['expert_config']['llm_config']['num_hidden_layers'] == 28
    self.model = model
    self.allow_full_fallback, self.require_packed = allow_full_fallback, require_packed
    self.late_graph_budget = late_graph_budget
    self.original_layer, self.original_prefill = model.text_layer, model.prefill
    self.original_padded = getattr(model, 'padded_prefill', None)
    self.enabled = True
    self.phase = None
    self.prompt_variable = None
    self.active_slots = None
    self.allowed = None
    self.shape = None
    self.metadata = None
    self.events = []
    self.decoders = []
    self.late_jits = OrderedDict()
    self.early_jit, self.full_jit = TinyJit(self._early), TinyJit(self._full)
    self.installed_layer = MethodType(lambda _, *a, **kw: self.layer(*a, **kw), model)
    model.text_layer = self.installed_layer
    model.prefill = self.prefill
    model._agent_token_pruner = self

  def prepare(self, prompt, frames):
    # Main prediction thread only. Never called by a future input's CPU worker.
    frames = tuple(frames)
    if any(frame.offset_us > 0 for frame in frames): raise ValueError('Unavailable future camera metadata')
    self.metadata = (int(prompt), frames)

  def register_decoder(self, decoder):
    if decoder not in self.decoders: self.decoders.append(decoder)

  def install_padded(self):
    self.original_padded = getattr(self.model, 'padded_prefill', None)
    self.model.padded_prefill = self.padded_prefill

  def restore(self):
    # Keep later pipeline wrappers intact. Their captured reference to this
    # layer becomes an arithmetic passthrough after disabling the controller.
    self.enabled = False
    if self.model.text_layer is self.installed_layer: self.model.text_layer = self.original_layer
    self.model.prefill = self.original_prefill
    if self.original_padded is not None: self.model.padded_prefill = self.original_padded
    elif hasattr(self.model, 'padded_prefill'): del self.model.padded_prefill
    for jit in (self.model.prefill_jit, self.model.decode_jit, self.model.greedy_decode_jit): jit.reset()
    self.model.trajectory_jits.clear()
    for decoder in self.decoders:
      for name in ('jits', 'verifiers', 'select_jits'):
        if hasattr(decoder, name): getattr(decoder, name).clear()
    if getattr(self.model, '_agent_token_pruner', None) is self: del self.model._agent_token_pruner

  @contextmanager
  def stage(self, phase, prompt=None, slots=None):
    old = self.phase, self.prompt_variable, self.active_slots
    self.phase, self.prompt_variable, self.active_slots = phase, prompt, slots
    try: yield
    finally: self.phase, self.prompt_variable, self.active_slots = old

  @staticmethod
  def _index(prefix, kind):
    start = 'vlm.model.language_model.layers.' if kind == 'language' else 'expert.expert.layers.'
    return int(prefix[len(start):]) if prefix.startswith(start) else None

  def _saliency(self, q, k):
    prompt = self.prompt_variable
    last = q[:, :, prompt-1:prompt].float()
    keys = k.float().repeat_interleave(q.shape[1]//k.shape[1], dim=1)
    scores = last.matmul(keys.transpose(-2, -1))*(q.shape[-1]**-0.5)
    scores = scores+(Tensor.arange(k.shape[2]) >= prompt).where(-math.inf, 0.0)
    return scores.softmax(-1).mean(axis=1).reshape(-1).contiguous().realize()

  def layer(self, x, cos, sin, prefix, cache=None, mask=None, cache_pos=None):
    if not self.enabled: return self.original_layer(x, cos, sin, prefix, cache, mask, cache_pos)
    language = self._index(prefix, 'language')
    expert = self._index(prefix, 'expert')
    special = (self.phase == 'early' and language == 1) or (self.phase == 'late' and language is not None)
    if not special:
      if self.phase is None and ((language is not None and language >= 2) or (expert is not None and expert >= 2)):
        if self.allowed is None or mask is None: raise RuntimeError('Selected-prefix validity is not initialized')
        allowed = self.allowed
        if expert is not None: allowed = allowed.cat(Tensor.ones(64, device=allowed.device, dtype=dtypes.bool))
        mask = mask+allowed[:mask.shape[-1]].where(0.0, -math.inf)
      return self.original_layer(x, cos, sin, prefix, cache, mask, cache_pos)
    # Same text-layer arithmetic and quantizer calls as frozen EdgeModel.text_layer.
    # The surrounding original packed wrapper supplies its existing scale state.
    model = self.model
    h = model.norm(x, f'{prefix}.input_layernorm', rms=True)
    quantized = model.quantize(h) if f'{prefix}.self_attn.q_proj.weight' in model.weight_scales else None
    q, k, v = (model.linear(h, f'{prefix}.self_attn.{name}_proj', quantized, heads=heads)
               .reshape(1, h.shape[1], heads, 128).transpose(1, 2)
               for name, heads in (('q', 16), ('k', 8), ('v', 8)))
    q, k = model.rope(q, k, cos, sin)
    if self.phase == 'early':
      self.score_output = self._saliency(q, k)
      store = cache[:, :, :, :x.shape[1]].uop.store(Tensor.stack(k, v).uop)
      assigned = Tensor(cache.uop.after(store))
      k, v = (assigned[i, :, :, :x.shape[1]] for i in range(2))
    else:
      # Explicit realization records the in-place effect in TinyJit, even though
      # attention reads compact current K/V rather than gathering the cache again.
      indexed_scatter(cache, Tensor.stack(k, v).contiguous(), self.active_slots).realize()
    h = model.attention(q, k, v, None, True).transpose(1, 2).reshape(1, x.shape[1], 2048)
    x = x+model.linear(h, f'{prefix}.self_attn.o_proj')
    h = model.linear(model.norm(x, f'{prefix}.post_attention_layernorm', rms=True), f'{prefix}.mlp.fc1').relu().square()
    return (x+model.linear(h, f'{prefix}.mlp.fc2')).realize()

  def _early(self, x, cos, sin, prompt):
    with self.stage('early', prompt):
      for i in range(2):
        x = self.model.text_layer(x, cos, sin, f'vlm.model.language_model.layers.{i}', self.model.cache[i], None, cache_pos=0)
      return x, self.score_output

  def _late(self, x, cos, sin, slots, last):
    x, cos, sin = (indexed_gather(value, slots).realize() for value in (x, cos, sin))
    with self.stage('late', slots=slots):
      for i in range(2, 28):
        x = self.model.text_layer(x, cos, sin, f'vlm.model.language_model.layers.{i}', self.model.cache[i], None, cache_pos=0)
      x = self.model.norm(x[:, last:last+1], 'vlm.model.language_model.norm', rms=True)
      return self.model.linear(x, 'vlm.lm_head').realize()

  def _full(self, x, cos, sin, prompt):
    with self.stage('full', prompt):
      for i in range(28):
        x = self.model.text_layer(x, cos, sin, f'vlm.model.language_model.layers.{i}', self.model.cache[i], None, cache_pos=0)
      x = self.model.norm(x[:, prompt-1:prompt], 'vlm.model.language_model.norm', rms=True)
      return self.model.linear(x, 'vlm.lm_head').realize()

  def _allocate(self, x, cos, capacity):
    model = self.model
    shape = (x.shape, cos.shape, capacity)
    if model.cache is None or model.cache[0].shape[3] != capacity:
      model.cache = [Tensor.zeros(2, 1, 8, capacity, 128, device=x.device, dtype=model.dtype).realize() for _ in range(28)]
      self.allowed = Tensor.ones(capacity, device=x.device, dtype=dtypes.bool).contiguous().realize()
      for jit in (model.prefill_jit, model.decode_jit, model.greedy_decode_jit): jit.reset()
      model.trajectory_jits.clear()
      for decoder in self.decoders:
        for name in ('jits', 'verifiers', 'select_jits'):
          if hasattr(decoder, name): getattr(decoder, name).clear()
      self.shape = None
    if self.allowed is None:
      self.allowed = Tensor.ones(capacity, device=x.device, dtype=dtypes.bool).contiguous().realize()
    if shape != self.shape:
      self.early_jit.reset(); self.full_jit.reset(); self.late_jits.clear()
      self.shape = shape
    model.prompt_length = x.shape[1]

  def _set_allowed(self, plan, device):
    active = np.zeros(plan.capacity, dtype=bool)
    active[plan.slots] = True
    active[plan.prompt:] = True  # Existing causal/length masks limit generated slots.
    self.allowed.assign(Tensor(active, device='CPU').to(device)).realize()

  def prefill(self, x, cos, sin, max_tokens):
    return self.padded_prefill(x, cos, sin, max_tokens, x.shape[1])

  def padded_prefill(self, x, cos, sin, max_tokens, length):
    length = int(length)
    if not 0 < length <= x.shape[1] or max_tokens <= 0: raise ValueError('Invalid original prompt or reasoning capacity')
    capacity = max(round_up(x.shape[1]+max_tokens, 256), getattr(self.model, 'minimum_cache_capacity', 0))
    self._allocate(x, cos, capacity)
    frames, reason = (), None
    if self.metadata is None or self.metadata[0] != length: reason = 'missing_current_metadata'
    else: frames = self.metadata[1]
    valid = UOp.variable('agent_prune_actual_prompt', 1, x.shape[1]).bind(length)
    event = {'classification': 'NUMERICAL whole-prompt-conditioned model; not exact', 'retention': 0.5,
             'full_language_layers': 2, 'original_prompt_tokens': length, 'original_prefill_rows': x.shape[1],
             'cache_capacity': capacity, 'prompt_prefix_invariance_pass': False, 'quality_passed': None}
    if reason is None:
      hidden, scores = self.early_jit(x, cos, sin, valid)
      # This fence/readback and CPU ranking are inside the caller's prefill timing.
      values = scores.numpy().copy()[:length]
      plan = reference.select_last_query(values, length, capacity, frames)
      reason = plan.fallback
      rows = round_up(len(plan.slots), 64)
      if self.require_packed and rows < 128: reason = 'compact_rows_below_packed_threshold'
      if rows not in self.late_jits and len(self.late_jits) >= self.late_graph_budget: reason = 'bounded_late_graph_budget'
    if reason is not None:
      if not self.allow_full_fallback: raise RuntimeError(f'Numerical prefill cannot proceed: {reason}')
      plan = reference.full_plan(length, capacity, reason)
      self._set_allowed(plan, x.device)
      logits = self.full_jit(x, cos, sin, valid)
      event.update(fallback=reason, late_rows=x.shape[1], retained_prompt_tokens=length, full_prefill_recomputed=True)
    else:
      self._set_allowed(plan, x.device)
      slots_array = np.pad(plan.slots.astype(np.int32), (0, rows-len(plan.slots)), constant_values=-1)
      slots = Tensor(slots_array, device='CPU').to(x.device).contiguous().realize()
      if rows not in self.late_jits: self.late_jits[rows] = TinyJit(self._late)
      last = UOp.variable('agent_prune_last_compact_row', 0, rows-1).bind(len(plan.slots)-1)
      logits = self.late_jits[rows](hidden, cos, sin, slots, last)
      event.update(fallback=None, late_rows=rows, retained_prompt_tokens=len(plan.slots), full_prefill_recomputed=False)
    self.last_plan = plan
    event['selected_slots_sha256'] = hashlib.sha256(plan.slots.astype('<i4').tobytes()).hexdigest()
    self.events.append(event)
    return logits
