"""Opt-in joint depth prototype and unchanged-teacher capture boundaries.

No defaults change. Language students use new BF16 names, outside packed FP8
weight dictionaries. Expert students require an explicit untrained-seed opt-in.
"""
from contextlib import contextmanager
from copy import deepcopy
import hashlib
from pathlib import Path
import struct
import json
from types import MethodType

LANGUAGE = 'vlm.model.language_model.layers.'
EXPERT = 'expert.expert.layers.'
GROUPS = tuple(tuple(range(i, i + 4)) for i in range(0, 28, 4))
VISION_GROUPS = tuple(tuple(range(i, min(i + 4, 27))) for i in range(0, 27, 4))
PILOT_GROUPS = tuple((i,) for i in range(12)) + (GROUPS[3],) + tuple((i,) for i in range(16, 28))
PILOT_SHA = 'cf5762ab5cfdb3b622ff8d12d2d6e7f20c3d6f75f7ed772dcca97653d85504fd'


def sha(path):
  with Path(path).open('rb') as stream: return hashlib.file_digest(stream, 'sha256').hexdigest()


def mapping(groups=PILOT_GROUPS):
  if tuple(i for group in groups for i in group) != tuple(range(28)):
    raise ValueError('Groups must partition all 28 ordered teacher layers')
  return [dict(student=i, teacher_group=list(group), weight_source=group[0],
               language_cache_slot=i, expert_cache_slot=i, residual_target_after=group[-1])
          for i, group in enumerate(groups)]


def reset_state(model, decoder=None, vision_cache=None):
  """Discard graph/cache objects when changing weights, mapping, or Python hooks."""
  owners = [model, getattr(model, 'padded_prefill', None), decoder, vision_cache]
  if owners[1] is not None: owners.append(getattr(owners[1], 'baseline', None))
  for owner in owners:
    if owner is None: continue
    for value in list(vars(owner).values()):
      if hasattr(value, 'reset') and hasattr(value, 'fxn'): value.reset()
      elif isinstance(value, dict):
        for item in value.values():
          if hasattr(item, 'reset') and hasattr(item, 'fxn'): item.reset()
  model.cache = None
  model.prompt_length = model.greedy_shape = None
  model.trajectory_jits.clear()
  for prefill in owners[1:2] + owners[4:]:
    if prefill is not None: prefill.shape = None
  if decoder is not None:
    for name in ('previous', 'history', 'statistics', 'prefill_draft_records', 'prefill_draft_old_history'):
      if hasattr(decoder, name): getattr(decoder, name).clear()
  if vision_cache is not None:
    vision_cache.cache.clear()
    vision_cache.encoders.clear()
    vision_cache.stitchers.clear()


def seed_block(checkpoint, prefix):
  """Read only the requested original BF16 block, without a model loader."""
  import numpy as np
  checkpoint = Path(checkpoint)
  index = json.loads((checkpoint/'model.safetensors.index.json').read_text())['weight_map']
  result = {}
  for full, filename in index.items():
    if not full.startswith(prefix): continue
    with (checkpoint/filename).open('rb') as stream:
      size = struct.unpack('<Q', stream.read(8))[0]
      header = json.loads(stream.read(size)); entry = header[full]
      assert entry['dtype'] == 'BF16'
      begin, end = entry['data_offsets']; stream.seek(8 + size + begin)
      bits = np.frombuffer(stream.read(end - begin), dtype=np.uint16)
    result[full.removeprefix(prefix)] = (bits.astype(np.uint32) << 16).view(np.float32).reshape(entry['shape'])
  if not result: raise ValueError('Requested BF16 seed block is absent')
  return result


def load_pilot(path, expected_sha=PILOT_SHA):
  import numpy as np
  if sha(path) != expected_sha: raise ValueError('Language checkpoint hash mismatch')
  with np.load(path, allow_pickle=False) as data: values = {name: data[name].copy() for name in data}
  if sum(v.size for v in values.values()) != 50335744: raise ValueError('Wrong language block width')
  if any(v.dtype != np.float32 or not np.isfinite(v).all() for v in values.values()):
    raise ValueError('Expected finite FP32 language checkpoint masters')
  return values


@contextmanager
def joint_pilot(model, language_weights, expert_weights, *, allow_untrained_expert=False, decoder=None, vision_cache=None):
  """Execute 25 paired blocks; compact slot12 replaces teacher blocks12..15.

  Original layer16 uses compact cache13, through original layer27/cache24.
  No caches for teacher layers13..15 exist. Every retained KV is current.
  """
  from tinygrad import Tensor
  if not allow_untrained_expert: raise ValueError('The expert seed is untrained; opt in explicitly')
  if getattr(model, 'joint_depth_mapping', None) is not None: raise ValueError('Already compressed')
  text = model.config['vlm_config']['text_config']
  expert = model.config['expert_config']['llm_config']
  if text['num_hidden_layers'] != 28 or expert['num_hidden_layers'] != 28: raise ValueError('Expected full teacher depths')
  plan = mapping()
  namespaces = {LANGUAGE: 'agent_joint.language.12', EXPERT: 'agent_joint.expert.12'}
  added = {}
  for source, arrays in ((LANGUAGE, language_weights), (EXPERT, expert_weights)):
    prefix = source + '12.'
    expected = {key.removeprefix(prefix): value.shape for key, value in model.weights.items() if key.startswith(prefix)}
    if set(arrays) != set(expected) or any(tuple(arrays[k].shape) != tuple(expected[k]) for k in arrays):
      raise ValueError('Student weights do not match original block geometry')
    device = model.weights[prefix+'input_layernorm.weight'].device
    for name, value in arrays.items():
      key = namespaces[source] + '.' + name
      if key in model.weights: raise ValueError('Student weight namespace already exists')
      added[key] = Tensor(value.copy(), device='CPU').to(device).cast(model.dtype).realize()
  old_config, original_layer = model.config, model.text_layer
  reset_state(model, decoder, vision_cache)
  model.weights.update(added)
  model.config = deepcopy(model.config)
  model.config['vlm_config']['text_config']['num_hidden_layers'] = len(plan)
  model.config['expert_config']['llm_config']['num_hidden_layers'] = len(plan)
  model.joint_depth_mapping = plan
  def layer(self, x, cos, sin, prefix, cache=None, mask=None, cache_pos=None):
    source = next((name for name in (LANGUAGE, EXPERT) if prefix.startswith(name)), None)
    if source is None: raise ValueError('Unexpected text layer namespace')
    slot = int(prefix.removeprefix(source))
    if not 0 <= slot < len(plan): raise ValueError('Unmapped student layer')
    target = namespaces[source] if slot == 12 else source + str(plan[slot]['weight_source'])
    if self.cache is not None and len(self.cache) != len(plan): raise ValueError('Stale teacher cache count')
    if source == LANGUAGE and cache is not self.cache[slot]: raise ValueError('Incorrect language cache slot')
    if source == EXPERT and (cache is None or len(cache) != 2): raise ValueError('Missing paired expert context')
    return original_layer(x, cos, sin, target, cache, mask, cache_pos)
  model.text_layer = MethodType(layer, model)
  try:
    yield dict(mapping=plan, language_student_dtype=str(model.dtype), expert_trained=False, quality_qualified=False)
  finally:
    model.text_layer, model.config = original_layer, old_config
    del model.joint_depth_mapping
    for name in added: del model.weights[name]
    reset_state(model, decoder, vision_cache)


@contextmanager
def teacher_targets(model, save, *, groups=(GROUPS[3],)):
  """Capture full teacher execution. Caller uses Context(JIT=0) for diagnostics.

  `save(name, tensor)` copies each tensor immediately. Its return value is metadata.
  All expert targets follow actual teacher rollout actions and current reasoning KV.
  """
  if getattr(model, 'joint_depth_mapping', None) is not None: raise ValueError('Targets require unchanged teacher')
  if model.config['vlm_config']['text_config']['num_hidden_layers'] != 28: raise ValueError('Teacher language depth changed')
  if model.config['expert_config']['llm_config']['num_hidden_layers'] != 28: raise ValueError('Teacher expert depth changed')
  if model.config['vlm_config']['vision_config']['num_hidden_layers'] != 27: raise ValueError('Teacher vision depth changed')
  starts = {g[0]: g for g in groups}; ends = {g[-1]: g for g in groups}
  old = {name: getattr(model, name) for name in ('text_layer', 'vision_layer', 'expert_step', 'linear')}
  if hasattr(model, 'batched_projector'): old['batched_projector'] = model.batched_projector
  state = dict(expert_steps=[], vision_batches=[], language_calls=0, language_orders=[], tensors={})
  active = dict(expert=None, vision=None, language=None)
  def keep(name, tensor):
    if name in state['tensors']: raise ValueError('Duplicate capture target '+name)
    state['tensors'][name] = save(name, tensor)
  def text_layer(self, x, cos, sin, prefix, cache=None, mask=None, cache_pos=None):
    source = LANGUAGE if prefix.startswith(LANGUAGE) else EXPERT
    index = int(prefix.removeprefix(source))
    if source == LANGUAGE:
      if index == 0:
        active['language'] = state['language_calls']; state['language_calls'] += 1
        state['language_orders'].append([])
      state['language_orders'][active['language']].append(index)
      stem = f'language{active["language"]}'
    else:
      if active['expert'] is None: raise ValueError('Expert layer outside actual expert_step')
      state['expert_steps'][active['expert']]['layer_order'].append(index)
      stem = f'expert{active["expert"]}'
    capture = source == EXPERT or (isinstance(cache_pos, int) and cache_pos == 0)
    if capture and index in starts:
      group = starts[index]; base = f'{stem}.group{index}'
      keep(base+'.x', x); keep(base+'.cos', cos); keep(base+'.sin', sin)
      if source == EXPERT:
        # Full capacity and the original mask preserve padding/anchor semantics.
        keep(base+'.context_k', cache[0]); keep(base+'.context_v', cache[1]); keep(base+'.mask', mask)
    y = old['text_layer'](x, cos, sin, prefix, cache, mask, cache_pos)
    if capture and index in ends:
      group = ends[index]; base = f'{stem}.group{group[0]}'
      keep(base+'.y', y)
      if source == LANGUAGE:
        # Includes current decode/draft writes. Store full current capacity and position.
        keep(base+'.k', self.cache[group[0]][0]); keep(base+'.v', self.cache[group[0]][1])
        if cache_pos is not None:
          position = cache_pos if isinstance(cache_pos, int) else cache_pos.unbind()[1]
          state.setdefault('language_positions', {})[stem] = dict(position=int(position), rows=x.shape[1])
    return y
  def vision_layer(self, x, index):
    if index == 0:
      active['vision'] = len(state['vision_batches'])
      state['vision_batches'].append(dict(images=x.shape[0], patches=x.shape[1], layer_order=[]))
    row = state['vision_batches'][active['vision']]; row['layer_order'].append(index)
    if index in starts: keep(f'vision{active["vision"]}.group{index}.x', x)
    y = old['vision_layer'](x, index)
    if index in ends: keep(f'vision{active["vision"]}.group{ends[index][0]}.y', y)
    return y
  def linear(self, x, name, quantized_input=None, heads=None):
    y = old['linear'](x, name, quantized_input, heads)
    if name == 'vlm.model.projector.linear_fc2':
      count = state.setdefault('projector_calls', 0); keep(f'projector{count}.features', y)
      state['projector_calls'] += 1
    return y
  def batched_projector(value):
    # The selected packed projector bypasses model.linear entirely.
    outputs = old['batched_projector'](value)
    for output in outputs:
      count = state.setdefault('projector_calls', 0); keep(f'projector{count}.features', output)
      state['projector_calls'] += 1
    return outputs
  def expert_step(self, actions, timestep, cos, sin, flat_cache, mask):
    if active['expert'] is not None: raise ValueError('Nested expert rollout')
    index = len(state['expert_steps']); active['expert'] = index
    state['expert_steps'].append(dict(layer_order=[]))
    keep(f'expert{index}.actions', actions); keep(f'expert{index}.timestep', timestep)
    try:
      y = old['expert_step'](actions, timestep, cos, sin, flat_cache, mask)
      keep(f'expert{index}.velocity', y)
      return y
    finally: active['expert'] = None
  for name, method in (('text_layer', text_layer), ('vision_layer', vision_layer), ('linear', linear), ('expert_step', expert_step)):
    setattr(model, name, MethodType(method, model))
  if 'batched_projector' in old: model.batched_projector = batched_projector
  try: yield state
  finally:
    for name, method in old.items(): setattr(model, name, method)


def validate_targets(state):
  if len(state['expert_steps']) != 3: raise ValueError('Expected three actual Euler evaluations')
  if any(row['layer_order'] != list(range(28)) for row in state['expert_steps']):
    raise ValueError('Incomplete unchanged expert execution')
  if not state['vision_batches'] or any(row['layer_order'] != list(range(27)) for row in state['vision_batches']):
    raise ValueError('Incomplete unchanged vision execution')
  if state['language_calls'] < 1 or any(order != list(range(28)) for order in state['language_orders']):
    raise ValueError('Incomplete current language context')
  if not state.get('projector_calls'): raise ValueError('Missing actual projector features')
  return True
