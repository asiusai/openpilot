"""CPU contracts for complete, current-target-verified padded-prefill drafts."""
import numpy as np

DRAFT_TEXT = '<NOMINAL>\n<|traj_future_start|>'
CHECKPOINT_DRAFT_IDS = (124549,11135,3174,4286,1561,135084)


def validate_draft(draft, anchor):
  draft = tuple(int(token) for token in draft)
  if len(draft) != 6 or draft[-1] != int(anchor) or int(anchor) in draft[:-1]:
    raise ValueError('Require six complete draft tokens with exactly one final trajectory marker')
  return draft


def padded_inputs(prepared, packed, length, draft, anchor):
  draft = validate_draft(draft,anchor)
  ids, positions, delta, vision_positions = packed
  actual = ids.shape[1]
  if ids.shape[0] != 1 or positions.shape != (3,1,actual) or not 0 < actual <= length:
    raise ValueError('Invalid actual prompt or fixed padded length')
  count = length-actual
  new_ids = np.pad(ids,((0,0),(0,count)))
  new_positions = np.pad(positions,((0,0),(0,0),(0,count)))
  eligible = count >= len(draft)
  if eligible:
    new_ids[0,actual:actual+len(draft)] = draft
    # This is the same scalar text position used by current decode RoPE.
    next_positions = actual+int(delta)+np.arange(len(draft))
    new_positions[:,:,actual:actual+len(draft)] = next_positions[None,None,:]
  metadata = dict(prepared,actual_prompt_length=actual,prefill_draft_eligible=eligible)
  return metadata,(new_ids,new_positions,int(delta)-count,vision_positions)


def accept_targets(predicted, draft, anchor, finite=True):
  draft = validate_draft(draft,anchor)
  values = np.asarray(predicted)
  return bool(finite and values.shape == (6,) and np.issubdtype(values.dtype,np.integer)
              and tuple(int(value) for value in values) == draft)


def target_rows(prompt):
  if prompt <= 0: raise ValueError('Prompt must be nonempty')
  return np.arange(prompt-1,prompt+5)


def cache_contract(prompt, generated, capacity, action_queries=64):
  if not 0 < prompt <= prompt+generated <= capacity: raise ValueError('Invalid cache extent')
  length = prompt+generated
  text_keys = np.arange(capacity)
  expert_keys = np.arange(capacity+action_queries)
  return dict(cache_length=length,action_rotary_offset=generated,
              valid_text=text_keys<length,expert_mask=(expert_keys<length)|(expert_keys>=capacity))
