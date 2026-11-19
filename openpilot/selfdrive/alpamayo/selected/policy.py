"""Prefill draft placement and target verification."""
import numpy as np

DRAFT_TEXT = '<NOMINAL>\n<|traj_future_start|>'
CHECKPOINT_DRAFT_IDS = (124549,11135,3174,4286,1561,135084)


class PromptTooLong(ValueError):
  """An individual input exceeds the selected model's fixed prompt capacity."""


def validate_draft(draft, anchor):
  draft = tuple(int(token) for token in draft)
  if len(draft) != 6 or draft[-1] != int(anchor) or int(anchor) in draft[:-1]:
    raise ValueError('Require six draft tokens with exactly one final trajectory marker')
  return draft


def padded_inputs(prepared, packed, length, draft):
  """Pad the prompt to `length` rows and place as much of `draft` as fits after it."""
  ids,positions,delta,vision_positions = packed
  actual = ids.shape[1]
  if actual > length:
    raise PromptTooLong(f'Prompt uses {actual} tokens; selected capacity is {length}')
  if ids.shape[0] != 1 or positions.shape != (3,1,actual) or not 0 < actual <= length:
    raise ValueError('Invalid actual prompt or fixed padded length')
  count = length-actual
  new_ids = np.pad(ids,((0,0),(0,count)))
  new_positions = np.pad(positions,((0,0),(0,0),(0,count)))
  placed = tuple(int(t) for t in draft[:count])
  if placed:
    new_ids[0,actual:actual+len(placed)] = placed
    next_positions = actual+int(delta)+np.arange(len(placed))
    new_positions[:,:,actual:actual+len(placed)] = next_positions[None,None,:]
  metadata = dict(prepared,actual_prompt_length=actual,prefill_draft=placed)
  return metadata,(new_ids,new_positions,int(delta)-count,vision_positions)


def accepted_prefix(targets, finite, draft, anchor):
  """Draft tokens confirmed by the prefill heads: stop at the first mismatch, a non-finite head, or the anchor."""
  accepted = []
  for target,ok,token in zip(targets,finite,draft):
    if not ok or int(target) != int(token):
      break
    accepted.append(int(token))
    if int(token) == int(anchor):
      break
  return accepted
