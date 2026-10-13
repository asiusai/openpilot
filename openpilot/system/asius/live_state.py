"""Batch changed state in the existing signed and encrypted peer envelope."""
import copy
import time
import uuid


def difference(old, new):
  if not isinstance(old, dict) or not isinstance(new, dict):
    return new
  result = dict.fromkeys(old.keys() - new.keys())
  for key, value in new.items():
    if key not in old or value != old[key]:
      result[key] = difference(old.get(key), value)
  return result


class StateEncoder:
  def __init__(self):
    self.stream = uuid.uuid4().hex
    self.sequence = 0
    self.previous = None
    self.full_at = -float("inf")

  def encode(self, snapshot, now=None):
    now = time.monotonic() if now is None else now
    full = self.previous is None or now - self.full_at >= 3
    data = snapshot if full else difference(self.previous, snapshot)
    if full:
      self.full_at = now
    self.sequence += 1
    result = {"stream": self.stream, "sequence": self.sequence, "base": None if full else self.sequence - 1,
              "changes": data, "full": full}
    self.previous = copy.deepcopy(snapshot)
    return result
