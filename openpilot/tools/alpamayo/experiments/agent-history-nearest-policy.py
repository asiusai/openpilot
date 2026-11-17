"""ONE numerical input policy: nearest native historical capture, constrained by query.

Original five offsets and both cameras remain fixed. Ties choose the past frame.
Selection has no access to images, model outputs, vehicle history, navigation, or GT.
"""
import bisect

OFFSETS_NS = (-2_000_000_000,-1_000_000_000,-500_000_000,-100_000_000,0)


def select_index(times, desired, query):
  if desired > query: raise ValueError('Desired history position is later than current query')
  floor = bisect.bisect_right(times,desired)-1
  if floor < 0: raise ValueError('No past historical capture')
  chosen = floor
  if floor+1 < len(times) and times[floor+1] <= query:
    if times[floor+1]-desired < desired-times[floor]: chosen += 1
  if times[chosen] > query or abs(times[chosen]-desired) > 150_000_000:
    raise ValueError('Unavailable or stale historical frame')
  return chosen
