"""Two declared NUMERICAL policies: latest full, older per-image50% or75%.

No route outputs, ground truth, proposed tokens, or future rows enter selection.
Whole-current-prompt conditioning still fails original prompt-prefix invariance.
"""
from dataclasses import dataclass
import importlib.util
import math
from numbers import Integral
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
POLICIES = {'latest-full-history-50': .5, 'latest-full-history-75': .75}


def load_frozen(name, namespace):
  spec = importlib.util.spec_from_file_location(namespace, HERE/(name+'.py'))
  module = importlib.util.module_from_spec(spec)
  sys.modules[namespace] = module
  spec.loader.exec_module(module)
  return module


reference = load_frozen('agent-token-prune-reference', 'agent_token_current_frozen_reference')


def integer(value, field):
  if isinstance(value, bool) or not isinstance(value, Integral): raise ValueError('Expected integer '+field)
  value = int(value)
  if not 0 <= value < 2**63: raise ValueError('Invalid '+field)
  return value


def canonical_records(query, records):
  """Use the actual adapter's canonical ordering, with strict front-history checks."""
  import torch
  from alpamayo2_edge.sample import canonical_frame_order
  query = integer(query, 'query_ns')
  records = tuple((integer(camera, 'camera'), integer(stamp, 'capture_ns')) for camera, stamp in records)
  if len(records) != 10 or {camera for camera, _ in records} != {1, 6}:
    raise ValueError('Require exactly both complete five-frame front histories')
  if any(stamp > query for _, stamp in records): raise ValueError('Unavailable future camera frame')
  cameras = torch.tensor([camera for camera, _ in records], dtype=torch.int64, device='cpu')
  stamps = torch.tensor([stamp for _, stamp in records], dtype=torch.int64, device='cpu')
  order = canonical_frame_order(cameras, stamps).tolist()
  result = tuple(records[index] for index in order)
  for camera in (1, 6):
    times = [stamp for view, stamp in result if view == camera]
    if len(times) != 5 or any(a >= b for a, b in zip(times, times[1:])):
      raise ValueError('Camera history must contain five unique chronological timestamps')
    if len({stamp//1000 for stamp in times}) != 5:
      raise ValueError('Camera timestamps collapse in the actual microsecond adapter')
  return result


@dataclass(frozen=True)
class Observations:
  query_ns: int
  frames: tuple
  source_frames: tuple

  # Preserve the frozen route's two-field observation interface.
  def __iter__(self): return iter((self.query_ns, self.frames))
  def __getitem__(self, index): return (self.query_ns, self.frames)[index]


@dataclass(frozen=True)
class ObservedFrame:
  begin: int
  end: int
  camera: int
  offset_us: int
  capture_ns: int
  query_ns: int


def current_observations(rows, sample_key):
  row = rows[sample_key]  # Addressed current row only; never enumerate rows or GT.
  query = integer(row['query_ns'], 'query_ns')
  source = tuple((integer(frame['view'], 'camera'), integer(frame['capture_ns'], 'capture_ns')) for frame in row['frames'])
  return Observations(query, canonical_records(query, source), source)


def prepared_observations(data, observations):
  """Match metadata to original pixel order, then the adapter's canonical order."""
  import torch
  cameras, stamps = data['camera_indices'], data['absolute_timestamps']
  if cameras.dtype != torch.int64 or cameras.device.type != 'cpu' or cameras.shape != (10,):
    raise ValueError('Require CPU int64 camera IDs for exactly ten images')
  if stamps.dtype != torch.int64 or stamps.device.type != 'cpu' or stamps.shape not in ((10,), (10, 1)):
    raise ValueError('Require CPU int64 microsecond capture timestamps')
  raw = tuple(zip(cameras.tolist(), stamps.reshape(-1).tolist(), strict=True))
  if raw != tuple((camera, stamp//1000) for camera, stamp in observations.source_frames):
    raise ValueError('Pixel metadata order does not match the addressed observation records')
  canonical = canonical_records(observations.query_ns, observations.source_frames)
  if canonical != observations.frames: raise ValueError('Canonical observation records changed')
  return observations


def frame_spans(ids, image_id, prompt, observations, spans, grids, *, strict=True):
  """Bind real image IDs/grid lengths to canonically validated camera records."""
  if not isinstance(observations, Observations): raise ValueError('Validated current observations required')
  ids = np.asarray(ids)
  if ids.ndim != 2 or ids.shape[0] != 1 or not 0 < prompt <= ids.shape[1]: raise ValueError('Invalid prompt IDs')
  records = canonical_records(observations.query_ns, observations.frames)
  if records != observations.frames or len(spans) != 10 or len(grids) != 10:
    raise ValueError('Canonical frame/span/grid mismatch')
  covered = np.zeros(prompt, dtype=bool)
  result, previous_end = [], 0
  for (camera, stamp), (begin, end), grid in zip(records, spans, grids, strict=True):
    begin, end = int(begin), int(end)
    grid = tuple(map(int, grid))
    if len(grid) != 3 or min(grid) <= 0 or math.prod(grid) % 4: raise ValueError('Invalid original image grid')
    if not previous_end <= begin < end < prompt or covered[begin:end].any(): raise ValueError('Invalid canonical image token span')
    count = math.prod(grid)//4
    if count != end-begin or (strict and (grid != (1, 20, 36) or count != 180)):
      raise ValueError('Require all original full-resolution projected visual tokens')
    covered[begin:end] = True
    result.append(ObservedFrame(begin, end, camera, (stamp-observations.query_ns)//1000, stamp, observations.query_ns))
    previous_end = end
  if not np.array_equal(covered, ids[0, :prompt] == int(image_id)): raise ValueError('Image IDs do not match canonical frame spans')
  return tuple(result)


def validate_bound_frames(prompt, frames, *, require_full_resolution=False):
  frames = tuple(frames)
  if len(frames) != 10 or not all(isinstance(frame, ObservedFrame) for frame in frames):
    raise ValueError('Require ten explicitly observed frame spans')
  queries = {frame.query_ns for frame in frames}
  if len(queries) != 1: raise ValueError('Mixed current query timestamps')
  query = next(iter(queries))
  records = tuple((frame.camera, frame.capture_ns) for frame in frames)
  # Slot order and camera order are independently checked; latest never uses a guessed span number.
  if canonical_records(query, records) != records: raise ValueError('Frame spans are not in actual canonical camera order')
  covered = np.zeros(prompt, dtype=bool)
  previous_end = 0
  for frame in frames:
    if frame.offset_us != (frame.capture_ns-query)//1000: raise ValueError('Inconsistent actual capture timestamp')
    if not previous_end <= frame.begin < frame.end < prompt: raise ValueError('Overlapping or unsorted image token spans')
    if require_full_resolution and frame.end-frame.begin != 180: raise ValueError('Original projected token count changed')
    covered[frame.begin:frame.end] = True
    previous_end = frame.end
  latest = {camera:max(stamp for view, stamp in records if view == camera) for camera in (1, 6)}
  return frames, covered, latest


def select(scores, prompt, capacity, frames, policy):
  if policy not in POLICIES: raise ValueError('Only the two predeclared retention policies are supported')
  if not 0 < prompt <= capacity: raise ValueError('Prompt exceeds cache capacity')
  frames, image, latest = validate_bound_frames(prompt, frames)
  scores = np.asarray(scores)
  if scores.shape != (prompt,) or not np.isfinite(scores).all() or np.any(scores < 0) or not scores.sum() > 0:
    return reference.full_plan(prompt, capacity, 'invalid_saliency')
  selected = []
  for frame in frames:
    indices = np.arange(frame.begin, frame.end)
    if frame.capture_ns == latest[frame.camera]: selected.extend(indices.tolist())
    else:
      count = math.ceil(POLICIES[policy]*len(indices))
      selected.extend(indices[np.lexsort((indices, -scores[indices]))[:count]].tolist())
  slots = np.sort(np.concatenate((np.flatnonzero(~image), np.asarray(selected, dtype=np.int64))))
  return reference.Plan(prompt, capacity, slots, np.flatnonzero(image), frames)


def frame_counts(plan):
  if not plan.frames: return []
  _, _, latest = validate_bound_frames(plan.prompt, plan.frames)
  return [dict(camera=frame.camera, capture_ns=frame.capture_ns, query_ns=frame.query_ns,
               begin=frame.begin, end=frame.end, latest=frame.capture_ns == latest[frame.camera],
               available=frame.end-frame.begin, retained=int(((plan.slots >= frame.begin) & (plan.slots < frame.end)).sum()))
          for frame in plan.frames]
