"""Inference contract, process-boundary snapshots, and durable shadow output."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import hashlib
import io
import json
import math
import os
from pathlib import Path
import shutil
import signal
import struct
import sys
import tempfile
import time
from typing import Protocol

import numpy as np

from openpilot.selfdrive.alpamayo.segments import FrontImages

from openpilot.selfdrive.alpamayo.frontend import CAMERAS, FRAME_TOLERANCE_NS, OFFSETS_NS, ROLLING_PERIODS_NS, Frame, Selection, State, Unavailable

INPUT_LIMIT = 32*1024**2
HEADER_LIMIT = 4*1024**2
INPUT_MAGIC = b'ALPAIN01'
INPUT_PREFIX = struct.Struct('<8sI')


class FixedSlots:
  """One admission attempt per monotonic slot, with skipped slots never replayed."""
  def __init__(self, period_ms=500):
    if type(period_ms) is not int or period_ms not in (100, 250, 500):
      raise ValueError('period_ms must be 100, 250, or 500')
    self.period_ms, self.period_ns = period_ms, period_ms*1_000_000
    self.next_ns = None
    self.missed = 0

  def take(self, now):
    if self.next_ns is None:
      self.next_ns = ((now+self.period_ns-1)//self.period_ns)*self.period_ns
    if now < self.next_ns:
      return None
    skipped = (now-self.next_ns)//self.period_ns
    self.missed += skipped
    start = self.next_ns+skipped*self.period_ns
    self.next_ns = start+self.period_ns
    return {'slot_start_ns': start, 'slot_deadline_ns': self.next_ns, 'period_ms': self.period_ms, 'missed_slots': self.missed}

  def skip(self):
    self.missed += 1


def memory_ok(upcoming=0):
  memory = dict(line.split(':', 1) for line in Path('/proc/meminfo').read_text().splitlines())
  return int(memory['MemAvailable'].split()[0])*1024-upcoming >= 768*1024**2


def input_storage_ok(path, upcoming=0):
  """Input snapshots require tmpfs, its own small reserve, and available system RAM."""
  if not Path(path).resolve().is_relative_to('/dev/shm'):
    return False
  mounts = Path('/proc/self/mountinfo').read_text().splitlines()
  if not any(row.split()[4] == '/dev/shm' and row.split(' - ', 1)[1].split()[0] == 'tmpfs' for row in mounts):
    return False
  return shutil.disk_usage(path).free-upcoming >= 8*1024**2 and memory_ok(upcoming)


def ram_handoff():
  """Private single-slot lifetime; never fall back to a disk-backed temp directory."""
  if not input_storage_ok('/dev/shm', INPUT_LIMIT):
    raise RuntimeError('Shadow input requires available tmpfs and system RAM reserves')
  return tempfile.TemporaryDirectory(prefix='alpamayo-', dir='/dev/shm')


class InferenceAdapter(Protocol):
  manifest: dict

  def predict(self, arrays: dict, metadata: dict) -> dict:
    """Return {query_ns, prediction: {xyz, rot, reasoning, ...}} without publishing."""
    ...

  def synchronize(self) -> None:
    """Wait for all inference and output copies to finish."""
    ...

  def close(self) -> None: ...


def storage_ok(path, upcoming=0):
  usage = shutil.disk_usage(path)
  return usage.free-upcoming >= max(usage.total*.1, 5*1024**3)


def private_output(path):
  from openpilot.common.hardware.hw import Paths
  path = Path(path).resolve()
  if path.is_relative_to(Path(Paths.log_root()).resolve()):
    raise ValueError('Shadow output must be outside normal route recording storage')
  return path


def json_bytes(value):
  return (json.dumps(value, allow_nan=False, separators=(',', ':'))+'\n').encode()


def atomic_write(path, raw):
  if not storage_ok(path.parent, len(raw)):
    raise RuntimeError('Shadow output reached the recording disk reserve')
  temporary = path.with_suffix('.tmp')
  try:
    with temporary.open('xb') as stream:
      stream.write(raw)
      stream.flush()
      os.fsync(stream.fileno())
    temporary.replace(path)
    fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
      os.fsync(fd)
    finally:
      os.close(fd)
  finally:
    temporary.unlink(missing_ok=True)
  return len(raw)


def atomic_json(path, value):
  return atomic_write(path, json_bytes(value))


class PredictionWriter:
  """One immutable prediction can be in flight. The worker owns all admission and error checks."""
  def __init__(self):
    self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix='prediction-log')
    self.pending = None
    self.written = 0

  def check(self, wait=False):
    if self.pending is not None and (wait or self.pending.done()):
      self.pending.result()
      self.pending = None

  def submit(self, path, result):
    self.check(wait=True)
    raw = json_bytes(result)
    if self.written+len(raw) > 256*1024**2:
      raise RuntimeError('Shadow session reached its 256 MiB prediction limit')
    if not memory_ok(len(raw)) or not storage_ok(path.parent, len(raw)):
      raise RuntimeError('Shadow prediction reached its memory or disk reserve')
    self.pending = self.pool.submit(atomic_write, path, raw)
    self.written += len(raw)

  def close(self):
    self.pool.shutdown(wait=True)
    self.check(wait=True)


def frame_layout(ref):
  """Shape and payload size; a frame the worker already holds ('cached') has no payload."""
  shape = ref['shape']
  if (ref['dtype'] != 'uint8' or not isinstance(shape, list) or len(shape) not in (2, 3) or
      any(type(n) is not int or n <= 0 for n in shape)):
    raise ValueError('Invalid frame dtype or dimensions')
  if not ((ref['source_format'] == 'rgb' and len(shape) == 3 and shape[0] == 3) or
          (ref['source_format'] == 'nv12' and len(shape) == 2 and shape[0] % 3 == 0 and shape[1] % 2 == 0)):
    raise ValueError('Invalid frame format or shape')
  size = math.prod(shape)
  if type(ref['nbytes']) is not int or size != ref['nbytes'] or size > 8*1024**2:
    raise ValueError('Invalid frame byte count')
  if ref.get('cached', False) not in (True, False):
    raise ValueError('Invalid cached flag')
  return tuple(shape), 0 if ref.get('cached') else size


def valid_snapshot_views(refs, context):
  views = [ref['view'] for ref in refs]
  if context.get('early') is True:
    return views in ([1, 6], [1, 6, 1, 6]) and type(context.get('slot_start_ns')) is int
  return views == [1]*5+[6]*5 or (views == [1, 1, 6, 6] and context.get('period_ms') in (250, 500) and type(context.get('slot_start_ns')) is int)


def save_selection(path, selected, context, held=None):
  """held: {camera: frame IDs the worker retains}; those frames are sent as references without pixels."""
  refs, payload = [], []
  for frame in selected.frames:
    pixels = frame.pixels
    cached = frame.frame_id in (held or {}).get(frame.camera, ())
    ref = dict(frame.reference(), shape=list(pixels.shape), dtype=str(pixels.dtype), nbytes=pixels.nbytes, cached=cached)
    frame_layout(ref)
    if not pixels.flags.c_contiguous or pixels.flags.writeable:
      raise ValueError('Input frames must be immutable and contiguous')
    refs.append(ref)
    if not cached:
      payload.append(pixels)
  if not valid_snapshot_views(refs, context):
    raise ValueError('Invalid front-camera snapshot layout')
  meta = {'query_ns': selected.query_ns, 'selected_ns': selected.selected_ns,
          'frames': refs, 'small_model': selected.small_model,
          'states': {kind: [vars(row) for row in rows] for kind, rows in selected.states.items()},
          'camera_gaps': selected.camera_gaps, 'context': context}
  header = json.dumps(meta, allow_nan=False, separators=(',', ':')).encode()
  size = INPUT_PREFIX.size+len(header)+sum(p.nbytes for p in payload)
  if len(header) > HEADER_LIMIT or size > INPUT_LIMIT or not input_storage_ok(path.parent, size):
    raise RuntimeError('Shadow input reached its RAM limit or reserve')
  temporary = path.with_suffix('.tmp')
  if path.name not in ('pending.bin', 'early.bin') or path.exists() or temporary.exists() or (path.parent/'working.bin').exists():
    raise RuntimeError('Shadow input slot is occupied')
  try:
    with temporary.open('xb') as stream:
      stream.write(INPUT_PREFIX.pack(INPUT_MAGIC, len(header)))
      stream.write(header)
      for pixels in payload:
        if stream.write(memoryview(pixels).cast('B')) != pixels.nbytes:
          raise OSError('Short input payload write')
    if temporary.stat().st_size != size:
      raise OSError('Incomplete input snapshot')
    if 'slot_deadline_ns' in context and time.monotonic_ns() >= context['slot_deadline_ns']:
      raise Unavailable('Input handoff missed its slot deadline')
    temporary.replace(path)
  finally:
    temporary.unlink(missing_ok=True)


def invalid_json_number(value):
  raise ValueError(f'Nonfinite input metadata: {value}')


@contextmanager
def load_selection(path):
  """Yield read-only mapped frames. All pixel views are valid only inside this context."""
  with Path(path).open('rb') as stream:
    size = os.fstat(stream.fileno()).st_size
    if not INPUT_PREFIX.size <= size <= INPUT_LIMIT:
      raise ValueError('Invalid input file size')
    magic, length = INPUT_PREFIX.unpack(stream.read(INPUT_PREFIX.size))
    if magic != INPUT_MAGIC or not 0 < length <= min(HEADER_LIMIT, size-INPUT_PREFIX.size):
      raise ValueError('Invalid input magic or header length')
    try:
      meta = json.loads(stream.read(length), parse_constant=invalid_json_number)
      if (not isinstance(meta, dict) or not isinstance(meta['frames'], list) or
          any(not isinstance(meta[key], dict) for key in ('states', 'small_model', 'camera_gaps', 'context')) or
          not valid_snapshot_views(meta['frames'], meta['context']) or
          any(type(meta[key]) is not int or meta[key] <= 0 for key in ('query_ns', 'selected_ns'))):
        raise ValueError('Invalid input metadata')
      layouts = [frame_layout(ref) for ref in meta['frames']]
      payload_size = sum(n for _, n in layouts)
      if INPUT_PREFIX.size+length+payload_size != size:
        raise ValueError('Input payload does not match exact file size')
      states = {kind: tuple(State(**row) for row in rows) for kind, rows in meta['states'].items()}
      gaps = {int(camera): count for camera, count in meta['camera_gaps'].items()}
      # Validate frame metadata before mapping; layout offsets are derived, never trusted from the header.
      for ref in meta['frames']:
        fields = ('frame_id', 'capture_ns', 'eof_ns', 'received_ns', 'ready_ns')
        if any(type(ref[key]) is not int for key in fields) or ref['frame_id'] < 0 or not (
            0 < ref['capture_ns'] <= ref['eof_ns'] <= ref['received_ns'] <= ref['ready_ns']):
          raise ValueError('Invalid frame metadata')
    except (KeyError, TypeError, ValueError, RecursionError) as error:
      raise ValueError('Invalid input snapshot header or payload') from error
    mapped = np.memmap(stream, dtype=np.uint8, mode='r', offset=INPUT_PREFIX.size+length, shape=(payload_size,)) if payload_size else None
  try:
    frames, offset = [], 0
    for ref, (shape, count) in zip(meta['frames'], layouts, strict=True):
      pixels = None if ref.get('cached') else mapped[offset:offset+count].reshape(shape)
      frames.append(Frame(ref['view'], ref['frame_id'], ref['capture_ns'], ref['eof_ns'], ref['received_ns'],
                          ref['ready_ns'], pixels, ref['source_format']))
      offset += count
    yield Selection(meta['query_ns'], meta['selected_ns'], tuple(frames), states, meta['small_model'], gaps), meta['context']
  finally:
    if mapped is not None:
      mapped._mmap.close()


def validate_input(arrays, meta):
  shapes = {'image_frames': ((10, 1, 3, 432, 768), np.uint8), 'camera_indices': ((10,), np.int64),
            'absolute_timestamps': ((10,), np.int64), 'ego_history_xyz': ((1, 20, 3), np.float32),
            'ego_history_rot': ((1, 20, 3, 3), np.float32)}
  if set(arrays) != {*shapes, 'nav_text'}:
    raise ValueError('Incomplete model inputs')
  for name, (shape, dtype) in shapes.items():
    value = arrays[name]
    segments = name == 'image_frames' and type(value) is FrontImages
    if (value.shape != shape or value.dtype != dtype or (not segments and not value.flags.c_contiguous) or
        (np.issubdtype(dtype, np.floating) and not np.isfinite(value).all())):
      raise ValueError(f'Invalid {name}')
  if arrays['camera_indices'].tolist() != [1]*5+[6]*5 or len(meta['frames']) != 10:
    raise ValueError('Keep both complete front-camera histories')
  rolling = any('anchor_slot_ns' in ref for ref in meta['frames'])
  period = meta.get('period_ms', 0)*1_000_000
  if rolling and (period not in ROLLING_PERIODS_NS or not 0 <= meta['slot_start_ns']-meta['query_ns'] <= FRAME_TOLERANCE_NS or
                  not meta['slot_start_ns'] <= meta['selected_ns'] < meta['slot_start_ns']+period):
    raise ValueError('Invalid half-second history context')
  for i, (ref, offset) in enumerate(zip(meta['frames'], OFFSETS_NS*2, strict=True)):
    desired = meta['query_ns']+offset
    capture = ref['capture_ns']
    causal = 0 <= desired-capture <= FRAME_TOLERANCE_NS
    if rolling and i % 5 < 3:
      # Camera jitter can put a real earlier slot's anchor on either side of the nominal offset.
      source = meta['slot_start_ns']+offset
      origin, watermark = ref.get('anchor_query_ns', 0), ref.get('anchor_selected_ns', 0)
      paired = meta['frames'][(i+5) % 10]
      causal = (ref.get('anchor_slot_ns') == source and 0 <= source-origin <= FRAME_TOLERANCE_NS and
                source <= watermark < source+period and capture <= ref['ready_ns'] <= watermark and
                origin == paired.get('anchor_query_ns') and watermark == paired.get('anchor_selected_ns') and
                0 <= origin-capture <= FRAME_TOLERANCE_NS and (ref['view'] != 6 or origin == capture) and
                abs(desired-capture) <= FRAME_TOLERANCE_NS and capture < meta['query_ns'])
    elif 'anchor_slot_ns' in ref:
      causal = False
    elif rolling:
      causal = causal and capture <= ref['ready_ns'] <= meta['selected_ns'] and (i != 9 or capture == meta['query_ns'])
    if (ref['view'] != CAMERAS[i//5] or ref['requested_ns'] != desired or
        not causal or arrays['absolute_timestamps'][i] != capture//1000 or
        not 0 <= capture-ref['image_calibration_ns'] <= 1_000_000_000):
      raise ValueError('Noncausal or mismatched camera input')
  if any(np.any(np.diff(arrays['absolute_timestamps'][i:i+5]) <= 0) for i in (0, 5)):
    raise ValueError('Camera histories must contain five distinct frames')
  if arrays['nav_text'] != [meta['navigation_text']] or not isinstance(meta['navigation_text'], str) or len(meta['navigation_text']) > 1024:
    raise ValueError('Navigation differs from its source metadata')


def input_hashes(arrays):
  """Canonical NPY SHA-256 of each model input."""
  hashes = {}
  for name, value in arrays.items():
    array = np.array(value[0]) if name == 'nav_text' else value
    header = io.BytesIO()
    desc = ({'descr':array.dtype.str,'shape':array.shape,'fortran_order':False} if type(array) is FrontImages
            else np.lib.format.header_data_from_array_1_0(array))
    np.lib.format.write_array_header_1_0(header, desc)
    digest = hashlib.sha256(header.getbuffer())
    if type(array) is FrontImages:
      for owner in array:
        digest.update(owner)  # One ordered read of all ten: canonical NPY SHA-256 is not a hash of frame hashes.
    else:
      digest.update(array.reshape(-1).view(np.uint8))
    hashes[name] = digest.hexdigest()
  return hashes


def run_prediction(adapter, arrays, metadata, clock=time.monotonic_ns, *, hash_inputs=True):
  """The same synchronous inference boundary serves live input and replay.

  With hash_inputs=False the caller must add input_sha256 itself (the live worker hashes after publishing the plan).
  """
  validate_input(arrays, metadata)
  metadata = dict(metadata, input_sha256=input_hashes(arrays)) if hash_inputs else dict(metadata)
  begin = clock()
  result = adapter.predict(arrays, metadata)
  adapter.synchronize()
  completed = clock()
  if result['query_ns'] != metadata['query_ns'] or completed < begin:
    raise ValueError('Mismatched output or invalid completion clock')
  prediction = result['prediction']
  for name, shape in (('xyz', (1, 64, 3)), ('rot', (1, 64, 3, 3))):
    value = np.asarray(prediction[name])
    if value.shape != shape or not np.isfinite(value).all():
      raise ValueError(f'Incomplete or nonfinite predicted {name}')
  if not isinstance(prediction['reasoning'], str):
    raise ValueError('Missing reasoning output')
  ids = prediction.get('generated_token_ids')
  if not isinstance(ids, list) or not ids or any(type(i) is not int or i < 0 for i in ids):
    raise ValueError('Missing complete generated token IDs')
  return {'schema': 1, 'input': metadata, 'prediction': prediction, 'started_ns': begin, 'completed_ns': completed,
          'inference_ms': (completed-begin)/1e6, 'trajectory_axes': 'forward/left/up',
          'trajectory_times_s': (np.arange(1, 65)*.1).tolist(), 'small_model_axes': 'forward/right/down'}


def live_timing(selected, worker_started_ns, output_ready_ns, slot=None):
  """Current-pair receipt through validated output, including selection and RAM handoff."""
  # Historical images were copied on earlier queries; do not count their two-second lookback as current processing.
  live_started_ns = min(selected.selected_ns, selected.frames[len(selected.frames)//2-1].received_ns, selected.frames[-1].received_ns)
  if not 0 < live_started_ns <= selected.selected_ns <= worker_started_ns <= output_ready_ns or output_ready_ns < selected.query_ns:
    raise ValueError('Invalid live processing timestamps')
  result = {'live_processing_started_ns': live_started_ns, 'selection_started_ns': selected.selected_ns,
            'worker_started_ns': worker_started_ns, 'output_ready_ns': output_ready_ns,
            'live_processing_ms': (output_ready_ns-live_started_ns)/1e6,
            'worker_processing_ms': (output_ready_ns-worker_started_ns)/1e6,
            'input_age_at_selection_ms': (selected.selected_ns-selected.query_ns)/1e6,
            'input_age_ms': (output_ready_ns-selected.query_ns)/1e6}
  if slot is not None:
    result.update({key: slot[key] for key in ('slot_start_ns', 'slot_deadline_ns', 'period_ms', 'missed_slots')})
    result.update(slot_deadline_met=output_ready_ns < slot['slot_deadline_ns'],
                  selection_lateness_ms=(selected.selected_ns-slot['slot_start_ns'])/1e6,
                  completion_lateness_ms=max(0, output_ready_ns-slot['slot_deadline_ns'])/1e6)
  return result


def load_adapter(config):
  settings = dict(config['inference'])
  variant = settings.pop('variant', 'default')
  if variant == 'default':
    from openpilot.selfdrive.alpamayo.inference import ShadowAdapter
  elif variant == 'selected':
    allowed = {'checkpoint', 'fp8_cache', 'prefill_calibration', 'initial_noise', 'device_timeout_ms', 'native_fresh_patches'}
    if extra := settings.keys()-allowed:
      raise ValueError(f'Unexpected selected inference fields: {sorted(extra)}')
    settings['cache_dir'] = str(Path(config['cache']).resolve())
    from openpilot.selfdrive.alpamayo.selected import ShadowAdapter
  else:
    raise ValueError(f'Unknown Alpamayo variant: {variant}')
  return ShadowAdapter(settings)


def worker(config, directory, handoff):
  from openpilot.selfdrive.alpamayo.frontend import Preparer, Priming, Unavailable
  from openpilot.selfdrive.alpamayo.selected.policy import PromptTooLong
  from openpilot.selfdrive.alpamayo.shadowd import planner_mode
  from openpilot.selfdrive.alpamayo.planner import packet
  root = Path(directory)
  handoff = Path(handoff)
  if not input_storage_ok(handoff):
    raise RuntimeError('Shadow worker requires a RAM-backed input slot')
  load_started = time.monotonic_ns()
  sampler = None
  if config['inference'].get('variant', 'default') == 'selected':
    from functools import partial
    from openpilot.selfdrive.alpamayo.geometry import lens_rays
    from openpilot.selfdrive.alpamayo.selected.camera import calibrated_image, library
    # The lens grid and sampler binary need no exposure or live calibration. Build them before the first timed camera pair.
    geometry_json = json.dumps(config['geometry'], sort_keys=True)
    for camera in CAMERAS:
      lens_rays(camera, geometry_json)
    library(str(config['cache']))
    sampler = partial(calibrated_image, cache_dir=config['cache'])
  segmented = config.get('retained_cpu_inputs', False)
  period_ms = config.get('period_ms', 500)
  if type(segmented) is not bool or (segmented and (sampler is None or period_ms not in (0, 250, 500))):
    raise ValueError('retained_cpu_inputs requires the selected free-running, 250 or 500 ms worker')
  preparer = Preparer(config['geometry'], sampler=sampler, segmented=segmented, free=period_ms == 0,
                      period_ns=period_ms*1_000_000 if period_ms in (250, 500) else 500_000_000)
  # Selected inference removes ALPAMAYO_* tuning variables during initialization.
  publish_plans = planner_mode()
  adapter = load_adapter(config)
  if segmented and hasattr(adapter, 'encode_ahead'):
    preparer.on_current = adapter.encode_ahead
  if hasattr(adapter, 'warmup'):
    adapter.warmup()  # Compile every steady-state graph shape before ready; no live slot pays a mid-drive compile.
  # Per-slot CPU work (projection, patch packing, dispatch) takes ~2.5x longer on the A55 cores 0-3.
  # Run this thread and the threads it creates on big cores 4-5 whenever they are online (onroad policy).
  def pin_big_cores():
    big = {4, 5} if all(Path(f'/sys/devices/system/cpu/cpu{c}/online').read_text().strip() == '1' for c in (4, 5)) else set()
    if big and os.sched_getaffinity(0) != big:
      os.sched_setaffinity(0, big)
      if hasattr(adapter, 'pin_threads'):
        adapter.pin_threads(big)
  def encode_early(path):
    # Project and encode the predicted -0.1 s pair before its slot opens. Failures only lose the shortcut.
    try:
      pin_big_cores()
      with load_selection(path) as (selected, context):
        if segmented and hasattr(adapter, 'prime') and time.monotonic_ns() < context['slot_start_ns']:
          views = preparer.early(selected)
          for pair in (views[:2], views[2:]):
            if pair:
              adapter.prime(pair, context['slot_start_ns']+400_000_000)
    except Exception as error:  # noqa: BLE001
      print(f'Early encoding skipped: {type(error).__name__}: {error}', file=sys.stderr, flush=True)
    finally:
      path.unlink(missing_ok=True)

  writer, count = PredictionWriter(), 0
  stopping = False
  def stop(*_):
    nonlocal stopping
    stopping = True  # Normal supervisor SIGTERM must drain the admitted write.
  previous = signal.signal(signal.SIGTERM, stop)
  try:
    atomic_json(root/'runtime.json', dict(adapter.manifest, geometry=config['geometry'], history_offsets_ns=OFFSETS_NS, period_ms=config.get('period_ms', 500),
                                         retained_cpu_inputs=segmented, source_sof='Original VisionIPC SOF; estimated from EOF on v0'))
    ready = time.monotonic_ns()
    atomic_json(root/'ready.json', {'ready': True, 'adapter_load_started_ns': load_started, 'worker_ready_ns': ready,
                                    'adapter_startup_ms': getattr(adapter, 'startup_ms', {})})
    while not stopping:
      writer.check()
      try:
        (handoff/'pending.bin').replace(handoff/'working.bin')
        pin_big_cores()
      except FileNotFoundError:
        if (handoff/'early.bin').exists():
          encode_early(handoff/'early.bin')
        else:
          time.sleep(.001)
        continue
      try:
        begin = time.monotonic_ns()
        with load_selection(handoff/'working.bin') as (selected, context):
          admitted = time.monotonic_ns()
          if not 0 <= admitted-selected.query_ns <= 500_000_000 or admitted >= context.get('slot_deadline_ns', admitted+1):
            raise Unavailable('Input expired before preparation')
          if context.get('anchor_only'):
            preparer.rolling_images(selected, context['slot_start_ns'], anchor_only=True)  # Raises Priming; no prediction for this slot.
          arrays, meta = preparer.prepare(selected, context['slot_start_ns'] if len(selected.frames) == 4 else None)
          meta.update(context)
          meta['prepared_ns'] = time.monotonic_ns()
          if meta['prepared_ns'] >= context.get('slot_deadline_ns', meta['prepared_ns']+1):
            raise Unavailable('Input expired during preparation')
          result = run_prediction(adapter, arrays, meta, hash_inputs=False)
          result.update(startup=count < 5, **live_timing(selected, begin, time.monotonic_ns(), context if 'slot_start_ns' in context else None))
          writer.check(wait=True)  # A failed previous write must stop this plan before publication.
          if stopping:
            return
          if publish_plans:
            temporary = handoff/'plan.tmp'
            temporary.write_text(json.dumps(packet(result), allow_nan=False))
            temporary.replace(handoff/'plan.json')
          # Audit hashes cover the exact inputs but are not needed to publish the plan; the source mapping is still open.
          result['input']['input_sha256'] = input_hashes(arrays)
          writer.submit(root/f'{selected.query_ns}.json', result)
          count += 1
      except (Unavailable, PromptTooLong) as error:
        if isinstance(error, Priming) and sampler is not None and not adapter.prime(error.anchors, context['slot_deadline_ns']):
          error = Unavailable('Front-camera feature priming missed its slot deadline')
        if not isinstance(error, Priming):
          preparer.anchors.clear()
        rejected = time.monotonic_ns()
        writer.check(wait=True)  # Serialize disk reserve checks for predictions and status.
        atomic_json(root/'unavailable.json', {'query_ns': selected.query_ns, 'selected_ns': selected.selected_ns,
                                            'slot_start_ns': context.get('slot_start_ns'), 'slot_deadline_ns': context.get('slot_deadline_ns'),
                                            'rejected_ns': rejected, 'type': type(error).__name__, 'reason': str(error)[:1024]})
        (handoff/'skipped').touch()
        continue
      finally:
        (handoff/'working.bin').unlink(missing_ok=True)
  finally:
    try:
      writer.close()
    finally:
      signal.signal(signal.SIGTERM, previous)
      adapter.close()
