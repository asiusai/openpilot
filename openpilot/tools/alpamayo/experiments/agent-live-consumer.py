"""Live consumer for explicitly initialized benchmark components.

No checkpoint loader, sample reader, model implementation, publisher, or warmup.
Components become exclusive to this consumer until close(). The caller still owns
device health/storage guards and must initialize the selected model exactly once.
"""
from collections import OrderedDict, deque
from dataclasses import dataclass
import functools
import hashlib
import inspect
import json
import marshal
from pathlib import Path
import threading
import time

import numpy as np
import torch


def fingerprint(value):
  """Stable metadata encoding; never serialize weights or array contents."""
  if isinstance(value, np.ndarray):
    return dict(shape=list(value.shape), dtype=str(value.dtype), sha256=hashlib.sha256(value.tobytes()).hexdigest())
  if isinstance(value, bytes):
    return {'bytes_sha256': hashlib.sha256(value).hexdigest()}
  if isinstance(value, dict):
    return {str(k): fingerprint(v) for k, v in sorted(value.items(), key=lambda pair: str(pair[0]))}
  if isinstance(value, (tuple, list)):
    return [fingerprint(v) for v in value]
  if isinstance(value, np.generic):
    return value.item()
  if value is None or isinstance(value, (str, bool, int, float)):
    return value
  raise TypeError(f'Unsupported provenance value: {type(value).__name__}')


def digest(value):
  return hashlib.sha256(json.dumps(fingerprint(value), sort_keys=True, allow_nan=False).encode()).hexdigest()


def callable_sources(*values):
  """Record actual function code, including instance hooks and wrapped closures."""
  pending, visited, result, helpers = list(values), set(), [], []
  while pending:
    obj = pending.pop()
    if id(obj) in visited:
      continue
    visited.add(id(obj))
    if len(visited) > 192:
      raise ValueError('Injected callable graph exceeds the provenance bound')
    if isinstance(obj, functools.partial):
      pending.append(obj.func)
      continue
    if inspect.ismethod(obj):
      obj = obj.__func__
    if not inspect.isfunction(obj):
      continue
    path = Path(obj.__code__.co_filename)
    result.append(dict(name=obj.__qualname__, path=str(path),
                       source_sha256=hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None,
                       code_sha256=hashlib.sha256(marshal.dumps(obj.__code__)).hexdigest()))
    if hasattr(obj, '__wrapped__'):
      pending.append(obj.__wrapped__)
    for cell in obj.__closure__ or ():
      try:
        value = cell.cell_contents
      except ValueError:
        continue
      if inspect.isfunction(value) or inspect.ismethod(value) or isinstance(value, functools.partial):
        pending.append(value)
      elif type(value).__name__ == 'StartupWarmup' and callable(getattr(value, 'active', None)):
        if all(value is not previous for previous in helpers):
          helpers.append(value)
  return sorted(result, key=lambda row: (row['path'], row['name'], row['code_sha256'])), helpers


def live_before_diffusion():
  """The live bridge selects inputs when idle; no offline prefetch callback."""


@dataclass(frozen=True)
class Budgets:
  records: int = 64
  vision_positions: int = 4
  selective_graphs: int = 128
  vision_features: int = 144
  vision_encoders: int = 16
  vision_stitchers: int = 8
  decoder_graphs: int = 32
  verifier_graphs: int = 16
  selector_graphs: int = 8
  trajectory_graphs: int = 64


class Consumer:
  def __init__(self, *, adapter, model, runner, settings, config, input_class, vision_cache,
               pad_inputs, pad_length, initializer_provenance, decoder, tokenizer=None, inputs=None,
               speculation_policy=None, speculation_draft=None, startup_helpers=(), seed=0,
               budgets=Budgets(), clock=time.monotonic_ns):
    if any(value <= 0 for value in vars(budgets).values()):
      raise ValueError('Positive ownership budgets are required')
    if not isinstance(pad_length, int) or pad_length <= 0 or not initializer_provenance:
      raise ValueError('Declare fixed prompt padding and the initialized model provenance')
    if config is not adapter.config or getattr(runner, '__self__', None) is not decoder or decoder.model is not model:
      raise ValueError('Use the existing adapter configuration and bound decoder.run for this model')
    if getattr(vision_cache, 'model', None) is not model:
      raise ValueError('The exact vision cache must belong to the same initialized model')
    if settings.vision_batch != 1 or not settings.greedy or settings.diffusion_steps != 3:
      raise ValueError('This consumer preserves single-image encoding, greedy reasoning, and three expert steps')
    if not 6 <= settings.max_tokens <= 256:
      raise ValueError('Reasoning capacity must remain within the declared live bound')
    if (getattr(adapter, 'camera_profile', None) != 'nvidia' or not adapter.allow_missing_cameras or
        adapter.history_pixel_scale != 1 or adapter.history_feature_scale != 1 or adapter.front_history_indices is not None or
        adapter.narrow_crop != 1 or adapter.wide_equidistant_focal != 0):
      raise ValueError('Use the complete calibrated two-front input adapter without extra crops or reduced history')
    if not callable(getattr(input_class, 'arrays', None)) or not callable(getattr(input_class, 'signature', None)):
      raise ValueError('Inject the current CompactInputs implementation, including installed upload hooks')
    self.adapter, self.model, self.runner, self.settings, self.config = adapter, model, runner, settings, config
    self.input_class, self.vision_cache, self.decoder = input_class, vision_cache, decoder
    self.tokenizer = adapter.tokenizer if tokenizer is None else tokenizer
    self.pad_inputs, self.pad_length, self.seed = pad_inputs, pad_length, seed
    self.inputs, self.budgets, self.clock = inputs, budgets, clock
    self.policy, self.draft = speculation_policy, speculation_draft
    candidate = hasattr(decoder, 'prefill_draft_records') or hasattr(model.padded_prefill, 'last_targets')
    if candidate != (self.policy is not None):
      raise ValueError('Declare target-verified prefill explicitly and inject its actual policy and padding hook')
    self.mode = 'target-verified-prefill' if candidate else 'ordinary-validated-runner'
    if candidate:
      self.draft = self.policy.validate_draft(self.draft, config.traj_ids['future_start'])
      if not hasattr(decoder, 'prefill_draft_records') or not hasattr(model.padded_prefill, 'last_targets'):
        raise ValueError('Both current speculative prefill and decoder hooks must already be installed')
      if getattr(model.padded_prefill, 'audit', False):
        raise ValueError('Speculation audit is diagnostic and is not supported by this live consumer')
    elif self.draft is not None:
      raise ValueError('An ordinary runner cannot consume a speculative draft')
    functions = (adapter.prepare, adapter.native_inputs, input_class.__init__, input_class.arrays,
                 input_class.signature, input_class.update, input_class.upload, vision_cache.prepare,
                 vision_cache.set_layout, runner, decoder.generate, decoder.before_diffusion,
                 pad_inputs, type(model.padded_prefill).__call__, model.synchronize)
    sources, discovered = callable_sources(*functions)
    self.startup_helpers = tuple({id(h): h for h in (*startup_helpers, *discovered)}.values())
    self._guard_startup()
    self._settings = digest(vars(settings))
    self._config = digest(config.to_dict())
    self._minimum_cache_capacity = getattr(model, 'minimum_cache_capacity', 0)
    self.provenance = dict(mode=self.mode, source=self._source(), callables=sources,
                           initializer=fingerprint(initializer_provenance), settings=fingerprint(vars(settings)),
                           config_sha256=self._config, pad_length=pad_length, seed=seed,
                           speculative_draft=self.draft, budgets=vars(budgets),
                           startup_warming=False, offline_prefetch_replaced=True)
    self.provenance_sha256 = digest(self.provenance)
    self.records = deque(maxlen=budgets.records)
    self.position_hashes = OrderedDict()
    self._lock = threading.Lock()
    self._pending = None
    self._owner_thread = None
    self._failed = self._closed = False
    self.last_query_ns = 0
    self.constructed_inputs = self.reused_inputs = 0
    # Explicit handover barrier. Do not replace a live upload owner until all
    # previous work is complete. This initialization is outside query timing.
    model.synchronize()
    decoder.before_diffusion = live_before_diffusion
    self._cache_identity = self._kv_identity()
    self._check_bounds()

  @staticmethod
  def _source():
    path = Path(__file__)
    return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())

  def _kv_identity(self):
    cache = getattr(self.model, 'cache', None)
    return None if cache is None else tuple(id(getattr(getattr(value, 'uop', None), 'base', value)) for value in cache)

  def _guard_startup(self):
    if getattr(self.decoder, 'warmup_graphs', False) or getattr(self.vision_cache, 'warmup_graphs', False):
      raise RuntimeError('Finish startup warming before binding or using live inputs')
    if any(helper.active() for helper in self.startup_helpers):
      raise RuntimeError('Active StartupWarmup can overwrite speculative KV; live execution is blocked')

  def _check_bounds(self):
    checks = [(self.vision_cache, 'cache', self.budgets.vision_features),
              (self.vision_cache, 'encoders', self.budgets.vision_encoders),
              (self.vision_cache, 'stitchers', self.budgets.vision_stitchers),
              (self.decoder, 'jits', self.budgets.decoder_graphs),
              (self.decoder, 'verifiers', self.budgets.verifier_graphs),
              (self.decoder, 'select_jits', self.budgets.selector_graphs),
              (self.model, 'trajectory_jits', self.budgets.trajectory_graphs),
              (self.decoder, 'history', 64), (self.decoder, 'prefill_draft_old_history', 64)]
    if self.inputs is not None:
      checks.extend(((self.inputs, 'selective_jits', self.budgets.selective_graphs),
                     (self.inputs, 'positions', self.budgets.vision_positions)))
    counts = {}
    for owner, field, maximum in checks:
      count = len(getattr(owner, field, ()))
      counts[f'{type(owner).__name__}.{field}'] = count
      if count > maximum:
        raise RuntimeError(f'{type(owner).__name__}.{field} exceeds its declared live bound: {count}/{maximum}')
    return counts

  def _data(self, arrays, meta, now):
    keys = {'image_frames', 'camera_indices', 'absolute_timestamps', 'ego_history_xyz', 'ego_history_rot', 'nav_text'}
    if set(arrays) != keys:
      raise ValueError('Consume the complete in-memory live bridge arrays only')
    q = meta['query_ns']
    if not isinstance(q, int) or not self.last_query_ns < q <= meta['selected_ns'] <= meta['prepared_ns'] <= now:
      raise ValueError('Query identity or live monotonic timing is stale or inconsistent')
    formats = {'image_frames': ((10, 1, 3, 432, 768), np.uint8), 'camera_indices': ((10,), np.int64),
               'absolute_timestamps': ((10,), np.int64), 'ego_history_xyz': ((1, 20, 3), np.float32),
               'ego_history_rot': ((1, 20, 3, 3), np.float32)}
    for key, (shape, dtype) in formats.items():
      value = arrays[key]
      if not isinstance(value, np.ndarray) or value.shape != shape or value.dtype != dtype or not value.flags.c_contiguous:
        raise ValueError(f'Invalid complete live array: {key}')
      if key.startswith('ego_') and not np.isfinite(value).all():
        raise ValueError('Nonfinite ego history')
    if arrays['camera_indices'].tolist() != [1]*5+[6]*5 or len(meta['frames']) != 10:
      raise ValueError('Both full front histories must retain camera-major order')
    offsets = (-2_000_000_000, -1_000_000_000, -500_000_000, -100_000_000, 0)*2
    for i, (ref, offset) in enumerate(zip(meta['frames'], offsets, strict=True)):
      stamp, desired = ref['capture_ns'], q+offset
      if (ref['view'] != int(arrays['camera_indices'][i]) or ref['requested_ns'] != desired or
          not 0 <= desired-stamp <= 150_000_000 or arrays['absolute_timestamps'][i] != stamp//1000 or
          not 0 <= stamp-ref['image_calibration_ns'] <= 1_000_000_000):
        raise ValueError('Frame timestamp, calibration, or causal history differs from live metadata')
    if meta['frames'][-1]['capture_ns'] != q:
      raise ValueError('The query must retain its actual current road capture timestamp')
    if any(np.any(np.diff(arrays['absolute_timestamps'][start:start+5]) <= 0) for start in (0, 5)):
      raise ValueError('Every front history requires five distinct increasing capture timestamps')
    nav = arrays['nav_text']
    if not isinstance(nav, (tuple, list)) or len(nav) != 1 or not isinstance(nav[0], str) or not 0 < len(nav[0]) <= 1024:
      raise ValueError('Explicit bounded live navigation text is required')
    if nav[0] != meta['navigation_text']:
      raise ValueError('Navigation differs between the live arrays and their metadata')
    return {k: [nav[0]] if k == 'nav_text' else torch.from_numpy(v if v.flags.writeable else v.copy()) for k, v in arrays.items()}

  def _bind_inputs(self, prepared, packed, upload):
    signature = self.input_class.signature(self.settings, self.config, prepared, packed)
    layout = (str(self.config._name_or_path), prepared['tokenized_data']['image_grid_thw'].numpy().tobytes())
    position_digest = digest(packed[3])
    if layout in self.position_hashes and self.position_hashes[layout] != position_digest:
      raise ValueError('Fixed vision-position contents changed under an existing cached layout')
    self.position_hashes[layout] = position_digest
    self.position_hashes.move_to_end(layout)
    while len(self.position_hashes) > self.budgets.vision_positions:
      self.position_hashes.popitem(last=False)
    reusable = self.inputs is not None and signature == self.inputs._signature and len(upload) == len(self.inputs.cpu)
    if reusable:
      # CompactInputs.update checks shape; also check dtype before its raw-buffer
      # write. Never copy bytes into a merely shape-compatible old allocation.
      reusable = all(
          tuple(array.shape) == old.shape and array.dtype.itemsize == old.dtype.itemsize and
          np.dtype(array.dtype).name == self._cpu_dtype(old) for array, old in zip(upload, self.inputs.cpu, strict=True))
    reused = bool(reusable and self.inputs.update(self.settings, self.config, prepared, packed, arrays=upload))
    if not reused:
      self.inputs = None  # The previous query already passed its completion barrier.
      self.inputs = self.input_class(self.settings, self.config, prepared, packed, arrays=upload)
      self.constructed_inputs += 1
    else:
      self.reused_inputs += 1
    self.inputs.selective_cache = self.vision_cache
    return reused, digest((signature, [(a.shape, str(a.dtype)) for a in upload]))

  @staticmethod
  def _cpu_dtype(value):
    # These CPU buffers store byte pixels, BF16 bit patterns, IDs, and FP32 noise.
    if isinstance(value.dtype, np.dtype):
      return value.dtype.name
    fmt = getattr(value.dtype, 'fmt', None)
    return np.dtype(fmt).name if fmt is not None else None

  def predict(self, arrays, metadata):
    if self._closed or self._failed:
      raise RuntimeError('Consumer is closed or failed; reinitialize before another query')
    if not self._lock.acquire(blocking=False):
      raise RuntimeError('One live query is still active; call synchronize before reusing buffers')
    began = self.clock()
    record = dict(query_ns=metadata.get('query_ns'), started_ns=began, completed=False, mode=self.mode,
                  provenance_sha256=self.provenance_sha256, stages_ms={}, stage_timing_kind='host intervals; completion barrier measured separately')
    self._pending = record
    self._owner_thread = threading.get_ident()
    metadata['live_consumer'] = record
    try:
      self._guard_startup()
      if digest(vars(self.settings)) != self._settings or digest(self.config.to_dict()) != self._config:
        raise ValueError('Initialized settings/configuration changed during consumer ownership')
      if getattr(self.model, 'minimum_cache_capacity', 0) != self._minimum_cache_capacity:
        raise ValueError('Initialized KV capacity policy changed')
      if self._kv_identity() != self._cache_identity:
        raise ValueError('KV allocation changed outside this consumer; captured decoder graphs need reinitialization')
      data = self._data(arrays, metadata, began)
      validated = self.clock()
      config, tokenizer, prepared = self.adapter.prepare(data)
      if config is not self.config or tokenizer is not self.adapter.tokenizer:
        raise ValueError('The adapter replaced its initialized configuration or tokenizer')
      packed = self.adapter.native_inputs(prepared)
      adapted = self.clock()
      actual = int(packed[0].shape[1])
      prepared, packed = self.pad_inputs(prepared, packed, self.pad_length)
      if prepared.get('actual_prompt_length') != actual or packed[0].shape != (1, self.pad_length):
        raise ValueError('The injected canonical padding hook violated its actual-prefix contract')
      if self.policy is None and 'prefill_draft_eligible' in prepared:
        raise ValueError('Speculative padding cannot be used with the ordinary runner')
      if self.policy is not None:
        eligible = self.pad_length-actual >= len(self.draft)
        if prepared.get('prefill_draft_eligible') is not eligible:
          raise ValueError('Use the installed speculative pad_inputs hook, including its eligibility metadata')
        if eligible and tuple(packed[0][0, actual:actual+len(self.draft)]) != self.draft:
          raise ValueError('Padded draft differs from the current target-verified decoder contract')
      grids = prepared['tokenized_data']['image_grid_thw'].numpy()
      if grids.shape != (10, 3) or not np.all(grids == (1, 20, 36)) or 'embedding_grid_thw' in prepared['tokenized_data']:
        raise ValueError('Keep all ten complete 720-patch front images and original feature grids')
      upload = self.input_class.arrays(self.settings, prepared, packed)
      packed_at = self.clock()
      # Preserve the canonical SHA-256 pixel-content keys and layout validation.
      self.vision_cache.set_layout(prepared)
      self.vision_cache.prepare(prepared)
      cache_at = self.clock()
      reused, signature = self._bind_inputs(prepared, packed, upload)
      staged = self.clock()
      self._guard_startup()
      if self.decoder.before_diffusion is not live_before_diffusion:
        raise RuntimeError('Offline or external before_diffusion callback was installed during live ownership')
      self._check_bounds()
      prior_spec = self.decoder.prefill_draft_records[-1] if self.policy is not None and self.decoder.prefill_draft_records else None
      prediction = self.runner(self.model, self.settings, self.config, self.tokenizer, prepared, packed,
                               seed=self.seed, inputs=self.inputs)
      returned = self.clock()
      xyz, rot = np.asarray(prediction['xyz']), np.asarray(prediction['rot'])
      if xyz.shape != (1, 64, 3) or rot.shape != (1, 64, 3, 3) or not np.isfinite(xyz).all() or not np.isfinite(rot).all():
        raise ValueError('The existing runner did not return a complete decoded trajectory')
      if self.policy is not None:
        current = self.decoder.prefill_draft_records[-1]
        if current is prior_spec or current['actual_prompt'] != actual or current['generated'] != prediction['generated_token_ids']:
          raise ValueError('Speculative verification record does not belong to the current runner output')
        if current.get('accepted') and not self.policy.accept_targets(np.asarray(current['targets']), self.draft,
                                                                     self.config.traj_ids['future_start'], current['all_targets_finite']):
          raise ValueError('The speculative hook accepted an unverified draft')
        record['speculation'] = dict(eligible=current['eligible'], accepted=current.get('accepted', False), actual_prompt=actual)
      record['stages_ms'] = dict(validate_and_wrap=(validated-began)/1e6, adapter_and_history_tokens=(adapted-validated)/1e6,
                                 pad_and_upload_arrays=(packed_at-adapted)/1e6, exact_vision_keys=(cache_at-packed_at)/1e6,
                                 input_buffers=(staged-cache_at)/1e6, runner_host=(returned-staged)/1e6)
      record.update(runner_returned_ns=returned, query_age_at_consumer_start_ms=(began-metadata['query_ns'])/1e6,
                    latest_camera_capture_ns=[metadata['frames'][i]['capture_ns'] for i in (4, 9)],
                    actual_prompt_tokens=actual, padded_prompt_tokens=self.pad_length, input_reused=reused,
                    input_signature_sha256=signature, upload_array_bytes=sum(a.nbytes for a in upload),
                    source_capture_timestamps_ns=[ref['capture_ns'] for ref in metadata['frames']],
                    runner_stage_timing_kind=prediction.get('stage_timing_kind', 'existing synchronized runner stages'))
      # Do not copy, rewrite, or add fields inside the canonical prediction.
      return dict(query_ns=metadata['query_ns'], prediction=prediction, live_consumer=record)
    except BaseException as error:
      self._failed = True
      record['error_type'] = type(error).__name__
      try:
        self.model.synchronize()
      except BaseException as sync_error:
        error.add_note(f'Completion synchronization also failed: {type(sync_error).__name__}')
      self.records.append(record)
      self._pending = None
      self._owner_thread = None
      self._lock.release()
      raise

  def synchronize(self):
    if self._pending is None:
      if self._failed:
        raise RuntimeError('Consumer failed; completion is not established')
      return
    record = self._pending
    if self._owner_thread != threading.get_ident() or 'runner_returned_ns' not in record:
      raise RuntimeError('Complete predict on its owning thread before requesting synchronization')
    begin = self.clock()
    try:
      self.model.synchronize()
      synced = self.clock()
      identity = self._kv_identity()
      if self._cache_identity is not None and identity != self._cache_identity:
        raise ValueError('Runner reallocated existing KV; reject this result and reinitialize captured graphs')
      self._cache_identity = identity
      # Only diagnostic lists are truncated. Keep every canonical text history,
      # feature entry, graph, and KV byte under its existing ownership contract.
      for field in ('statistics', 'prefill_draft_records'):
        values = getattr(self.decoder, field, None)
        if values is not None:
          del values[:-self.budgets.records]
      counts = self._check_bounds()
      done = self.clock()
      record.update(completed=True, synchronized_ns=synced, completed_ns=done,
                    completion_sync_ms=(synced-begin)/1e6, post_sync_checks_ms=(done-synced)/1e6,
                    consumer_to_completion_ms=(done-record['started_ns'])/1e6,
                    query_to_completion_ms=(done-record['query_ns'])/1e6,
                    current_camera_age_at_completion_ms=[(done-stamp)/1e6 for stamp in record['latest_camera_capture_ns']],
                    cache_entries=counts, owned_input_objects=int(self.inputs is not None))
      self.last_query_ns = record['query_ns']
    except BaseException as error:
      self._failed = True
      record['error_type'] = type(error).__name__
      raise
    finally:
      self.records.append(record)
      self._pending = None
      self._owner_thread = None
      self._lock.release()

  def close(self):
    if self._pending is not None:
      self.synchronize()
    # A synchronization failure retains the input owner until explicit caller
    # recovery. Releasing potentially active upload buffers would be unsafe.
    if not self._failed:
      self.inputs = None
    self._closed = True
