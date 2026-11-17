"""Initialization only. The caller supplies exactly one current in-memory input.

No sample loading, token forcing, generation replacement, or background work.
Unseen layouts fail closed after handoff. GPU qualification remains mandatory.
"""
from collections import OrderedDict
from contextlib import contextmanager
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import time


def load(path):
  spec = importlib.util.spec_from_file_location(Path(path).stem.replace('-', '_'), path)
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module


def sha(path):
  return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class ClosedGraphs(OrderedDict):
  """Reject new JIT keys before their first invocation can compile anything."""
  def __init__(self, values):
    self.sealed = False
    super().__init__(values)
    self.sealed = True

  def __setitem__(self, key, value):
    if self.sealed and (key not in self or self[key] is not value):
      raise RuntimeError('Unwarmed graph/layout: stop and reinitialize outside live capture')
    super().__setitem__(key, value)


@contextmanager
def decoder_state(decoder):
  """Restore histories and diagnostics, including replacement of list attributes."""
  fields = ('history', 'previous', 'statistics', 'prefill_draft_records', 'prefill_draft_old_history')
  saved = {key: (getattr(decoder, key), copy.deepcopy(getattr(decoder, key)))
           for key in fields if hasattr(decoder, key)}
  callback = decoder.before_diffusion
  decoder.before_diffusion = lambda: None  # No offline prefetch during initialization.
  try:
    yield
  finally:
    for key, (original, value) in saved.items():
      if isinstance(original, list):
        original[:] = value
        setattr(decoder, key, original)
      else:
        setattr(decoder, key, value)
    decoder.before_diffusion = callback


def restore_prefill(cache, snapshots, prefill, *, identity, equal, copy_into):
  """Restore the same allocations, then regenerate actual logits and draft payload."""
  before = identity()
  for current, saved in zip(cache, snapshots, strict=True):
    copy_into(current, saved)
  logits = prefill()
  if identity() != before:
    raise AssertionError('Prefill replaced allocations held by warmed decoder graphs')
  for current, saved in zip(cache, snapshots, strict=True):
    equal(current, saved)
  return logits


def input_class(base, helper, state):
  class CurrentInputs(base):
    @staticmethod
    def signature(args, config, prepared, packed):
      if not prepared.get('prefill_draft_eligible'):
        raise ValueError('Unwarmed non-speculative prefill layout; initialize separately')
      return base.signature(args, config, prepared, packed)

    def __init__(self, *args, **kwargs):
      if state['ready']:
        raise RuntimeError('Current input owner/layout changed; stop before new allocation or compilation')
      super().__init__(*args, **kwargs)
  helper.StartupWarmup(lambda: False).patch_inputs(CurrentInputs)
  return CurrentInputs


def initialize(components, args, arrays, *, source_root, memory_limit=7*1024**3, stage_limit_seconds=900):
  """Finish all work before runtime.bind_components. Never call a live Consumer."""
  import numpy as np
  import torch
  from tinygrad import Tensor, TinyJit, dtypes
  from tinygrad.helpers import GlobalCounters

  c, model, decoder, vision = components, components.model, components.decoder, components.vision_cache
  if (getattr(c, '_live_bound', False) or getattr(c, '_startup_spec', None) is not None or c.inputs is not None or
      c.startup is not None or any(c.startup_active) or decoder.warmup_graphs or vision.warmup_graphs):
    raise ValueError('Require one fresh, unbound model with all old startup hooks disabled')
  if (args.vision_calibration is not None or args.pad_prompt_to != 2048 or args.vision_batch != 1 or
      args.max_tokens < 64 or args.max_tokens > 256 or c.settings.diffusion_steps != 3 or
      not decoder.compact_io or not vision.grouped or vision.group_limit != 2 or vision.verify or
      getattr(decoder, 'fallback_block', None) != 6 or c.settings.initial_noise is None or model.padded_prefill.audit):
    raise ValueError('Require selected dynamic vision, pad2048, serial6, compact lookup, fixed noise, and three Euler steps')
  if not hasattr(decoder, 'prefill_draft_records') or 'generate' not in vars(decoder):
    raise ValueError('Require the actual installed nominal-prefill instance decoder')
  if any(getattr(decoder, k, ()) for k in ('history', 'statistics', 'prefill_draft_records')):
    raise ValueError('Initialization cannot consume previous prediction history')
  if not 0 < memory_limit <= 7*1024**3 or not 0 < stage_limit_seconds <= 900:
    raise ValueError('Startup memory/time bounds exceed the reviewed limits')
  root = Path(source_root)
  helper = load(root/'experiments/agent-pipeline-warmup.py')
  begun, before_bytes = time.monotonic(), GlobalCounters.mem_used
  report = dict(schema=1, ready=False, initialization_only=True, current_inputs=1, future_inputs=0,
    predictions_published=0, forced_tokens=0, memory_before_bytes=before_bytes, memory_limit_bytes=memory_limit,
    stage_limit_seconds=stage_limit_seconds, stages=[], graph_counts={},
    source_sha256={p.name: sha(p) for p in (Path(__file__), root/'model.py',
      *(root/'experiments'/name for name in ('agent-pipeline-warmup.py', 'agent-speculation-prefill.py',
      'compact-inputs.py', 'vision-cache.py', 'lookup-decode.py', 'draft-decode.py', 'block-decode.py')))})
  c._startup_spec = report
  state = {'ready': False}
  original_class, original_encode = c.input_class, model.encode
  original_inputs = c.inputs
  peak = before_bytes

  def stage(name):
    nonlocal peak
    model.synchronize()
    elapsed = time.monotonic()-begun
    peak = max(peak, GlobalCounters.mem_used)
    report['stages'].append(dict(name=name, elapsed_ms=elapsed*1000, live_allocation_bytes=GlobalCounters.mem_used))
    print(json.dumps({'startup_spec_stage': report['stages'][-1]}), flush=True)
    if elapsed > stage_limit_seconds or peak > memory_limit:
      raise RuntimeError('Startup bound exceeded; do not bind this model')

  def identity():
    return tuple(id(t._buffer().base) for t in model.cache)

  def exact(left, right):
    if left.shape != right.shape or left.dtype != right.dtype:
      raise AssertionError('KV/logit/payload shape or dtype changed')
    # Integer bitcasts retain signed zero and every payload bit. No float tolerance.
    dtype = {1: dtypes.uint8, 2: dtypes.uint16, 4: dtypes.uint32}[left.dtype.itemsize]
    if (left.bitcast(dtype) != right.bitcast(dtype)).any().item():
      raise AssertionError('Startup changed actual prefill/KV/payload bits')

  try:
    c.input_class = input_class(original_class, helper, state)
    data = {key: list(value) if key == 'nav_text' else torch.from_numpy(value.copy()) for key, value in arrays.items()}
    config, tokenizer, prepared = c.adapter.prepare(data)
    if config is not c.config or tokenizer is not c.tokenizer:
      raise ValueError('Current adapter replaced its initialized owners')
    packed = c.adapter.native_inputs(prepared)
    prepared, packed = c.pad_inputs(prepared, packed, args.pad_prompt_to)
    if not prepared.get('prefill_draft_eligible'):
      raise ValueError('Current prompt has no complete nominal suffix slots; unsupported startup layout')
    if not np.all(prepared['tokenized_data']['image_grid_thw'].numpy() == (1, 20, 36)):
      raise ValueError('Only the selected complete front image grid is supported')
    vision.prepare(prepared)
    owner = c.input_class(c.settings, config, prepared, packed)
    owner.selective_cache = vision
    c.inputs = owner
    if len(owner.batches) != 10 or len({b-a for a,b,_ in owner.batches}) != 1:
      raise ValueError('Require ten current full-grid images')
    # Link every count, not just the two TinyJit capture calls. Destinations remain symbolic.
    for count in range(11):
      jit, values = helper.selection_call(owner, tuple(range(count)))
      for _ in range(3):
        Tensor.realize(*jit(*values))
      stage('upload_count_'+str(count))
    pixels, positions, *_ = owner.upload(c.settings.initial_noise)
    # Both image-group counts use views from this same current input only.
    for count in (1, 2):
      pairs = [(vision.image_view(t,a,b) if vision.views else t[a:b].unsqueeze(0).clone())
               for a,b,_ in owner.batches[:count] for t in (pixels, positions)]
      Tensor.realize(*pairs)
      key = tuple(t.shape for t in pairs)
      if key not in vision.encoders:
        vision.encoders[key] = TinyJit(vision.batch_encoder or
          (lambda *values: tuple(model._vision(p,v) for p,v in zip(values[::2], values[1::2]))))
      for _ in range(3):
        Tensor.realize(*vision.encoders[key](*pairs))
      stage('image_group_'+str(count))
    predictions = []
    # The original runner warms trajectory and CPU action decoding. Each call sees
    # original empty history; nothing is accepted into the live history.
    for index in range(3):
      vision.cache.clear()
      with decoder_state(decoder):
        result = c.runner(model, c.settings, config, tokenizer, prepared, packed, seed=0, inputs=owner)
        predictions.append({key: value for key, value in result.items() if key != 'timings_ms'})
      stage('actual_current_runner_'+str(index))
    if predictions[1:] != predictions[:-1]:
      raise AssertionError('Repeated actual current predictions changed during JIT capture')
    del predictions
    pixels, positions, ids, cos, sin, dc, ds, *_ = owner.upload(c.settings.initial_noise)
    embedding = model.encode(pixels, positions, ids, owner.batches, owner.spans)
    prompt = int(prepared['actual_prompt_length'])
    prefill = lambda: model.padded_prefill(embedding, cos, sin, c.settings.max_tokens, prompt)
    logits = prefill()
    snapshots = [cache.clone().realize() for cache in model.cache]
    first = logits.clone().realize()
    payload = model.padded_prefill.last_targets.clone().realize()
    allocation = identity()
    stage('complete_actual_prefill_snapshot')
    try:
      with decoder_state(decoder):
        report['decoder_plan'] = helper.warm_decoder(decoder, model, c.settings, config, logits, dc, ds, prompt)
        decoder.first_token(logits)  # Link compact selection after its capture.
      stage('decoder_serial_1_to_6_verifier_16_to_64')
    finally:
      logits = restore_prefill(model.cache, snapshots, prefill, identity=identity, equal=exact,
                               copy_into=lambda target, saved: target.assign(saved).realize())
      exact(logits, first)
      exact(model.padded_prefill.last_targets, payload)
      model.synchronize()
    if identity() != allocation:
      raise AssertionError('A warmed graph retained a retired KV allocation')
    report.update(full_kv_bits_restored=True, actual_first_logits_bits_restored=True,
                  target_payload_bits_restored=True, snapshot_bytes=sum(t.numel()*t.dtype.itemsize for t in snapshots))
    del snapshots, first, payload
    # Keep captured graphs and the exact input owner, but restore cold logical
    # vision-cache state. This preserves the first real query's cache accounting.
    vision.cache.clear()
    vision.hits = vision.misses = 0
    for target, name in ((owner,'selective_jits'), (vision,'encoders'), (vision,'stitchers'),
                         (decoder,'jits'), (decoder,'verifiers'), (decoder,'select_jits'), (model,'trajectory_jits')):
      values = getattr(target, name)
      if not values or any(jit.cnt < 3 for jit in values.values()):
        raise AssertionError(f'Unlinked graph at handoff: {name}')
      report['graph_counts'][name] = len(values)
      setattr(target, name, ClosedGraphs(values))
    if decoder.choose_jit.cnt < 3 or model.padded_prefill.jit.cnt < 3:
      raise AssertionError('Unlinked compact selector or real prefill')
    state['ready'] = True
    stage('ready_before_consumer_binding')
    report.update(ready=True, initialization_ms=(time.monotonic()-begun)*1000,
      memory_after_bytes=GlobalCounters.mem_used, sampled_peak_live_bytes=peak,
      memory_metric='tinygrad live buffers sampled at barriers; not allocator-reserved VRAM or a hard preallocation cap',
      cold_layout_policy='fail closed before new graph invocation; initialize a new owner outside capture',
      warmed_owner_retained=True, actual_prompt=prompt, accepted_suffix_rows=6,
      limitation='No promise for unseen image duplicate partitions or changed input layouts; no full-route stall claim')
    return report
  except BaseException as error:
    report['error'] = repr(error)
    report['ready'] = False
    c.input_class, c.inputs = original_class, original_inputs
    # Some graph state may have changed. Failure is terminal for this model.
    raise
  finally:
    model.encode = original_encode
