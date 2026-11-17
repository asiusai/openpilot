"""Observe pipeline scheduling without extra GPU work or tensor readbacks.

Default mode audits saved reports using only the Python standard library.
--invocation replays a saved benchmark through the parent's GPU trial wrapper.
--defer-first-measured delays input preparation across the startup GC barrier.
No decoding, masks, tokens, image bytes, or numerical settings are changed.
"""
from __future__ import annotations

import argparse
from collections import Counter, deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
import functools
import gc
import importlib.util
import json
import math
import os
from pathlib import Path
import resource
import statistics
import sys
import threading
import time
from unittest.mock import patch
import weakref


def percentile(values, fraction):
  values = sorted(values)
  if not values:
    return None
  index = (len(values)-1)*fraction
  a, b = math.floor(index), math.ceil(index)
  return values[a] + (values[b]-values[a])*(index-a)


def summarize(values):
  return {'count':len(values), 'median':statistics.median(values) if values else None,
          'p95':percentile(values, .95), 'max':max(values) if values else None}


def audit(path):
  """Keep startup outliers visible; identify costs excluded from service time."""
  report = json.loads(Path(path).read_text())
  runs, warmup = report['runs'], report.get('warmup', [])
  freeze_ms = report.get('warmup_gc_freeze', {}).get('seconds', 0)*1000
  med_service = statistics.median(row['service_ms'] for row in runs)
  rows = []
  all_rows = warmup + runs
  for index, row in enumerate(runs):
    previous = all_rows[len(warmup)+index-1] if len(warmup)+index else None
    prior_times = previous['prediction']['timings_ms'] if previous else {}
    age_before_service = row['pipeline_ms']-row['service_ms']
    # This is only a residual estimate. It is not a timestamped attribution.
    prior_late_tail = prior_times.get('diffusion', 0)+prior_times.get('trajectory_decode', 0)
    residual = age_before_service-prior_late_tail-(freeze_ms if index == 0 else 0)
    if index == 0 or row['pipeline_ms'] > med_service*1.8 or row.get('gc_ms', 0) > 10:
      rows.append({'run':index, 'sample':row['sample'],
                   'pipeline_ms':row['pipeline_ms'], 'service_ms':row['service_ms'],
                   'age_before_service_ms':age_before_service,
                   'prepared_queue_ms':row.get('prepared_queue_ms'),
                   'load_ms':row.get('load_ms'), 'preprocess_ms':row.get('preprocess_ms'),
                   'gc_inside_service_ms':row.get('gc_ms'),
                   'startup_collect_freeze_reported_ms':freeze_ms if index == 0 else 0,
                   'residual_after_prior_late_tail_and_freeze_ms':residual if report.get('late_inputs') else None})
  deltas = [b['input_latest_timestamp_us']-a['input_latest_timestamp_us'] for a,b in zip(runs,runs[1:])]
  wall_ms = report.get('summary', {}).get('measured_wall_seconds', 0)*1000
  source_span_ms = (runs[-1]['input_latest_timestamp_us']-runs[0]['input_latest_timestamp_us'])/1000 if runs else 0
  allocations = [row.get('allocated_bytes_by_device', {}).get('AMD') for row in runs]
  allocations = [value for value in allocations if value is not None]
  return {'report':str(path), 'measurement_boundary':report.get('timing_boundary'),
          'profiling_enabled':report.get('summary', {}).get('profiling_enabled'),
          'pipeline_ms':summarize([row['pipeline_ms'] for row in runs]),
          'service_ms':summarize([row['service_ms'] for row in runs]),
          'prepared_queue_ms':summarize([row['prepared_queue_ms'] for row in runs]),
          'explicit_gc_ms':{**summarize([row.get('gc_ms',0) for row in runs]),
                            'sum':sum(row.get('gc_ms',0) for row in runs),
                            'over_10ms':sum(row.get('gc_ms',0) > 10 for row in runs)},
          'generated_tokens':summarize([row['prediction']['generated_tokens'] for row in runs]),
          'startup_collect_freeze_reported_ms':freeze_ms,
          'outliers':rows,
          'source_timestamp_delta_ms':summarize([delta/1000 for delta in deltas]),
          'measured_wall_ms':wall_ms, 'source_span_ms':source_span_ms,
          'cadence_note':'Source deltas describe fixtures. The benchmark does not pace input arrival or measure sensor age.',
          'amd_allocation_bytes':{'first':allocations[0], 'last':allocations[-1], 'peak':max(allocations)} if allocations else None,
          'limits':['Startup outliers remain included.',
                    'Residuals need direct event timing before assigning an exact cause.',
                    'Allocation counters exclude some driver and allocator retention.']}


def compare(reference_path, candidate_path):
  """Require exact outputs for scheduling-only changes, including every warmup."""
  reference = json.loads(Path(reference_path).read_text())
  candidate = json.loads(Path(candidate_path).read_text())
  checked = 0
  for section in ('validation','warmup','runs'):
    before, after = reference.get(section,[]), candidate.get(section,[])
    if len(before) != len(after):
      raise AssertionError(f'{section}: prediction count changed')
    for index, (old,new) in enumerate(zip(before,after,strict=True)):
      if old['sample'] != new['sample']:
        raise AssertionError(f'{section}[{index}]: input ordering changed')
      if section == 'validation':
        if old != new:
          raise AssertionError(f'{section}[{index}]: saved-control validation changed')
        continue
      for key in ('generated_token_ids','reasoning','xyz','rot','prompt_tokens','prefill_tokens'):
        if old['prediction'].get(key) != new['prediction'].get(key):
          raise AssertionError(f'{section}[{index}]: {key} changed')
      checked += 1
  for key in ('initial_noise_sha256','prefill_calibration_sha256','settings'):
    if reference.get(key) != candidate.get(key):
      raise AssertionError(f'{key} changed')
  old_sources, new_sources = reference.get('source_sha256',{}), candidate.get('source_sha256',{})
  changed_sources = [key for key in set(old_sources)|set(new_sources)
                     if not key.startswith('agent-pipeline') and old_sources.get(key) != new_sources.get(key)]
  if changed_sources:
    raise AssertionError(f'Non-helper sources changed: {sorted(changed_sources)}')
  return {'status':'passed','predictions_checked':checked,
          'reference':str(reference_path),'candidate':str(candidate_path),
          'checks':'Exact tokens, reasoning, XYZ, rotations, prompt lengths, noise, calibration, settings, and non-helper sources.',
          'limit':'This checks paired saved outputs. Full-route and target-device validation remain separate.'}


class DeferredResult:
  """Submit exactly once, when the consumer requests the first measured input."""
  def __init__(self, submit):
    self.submit = submit
    self.future = None
    self.lock = threading.Lock()

  def result(self, timeout=None):
    started = time.perf_counter()
    with self.lock:
      if self.future is None:
        self.future = self.submit()
    remaining = None if timeout is None else max(0, timeout-(time.perf_counter()-started))
    return self.future.result(timeout=remaining)


class Observer:
  """Bounded metadata only. Never retain Tensors, futures, prepared inputs, or frames."""
  def __init__(self, capacity=8192):
    if capacity <= 0:
      raise ValueError('Event capacity must be positive')
    self.events = deque(maxlen=capacity)
    self.dropped = 0
    self.counts = Counter()
    self.lock = threading.RLock()
    self.started = time.perf_counter_ns()
    self.run = 0
    self.vision = None
    self.inventory_limit = 128
    self.allocators = {}

  def record(self, kind, **values):
    stamp = time.perf_counter_ns()-self.started
    with self.lock:
      if len(self.events) == self.events.maxlen:
        self.dropped += 1
      self.events.append({'t_ns':stamp, 'kind':kind, 'thread':threading.get_ident(), **values})
      self.counts[kind] += 1

  def memory(self):
    result = {'python_allocated_blocks':sys.getallocatedblocks(),
              'gc_count':list(gc.get_count()), 'peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
    try:
      pages = Path('/proc/self/statm').read_text().split()
      result['rss_bytes'] = int(pages[1])*os.sysconf('SC_PAGE_SIZE')
    except (OSError, ValueError, IndexError):
      pass
    # Read host counters only. Never create or synchronize a Device.
    helpers = sys.modules.get('tinygrad.helpers')
    if helpers is not None:
      result['allocated_bytes_by_device'] = dict(helpers.GlobalCounters.mem_used_per_device)
    cached = {}
    for ref in tuple(self.allocators.values()):
      if (allocator := ref()) is None:
        continue
      name = getattr(allocator.dev,'device',type(allocator.dev).__name__)
      entries = list(allocator.cache.items())
      cached[name] = {'bytes':sum(size*len(values) for (size,_),values in entries),
                      'buffers':sum(len(values) for _,values in entries),
                      'size_classes':len(entries)}
    result['allocator_cache'] = cached
    return result

  @staticmethod
  def cache_identity(model):
    cache = getattr(model,'cache',None)
    return {'tensor_object_ids':[id(value) for value in cache], 'buffer_uop_ids':[id(value.uop.base) for value in cache],
            'first_shape':list(cache[0].shape)} if cache else None

  def inventory(self, **owners):
    result = {}
    for owner_name, owner in owners.items():
      if owner is None:
        continue
      entries, total = [], 0
      for field, value in vars(owner).items():
        candidates = value.items() if isinstance(value, dict) else [(None,value)]
        for key, jit in candidates:
          if not hasattr(jit, 'cnt') or not hasattr(jit, 'captured'):
            continue
          total += 1
          if len(entries) < self.inventory_limit:
            captured = jit.captured
            entries.append({'field':field, 'key':repr(key)[:256], 'calls':jit.cnt,
                            'captured':captured is not None,
                            'linked_alias_variants':len(getattr(captured, '_linears', {})) if captured is not None else 0})
      result[owner_name] = {'total_jits':total, 'omitted_jits':max(0,total-len(entries)), 'jits':entries}
    return result

  def gc_callback(self, phase, info):
    self.record('gc_'+phase, generation=info['generation'], collected=info.get('collected'),
                uncollectable=info.get('uncollectable'))

  def gc_wrapper(self, name, original):
    @functools.wraps(original)
    def wrapped(*args, **kwargs):
      self.record(name+'_begin')
      started = time.perf_counter_ns()
      try:
        return original(*args, **kwargs)
      finally:
        self.record(name+'_end', elapsed_ns=time.perf_counter_ns()-started)
    return wrapped

  def pool_type(self, defer_ordinal):
    observer = self
    class ObservedPool(ThreadPoolExecutor):
      def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self._max_workers != 1:
          raise ValueError('Pipeline observer requires one CPU preparation worker')
        self.ordinal = 0

      def submit(self, fn, /, *args, **kwargs):
        ordinal = self.ordinal
        self.ordinal += 1
        observer.record('input_requested', ordinal=ordinal, sample=str(args[0]) if args else None)
        def prepare():
          observer.record('input_started', ordinal=ordinal)
          try:
            result = fn(*args, **kwargs)
            observer.record('input_ready', ordinal=ordinal)
            return result
          except BaseException as error:
            observer.record('input_error', ordinal=ordinal, error=type(error).__name__)
            raise
        def submit():
          observer.record('input_submitted', ordinal=ordinal)
          return super(ObservedPool,self).submit(prepare)
        if ordinal == defer_ordinal:
          observer.record('input_deferred_for_startup_barrier', ordinal=ordinal)
          return DeferredResult(submit)
        return submit()
    return ObservedPool

  def wrap_runner(self, cls):
    original = cls.run
    if getattr(original, '_agent_pipeline_observer', False):
      return
    observer = self
    @functools.wraps(original)
    def run(decoder, model, args, config, tokenizer, prepared, packed, seed, inputs):
      number = observer.run
      observer.run += 1
      vision = observer.vision() if observer.vision is not None else None
      observer.record('prediction_begin', ordinal=number, memory=observer.memory(),
                      conditioning_cache=observer.cache_identity(model),
                      inventory=observer.inventory(model=model, decoder=decoder, inputs=inputs, vision=vision))
      started = time.perf_counter_ns()
      try:
        return original(decoder,model,args,config,tokenizer,prepared,packed,seed,inputs)
      finally:
        observer.record('prediction_end', ordinal=number, elapsed_ns=time.perf_counter_ns()-started,
                        memory=observer.memory(),
                        conditioning_cache=observer.cache_identity(model),
                        inventory=observer.inventory(model=model, decoder=decoder, inputs=inputs, vision=vision))
    run._agent_pipeline_observer = True
    cls.run = run


def flag_value(command, flag, default=None):
  return command[command.index(flag)+1] if flag in command else default


def replay(args):
  invocation = json.loads(args.invocation.read_text())
  command = invocation['command']
  program = Path(command[1]).resolve()
  if program != Path(__file__).with_name('benchmark-pipeline.py').resolve():
    raise ValueError('Only the adjacent benchmark-pipeline.py invocation is supported')
  warmup = int(flag_value(command,'--warmup','2'))
  if args.warmup is not None:
    warmup = args.warmup
  if warmup < 0 or (args.warm_current_shapes and warmup == 0):
    raise ValueError('Current-shape warmup needs at least one declared startup input')
  validation_path = flag_value(command,'--validation-reference')
  validations = len(json.loads(Path(validation_path).read_text())) if validation_path else 0
  if args.defer_first_measured and not all(flag in command for flag in ('--pipeline-inputs','--late-inputs','--freeze-warmup-gc')):
    raise ValueError('Startup barrier requires pipeline-inputs, late-inputs, and freeze-warmup-gc')
  if args.warm_current_shapes and not all(flag in command for flag in ('--lookup-draft','--compact-inputs','--selective-pixels')):
    raise ValueError('Current-shape warmup requires lookup drafts and compact selective inputs')
  for key, value in invocation['environment'].items():
    if value is None:
      os.environ.pop(key,None)
    else:
      os.environ[key] = value
  for path in reversed(os.environ.get('PYTHONPATH','').split(os.pathsep)):
    if path:
      sys.path.insert(0,path)
  sys.path.insert(0,str(program.parent))
  spec = importlib.util.spec_from_file_location('agent_pipeline_benchmark', program)
  pipeline = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(pipeline)
  from tinygrad.engine.jit import _TinyJit
  from tinygrad.device import Allocator

  observer = Observer(args.event_capacity)
  startup = None
  if args.warm_current_shapes:
    warm_spec = importlib.util.spec_from_file_location('agent_pipeline_warmup',Path(__file__).with_name('agent-pipeline-warmup.py'))
    warm_module = importlib.util.module_from_spec(warm_spec)
    warm_spec.loader.exec_module(warm_module)
    startup = warm_module.StartupWarmup(lambda:observer.run <= warmup+validations,observer.record,args.graph_budget)
  original_jit_call = _TinyJit.__call__
  original_allocator_init = Allocator.__init__
  @functools.wraps(original_allocator_init)
  def allocator_init(allocator, dev):
    original_allocator_init(allocator,dev)
    if len(observer.allocators) < 16:
      observer.allocators[id(allocator)] = weakref.ref(allocator)

  @functools.wraps(original_jit_call)
  def jit_call(jit, *values, **kwargs):
    phase = 'first_run' if jit.cnt == 0 else 'capture_link' if jit.cnt == 1 else 'replay'
    # Replay events are deliberately omitted. Cold phases expose graph stalls.
    if jit.cnt >= 2:
      before = len(getattr(jit.captured,'_linears',{}))
      started = time.perf_counter_ns()
      result = original_jit_call(jit,*values,**kwargs)
      after = len(getattr(jit.captured,'_linears',{}))
      if after > before:
        observer.record('jit_new_alias_variant', jit=id(jit), variants=after,
                        function=getattr(jit.fxn,'__qualname__',type(jit.fxn).__name__),
                        elapsed_ns=time.perf_counter_ns()-started)
      return result
    name = getattr(jit.fxn,'__qualname__',type(jit.fxn).__name__)
    observer.record('jit_cold_begin', jit=id(jit), phase=phase, function=name)
    started = time.perf_counter_ns()
    try:
      return original_jit_call(jit,*values,**kwargs)
    finally:
      observer.record('jit_cold_end', jit=id(jit), phase=phase, function=name,
                      elapsed_ns=time.perf_counter_ns()-started)

  original_loader = pipeline.local_module
  def load(name):
    module = original_loader(name)
    if name in ('block-decode','draft-decode','lookup-decode'):
      cls = getattr(module,{'block-decode':'BlockDecoder','draft-decode':'DraftDecoder','lookup-decode':'LookupDecoder'}[name])
      observer.wrap_runner(cls)
      if startup is not None:
        startup.patch_decoder(cls)
    if name == 'compact-inputs' and startup is not None:
      startup.patch_inputs(module.CompactInputs)
    if name == 'vision-cache':
      original_init = module.VisionCache.__init__
      @functools.wraps(original_init)
      def init(owner,*values,**kwargs):
        original_init(owner,*values,**kwargs)
        observer.vision = weakref.ref(owner)
      module.VisionCache.__init__ = init
      if startup is not None:
        startup.patch_vision(module.VisionCache)
    return module

  argv = command[1:]
  argv[argv.index('--output')+1] = str(args.output)
  for name,value in (('warmup',args.warmup),('runs',args.runs)):
    if value is not None:
      if name == 'runs' and value <= 0:
        raise ValueError('Measured run count must be positive')
      flag = '--'+name
      if flag in argv:
        argv[argv.index(flag)+1] = str(value)
      else:
        argv.extend([flag,str(value)])
  if args.reasoning_budget is not None:
    if '--max-tokens' in argv:
      argv[argv.index('--max-tokens')+1] = str(args.reasoning_budget)
    else:
      argv.extend(['--max-tokens',str(args.reasoning_budget)])
  if (args.warmup_graphs or args.warm_current_shapes) and '--warmup-graphs' not in argv:
    argv.append('--warmup-graphs')
  args.output.parent.mkdir(parents=True,exist_ok=True)
  diagnostics = args.output.with_suffix('.pipeline.json')
  if args.output.exists() or diagnostics.exists():
    raise FileExistsError('Use a new output path for this trial')
  started = time.perf_counter_ns()
  status = 'failed'
  try:
    with ExitStack() as stack:
      stack.enter_context(patch.object(pipeline,'ThreadPoolExecutor',observer.pool_type(warmup+validations if args.defer_first_measured else None)))
      stack.enter_context(patch.object(pipeline,'local_module',load))
      stack.enter_context(patch.object(_TinyJit,'__call__',jit_call))
      stack.enter_context(patch.object(Allocator,'__init__',allocator_init))
      stack.enter_context(patch.object(sys,'argv',argv))
      for name in ('collect','freeze','get_freeze_count'):
        stack.enter_context(patch.object(gc,name,observer.gc_wrapper(name,getattr(gc,name))))
      gc.callbacks.append(observer.gc_callback)
      stack.callback(gc.callbacks.remove,observer.gc_callback)
      pipeline.main()
      generated = json.loads(args.output.read_text())
      generated['agent_pipeline'] = {'observed':True, 'diagnostics':str(diagnostics),
                                     'defer_first_measured':args.defer_first_measured,
                                     'warm_current_shapes':args.warm_current_shapes}
      generated['profiling'] = True
      generated.setdefault('summary',{})['profiling_enabled'] = True
      args.output.write_text(json.dumps(generated,indent=2)+'\n')
      status = 'complete'
  finally:
    result = {'schema':1, 'status':status, 'source_invocation':str(args.invocation),
              'command':argv, 'defer_first_measured':args.defer_first_measured,
              'warm_current_shapes':args.warm_current_shapes,
              'first_measured_ordinal':warmup+validations,
              'process_observed_ns':time.perf_counter_ns()-started,
              'instrumentation':('Diagnostic overhead plus explicit current-input startup graph warmup. No future frames are opened.' if startup is not None else
                                 'Diagnostic overhead; no additional GPU fences, readbacks, kernels, or model invocations.'),
              'boundary':'Monotonic host events. Input starts at RGB preparation, not sensor capture.',
              'counts':dict(observer.counts), 'dropped_events':observer.dropped,
              'events':list(observer.events), 'final_memory':observer.memory()}
    diagnostics.write_text(json.dumps(result,indent=2)+'\n')


def self_check():
  """Check deferred ordering, value preservation, failure propagation, and bounds."""
  import random
  randomizer = random.Random(9060)
  cases = 0
  for warmup in (0,1,3):
    observer = Observer(128)
    prepared = []
    payloads = [[randomizer.randrange(100000) for _ in range(length)] for length in (1,6,64,65,127,256,519)]
    with observer.pool_type(warmup)(max_workers=1) as pool:
      for index, payload in enumerate(payloads):
        future = pool.submit(lambda value: (prepared.append(index),value)[1],payload)
        if index == warmup:
          if index in prepared:
            raise AssertionError('Measured input prepared before startup barrier')
          observer.record('fake_startup_barrier_complete')
        first = future.result(timeout=5)
        second = future.result(timeout=5)
        if first is not payload or second is not first or prepared.count(index) != 1:
          raise AssertionError('Deferred preparation changed values or repeated work')
        cases += 1
    events = list(observer.events)
    barrier = next(row['t_ns'] for row in events if row['kind'] == 'fake_startup_barrier_complete')
    start = next(row['t_ns'] for row in events if row['kind'] == 'input_started' and row['ordinal'] == warmup)
    if start < barrier:
      raise AssertionError('Preparation crossed startup barrier')
  observer = Observer(2)
  with observer.pool_type(0)(max_workers=1) as pool:
    def fail():
      raise LookupError('expected failure')
    try:
      pool.submit(fail).result(timeout=5)
    except LookupError:
      pass
    else:
      raise AssertionError('Worker exception was lost')
  if len(observer.events) > 2 or observer.dropped == 0:
    raise AssertionError('Diagnostics are unbounded')
  return {'status':'passed', 'random_payload_cases':cases, 'largest_payload_length':519,
          'checks':['no preparation before barrier','identical returned objects','one preparation per input',
                    'worker exception propagation','bounded event retention'],
          'limit':'CPU helper checks only. No model, tokenizer, target-verification, GPU, or camera validation.'}


def gc_probe(count):
  """Run only in a separate CLI process; unfreeze its heap before exit."""
  if not 1 <= count <= 500000:
    raise ValueError('CPU GC probe allows at most 500000 tracked objects')
  initial_frozen = gc.get_freeze_count()
  was_enabled = gc.isenabled()
  gc.disable()
  objects, measured = [], {}
  try:
    for _ in range(count):
      value = []
      value.append(value)
      objects.append(value)
    for name in ('collect','freeze','get_freeze_count'):
      start = time.perf_counter_ns()
      result = getattr(gc,name)()
      measured[name] = {'elapsed_ns':time.perf_counter_ns()-start, 'result':result}
  finally:
    gc.unfreeze()
    objects.clear()
    value = None
    gc.collect()
    if was_enabled:
      gc.enable()
  return {'objects':count,'initial_frozen_objects':initial_frozen,'measurements':measured,
          'limit':'Synthetic CPU graph. This does not reproduce actual model heap traversal cost.'}


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  mode = parser.add_mutually_exclusive_group(required=True)
  mode.add_argument('--audit',nargs='+',type=Path)
  mode.add_argument('--compare',nargs=2,type=Path,metavar=('REFERENCE','CANDIDATE'))
  mode.add_argument('--invocation',type=Path)
  mode.add_argument('--self-check',action='store_true')
  mode.add_argument('--gc-probe',type=int)
  parser.add_argument('--output',type=Path)
  parser.add_argument('--warmup',type=int,help='Explicit declared startup count override')
  parser.add_argument('--runs',type=int,help='Explicit measured count override; uses the same chronological sample prefix')
  parser.add_argument('--defer-first-measured',action='store_true')
  parser.add_argument('--warmup-graphs',action='store_true',help='Use existing bounded graph warmup on current startup inputs only')
  parser.add_argument('--warm-current-shapes',action='store_true',help='Experimental current-input decoder, vision-cardinality, and dynamic upload graphs')
  parser.add_argument('--graph-budget',type=int,default=32,help='Maximum startup graphs per family; does not cap reasoning length')
  parser.add_argument('--event-capacity',type=int,default=8192)
  parser.add_argument('--reasoning-budget',type=int,
                      help='Explicit optional budget override; no cap or fixed reasoning length is imposed')
  args = parser.parse_args()
  if args.invocation:
    if args.output is None:
      parser.error('Replay requires a new output path')
    replay(args)
    return
  if (args.defer_first_measured or args.reasoning_budget is not None or args.warmup_graphs or args.warm_current_shapes
      or args.warmup is not None or args.runs is not None):
    parser.error('Replay-only settings need --invocation')
  result = ([audit(path) for path in args.audit] if args.audit else
            compare(*args.compare) if args.compare else
            gc_probe(args.gc_probe) if args.gc_probe is not None else self_check())
  text = json.dumps(result,indent=2)+'\n'
  if args.output:
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(text)
  else:
    print(text,end='')


if __name__ == '__main__':
  main()
