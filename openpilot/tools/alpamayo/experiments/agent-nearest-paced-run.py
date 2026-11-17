"""Optional parent-only nearest paced replay. Default CPU admission; no parity rerun."""
import argparse, copy, fcntl, importlib.util, json, os, sys
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('nearest_paced_admit', HERE/'agent-nearest-paced-admit.py')
a = importlib.util.module_from_spec(spec); spec.loader.exec_module(a)
DEFAULT = HERE/'agent-nearest-paced-parent-run/report.json'

def options(plan, output):
  return ['--task','paced','--base-source',plan['base_source'],'--reference-report',plan['startup_reference'],
    '--consumer-parity-report',plan['startup_candidate'],'--invocation',plan['invocation'],'--mode','run','--output',str(output),
    '--hz','10','--start-seconds','35','--seconds','8','--inference-budget-seconds','180','--raw-frames',plan['raw_frames'],'--warp','native']

def before_binding(c, components, clock, saved, replay):
  if clock.anchor is not None: raise ValueError('Source clock started before qualified Consumer binding')
  actual = dict(source_sha256=components.report['source_sha256'], settings=replay.tree(vars(components.settings)),
    runtime_environment=components.report['runtime_environment'])
  if actual != saved['initialized']: raise ValueError('Actual initialized source/settings/environment changed')
  state = components._startup_spec
  if not all(state.get(key) is True for key in c.RESTORE): raise ValueError('Actual startup restoration/owner gate failed')
  if (state['current_inputs'] != 1 or state['future_inputs'] or state['forced_tokens'] or state['predictions_published'] or
      any(components.startup_active)): raise ValueError('Actual startup retained active or future work')
  for key in ('source_sha256','graph_counts','decoder_plan'):
    if state[key] != saved['startup_spec'][key]: raise ValueError('Actual startup graph/source contract differs: '+key)
  decoder, vision = components.decoder, components.vision_cache
  if decoder.history or decoder.statistics or decoder.prefill_draft_records or decoder.prefill_draft_old_history:
    raise ValueError('Actual decoder history is not empty at initial handoff')
  if vision.cache or vision.hits or vision.misses: raise ValueError('Actual logical vision cache is not cold')
  return dict(initialized=actual, startup_restoration=True, empty_decoder_history=True, cold_logical_vision_cache=True,
    source_clock_started=False, input_owner_retained=True, graph_counts=state['graph_counts'])

@contextmanager
def observer(c, plan, saved, output, records):
  """Read-only bind checks plus compiler-cache pathname isolation. No model substitutions."""
  root = c.BASE/'experiments'; original = importlib.util.spec_from_file_location; restored=[]
  def spec(name, location, *args, **kwargs):
    result = original(name, location, *args, **kwargs)
    if Path(location).resolve() != root/'agent-paced-replay.py': return result
    loader = result.loader
    class Loader:
      def create_module(self, value): return loader.create_module(value)
      def exec_module(self, module):
        loader.exec_module(module); local = module.local
        def checked_local(key):
          value = local(key)
          if key == 'agent-live-replay':
            normalize = value.normalize
            def isolated(*args, **kwargs):
              normalized, parsed, protocol = normalize(*args, **kwargs)
              if protocol != saved['qualified_factory_protocol']: raise ValueError('Actual paced factory recipe changed')
              normalized = copy.deepcopy(normalized)
              before = normalized['environment']['CACHEDB']; after = str(output.parent/'compiler-cache.db')
              normalized['environment']['CACHEDB'] = after
              records['compiler_cache_relocation'].append(dict(before=before, after=after))
              return normalized, parsed, protocol
            value.normalize = isolated
          if key == 'agent-live-runtime':
            bind = value.bind_components
            def checked(components, args, **kwargs):
              replay = local('agent-live-replay')
              record = before_binding(c, components, kwargs['clock'], saved, replay)
              owner = components.inputs
              consumer = bind(components, args, **kwargs)
              if consumer.inputs is not owner: raise ValueError('Actual Consumer replaced warmed owner')
              nearest = consumer.provenance['nearest_live_policy']
              if nearest['policy'] != plan['policy'] or nearest['source_sha256'] != a.sha(c.FROZEN/'agent-nearest-live-policy.py'):
                raise ValueError('Actual Consumer did not bind the qualified received-only nearest policy')
              record['consumer_nearest_policy'] = nearest
              records['bindings'].append(record)
              if len(records['bindings']) != 1: raise ValueError('Only one model/Consumer binding is admitted')
              return consumer
            value.bind_components = checked
          return value
        module.local = checked_local; restored.append((module, local))
    result.loader = Loader(); return result
  try:
    with patch.object(importlib.util, 'spec_from_file_location', spec): yield
  finally:
    for module, local in restored: module.local = local

def verify_frozen():
  protocol = a.read(a.PROTOCOL)
  if protocol['source_sha256'] != a.sources(): raise ValueError('New wrapper changed since CPU freeze')
  a.check_pins(protocol['artifact_sha256'])
  checks = a.read(HERE/'agent-nearest-paced-checks.json')
  if not checks['passed'] or checks['gpu_executed'] or checks['source_sha256'] != a.sources():
    raise ValueError('Current CPU adapter/admission controls required')
  return protocol

def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--execute-gpu',action='store_true'); parser.add_argument('--output',type=Path,default=DEFAULT)
  args = parser.parse_args(); a.setup_cpu()
  c, plan, policy, startup, dynamic, order_protocol, reference, saved, admission = a.verify()
  frozen = verify_frozen()
  if frozen['admission'] != admission: raise ValueError('Frozen independent admission changed')
  if not args.execute_gpu:
    print(json.dumps(dict(cpu_only=True,gpu_executed=False,admission=admission,requested_hz_ceiling=10,
      source_window_seconds=[35,43],rgb_frames=450,active_budget_seconds=180,outer_timeout_seconds=900,
      full_qualified_startup=True,automatic_parity_rerun=False,automatic_adoption=False),indent=2)); return
  if args.output.parent.parent.resolve() != HERE or not args.output.parent.name.startswith('agent-nearest-paced-'):
    raise ValueError('Parent output must use a new owned agent-nearest-paced-* directory')
  if args.output.parent.exists(): raise FileExistsError('Use a new directory; never overwrite earlier evidence')
  with Path('/tmp/opencode/alpamayo-chestnut-speed.lock').open('a') as lock:
    try: fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError: pass
    else: raise RuntimeError('Parent must hold the shared GPU lock')
  if os.environ.get('AGENT_NEAREST_PACED_PARENT_TIMEOUT') != '1': raise RuntimeError('Use the reviewed900s parent timeout')
  args.output.parent.mkdir()
  record = dict(schema=1,mode='nearest-paced-independent-admission',gpu_execution_requested=True,
    source_sha256=a.sources(),protocol_sha256=a.sha(a.PROTOCOL),admission=admission,
    command_options=options(plan,args.output),compiler_cache_relocation=[],bindings=[],
    automatic_adoption=False,extra_parity_runs=0,goal_qualified=False)
  a.save(args.output.with_suffix('.admission.json'),record)
  try:
    with policy.imports(c.BASE),observer(c,plan,saved,args.output,record):
      startup.main(options(plan,args.output))
    result = a.read(args.output)
    if len(record['bindings']) != 1 or len(record['compiler_cache_relocation']) != 1: raise ValueError('Unexpected factory call count')
    if (result['requested_hz'] != 10 or result['source_window_s'] != 8 or result['inference_budget_seconds'] != 180 or
        result['maximum_active_inference'] != 1 or result['queued_prediction_snapshots'] != 0 or
        result['startup_spec_sources'] != startup.own_sources()): raise ValueError('Completed paced bounds/source changed')
    state=result['startup_spec']
    if not all(state.get(key) is True for key in c.RESTORE) or state['source_clock_started'] is not False:
      raise ValueError('Completed paced startup contract failed')
    if state['current_query_ns'] > result['source_start_ns']: raise ValueError('Startup used a future query')
    record.update(completed=True,report_sha256=a.sha(args.output),gpu_executed=True,
      recorded_replay_completion_hz=result['recorded_replay_completion_hz'],completed_queries=result['completed_queries'],
      limit='Recorded PC replay; requested10Hz is a ceiling, not achieved throughput or a device/quality qualification')
  except BaseException as error:
    record.update(completed=False,error=repr(error)); raise
  finally: a.save(args.output.with_suffix('.nearest-admission-result.json'),record)

if __name__ == '__main__': main()
