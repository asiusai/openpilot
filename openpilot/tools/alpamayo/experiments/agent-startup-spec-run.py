"""Standalone source-bound startup experiment over the frozen dynamic policy.

This file changes no existing source. Exact function substitutions only insert
initialization before Audit/Consumer ownership and before the paced source clock.
"""
import argparse
from contextlib import contextmanager
import hashlib
import importlib.util
import inspect
import json
import os
from pathlib import Path
import sys
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
DEFAULT = Path('/mnt/nvme-500/tmp/alpamayo-speed-20261116/agent-paced-dynamic-source-cpu02/alpamayo')
PINS = {'agent-live-replay.py': '4aac2b9b092c2c1b90a881b17dc1ab2f8e2be39fe0e5eb4d044eaa26438d22ec',
        'agent-paced-replay.py': '8af0ee0ff255c2c928eca41a706b4d370340d410444723565a424a2a6144eee5'}


def sha(path):
  return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def local(path):
  spec = importlib.util.spec_from_file_location(Path(path).stem.replace('-', '_'), path)
  value = importlib.util.module_from_spec(spec)
  sys.modules[spec.name] = value
  spec.loader.exec_module(value)
  return value


def own_sources():
  return {name: sha(HERE/name) for name in ('agent-startup-spec-run.py', 'agent-startup-spec-warm.py')}


def transform(module, name, changes, extra):
  path = Path(module.__file__)
  if sha(path) != PINS[path.name]:
    raise ValueError('Reviewed source changed; startup adapter requires a new review')
  source = inspect.getsource(getattr(module, name))
  for before, after in changes:
    if source.count(before) != 1:
      raise ValueError('Startup source substitution is ambiguous')
    source = source.replace(before, after, 1)
  namespace = dict(module.__dict__, **extra)
  exec(compile(source, str(HERE/'agent-startup-spec-run.py')+':'+name, 'exec'), namespace)
  setattr(module, name, namespace[name])
  return hashlib.sha256(source.encode()).hexdigest()


def qualify(reference, candidate, policy):
  policy.qualified_gate(candidate, reference=reference)
  if reference.get('startup_spec', {}).get('enabled') is not False:
    raise ValueError('Require an un-warmed original-runner reference')
  if reference.get('startup_spec_sources') != own_sources() or candidate.get('startup_spec_sources') != own_sources():
    raise ValueError('Startup sources changed after paired qualification')
  warm = candidate.get('startup_spec', {})
  if not all(warm.get(k) is True for k in ('ready', 'full_kv_bits_restored', 'actual_first_logits_bits_restored',
                                        'target_payload_bits_restored', 'warmed_owner_retained')):
    raise ValueError('Require completed byte-exact startup restoration proof')
  records = [row['speculation'] for row in candidate['rows']]
  accepted = sum(bool(row.get('accepted')) for row in records)
  fallback = sum(row.get('accepted') is False for row in records)
  if not accepted or not fallback:
    raise ValueError('The actual 64-query parity gate must cover acceptance AND rejection/fallback')
  if candidate['rows'][0]['consumer_protocol']['input_reused'] is not True:
    raise ValueError('Consumer replaced the warmed current input owner')
  return dict(exact=True, queries=64, accepted=accepted, fallback=fallback,
              full_kv_restore=True, current_owner_reused=True)


def install_replay(module, root, warm, startup, transforms):
  def before_audit(components, effective, dataset, mode):
    if mode == 'reference':
      value = {'enabled': False, 'ready': False, 'current_inputs': 0}
    else:
      # Only row zero enters startup. Dataset integrity checking is not model warming.
      try:
        value = warm.initialize(components, effective, dataset.load(dataset.rows[0]), source_root=root)
      finally:
        startup.update(getattr(components, '_startup_spec', {}))
      value['enabled'] = True
    startup.update(value)
    return value
  key = '      audit.install(components)'
  transforms['replay'] = transform(module, 'execute_existing_launcher', [(key,
    "      result['startup_spec'] = _startup_before_audit(components, values[0], dataset, mode)\n"+key)],
    {'_startup_before_audit': before_audit})


def install_paced(module, root, warm, startup, transforms):
  def before_binding(components, effective, runner, producer, clock):
    if clock.anchor is not None or producer.thread is not None:
      raise ValueError('Prime only current/past inputs before source-clock start')
    selected = runner.store.select(clock(), 0, {}, runner.max_input_age_ns, runner.require_small_model)
    arrays, metadata = runner.preparer.prepare(selected)
    arrays['nav_text'] = [runner.navigation]
    if selected.query_ns > clock() or any(frame.capture_ns > selected.query_ns for frame in selected.frames):
      raise ValueError('Startup attempted to condition on a future capture')
    before_counts = dict(runner.counts)
    try:
      result = warm.initialize(components, effective, arrays, source_root=root)
    finally:
      startup.update(getattr(components, '_startup_spec', {}))
    if dict(runner.counts) != before_counts or runner.last_query != 0 or clock.anchor is not None:
      raise AssertionError('Initialization advanced the live scheduler or published a prediction')
    result.update(enabled=True, current_query_ns=selected.query_ns, source_clock_started=False,
                  source_receipts='past/current priming only; producer thread starts after readiness')
    startup.update(result)
  bind = '    consumer = runtime.bind_components(components, replay.binding_args(effective), clock=clock,'
  old_prime = "    producer.prime_history()\n    output.save(dict(kind='past_only_priming', label=LABEL, **producer.prime))"
  changes = [(old_prime, "    output.save(dict(kind='past_only_priming', label=LABEL, **producer.prime))"),
    (bind, "    producer.prime_history()\n    _startup_before_binding(components, effective, runner, producer, clock)\n"+bind),
    ('active_startup_warming=False,', 'active_startup_warming=True,'),
    ("initial_priming='only raw frames and states released at/before source start; no feature warming',",
     "initial_priming='bounded startup uses only first current input, before Consumer binding and source clock',"),
    ('    output.save(summary)',
     "    summary['startup_spec'] = dict(_startup_report)\n"
     "    summary['first_completion_after_ready_ms'] = ((summary['rows'][0]['consumer_completed_ns']-trace.start)/1e6\n"
     "      if summary['rows'] else None)\n"
     "    output.save(summary)")]
  transforms['paced'] = transform(module, 'model_run', changes,
    {'_startup_before_binding': before_binding, '_startup_report': startup})


@contextmanager
def imports(root, warm, startup, transforms):
  original = importlib.util.spec_from_file_location
  def spec(name, location, *values, **options):
    result = original(name, location, *values, **options)
    path = Path(location).resolve()
    if path.parent != root/'experiments' or path.name not in PINS:
      return result
    loader = result.loader
    class Loader:
      def create_module(self, value): return loader.create_module(value)
      def exec_module(self, value):
        loader.exec_module(value)
        (install_replay if path.name == 'agent-live-replay.py' else install_paced)(value, root, warm, startup, transforms)
    result.loader = Loader()
    return result
  with patch.object(importlib.util, 'spec_from_file_location', spec):
    yield


def main(argv=None):
  parser = argparse.ArgumentParser(add_help=False)
  parser.add_argument('--task', choices=('replay', 'paced', 'qualify', 'check-adapters'), required=True)
  parser.add_argument('--base-source', type=Path, default=DEFAULT)
  parser.add_argument('--reference-report', type=Path)
  parser.add_argument('--candidate-report', type=Path)
  cli, options = parser.parse_known_args(argv)
  os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false')
  sys.dont_write_bytecode = True
  root = cli.base_source.resolve()
  sys.path[:0] = [str(root), str(root/'experiments')]
  policy = local(root/'experiments/agent-paced-dynamic-policy.py')
  policy.verify_snapshot()
  warm = local(HERE/'agent-startup-spec-warm.py')
  startup, transforms = {}, {}
  if cli.task == 'qualify':
    result = qualify(json.loads(cli.reference_report.read_text()), json.loads(cli.candidate_report.read_text()), policy)
    print(json.dumps(result, indent=2))
    return result
  if cli.task == 'check-adapters':
    with imports(root, warm, startup, transforms), policy.imports():
      local(root/'experiments/agent-live-replay.py')
      local(root/'experiments/agent-paced-replay.py')
    result = dict(cpu_only=True, transformed_functions=transforms, startup_sources=own_sources(), gpu_executed=False)
    print(json.dumps(result, indent=2))
    return result
  output = Path(options[options.index('--output')+1])
  if cli.task == 'paced':
    if cli.reference_report is None:
      raise ValueError('Warm paced run requires the exact paired reference as well as the Consumer gate')
    candidate = Path(options[options.index('--consumer-parity-report')+1])
    qualify(json.loads(cli.reference_report.read_text()), json.loads(candidate.read_text()), policy)
  try:
    with imports(root, warm, startup, transforms), policy.imports():
      module = local(root/'experiments'/('agent-live-replay.py' if cli.task == 'replay' else 'agent-paced-replay.py'))
      module.main(options)
    if output.exists():
      report = json.loads(output.read_text())
      report['startup_spec_sources'] = own_sources()
      report['startup_spec'] = dict(startup)
      report['startup_spec_transforms'] = transforms
      output.write_text(json.dumps(report, indent=2)+'\n')
  finally:
    # Failure evidence survives even when the original runner refuses qualification.
    sidecar = output.with_suffix('.startup-spec.json')
    with sidecar.open('x') as stream:
      json.dump(dict(startup=startup, transforms=transforms, sources=own_sources()), stream, indent=2)
      stream.write('\n')


if __name__ == '__main__':
  main()
