"""Frozen paired route runner for the qualified expert PV operation only."""
import argparse
from contextlib import contextmanager, ExitStack
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import runpy
import struct
import sys

HERE = Path(__file__).resolve().parent
ROW_MEASUREMENTS = {'load_ms', 'preprocess_ms', 'prepared_queue_ms', 'cpu_buffers_ms', 'pipeline_ms', 'service_ms',
                    'gc_ms', 'native_calibration_ms', 'allocated_bytes_by_device'}
TOP_MEASUREMENTS = {'model_load_ms', 'startup_through_first_prediction_s', 'startup_before_measured_input_s',
                    'process_elapsed_s', 'summary', 'warmup_gc_freeze', 'source_snapshot', 'warmup', 'runs', 'agent_expert_pv'}


def sha(data): return hashlib.sha256(data).hexdigest()


def local(name):
  spec = importlib.util.spec_from_file_location(name.replace('-', '_'), HERE/(name+'.py'))
  module = importlib.util.module_from_spec(spec)
  sys.modules[spec.name] = module
  spec.loader.exec_module(module)
  return module


def option(command, flag, default=None):
  assert command.count(flag) <= 1
  return command[command.index(flag)+1] if flag in command else default


def plan(invocation, output):
  commands, chain = [], []
  path = Path(invocation)
  for _ in range(5):
    raw = path.read_bytes()
    data = json.loads(raw)
    assert data['environment']['AM_POWER_LIMIT'] == '144' and data['environment']['AM_PERF_LEVEL'] == 'auto'
    command = data['command'][1:]
    program = Path(command[0])
    assert program.parent == HERE, 'All invocation sources must be recursively relocated to this frozen tree'
    assert program.name in ('agent-speculation-run.py', 'benchmark-pipeline.py')
    assert not any(x.split('=')[0] in ('--vision-calibration', '--capture-vision-scales', '--audit', '--online-expert',
      '--profile-kernels', '--profile-host', '--warm-current-shapes') for x in command)
    chain.append({'path': str(path), 'sha256': sha(raw), 'program': str(program), 'program_sha256': sha(program.read_bytes()),
                  'environment': data['environment']})
    commands.append(command)
    if program.name == 'benchmark-pipeline.py': break
    assert option(command, '--mode') == 'candidate' and option(command, '--prefill-attention') == 'shuffle-fma'
    assert option(command, '--pad-prompt-to') == '2048'
    path = Path(option(command, '--invocation'))
  else: raise AssertionError('Invocation recursion exceeds bound')
  assert len(commands) == 2 and Path(commands[0][0]).name == 'agent-speculation-run.py'
  terminal, effective = commands[-1], commands[0].copy()
  assert option(terminal, '--steps') == '3' and option(terminal, '--trajectory-solver', 'euler') == 'euler'
  assert option(terminal, '--pixel-scale', '1') == '1' and option(terminal, '--bf16-vision-layers') == '2'
  for flag in ('--expert-kernels', '--llvm-wgp', '--packed-quantize', '--prefill-calibration', '--quantize-reduction',
               '--grouped-quantize-reduction', '--batched-vision', '--online-attention', '--initial-noise',
               '--native-manifest', '--native-calibrations', '--native-frame-cache', '--reduced-vocab'):
    assert flag in terminal
  effective[effective.index('--output')+1] = str(output)
  samples_path = option(effective, '--samples-json')
  samples = json.loads(Path(samples_path).read_text()) if samples_path else terminal[terminal.index('--samples')+1:terminal.index('--initial-noise')]
  warmup, runs = int(option(effective, '--warmup')), int(option(effective, '--runs'))
  assert (warmup, runs) in ((2, 62), (12, 24)) and len(samples) >= warmup+runs
  sources = {p.name: sha(p.read_bytes()) for p in HERE.glob('agent-expert-pv-route-*.py')}
  comparable = effective.copy(); comparable[comparable.index('--output')+1] = '<output>'
  identity = {'chain': chain, 'command': comparable, 'sources': sources, 'samples': samples[:warmup+runs]}
  return {'source_chain': chain, 'effective_command': effective, 'terminal_command': terminal,
          'scheduled_samples': samples[:warmup+runs], 'warmup': warmup, 'runs': runs,
          'pair_identity_sha256': sha(json.dumps(identity, sort_keys=True).encode()),
          'vision_policy': 'dynamic', 'expert_steps': 3, 'solver': 'euler', 'hook_sources': sources}


def canonical(value):
  """Keep every field, type, ordering, and finite float bit pattern."""
  if isinstance(value, dict): return ('dict', tuple((k, canonical(v)) for k, v in sorted(value.items())))
  if isinstance(value, list): return ('list', tuple(canonical(v) for v in value))
  if isinstance(value, float):
    assert math.isfinite(value), 'Nonfinite canonical result'
    return ('float', struct.pack('<d', value))
  assert value is None or type(value) in (str, bool, int)
  return (type(value).__name__, value)


def canonical_row(row):
  result = {k: v for k, v in row.items() if k not in ROW_MEASUREMENTS}
  result['prediction'] = {k: v for k, v in row['prediction'].items() if k != 'timings_ms'}
  return canonical(result)


def compare_reports(before, after, planned, before_spec, after_spec):
  for key in ('warmup', 'runs'):
    assert len(before[key]) == len(after[key]) == planned[key]
  a, b = before['warmup']+before['runs'], after['warmup']+after['runs']
  assert [r['sample'] for r in a] == [r['sample'] for r in b] == planned['scheduled_samples']
  for x, y in zip(a, b, strict=True): assert canonical_row(x) == canonical_row(y), f'Canonical row differs: {x["sample"]}'
  for obj in (before, after):
    assert obj['validation'] == [] and obj['pad_prompt_to'] == 2048 and obj['vision_calibration_sha256'] is None
    assert not obj['profiling'] and not obj['capture_vision_scales']
  old = {k: v for k, v in before.items() if k not in TOP_MEASUREMENTS}
  new = {k: v for k, v in after.items() if k not in TOP_MEASUREMENTS}
  assert canonical(old) == canonical(new), 'Canonical model/input/history policy or provenance differs'
  assert canonical(before_spec) == canonical(after_spec), 'Complete target verification or fallback records differ'
  assert len(after_spec['records']) == len(a) and after_spec['queries'] == len(a)
  assert after_spec['prefill_attention'] == 'shuffle-fma' and not after_spec['audit']
  return {'all_exact': True, 'samples': len(a), 'all_canonical_prediction_input_history_fields': True,
          'all_speculation_records': True, 'float_bits_exact': True,
          'excluded_row_measurements': sorted(ROW_MEASUREMENTS), 'excluded_top_measurements': sorted(TOP_MEASUREMENTS)}


@contextmanager
def interceptor(hook, candidate, evidence, handles):
  original = importlib.util.spec_from_file_location
  with ExitStack() as stack:
    def spec(name, location, *args, **kwargs):
      result = original(name, location, *args, **kwargs)
      if Path(location).resolve() != (HERE/'expert-kernels.py').resolve(): return result
      loader = result.loader
      class Loader:
        def create_module(self, value): return loader.create_module(value)
        def exec_module(self, module):
          loader.exec_module(module)
          previous = module.install
          def install(model, *values, **options):
            result = previous(model, *values, **options)
            handle = hook.install(model, module, candidate=candidate, evidence=evidence)
            handles.append(handle)
            stack.enter_context(handle.observe_programs())
            return result
          module.install = install
      result.loader = Loader()
      return result
    importlib.util.spec_from_file_location = spec
    try: yield
    finally:
      importlib.util.spec_from_file_location = original
      for handle in reversed(handles): handle.close()


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--output', type=Path, required=True)
  parser.add_argument('--invocation', type=Path, required=True)
  parser.add_argument('--variant', choices=('current', 'candidate'), required=True)
  parser.add_argument('--reference', type=Path)
  parser.add_argument('--evidence', type=Path, required=True)
  parser.add_argument('--plan', action='store_true')
  args = parser.parse_args()
  assert sys.flags.optimize == 0
  hook = local('agent-expert-pv-route-hook')
  planned = plan(args.invocation, args.output)
  evidence = hook.verify_evidence(args.evidence)
  meta = {'variant': args.variant, 'plan_only': args.plan, 'plan': planned, 'evidence': evidence,
          'route_completed': False, 'exactness_validated': False, 'coverage_validated': False, 'hooks': []}
  args.output.parent.mkdir(parents=True, exist_ok=True)
  metadata = args.output.with_suffix('.agent-expert-pv.json')
  def save(): metadata.write_text(json.dumps(meta, indent=2)+'\n')
  if args.plan:
    args.output.write_text(json.dumps(meta, indent=2)+'\n')
    print(json.dumps({'plan_only': True, 'gpu_executed': False, 'scheduled_samples': len(planned['scheduled_samples'])}))
    return
  reference = json.loads(args.reference.read_text()) if args.reference else None
  if args.variant == 'candidate':
    assert reference is not None
    old = reference['agent_expert_pv']
    assert old['variant'] == 'current' and old['route_completed'] and old['coverage_validated'] and not old.get('failure')
    assert old['plan']['pair_identity_sha256'] == planned['pair_identity_sha256']
  env, argv, paths = os.environ.copy(), sys.argv[:], sys.path[:]
  handles = []
  save()
  try:
    for k, v in json.loads(args.invocation.read_text())['environment'].items():
      if v is None: os.environ.pop(k, None)
      else: os.environ[k] = v
    os.environ.update(PROFILE='0', PYTHONDONTWRITEBYTECODE='1')
    sys.path.insert(0, str(HERE.parent))
    sys.argv = planned['effective_command']
    with interceptor(hook, args.variant == 'candidate', args.evidence, handles):
      runpy.run_path(sys.argv[0], run_name='__main__')
      assert len(handles) == 1
      info = handles[0].info
      assert info['traces'] and info['captures'] and info['trajectory_calls'] >= len(planned['scheduled_samples'])
      assert info['unqualified_trajectory_calls'] == info['outside_scope_selected'] == 0
      assert all(t == {'pv': 84, 'qk': 84} for t in info['traces'])
      assert all(c['pv_calls'] == c['qk_calls'] == c['softmax_calls'] == 84 for c in info['captures'])
      meta['hooks'] = [info]
      result = json.loads(args.output.read_text())
      spec = json.loads(args.output.with_suffix('.speculation.json').read_text())
      compare_reports(result, result, planned, spec, spec)
      meta.update(route_completed=True, coverage_validated=True)
      if reference is not None:
        previous_spec = json.loads(args.reference.with_suffix('.speculation.json').read_text())
        meta['comparison'] = compare_reports(reference, result, planned, previous_spec, spec)
        meta['exactness_validated'] = True
      hook.verify_evidence(args.evidence)
  except BaseException as exc:
    meta['failure'] = {'type': type(exc).__name__, 'message': str(exc)}
    meta['exactness_validated'] = False
    raise
  finally:
    os.environ.clear(); os.environ.update(env)
    sys.argv, sys.path = argv, paths
    save()
    if args.output.exists():
      result = json.loads(args.output.read_text())
      result['agent_expert_pv'] = meta
      args.output.write_text(json.dumps(result, indent=2)+'\n')


if __name__ == '__main__': main()
