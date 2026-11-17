"""Paired expert-tile route runner. --plan performs CPU-only policy/provenance checks."""
import argparse
from contextlib import contextmanager
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import runpy
import sys

HERE = Path(__file__).resolve().parent


def sha(data): return hashlib.sha256(data).hexdigest()


def local(name):
  spec = importlib.util.spec_from_file_location(name.replace('-', '_'), HERE/(name+'.py'))
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module


def option(command, flag, default=None):
  assert command.count(flag) <= 1, f'Duplicate option: {flag}'
  return command[command.index(flag)+1] if flag in command else default


def replace_option(command, flag, value):
  assert command.count(flag) <= 1
  if flag in command: command[command.index(flag)+1] = str(value)
  else: command.extend((flag, str(value)))


def inspect_invocation(path):
  chain, commands = [], []
  for _ in range(5):
    path = Path(path)
    raw = path.read_bytes()
    invocation = json.loads(raw)
    assert invocation['environment'].get('AM_POWER_LIMIT') == '144' and invocation['environment'].get('AM_PERF_LEVEL') == 'auto'
    command = invocation['command'][1:]
    program = Path(command[0])
    assert program.is_file()
    assert not any(word.split('=')[0] in ('--vision-calibration', '--capture-vision-scales', '--audit') for word in command)
    chain.append({'path': str(path), 'sha256': sha(raw), 'program': str(program), 'program_sha256': sha(program.read_bytes()),
      'environment': invocation['environment']})
    commands.append(command)
    if program.name == 'benchmark-pipeline.py': break
    assert program.name in ('agent-speculation-run.py', 'agent-vision-tiles-route.py'), 'Unsupported wrapper'
    if program.name == 'agent-speculation-run.py':
      assert option(command, '--mode', 'candidate') == 'candidate'
      assert option(command, '--prefill-attention', 'baseline') == 'shuffle-fma', 'Preserve the current exact language attention'
    path = option(command, '--invocation')
    assert path is not None
  else: raise AssertionError('Invocation nesting exceeds five levels')
  terminal = commands[-1]
  assert option(terminal, '--steps', '10') == '3' and option(terminal, '--trajectory-solver', 'euler') == 'euler'
  assert option(terminal, '--pixel-scale', '1') == '1' and option(terminal, '--bf16-vision-layers', '0') == '2'
  assert all(flag in terminal for flag in ('--expert-kernels', '--llvm-wgp', '--packed-quantize', '--prefill-calibration',
    '--quantize-reduction', '--grouped-quantize-reduction', '--batched-vision', '--online-attention', '--initial-noise'))
  assert '--online-expert' not in terminal and '--profile-kernels' not in terminal
  assert any(Path(command[0]).name == 'agent-speculation-run.py' for command in commands), 'Recorded speculation wrapper required'
  return chain, commands


def plan(invocation, output, *, warmup=None, runs=None, samples_json=None):
  chain, commands = inspect_invocation(invocation)
  command = commands[0].copy()
  replace_option(command, '--output', output)
  for flag, value in (('--warmup', warmup), ('--runs', runs)):
    if value is not None: replace_option(command, flag, value)
  if samples_json is not None:
    assert Path(command[0]).name == 'agent-speculation-run.py', 'Sample override requires the recorded speculation wrapper at the outer level'
    replace_option(command, '--samples-json', samples_json)
  terminal = commands[-1]
  samples = terminal[terminal.index('--samples')+1:terminal.index('--initial-noise')]
  # Only the existing wrapper handles sample replacement. Resolve its effective
  # sample list here to record an exact input schedule before any GPU access.
  for item in commands:
    if override := option(item, '--samples-json'): samples = json.loads(Path(override).read_text()); break
  if samples_json is not None: samples = json.loads(Path(samples_json).read_text())
  effective_commands = [command, *commands[1:]]
  def inherited(flag, default):
    return next((option(item, flag) for item in effective_commands if flag in item), default)
  n_warm, n_runs = int(inherited('--warmup', '12')), int(inherited('--runs', '24'))
  assert n_warm >= 0 and n_runs > 0 and n_warm+n_runs <= len(samples), 'Require a complete unrepeated sample schedule'
  comparable = command.copy()
  replace_option(comparable, '--output', '<paired-output>')
  sources = {str(p): sha(p.read_bytes()) for p in (Path(__file__), HERE/'agent-expert-tiles.py')}
  identity = {'source_chain': chain, 'command': comparable, 'samples': samples, 'warmup': n_warm, 'runs': n_runs, 'hook_sources': sources,
    'vision_policy': 'dynamic', 'expert_steps': 3, 'solver': 'euler'}
  return {'source_chain': chain, 'effective_command': command, 'terminal_command': terminal, 'samples': samples,
    'scheduled_samples': samples[:n_warm+n_runs], 'warmup': n_warm, 'runs': n_runs,
    'pair_identity_sha256': sha(json.dumps(identity, sort_keys=True).encode()), 'hook_sources': sources,
    'vision_policy': 'dynamic', 'expert_steps': 3, 'solver': 'euler'}


def compare_reports(before, after, expected_plan):
  import numpy as np
  expected, actual = before['warmup']+before['runs'], after['warmup']+after['runs']
  assert len(before['warmup']) == len(after['warmup']) == expected_plan['warmup']
  assert len(before['runs']) == len(after['runs']) == expected_plan['runs']
  assert [row['sample'] for row in expected] == expected_plan['scheduled_samples']
  assert [row['sample'] for row in actual] == expected_plan['scheduled_samples']
  checks = []
  for a, b in zip(expected, actual, strict=True):
    x, y = a['prediction'], b['prediction']
    assert x['generated_token_ids'] == y['generated_token_ids']
    assert x['reasoning'] == y['reasoning']
    for field in ('xyz', 'rot'):
      av, bv = np.asarray(x[field], np.float64), np.asarray(y[field], np.float64)
      assert av.shape == bv.shape and av.size and np.isfinite(av).all() and np.isfinite(bv).all()
      np.testing.assert_array_equal(av.view(np.uint64), bv.view(np.uint64), err_msg=f'{a["sample"]}: {field}')
    checks.append({'sample': a['sample'], 'tokens_exact': True, 'text_exact': True, 'xyz_bits_exact': True, 'rot_bits_exact': True})
  return {'all_exact': True, 'samples': checks}


@contextmanager
def install_interceptor(candidate, variant, evidence, expected_expert, handles, records):
  original = importlib.util.spec_from_file_location
  def hooked(name, location, *values, **options):
    spec = original(name, location, *values, **options)
    if Path(location).resolve() != expected_expert.resolve(): return spec
    candidate.checked_bytes('expert-kernels.py', Path(location).read_bytes())
    loader = spec.loader
    class Loader:
      def create_module(self, value): return loader.create_module(value)
      def exec_module(self, module):
        loader.exec_module(module)
        prior = module.install
        def install(model, *args, **kwargs):
          value = prior(model, *args, **kwargs)
          if variant == 'candidate':
            handle = candidate.install(model, evidence=evidence)
            handles.append(handle)
            records.append(handle.info)
          else: records.append({'variant': 'current', 'eligible_expert_projections': value})
          return value
        module.install = install
    spec.loader = Loader()
    return spec
  importlib.util.spec_from_file_location = hooked
  try: yield
  finally:
    importlib.util.spec_from_file_location = original
    for handle in reversed(handles): handle.close()


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--output', type=Path, required=True)
  parser.add_argument('--invocation', type=Path, required=True)
  parser.add_argument('--variant', choices=('current', 'candidate'), default='candidate')
  parser.add_argument('--reference', type=Path)
  parser.add_argument('--evidence', type=Path, default=Path('/mnt/nvme-500/tmp/alpamayo-speed-20261116/agent-expert-gemm-fc2-v2/report.json'))
  parser.add_argument('--warmup', type=int)
  parser.add_argument('--runs', type=int)
  parser.add_argument('--samples-json', type=Path)
  parser.add_argument('--plan', action='store_true')
  args = parser.parse_args()
  candidate = local('agent-expert-tiles')
  planned = plan(args.invocation, args.output, warmup=args.warmup, runs=args.runs, samples_json=args.samples_json)
  evidence = candidate.verify_evidence(args.evidence)
  expected_expert = Path(planned['terminal_command'][0]).with_name('expert-kernels.py')
  candidate.checked_bytes('expert-kernels.py', expected_expert.read_bytes())
  candidate.checked_bytes('amd_gemm.py', (expected_expert.parent.parent/'amd_gemm.py').read_bytes())
  reference = json.loads(args.reference.read_text()) if args.reference else None
  if reference is not None:
    paired = reference['agent_expert_tiles']
    assert paired['variant'] == 'current' and paired['route_completed'] and not paired.get('failure')
    assert paired['plan']['pair_identity_sha256'] == planned['pair_identity_sha256'], 'Reference settings or sources differ'
  report = {'variant': args.variant, 'plan_only': args.plan, 'plan': planned, 'matrix_evidence': evidence,
    'route_completed': False, 'exactness_validated': False, 'pipeline_speed_validated': False, 'hooks': []}
  args.output.parent.mkdir(parents=True, exist_ok=True)
  metadata = args.output.with_suffix('.agent-expert-tiles.json')
  def save(): metadata.write_text(json.dumps(report, indent=2)+'\n')
  save()
  if args.plan:
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({'plan_only': True, 'gpu_executed': False, 'scheduled_samples': len(planned['scheduled_samples']), 'vision_policy': 'dynamic'}))
    return
  if args.variant == 'candidate': assert reference is not None, 'Candidate route requires its matched current report'
  previous_environment, previous_argv, previous_path = os.environ.copy(), sys.argv[:], sys.path[:]
  handles = []
  try:
    invocation = json.loads(args.invocation.read_text())
    for key, value in invocation['environment'].items():
      if value is None: os.environ.pop(key, None)
      else: os.environ[key] = value
    os.environ['PROFILE'] = '0'
    sys.argv = planned['effective_command']
    # Match the terminal source checkout even when this hook was relocated.
    sys.path.insert(0, str(expected_expert.parent.parent))
    with install_interceptor(candidate, args.variant, args.evidence, expected_expert, handles, report['hooks']):
      runpy.run_path(sys.argv[0], run_name='__main__')
      assert len(report['hooks']) == 1, 'Expected exactly one original expert installer'
      if args.variant == 'candidate':
        expected = {f'expert.expert.layers.{layer}.{suffix}' for layer in range(28) for suffix in candidate.TILES}
        assert set(handles[0].info['trace_calls']) == expected, 'Both projections must trace in all 28 layers'
      result = json.loads(args.output.read_text())
      assert len(result['warmup']) == planned['warmup'] and len(result['runs']) == planned['runs']
      assert [row['sample'] for row in result['warmup']+result['runs']] == planned['scheduled_samples']
      report['route_completed'] = True
      if reference is not None:
        report['comparison'] = compare_reports(reference, result, planned)
        report['exactness_validated'] = True
  except BaseException as exc:
    report['failure'] = {'type': type(exc).__name__, 'message': str(exc)}
    raise
  finally:
    os.environ.clear()
    os.environ.update(previous_environment)
    sys.argv, sys.path = previous_argv, previous_path
    save()
    if args.output.exists():
      result = json.loads(args.output.read_text())
      result['agent_expert_tiles'] = report
      args.output.write_text(json.dumps(result, indent=2)+'\n')


if __name__ == '__main__': main()
