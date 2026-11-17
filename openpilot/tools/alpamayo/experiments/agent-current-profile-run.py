"""Plan or run a bounded diagnostic profile through the canonical speculation launcher.

The parent owns GPU serialization and the 300-second subprocess timeout.
Planning and provenance checks use only the CPU and Python standard library.
"""
import argparse
import ast
import atexit
import copy
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import sys

EXP = Path(__file__).resolve().parent
WORK = Path('/mnt/nvme-500/tmp/alpamayo-speed-20261116')
BASE_INVOCATION = WORK/'full-native-5hz-fixed-vision/invocation.json'
OUTER_INVOCATION = WORK/'combined-prefill-native-5hz/invocation.json'
LAUNCHER = EXP/'agent-speculation-run.py'
PINNED = {'agent-speculation-run.py': 'dfdecbaed0e80f3ebeb60012f2daad491d218a48e83efc65805f563137f9d9fa',
          'benchmark-pipeline.py': 'bcf890f8321dc0b0ff499380e01a42956f3a2395015efe091648a2a04e04cf56',
          'profile-kernels.py': '0631a76453e68f31188bc9febeb4d15d6de12c27bab16e1e376880034bde4d87'}


def digest(path):
  return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
  path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def option(command, flag):
  if command.count(flag) != 1: raise ValueError(f'Expected exactly one {flag}')
  return command[command.index(flag)+1]


def replace(command, flag, value):
  option(command, flag)
  command[command.index(flag)+1] = str(value)


def remove(command, flag, has_value=False):
  if command.count(flag) != 1: raise ValueError(f'Expected exactly one {flag}')
  index = command.index(flag)
  del command[index:index+1+has_value]


def profile_ast(source, enabled=True):
  """Change exactly one constant. Preserve all checks, hooks, and model operations."""
  original = ast.parse(source, filename=str(LAUNCHER))
  tree = copy.deepcopy(original)
  matches = [node for node in ast.walk(tree) if isinstance(node, ast.Assign) and len(node.targets) == 1
             and ast.dump(node.targets[0]) == ast.dump(ast.parse("os.environ['PROFILE'] = '0'").body[0].targets[0])]
  if len(matches) != 1 or not isinstance(matches[0].value, ast.Constant) or matches[0].value.value != '0':
    raise ValueError('Canonical PROFILE assignment changed')
  matches[0].value.value = '1' if enabled else '0'
  changed = ast.dump(tree, include_attributes=False)
  matches[0].value.value = '0'
  if ast.dump(tree, include_attributes=False) != ast.dump(original, include_attributes=False):
    raise AssertionError('Unexpected launcher AST change')
  matches[0].value.value = '1' if enabled else '0'
  return tree, hashlib.sha256(changed.encode()).hexdigest()


def normalized(command):
  command = command[:]
  for flag in ('--output', '--warmup', '--runs', '--pad-prompt-to'):
    replace(command, flag, '<diagnostic-boundary>')
  for flag in ('--vision-calibration', '--profile-kernels', '--coarse-timing'):
    if flag in command: remove(command, flag, flag == '--vision-calibration')
  return command


def derive(recorded, output, warmup, runs):
  if not 1 <= warmup <= 8 or runs != 2: raise ValueError('Require 1..8 prefix warmups and exactly two measured queries')
  if Path(recorded['command'][1]).resolve() != (EXP/'benchmark-pipeline.py').resolve():
    raise ValueError('Base must be a direct canonical benchmark invocation; nested outer launchers are not a base')
  original = recorded['command']
  required = ('--packed-quantize', '--compact-inputs', '--lookup-draft', '--reduced-vocab', '--prefill-calibration',
              '--native-manifest', '--native-calibrations', '--native-frame-cache', '--online-attention',
              '--coarse-timing', '--vision-calibration', '--vision-cache', '--group-vision', '--batched-vision',
              '--packed-projector', '--vision-views', '--bf16-expert-projections', '--packed-bf16-vision',
              '--direct-byte-inputs', '--late-inputs', '--cache-rotary-inputs')
  if any(flag not in original for flag in required): raise ValueError('Reference recipe is missing a required input or precision setting')
  for flag, expected in (('--steps', '3'), ('--pixel-scale', '1'), ('--vision-batch', '1'),
                         ('--bf16-vision-layers', '2'), ('--decode-block', '6'), ('--lookup-fallback', '6'),
                         ('--camera-profile', 'nvidia')):
    if option(original, flag) != expected: raise ValueError(f'Incompatible reference setting: {flag}')
  forbidden = ('--profile-kernels', '--profile-host', '--capture-prefill-scales', '--capture-vision-scales',
               '--validation-reference', '--history-pixel-scale', '--history-feature-scale', '--camera-pixel-scales',
               '--front-history-indices', '--trajectory-solver', '--warmup-graphs', '--warm-current-shapes')
  if any(flag in original for flag in forbidden): raise ValueError('Unexpected baseline profiling, reduced input, or solver override')
  if recorded['environment'].get('PROFILE') != '0': raise ValueError('Canonical reference must record PROFILE=0')
  control = copy.deepcopy(recorded)
  command = control['command']
  remove(command, '--vision-calibration', True)
  for flag, value in (('--output', output), ('--warmup', warmup), ('--runs', runs), ('--pad-prompt-to', 2048)):
    replace(command, flag, value)
  profiled = copy.deepcopy(control)
  remove(profiled['command'], '--coarse-timing')
  profiled['command'].append('--profile-kernels')
  profiled['environment']['PROFILE'] = '1'
  if normalized(original) != normalized(command) or normalized(command) != normalized(profiled['command']):
    raise AssertionError('An unrelated invocation setting changed')
  if {**profiled['environment'], 'PROFILE': '0'} != control['environment']:
    raise AssertionError('An unrelated environment setting changed')
  return control, profiled


def prefix_provenance(command, count):
  start = command.index('--samples')+1
  end = command.index('--initial-noise')
  samples = command[start:end][:count]
  if len(samples) != count or len(set(samples)) != count: raise ValueError('Prefix would reuse or omit queries')
  manifest_path = Path(option(command, '--native-manifest'))
  manifest = json.loads(manifest_path.read_text())
  if manifest['camera_subset'] != [1, 6] or manifest['camera_input_profile'] != 'calibrated_vehicle_history':
    raise ValueError('Unexpected native camera/history recipe')
  scheduled = {str(Path(row['sample']).resolve()): row for row in manifest['schedule'] if 'sample' in row}
  result = []
  for sample in samples:
    row = scheduled[str(Path(sample).resolve())]
    frames = row['frames']
    if len(frames) != 10 or any(sum(f['view'] == view for f in frames) != 5 for view in (1, 6)):
      raise ValueError('Expected full five-frame history from each camera')
    if any(frame['capture_ns'] > row['query_ns'] or frame['requested_ns'] > row['query_ns'] for frame in frames):
      raise ValueError('Future camera data in native query')
    expected_history = [-2_000_000_000, -1_000_000_000, -500_000_000, -100_000_000, 0]
    if any(sorted(f['requested_ns']-row['query_ns'] for f in frames if f['view'] == view) != expected_history for view in (1, 6)):
      raise ValueError('Original camera history offsets changed')
    inputs = {p.name: digest(p) for p in sorted(Path(sample).glob('*.npy')) if p.stem != 'image_frames'}
    if not inputs: raise ValueError('Missing sample CPU input files')
    result.append(dict(sample=sample, query_ns=row['query_ns'], frames=frames, cpu_input_file_sha256=inputs))
  intervals = [b['query_ns']-a['query_ns'] for a, b in zip(result, result[1:])]
  if any(not 190_000_000 <= delta <= 210_000_000 for delta in intervals):
    raise ValueError('Expected strictly advancing native 5 Hz query prefixes')
  return dict(manifest_sha256=digest(manifest_path), camera_subset=[1, 6], frames_per_camera=5,
              nominal_sample_hz=5, query_intervals_ns=intervals, queries=result)


def build_plan(output, invocation=BASE_INVOCATION, warmup=8, runs=2):
  for name, expected in PINNED.items():
    if digest(EXP/name) != expected: raise ValueError(f'Pinned diagnostic source changed: {name}')
  recorded = json.loads(invocation.read_text())
  control, profiled = derive(recorded, output, warmup, runs)
  checkpoint = Path(option(profiled['command'], '--checkpoint'))
  sources = {str(p): digest(p) for p in sorted([*EXP.glob('*.py'), *EXP.glob('*.c'), *EXP.glob('*.h'), *EXP.parent.glob('*.py')])}
  artifacts = [Path(option(profiled['command'], key)) for key in
               ('--prefill-calibration', '--native-calibrations', '--initial-noise', '--reduced-vocab')]
  artifacts += [checkpoint/'config.json', checkpoint/'model.safetensors.index.json']
  tinygrad = Path(recorded['environment']['PYTHONPATH'].split(os.pathsep)[0])
  revision = subprocess.check_output(['git', '-C', str(tinygrad), 'rev-parse', 'HEAD'], text=True).strip()
  if revision != '89846dcfb383a98ee01cfec3bba440ae7dcf9b1e': raise ValueError('Unexpected tinygrad revision')
  tinygrad_sources = {str(tinygrad/name): digest(tinygrad/name) for name in
                      ('tinygrad/device.py', 'tinygrad/helpers.py', 'tinygrad/runtime/ops_amd.py')}
  tree, transformed_hash = profile_ast(LAUNCHER.read_text())
  del tree
  outer = json.loads(OUTER_INVOCATION.read_text())
  if Path(outer['command'][1]).name != LAUNCHER.name: raise ValueError('Unexpected historical outer launcher')
  inner_path = Path(option(outer['command'], '--invocation'))
  plan = dict(schema=1, diagnostic_only=True, timings_valid_for_throughput=False,
              gpu_executed=False, completed=False, parent_timeout_seconds=300, warmup=warmup, runs=runs,
              base_invocation=dict(path=str(invocation), sha256=digest(invocation)),
              historical_outer=dict(path=str(OUTER_INVOCATION), sha256=digest(OUTER_INVOCATION),
                                    nested_invocation=str(inner_path), nested_sha256=digest(inner_path),
                                    used_as_base=False),
              source_sha256=sources, artifact_sha256={str(p): digest(p) for p in artifacts},
              tinygrad=dict(path=str(tinygrad), revision=revision, source_sha256=tinygrad_sources),
              checkpoint=dict(path=str(checkpoint), identity='Config and shard index hashes; tensor bytes are not reverified here'),
              launcher_ast=dict(change="os.environ['PROFILE']: '0' -> '1'", transformed_ast_sha256=transformed_hash,
                                all_other_nodes_equal=True, canonical_checks_preserved=True),
              native_prefix=prefix_provenance(profiled['command'], warmup+runs),
              current_recipe=dict(vision='dynamic FP8; first two layers and selected projections retain original BF16',
                                  language='existing fixed FP8 calibration', padded_prompt_bound=2048,
                                  prefill='canonical complete six-token target verification, with original rejection fallback',
                                  prefill_attention='shuffle-fma', history='five original frames per camera',
                                  trajectory_solver='three Euler evaluations'),
              delta_from_current_control=['PROFILE=1', 'add --profile-kernels', 'remove --coarse-timing'],
              profiling_limits=['USB timestamp readbacks and additional stage fences perturb wall times.',
                                'Full GPU event sums include separately identified transfer events.',
                                'No throughput, route quality, live-device, or sub-100ms claim.',
                                'Generic tinygrad profile export is unregistered after collection to avoid extra calibration GPU work.'])
  return control, profiled, plan


def execute_canonical(command, environment, source=None, enabled=True):
  """Reuse canonical hooks. Restore import, profiler, argv, path, and environment changes."""
  if any(name == 'tinygrad' or name.startswith('tinygrad.') for name in sys.modules):
    raise RuntimeError('PROFILE must be set before any tinygrad import; start a fresh process')
  tree, _ = profile_ast(LAUNCHER.read_text() if source is None else source, enabled)
  old_spec, old_argv, old_path = importlib.util.spec_from_file_location, sys.argv, sys.path[:]
  old_environment = dict(os.environ)
  undo = []

  def preserve(obj, name):
    own = name in vars(obj)
    value = vars(obj).get(name)
    def restore():
      if own: setattr(obj, name, value)
      elif name in vars(obj): delattr(obj, name)
    undo.append(restore)

  def spec_hook(name, location, *values, **options):
    spec = old_spec(name, location, *values, **options)
    if Path(location).resolve() != (EXP/'profile-kernels.py').resolve(): return spec
    loader = spec.loader
    class Loader:
      def create_module(self, value): return loader.create_module(value)
      def exec_module(self, module):
        loader.exec_module(module)
        original = module.Recorder
        class Recorder(original):
          def __init__(self, device, benchmark):
            for obj, key in ((device, 'collect_prof'), (device, '_at_profile_finalize'), (benchmark, 'timed')):
              preserve(obj, key)
            super().__init__(device, benchmark)
        module.Recorder = Recorder
    spec.loader = Loader()
    return spec

  try:
    for key, value in environment.items():
      if value is None: os.environ.pop(key, None)
      else: os.environ[key] = value
    os.environ['PROFILE'] = '1' if enabled else '0'
    sys.path[:0] = [p for p in os.environ.get('PYTHONPATH', '').split(os.pathsep) if p]
    importlib.util.spec_from_file_location = spec_hook
    sys.argv = command
    exec(compile(tree, str(LAUNCHER), 'exec'), {'__file__': str(LAUNCHER), '__name__': '__main__'})
  finally:
    # Recorder replaces generic USB-unsafe host-clock calibration. Do not run it at exit after restoration.
    device_module = sys.modules.get('tinygrad.device')
    if enabled and device_module is not None and hasattr(device_module, 'finalize_profile'):
      atexit.unregister(device_module.finalize_profile)
    for restore in reversed(undo): restore()
    importlib.util.spec_from_file_location = old_spec
    sys.argv, sys.path[:] = old_argv, old_path
    os.environ.clear()
    os.environ.update(old_environment)


def validate_report(report, plan):
  if not report['profiling'] or not report['summary']['profiling_enabled']: raise ValueError('Missing kernel profiling')
  if report.get('vision_calibration_sha256') is not None: raise ValueError('Fixed vision was accidentally enabled')
  if report['pad_prompt_to'] != 2048 or report['trajectory_solver'] != 'euler': raise ValueError('Recipe changed')
  if report['settings']['diffusion_steps'] != 3: raise ValueError('Expert evaluation count changed')
  if report['settings'].get('coarse_timing'): raise ValueError('Coarse timing is incompatible with stage profiling')
  if report['prefill_calibration_sha256'] != plan['artifact_sha256'][option_from_plan(plan, '--prefill-calibration')]:
    raise ValueError('Language calibration changed')
  spec = report['prefill_speculation']
  if spec['mode'] != 'candidate' or spec['audit'] or spec['prefill_attention'] != 'shuffle-fma' or spec['timings_valid_for_throughput']:
    raise ValueError('Unexpected speculation or profiling mode')
  if spec['padded_prompt_bound'] != 2048 or spec['draft'] != [124549, 11135, 3174, 4286, 1561, 135084]:
    raise ValueError('Complete nominal draft or padded prompt bound changed')
  if len(report['warmup']) != plan['warmup'] or len(report['runs']) != plan['runs']: raise ValueError('Incomplete query prefix')
  entries = report['warmup']+report['runs']
  if [e['sample'] for e in entries] != [q['sample'] for q in plan['native_prefix']['queries']]:
    raise ValueError('Native query prefix changed')
  for path, expected in plan['source_sha256'].items():
    # The existing benchmark snapshots experiment sources and model.py. Its benchmark.py is recorded by this wrapper.
    key = Path(path).name
    active_agents = {'agent-speculation-run.py', 'agent-speculation-policy.py', 'agent-speculation-prefill.py',
                     'agent-attention-rowsoftmax.py', 'agent-current-profile-run.py', 'agent-current-profile-summary.py'}
    # Other owners can edit independent experiments while this diagnostic loads its original model.
    if key.startswith('agent-') and key not in active_agents: continue
    if key in report['source_sha256'] and report['source_sha256'][key] != expected:
      raise ValueError(f'Runtime source snapshot changed: {key}')
  for entry in entries:
    prediction = entry['prediction']
    def finite(values):
      if isinstance(values, list): return bool(values) and all(finite(value) for value in values)
      return isinstance(values, (int, float)) and math.isfinite(values)
    if not finite(prediction['xyz']) or not finite(prediction['rot']): raise ValueError('Nonfinite or empty trajectory')
    record = entry.get('draft', {}).get('prefill_draft', {})
    if record.get('accepted') and (not record['all_targets_finite'] or record['targets'] != spec['draft']):
      raise ValueError('Accepted draft lacks complete finite target verification')
  for entry in report['runs']:
    for stage in ('vision', 'prefill', 'diffusion'):
      if not entry['device_profile']['stages'].get(stage, {}).get('kernels'):
        raise ValueError(f'Missing measured stage GPU events: {stage}')
  stamps = [entry['input_latest_timestamp_us'] for entry in entries]
  if any(b <= a for a, b in zip(stamps, stamps[1:])): raise ValueError('Input timestamps did not advance')


def option_from_plan(plan, flag):
  return option(plan['profiled_invocation']['command'], flag)


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--output', type=Path, required=True)
  parser.add_argument('--mode', choices=('plan', 'run'), default='plan')
  parser.add_argument('--invocation', type=Path, default=BASE_INVOCATION)
  parser.add_argument('--warmup', type=int, default=8)
  parser.add_argument('--runs', type=int, default=2)
  args = parser.parse_args()
  output = args.output.resolve()
  if output.exists(): raise FileExistsError(f'Refusing to replace an existing report: {output}')
  control, profiled, plan = build_plan(output, args.invocation, args.warmup, args.runs)
  output.parent.mkdir(parents=True, exist_ok=True)
  invocation_path = output.with_suffix('.current-profile-invocation.json')
  plan_path = output.with_suffix('.current-profile-plan.json')
  control_path = output.with_suffix('.current-profile-control-invocation.json')
  launcher_command = [str(LAUNCHER), '--output', str(output), '--invocation', str(invocation_path),
                      '--mode', 'candidate', '--prefill-attention', 'shuffle-fma', '--pad-prompt-to', '2048',
                      '--warmup', str(args.warmup), '--runs', str(args.runs)]
  plan.update(launcher_command=launcher_command, profiled_invocation=profiled, current_control_invocation=control)
  write_json(invocation_path, profiled)
  write_json(control_path, control)
  write_json(plan_path, plan)
  if args.mode == 'plan':
    print(json.dumps(dict(mode='plan', gpu_executed=False, plan=str(plan_path), queries=args.warmup+args.runs), indent=2))
    return
  plan.update(gpu_executed=None, run_requested=True)
  write_json(plan_path, plan)
  execute_canonical(launcher_command, profiled['environment'])
  report = json.loads(output.read_text())
  validate_report(report, plan)
  spec = importlib.util.spec_from_file_location('agent_current_profile_summary', EXP/'agent-current-profile-summary.py')
  summary = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(summary)
  result = summary.summarize(report)
  result['plan'] = str(plan_path)
  summary_path = output.with_suffix('.current-profile-kernels.json')
  write_json(summary_path, result)
  report['current_profile'] = dict(diagnostic_only=True, timings_valid_for_throughput=False,
                                  plan=str(plan_path), kernel_summary=str(summary_path),
                                  prefill_target_verification=[dict(sample=e['sample'], record=e.get('draft', {}).get('prefill_draft'),
                                                                   limit='No per-entry record means original decoder fallback; inspect speculation sidecar')
                                                              for e in report['warmup']+report['runs']])
  report['summary']['timings_valid_for_throughput'] = False
  write_json(output, report)
  if not result['measured']['valid_timestamp_coverage']: raise ValueError('Invalid GPU timestamps retained; profile is incomplete')
  plan.update(gpu_executed=True, completed=True)
  write_json(plan_path, plan)
  print(json.dumps(dict(kernel_summary=str(summary_path), categories=result['measured']['categories'],
                        timings_valid_for_throughput=False), indent=2))


if __name__ == '__main__': main()
