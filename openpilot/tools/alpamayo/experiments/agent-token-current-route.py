"""Optional NUMERICAL latest-full/history50-or75 wrapper for the parent GPU trial.

Uses frozen pruning hooks and controller arithmetic. Does not edit or replace
the pipeline initializer. Quality remains unqualified until parent paired gates.
"""
import argparse
from contextlib import ExitStack
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
POLICIES = ('latest-full-history-50', 'latest-full-history-75')
sys.dont_write_bytecode = True


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def module(name, namespace):
  spec = importlib.util.spec_from_file_location(namespace, HERE/(name+'.py'))
  value = importlib.util.module_from_spec(spec)
  sys.modules[namespace] = value
  spec.loader.exec_module(value)
  return value


def qualify(saved, output, runs=None, warmup=None):
  frozen = module('agent-token-prune-route', 'agent_token_current_frozen_route_qualify')
  command = frozen.qualify_command(saved, output, runs, warmup)
  if frozen.value(command, '--camera-profile') != 'nvidia' or '--allow-missing-cameras' not in command:
    raise ValueError('Require the actual two-front-camera NVIDIA profile with complete histories')
  if frozen.value(command, '--steps', '3') != '3' or frozen.value(command, '--trajectory-solver', 'euler') != 'euler':
    raise ValueError('Keep the original three-evaluation Euler action expert')
  return command


def install_pipeline_hooks(pipeline, runtime, stack, installed):
  frozen = module('agent-token-prune-route', 'agent_token_current_frozen_route_hooks')
  # Use the policy implementation attached to this runtime's observations.
  # Bind the function, not a second Observations class from another import.
  frozen.prepared_observations = runtime.prepared_observations
  initializer = getattr(pipeline, 'initialize', None)
  frozen.install_pipeline_hooks(pipeline, runtime, stack, installed)
  if getattr(pipeline, 'initialize', None) is not initializer: raise AssertionError('Protected initializer was replaced')
  return initializer


def verify_qualification(path, hook_path, compiler_path):
  report = json.loads(path.read_text())
  if (report.get('classification') != 'NUMERICAL NEW MODEL' or report.get('prompt_prefix_invariance_pass') is not False
      or report.get('policies') != list(POLICIES) or not report.get('cpu_qualified')
      or report.get('actual_cpu_inputs_checked') != 64 or report.get('gpu_quality_or_speed_validated') is not False):
    raise ValueError('Missing explicit two-policy CPU qualification on all64 actual inputs')
  if set(report['source_sha256']) != {p.name for p in HERE.glob('agent-token-current*.py')}:
    raise ValueError('CPU qualification does not cover the current standalone source set')
  for group in ('source_sha256', 'protected_source_sha256', 'contract_source_sha256'):
    for name, digest in report[group].items():
      path = Path(name) if name.startswith('/') else HERE/name
      if sha(path) != digest: raise ValueError('CPU qualification source changed: '+str(path))
  hook = json.loads(hook_path.read_text())
  if hook.get('policies') != list(POLICIES) or hook.get('opened_devices') != [] or not hook.get('hooks_restored'):
    raise ValueError('Missing both-policy CPU hook/lifecycle qualification')
  if hook['source_sha256'] != report['source_sha256']: raise ValueError('Stale hook qualification sources')
  compiler = json.loads(compiler_path.read_text())
  if (compiler.get('opened_devices') != [] or compiler.get('gpu_executed') is not False
      or compiler.get('inherited_runtime_sha256') != sha(HERE/'agent-token-prune-runtime.py')
      or len(compiler.get('records',[])) != 6):
    raise ValueError('Missing both-policy CPU indexed-copy compilation evidence')
  return report


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--policy', choices=POLICIES, required=True)
  parser.add_argument('--invocation', required=True, type=Path)
  parser.add_argument('--output', required=True, type=Path)
  parser.add_argument('--cpu-report', type=Path, default=HERE/'agent-token-current-cpu-report.json')
  parser.add_argument('--hook-report', type=Path, default=HERE/'agent-token-current-hook-report.json')
  parser.add_argument('--compiler-report', type=Path, default=HERE/'agent-token-current-compiler-report.json')
  parser.add_argument('--runs', type=int)
  parser.add_argument('--warmup', type=int)
  parser.add_argument('--dry-run', action='store_true', help='CPU preflight; do not import model or device runtime')
  args = parser.parse_args()
  qualification = verify_qualification(args.cpu_report, args.hook_report, args.compiler_report)
  saved = json.loads(args.invocation.read_text())
  command = qualify(saved, args.output, args.runs, args.warmup)
  source = module('agent-token-prune-route', 'agent_token_current_frozen_route_values')
  checkpoint = Path(source.value(command, '--checkpoint'))
  provenance = dict(classification='NUMERICAL NEW MODEL', policy=args.policy,
    latest_image_retention=1.0, older_image_retention=.5 if args.policy == POLICIES[0] else .75,
    full_language_layers=2, full_visual_encoder_and_five_frames_per_camera=True,
    prompt_prefix_invariance_pass=False, generation_target='changed numerical model', quality_passed=None,
    rejected_uniform50_quality=dict(ADE=[2.71346, 2.84233], FDE=[6.81482, 7.17402],
                                    right_turn_ADE=[4.086, 4.420], failed_windows=3, gate_percent=5, adopted=False),
    cpu_report_sha256=sha(args.cpu_report), invocation_sha256=sha(args.invocation),
    hook_report_sha256=sha(args.hook_report), compiler_report_sha256=sha(args.compiler_report),
    calibration_sha256=sha(source.value(command, '--prefill-calibration')),
    checkpoint_config_sha256=sha(checkpoint/'config.json'), checkpoint_index_sha256=sha(checkpoint/'model.safetensors.index.json'),
    source_sha256=qualification['source_sha256'], protected_initializer_unchanged=True,
    command=command, required_parent_gates=['same paired64 ADE/FDE within5% per window', 'full-route ADE/FDE',
                                          'trajectory shape', 'speed', 'slowing', 'reasoning'],
    selection='current available original layer1 attention only; latest by camera ID and actual timestamp maximum')
  if args.dry_run:
    print(json.dumps(dict(dry_run=True, no_gpu_opened=True, provenance=provenance), indent=2))
    return
  if args.output.exists(): raise FileExistsError('Use a new parent trial output')
  for key, setting in saved['environment'].items():
    if setting is None: os.environ.pop(key, None)
    else: os.environ[key] = setting
  for path in reversed(os.environ.get('PYTHONPATH', '').split(os.pathsep)):
    if path: sys.path.insert(0, path)
  sys.path.insert(0, str(HERE))
  factory = module('agent-token-current-runtime', 'agent_token_current_route_runtime')
  runtime = factory.make_runtime(args.policy)
  pipeline = module('benchmark-pipeline', 'agent_token_current_pipeline')
  installed = []
  with ExitStack() as stack:
    initializer = install_pipeline_hooks(pipeline, runtime, stack, installed)
    stack.enter_context(patch.object(sys, 'argv', command))
    pipeline.main()
    if pipeline.initialize is not initializer: raise AssertionError('Protected initializer changed')
    if len(installed) != 1 or not installed[0].events: raise AssertionError('Current-history policy did not execute')
    if any(event.get('policy') != args.policy for event in installed[0].events): raise AssertionError('Wrong runtime policy')
    report = json.loads(args.output.read_text())
    report['token_current_numerical_policy'] = dict(provenance, events=installed[0].events,
      measurement='Existing full pipeline boundary includes original full vision, early layers, score readback, ranking, gather/scatter, masks, and genuine fallback')
    args.output.write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__': main()
