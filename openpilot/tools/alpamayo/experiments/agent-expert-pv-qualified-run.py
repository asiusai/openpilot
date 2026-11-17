"""Fresh exact route pair for current CU PV and qualified complete WGP PV."""
import argparse
import copy
import importlib.util
import json
import os
from pathlib import Path
import runpy
import sys

HERE = Path(__file__).resolve().parent
COUNTS = {'quality': (2, 62), 'native': (12, 24)}


def common():
  spec = importlib.util.spec_from_file_location('qualified_run_common', HERE/'agent-expert-pv-qualified-common.py')
  value = importlib.util.module_from_spec(spec); spec.loader.exec_module(value)
  return value


def original(C): return C.usb().load('agent-expert-pv-route-run')


def plan(C, phase, output):
  result = original(C).plan(C.SNAP.parent/'invocations'/f'{phase}-0.json', output)
  assert (result['warmup'], result['runs']) == COUNTS[phase]
  return result


def validate(C, report, sidecar, planned, candidate, evidence, protocol_sha):
  meta = report['agent_expert_pv']
  assert meta['protocol_sha256'] == protocol_sha and meta['variant'] == ('candidate' if candidate else 'current')
  assert meta['route_completed'] and meta['coverage_validated'] and not meta.get('failure') and not meta['plan_only']
  assert meta['plan'] == planned and meta['evidence'] == evidence
  assert len(meta['hooks']) == 1
  info = meta['hooks'][0]
  assert info['candidate'] is candidate and info['evidence'] == evidence
  assert info['cleanup_exact'] and info['observer_cleanup_exact'] and not info['active']
  assert info['trajectory_calls'] >= planned['warmup']+planned['runs']
  assert info['unqualified_trajectory_calls'] == info['outside_scope_selected'] == 0
  assert info['traces'] and all(t == {'pv': 84, 'qk': 84} for t in info['traces'])
  assert info['bound_program_calls'] == (84*len(info['traces']) if candidate else 0)
  assert info['captures']
  expected = evidence['candidate' if candidate else 'current']
  for capture in info['captures']:
    assert capture['pv_calls'] == capture['qk_calls'] == capture['softmax_calls'] == 84
    assert capture['target'] == evidence['target'] and capture['program_keys'] == [expected['program_key']]
    assert capture['pv_full_description'] == expected
    assert capture['binaries'][expected['name']] == [expected['binary_sha256']]
  assert report['arch'] == 'gfx1200' and report['settings']['diffusion_steps'] == 3 and report['trajectory_solver'] == 'euler'
  assert report['llvm_wgp'] and report['expert_kernels'] and report['pixel_scale'] == 1 and not report['online_expert']
  assert report['vision_calibration_sha256'] is None and not report['summary']['profiling_enabled']
  assert report['runtime_environment']['AM_POWER_LIMIT'] == '144' and report['runtime_environment']['AM_PERF_LEVEL'] == 'auto'
  sources = [*(C.SNAP/'experiments').glob('*.py'), *(C.SNAP/'experiments').glob('*.c'), C.SNAP/'model.py']
  hashes = {p.name: C.sha(p.read_bytes()) for p in sources}
  assert report['source_sha256'] == hashes
  for name, digest in hashes.items(): assert C.sha((Path(report['source_snapshot'])/name).read_bytes()) == digest
  R = original(C)
  terminal = planned['terminal_command']
  assert report['prefill_calibration_sha256'] == C.sha(Path(R.option(terminal, '--prefill-calibration')).read_bytes())
  for field, flag in (('manifest', '--native-manifest'), ('calibrations', '--native-calibrations')):
    assert report['native_preprocessing'][field+'_sha256'] == C.sha(Path(R.option(terminal, flag)).read_bytes())
  return R.compare_reports(report, report, planned, sidecar, sidecar)


def compare(C, before, after, planned, before_spec, after_spec):
  result = original(C).compare_reports(before, after, planned, before_spec, after_spec)
  def hashes(report, name):
    return {value for capture in report['agent_expert_pv']['hooks'][0]['captures'] for value in capture['binaries'][name]}
  for name in ('alpamayo_attention_scores', 'alpamayo_attention_softmax'):
    assert hashes(before, name) == hashes(after, name), 'Non-PV arithmetic changed: '+name
  return result


def main():
  p = argparse.ArgumentParser(description=__doc__)
  p.add_argument('--output', type=Path, required=True)
  p.add_argument('--phase', choices=COUNTS, required=True)
  p.add_argument('--variant', choices=('current', 'candidate'), required=True)
  p.add_argument('--protocol', type=Path, required=True)
  p.add_argument('--protocol-sha', required=True)
  p.add_argument('--reference', type=Path)
  p.add_argument('--plan', action='store_true')
  args = p.parse_args()
  assert sys.flags.optimize == 0
  C = common()
  protocol, evidence = C.protocol(args.protocol, args.protocol_sha)
  C.paths()
  planned = plan(C, args.phase, args.output)
  meta = {'variant': args.variant, 'phase': args.phase, 'protocol_sha256': args.protocol_sha, 'plan_only': args.plan,
    'plan': planned, 'evidence': evidence, 'route_completed': False, 'coverage_validated': False,
    'exactness_validated': False, 'hooks': []}
  assert not args.output.exists()
  metadata = args.output.with_suffix('.agent-expert-pv-qualified.json')
  def save(): C.usb().old().write(metadata, meta)
  if args.plan:
    C.usb().old().write(args.output, {**meta, 'gpu_executed': False, 'import_path_checked': True})
    return
  reference = json.loads(args.reference.read_text()) if args.reference else None
  reference_spec = json.loads(args.reference.with_suffix('.speculation.json').read_text()) if args.reference else None
  if args.variant == 'candidate':
    assert reference is not None
    reference_plan = plan(C, args.phase, args.reference)
    validate(C, reference, reference_spec, reference_plan, False, evidence, args.protocol_sha)
    assert reference['agent_expert_pv']['plan']['pair_identity_sha256'] == planned['pair_identity_sha256']
  else: assert reference is None
  env, argv, search = os.environ.copy(), sys.argv[:], sys.path[:]
  handles = []
  save()
  try:
    invocation = json.loads((C.SNAP.parent/'invocations'/f'{args.phase}-0.json').read_text())
    for key, value in invocation['environment'].items():
      if value is None: os.environ.pop(key, None)
      else: os.environ[key] = value
    os.environ.update(PROFILE='0', PYTHONDONTWRITEBYTECODE='1')
    C.paths()
    sys.argv = planned['effective_command']
    H = C.local('agent-expert-pv-qualified-hook')
    with original(C).interceptor(H, args.variant == 'candidate', evidence, handles):
      runpy.run_path(sys.argv[0], run_name='__main__')
    assert len(handles) == 1
    meta['hooks'] = [h.info for h in handles]
    meta.update(route_completed=True, coverage_validated=True)
    result = json.loads(args.output.read_text())
    result['agent_expert_pv'] = meta
    sidecar = json.loads(args.output.with_suffix('.speculation.json').read_text())
    validate(C, result, sidecar, planned, args.variant == 'candidate', evidence, args.protocol_sha)
    if reference is not None:
      meta['comparison'] = compare(C, reference, result, planned, reference_spec, sidecar)
      meta['exactness_validated'] = True
    C.protocol(args.protocol, args.protocol_sha)
  except BaseException as exc:
    meta['failure'] = {'type': type(exc).__name__, 'message': str(exc)}
    meta['coverage_validated'] = meta['exactness_validated'] = False
    raise
  finally:
    meta['hooks'] = [h.info for h in handles]  # Retain bindings even on failure.
    os.environ.clear(); os.environ.update(env)
    sys.argv, sys.path = argv, search
    save()
    if args.output.exists():
      result = json.loads(args.output.read_text()); result['agent_expert_pv'] = meta
      C.usb().old().write(args.output, result)


if __name__ == '__main__': main()
