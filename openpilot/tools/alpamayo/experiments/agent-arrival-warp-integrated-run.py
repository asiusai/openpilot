"""Parent-only continuous entrypoint using the unchanged selected Consumer composition."""
import argparse
from contextlib import ExitStack
import copy
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('arrival_integrated_common', HERE/'agent-arrival-warp-integrated-common.py')
A = importlib.util.module_from_spec(spec); spec.loader.exec_module(A)
C = A.C


def run(args):
  assert not sys.flags.optimize
  A.configure(); data = A.verify(args.protocol, args.protocol_sha, cpu_only=args.cpu_preflight)
  assert args.output.resolve().is_relative_to(A.WORK)
  assert not any(args.output.parent.glob(args.output.stem+'*'))
  side = args.output.parent/'arrival-warp-integrated-status.json'
  assert not side.exists()
  C.N.reserve(args.output.parent); args.output.parent.mkdir(parents=True, exist_ok=True)
  admission = args.output.parent/'arrival-cache-admission.json'
  env = {**os.environ, **C.N.storage(), 'DEV':'CPU', 'ALLOW_DEVICE_USAGE':'0', 'TC_OPT':'2', 'CCACHE':'0', 'PYTHONDONTWRITEBYTECODE':'1'}
  child = subprocess.run([C.N.PY, str(HERE/'agent-arrival-warp-integrated-check.py'), '--admit', '--output', str(admission)],
    env=env, capture_output=True, text=True, timeout=120)
  admission.with_suffix('.log').write_text(child.stdout+child.stderr); child.check_returncode()
  proof = C.read(admission)
  assert proof['complete'] and not proof['gpu_executed'] and proof['factory']['real_device_open_attempts'] == []
  m, manifest = C.N.live()
  status = dict(passed=False, gpu_executed=False, protocol_sha256=args.protocol_sha,
    selected_consumer_sha256=A.PINS[C.output('consumer')], selected_paced_sha256=A.PINS[C.output('paced')],
    cpu_only=args.cpu_preflight, unchanged_numerical_installers=True, hooks_restored=False)
  C.write(side, status, new=True)
  previous = importlib.util.spec_from_file_location
  hooks = None
  try:
    with ExitStack() as stack:
      stack.enter_context(patch.dict(os.environ)); stack.enter_context(patch.object(sys, 'path', sys.path[:]))
      for key, value in m.read(m.INVOCATION)['environment'].items():
        if value is None: os.environ.pop(key, None)
        else: os.environ[key] = str(value)
      A.configure()
      hooks = A.load(HERE/'agent-arrival-warp-integrated-install.py', 'arrival_integrated_install').factory(A)('paced')
      with hooks:
        module = m.load(m.ROOT/'experiments/agent-paced-replay.py', 'arrival_integrated_paced')
        module.main(['--mode','run','--invocation',str(m.INVOCATION),'--consumer-parity-report',str(C.output('consumer')),
          '--hz','10','--start-seconds','35','--seconds','8','--inference-budget-seconds','180',
          '--raw-frames',str(m.RGB),'--warp','native','--output',str(args.output)])
        report = C.read(args.output)
        status.update(gpu_executed=True, storage_events=copy.deepcopy(hooks.storage_events),
          transforms=dict(hooks.transforms), startup_spec=copy.deepcopy(hooks.startup_report))
        m.equal(hooks.transforms, manifest['expected_transforms'], 'original_startup_and_policy_transforms')
        status['storage_proof'] = C.S.check_storage(hooks.storage_events, paced=True)
        status['coverage'] = hooks.coverage()
        status['quantizer_idle'] = hooks.composition.session.frame is None and not hooks.composition.session.state['prefill']
        assert status['quantizer_idle']
        report['startup_spec'] = copy.deepcopy(hooks.startup_report)
      status.update(pv=hooks.composition.pv_handle.info, composition_restorations=hooks.composition.restorations,
        constructor_restored=hooks.composition.constructor_restored, cpu_hook=copy.deepcopy(hooks.cpu_info),
        hooks_restored=importlib.util.spec_from_file_location is previous and hooks.composition.session.stats['methods_restored'],
        passed=True)
      # Retain every original numerical/storage/startup/Consumer gate.
      C.validate_status(report, status, 'paced', args.output)
      arrival = hooks.arrival.report()
      assert arrival['closed'] and arrival['producer_binding_restored'] and arrival['cache_cleanup_bytes'] == 0
      assert arrival['active']['frames'] > 0 and arrival['prime']['frames'] > 0
      assert len(arrival['arrivals']) == report['bounds']['counts']['received_frames']
      status['arrival'] = arrival
      status['source_after_exact'] = A.verify(args.protocol, args.protocol_sha)['files'] == data['files']
      report['agent_arrival_warp_integrated'] = status
      C.write(args.output, report)
  except BaseException as error:
    status.update(passed=False, failure=dict(type=type(error).__name__, message=str(error)))
    raise
  finally:
    status['hooks_restored'] = importlib.util.spec_from_file_location is previous and (hooks is None or
      hooks.composition.session is None or hooks.composition.session.stats['methods_restored'])
    if hooks is not None:
      status.update(arrival=hooks.arrival.report(), cpu_hook=copy.deepcopy(hooks.cpu_info),
        storage_events=copy.deepcopy(hooks.storage_events), constructor_restored=hooks.composition.constructor_restored)
    C.write(side, status)


if __name__ == '__main__':
  p = argparse.ArgumentParser(description=__doc__)
  p.add_argument('--protocol', type=Path, required=True); p.add_argument('--protocol-sha', required=True)
  p.add_argument('--output', type=Path, required=True); p.add_argument('--cpu-preflight', action='store_true')
  run(p.parse_args())
