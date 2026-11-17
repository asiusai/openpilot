"""Invoke the unchanged arrival runner with a separately source-bound ownership adapter."""
import argparse
import importlib.util
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('arrival_ownership_common', HERE/'agent-arrival-warp-ownership-common.py')
A = importlib.util.module_from_spec(spec); spec.loader.exec_module(A)


def run(args):
  A.configure()
  original = A.original_load(HERE/'agent-arrival-warp-integrated-run.py', 'arrival_ownership_original_runner')
  original.A, original.C = A, A.C
  side = args.output.parent/'arrival-warp-ownership-status.json'
  assert not side.exists()
  state = dict(complete=False, gpu_executed=False, cpu_only=args.cpu_preflight, protocol_sha256=args.protocol_sha)
  try:
    original.run(args)
    assert len(A.ownership) == 1
    info = A.ownership[0]
    assert info['entered'] and info['checks'] == info['completed'] == info['scoped_restorations'] == 1 and info['finally_restored']
    report = A.read(args.output)
    assert report['agent_arrival_warp_integrated']['passed'] and report['agent_arrival_warp_integrated']['source_after_exact']
    state.update(complete=True, gpu_executed=True, report_sha256=A.sha(args.output))
  except BaseException as error:
    state['failure'] = dict(type=type(error).__name__, message=str(error)); raise
  finally:
    state['ownership'] = A.ownership
    if args.output.parent.exists(): A.write(side, state, new=True)


if __name__ == '__main__':
  p = argparse.ArgumentParser(description=__doc__)
  p.add_argument('--protocol',type=Path,required=True); p.add_argument('--protocol-sha',required=True)
  p.add_argument('--output',type=Path,required=True); p.add_argument('--cpu-preflight',action='store_true')
  run(p.parse_args())
