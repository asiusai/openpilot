"""One explicitly requested paced run, reusing the exact passed Consumer pair."""
import argparse
import importlib.util
from pathlib import Path
from types import SimpleNamespace

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('stem_run_common',HERE/'agent-expert-paced-stem-common.py')
C=importlib.util.module_from_spec(spec);spec.loader.exec_module(C)


def run(args):
  C.configure();data=C.verify(args.protocol,args.protocol_sha)
  assert args.reviewed_parity_sha256==C.CANDIDATE_SHA and args.output==Path(data['output'])
  assert not args.output.exists(),'Never overwrite an existing canonical report'
  C.validate_paths(args.output,{k:Path(v) for k,v in data['metadata'].items()})
  paths=C.metadata(args.output)
  assert not any(path.exists() for path in paths.values()),'Never overwrite path-repair metadata'
  args.output.parent.mkdir(parents=True,exist_ok=True)
  status=dict(passed=False,gpu_executed=False,protocol_sha256=args.protocol_sha,candidate_sha256=C.CANDIDATE_SHA,
    candidate_admission=data['candidate_admission'],artifact_overlay=data['artifact_overlay'],
    overlay_helper_sha256=data['overlay_helper_sha256'],no_parity_rerun=True,automatic_paced=False,cpu_only=data['cpu_only'])
  execute,runtime,proof=C.build(args.output,repair=data['repair']);assert proof==data['artifact_overlay']
  values=SimpleNamespace(arm='paced',protocol=C.QUALIFIED_PROTOCOL,protocol_sha=C.QUALIFIED_SHA,
    output=args.output,reviewed_parity_sha256=args.reviewed_parity_sha256)
  try:
    execute(values)
    report=C.read(args.output);old=report['agent_expert_live_storage'];overlay=report['agent_expert_live_compare']
    assert not data['cpu_only'] and old['passed'] and old['gpu_executed'] and old['hooks_restored'] and overlay['passed']
    runtime.check_storage(old['storage_events'],paced=True)
    C.N.validate_startup(report,paced=True)
    C.N.validate_pv(old['pv'],True,3+len(report['rows']),C.read(C.QUALIFIED_PROTOCOL)['evidence']['pv'])
    status.update(passed=True,gpu_executed=True)
  except BaseException as error:
    status['failure']=dict(type=type(error).__name__,message=str(error));raise
  finally:
    if args.output.exists():
      report=C.read(args.output);status['gpu_executed']=report.get('agent_expert_live_storage',{}).get('gpu_executed',False)
      report['agent_expert_paced_stem']=status;C.write(args.output,report)
    C.write(paths['status'],status,new=True)


if __name__=='__main__':
  p=argparse.ArgumentParser(description=__doc__)
  p.add_argument('--protocol',type=Path,required=True);p.add_argument('--protocol-sha',required=True)
  p.add_argument('--reviewed-parity-sha256',required=True);p.add_argument('--output',type=Path,required=True)
  run(p.parse_args())
