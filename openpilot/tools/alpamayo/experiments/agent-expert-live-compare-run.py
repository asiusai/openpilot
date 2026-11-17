"""Execute the byte-pinned storage runner with explicit path/comparison overlays only."""
import argparse
import importlib.util
from pathlib import Path

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('compare_run_common',HERE/'agent-expert-live-compare-common.py')
V=importlib.util.module_from_spec(spec);spec.loader.exec_module(V)


def run(args):
  V.configure();data=V.verify(args.protocol,args.protocol_sha)
  assert args.arm in ('candidate','paced')
  assert args.output==V.WORK/f'agent-expert-live-compare-{args.arm}/report.json'
  assert not any(args.output.parent.glob('report*'))
  execute,runtime,overlay=V.runtime_module();assert overlay==data['runtime_overlay']
  status=dict(passed=False,gpu_executed=False,arm=args.arm,protocol_sha256=args.protocol_sha,
    runtime_overlay=overlay,overlay_helper_sha256=data['overlay_helper_sha256'],
    current_admission_sha256=V.sha(V.ADMISSION),admitted_current_sha256=V.CURRENT_SHA,
    original_current_passed=False,automatic_paced=False,current_gpu_rerun=False)
  args.output.parent.mkdir(exist_ok=True)
  V.write(args.output.parent/'agent-expert-live-compare-overlay.json',status,new=True)
  try:
    execute(args)
    report=V.read(args.output)
    assert report['agent_expert_live_storage']['passed']
    if args.arm=='candidate':status['paired_exact']=V.paired(report,data,args.protocol_sha)
    else:
      assert args.reviewed_parity_sha256==V.sha(V.CANDIDATE)
      V.N.validate_startup(report,paced=True)
      runtime.check_storage(report['agent_expert_live_storage']['storage_events'],paced=True)
    status['passed']=True;status['gpu_executed']=True
  except BaseException as error:
    status['failure']=dict(type=type(error).__name__,message=str(error));raise
  finally:
    if args.output.exists():
      report=V.read(args.output)
      status['gpu_executed']=report.get('gpu_executed',report['agent_expert_live_storage']['gpu_executed'])
      report['agent_expert_live_compare']=status;V.write(args.output,report)
    V.write(args.output.with_suffix('.agent-expert-live-compare.json'),status)


if __name__=='__main__':
  p=argparse.ArgumentParser(description=__doc__)
  p.add_argument('--arm',choices=('candidate','paced'),required=True)
  p.add_argument('--protocol',type=Path,required=True);p.add_argument('--protocol-sha',required=True)
  p.add_argument('--reviewed-parity-sha256');p.add_argument('--output',type=Path,required=True)
  run(p.parse_args())
