"""Parent-reviewed native pair. Default is CPU preflight; no GPU on import."""
import argparse
from contextlib import ExitStack
import importlib.util
import json
import os
from pathlib import Path
import sys
from unittest.mock import patch

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('agent_token75_timing_run_common',HERE/'agent-token75-timing-common.py')
common=importlib.util.module_from_spec(spec);spec.loader.exec_module(common)


def main():
  parser=argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--arm',choices=('reference','candidate'),required=True)
  parser.add_argument('--output',type=Path,required=True)
  parser.add_argument('--protocol',type=Path,default=common.PROTOCOL)
  parser.add_argument('--execute-gpu',action='store_true',help='PARENT ONLY, after trajectory/speed/slowing/reasoning review')
  args=parser.parse_args();plan=common.read(args.protocol);common.verify(plan)
  saved=common.read(plan['invocations'][args.arm])
  route=common.module(common.FEXP/'agent-token-current-route.py','agent_token75_timing_frozen_route')
  route.verify_qualification(Path(plan['relocated_cpu_report']),common.FEXP/'agent-token-current-hook-report.json',
                             common.FEXP/'agent-token-current-compiler-report.json')
  command=route.qualify(saved,args.output)
  if not args.execute_gpu:
    print(json.dumps(dict(cpu_only=True,gpu_executed=False,arm=args.arm,command=command,
      policy=common.POLICY if args.arm=='candidate' else 'unpruned ordinary lookup',
      recipe=plan['recipe'],parent_review_required=True,compatibility_gpu_gate_required=False),indent=2))
    return
  if args.output.exists(): raise FileExistsError('Never overwrite timing evidence')
  for key,value in saved['environment'].items():
    if value is None: os.environ.pop(key,None)
    else: os.environ[key]=value
  for path in reversed(saved['environment']['PYTHONPATH'].split(os.pathsep)):
    if path:sys.path.insert(0,path)
  sys.path.insert(0,str(common.FEXP))
  pipeline=common.module(common.FEXP/'benchmark-pipeline.py','agent_token75_timing_pipeline')
  initializer=pipeline.initialize;owners=[]
  with ExitStack() as stack:
    if args.arm=='candidate':
      factory=common.module(common.FEXP/'agent-token-current-runtime.py','agent_token75_timing_frozen_factory')
      runtime=factory.make_runtime(common.POLICY)
      route.install_pipeline_hooks(pipeline,runtime,stack,owners)
    stack.enter_context(patch.object(sys,'argv',command))
    pipeline.main()
    common.require(pipeline.initialize is initializer,'Protected initializer changed')
    report=common.read(args.output)
    if args.arm=='candidate':
      common.require(len(owners)==1 and len(owners[0].events)==36,'Missing native policy decisions')
      report['token75_timing_policy_events']=owners[0].events
    common.validate_output(report,plan,args.arm)
    report['token75_native_timing_protocol']=dict(arm=args.arm,protocol_sha256=common.sha(args.protocol),
      classification='NUMERICAL75% model candidate' if args.arm=='candidate' else 'Unpruned paired reference',
      fullroute_quality_proven=False,quality_adoption_passed=None,native_continuous_Asius_validated=False,
      measurement='Native RGB load/calibration through complete output, one CPU preparation overlap; archival5Hz fixture, not live exposure latency',
      gather_measurement='All score readback/selection/gather/scatter costs included in prefill and outer service; isolated GPU gather time unavailable without profiling')
    args.output.write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
