"""Frozen native36 pair. Default: CPU preflight. Parent must review quality before launch."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import subprocess

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('agent_token75_timing_queue_common',HERE/'agent-token75-timing-common.py')
c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c)


def main():
  parser=argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--launch-after-parent-review',action='store_true',help='PARENT ONLY: confirms completed trajectory/speed/slowing/reasoning review')
  args=parser.parse_args();plan=c.read(c.PROTOCOL);c.verify(plan)
  route=c.module(c.FEXP/'agent-token-current-route.py','agent_token75_timing_queue_route')
  route.verify_qualification(Path(plan['relocated_cpu_report']),c.FEXP/'agent-token-current-hook-report.json',c.FEXP/'agent-token-current-compiler-report.json')
  commands=[]
  for job in plan['jobs']:
    invocation=c.read(plan['invocations'][job['arm']])
    command=route.qualify(invocation,Path(job['report']))
    c.require(c.paths(command)==plan['samples'],'Paired native input order changed')
    commands.append(command)
  normalized=[]
  for command in commands:
    value=command.copy();value[value.index('--output')+1]='<arm-output>';normalized.append(value)
  c.require(normalized[0]==normalized[1],'Reference and candidate recipes differ')
  if not args.launch_after_parent_review:
    print(json.dumps(dict(cpu_only=True,gpu_executed=False,parent_review_required=True,policy=c.POLICY,
      jobs=plan['jobs'],comparison_command=plan['comparison_command'],recipe=plan['recipe'],automatic_adoption_or_fullroute=False),indent=2))
    return
  # The existing runner owns the serial GPU lock, device identity checks, ACL restoration, and child900s cap.
  for job in plan['jobs']:
    c.require(not Path(job['report']).parent.exists(),'Never overwrite or retry an existing timing trial automatically')
  env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1')
  for job in plan['jobs']:
    c.verify(plan)
    subprocess.run(job['command'],env=env,check=True)
  subprocess.run(plan['comparison_command'],env=dict(env,DEV='CPU',ALLOW_DEVICE_USAGE='0'),check=True)


if __name__=='__main__':main()
