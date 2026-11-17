"""Parent-only two-child quality sequence. No continuous run or adoption."""
import argparse
import importlib.util
from pathlib import Path
import subprocess
import sys

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('adaptive_parent_run',HERE/'agent-adaptive-prefill-run.py')
R=importlib.util.module_from_spec(spec);spec.loader.exec_module(R)


def main():
  p=argparse.ArgumentParser(description=__doc__);p.add_argument('--plan',action='store_true');args=p.parse_args()
  plan=R.verify_plan()
  outputs={arm:R.WORK/f'agent-adaptive-prefill-quality-{arm}'/'report.json' for arm in ('reference','candidate')}
  commands=[]
  for arm,path in outputs.items():
    if path.parent.exists():raise FileExistsError('Use the declared exclusive unused output: '+str(path.parent))
    command=[R.PY,str(R.WRAPPER),path.parent.name,'--program',str(HERE/'agent-adaptive-prefill-run.py'),
      '--timeout-seconds','900','--set','arm',arm,
      '--env','PYTHONDONTWRITEBYTECODE','1',
      '--env','PYTHONPATH','/tmp/opencode/alpamayo-asius-20261114/tinygrad',
      '--env','CACHEDB',str(R.ARTIFACTS/'gpu-cache.db'),'--env','TMPDIR',str(R.ARTIFACTS/'tmp')]
    if arm=='candidate':command.extend(['--set','reference',str(outputs['reference'])])
    commands.append(command)
  if args.plan:
    import json
    print(json.dumps({'plan_sha256':R.sha(R.MANIFEST),'queries_each':18,'commands':commands,
      'automatic_paced':False,'numerical_candidate':True},indent=2));return
  status_path=R.WORK/'agent-adaptive-prefill-queue-result.json'
  if status_path.exists():raise FileExistsError(status_path)
  rows=[]
  for arm,command in zip(outputs,commands,strict=True):
    result=subprocess.run(command,timeout=960)
    rows.append({'arm':arm,'returncode':result.returncode})
    R.write(status_path,{'children':rows,'complete':False,'automatic_paced':False},new=len(rows)==1)
    if result.returncode:raise SystemExit(result.returncode)
    status=R.read(outputs[arm].parent/'adaptive-prefill-status.json')
    assert status['complete'] and status['import_hook_restored'] and status['adaptive_hook_restored']
  R.write(status_path,{'children':rows,'complete':True,'automatic_paced':False,
    'candidate_status':str(outputs['candidate'].parent/'adaptive-prefill-status.json'),
    'quality':'Numerical differences require parent review; no automatic accuracy approval or performance claim'},new=False)


if __name__=='__main__':main()
