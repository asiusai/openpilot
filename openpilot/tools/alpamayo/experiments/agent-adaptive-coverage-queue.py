"""Parent-only one-child candidate check; observed reference predictions are reused."""
import argparse
import importlib.util
import json
from pathlib import Path
import subprocess

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('coverage_queue_runner',HERE/'agent-adaptive-coverage-run.py')
R=importlib.util.module_from_spec(spec);spec.loader.exec_module(R)


def main():
  p=argparse.ArgumentParser();p.add_argument('--plan',action='store_true');args=p.parse_args()
  R.verify();old,_,_,C,_=R.K.frozen();receipt=R.K.reference_receipt(C)
  assert receipt['reference_predictions_admitted'] and receipt['numerical_reference_only']
  output=R.K.WORK/'agent-adaptive-coverage-quality-candidate'
  if output.exists():raise FileExistsError('Use the declared new output directory: '+str(output))
  command=[old.PY,str(old.WRAPPER),output.name,'--program',str(HERE/'agent-adaptive-coverage-run.py'),
    '--timeout-seconds','900','--env','PYTHONDONTWRITEBYTECODE','1',
    '--env','PYTHONPATH','/tmp/opencode/alpamayo-asius-20261114/tinygrad',
    '--env','CACHEDB',str(R.ARTIFACTS/'gpu-cache.db'),'--env','TMPDIR',str(R.ARTIFACTS/'tmp')]
  if args.plan:
    print(json.dumps({'candidate_only':True,'queries':18,'timeout_seconds':900,'command':command,
      'manifest_sha256':R.K.sha(R.MANIFEST),'reference_receipt_sha256':R.K.sha(R.K.RECEIPT),'automatic_paced':False},indent=2));return
  path=R.K.WORK/'agent-adaptive-coverage-queue-result.json'
  if path.exists():raise FileExistsError(path)
  child=subprocess.run(command,timeout=960)
  R.K.write(path,{'returncode':child.returncode,'candidate_only':True,'reference_predictions_reused':True,
    'automatic_paced':False,'status':str(output/'adaptive-coverage-status.json')},new=True)
  raise SystemExit(child.returncode)


if __name__=='__main__':main()
