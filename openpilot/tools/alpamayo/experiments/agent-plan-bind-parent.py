"""Prepare WORK-only storage and optionally submit the bounded parent diagnostic."""
import argparse
import json
import os
from pathlib import Path
import re
import shutil
import sqlite3
import subprocess

WORK=Path('/mnt/nvme-500/tmp/alpamayo-speed-20261116')
EXP=Path(__file__).resolve().parent
TG=Path('/tmp/opencode/alpamayo-asius-20261114/tinygrad')
SOURCE=WORK/'agent-paced-dynamic-source-cpu02/alpamayo'


def main():
  parser=argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--tag',default='capture1')
  parser.add_argument('--execute-gpu',action='store_true')
  args=parser.parse_args()
  if not re.fullmatch('[A-Za-z0-9_-]+',args.tag): raise ValueError('Invalid tag')
  storage=WORK/('agent-plan-bind-'+args.tag+'-storage')
  if storage.exists(): raise FileExistsError('Use a new tag; all previous artifacts remain intact')
  if shutil.disk_usage(WORK).free<5*1024**3: raise RuntimeError('WORK has insufficient diagnostic reserve')
  storage.mkdir()
  for name in ('tmp','xdg','torch'): (storage/name).mkdir()
  cache=storage/'compiler.db'
  original=Path('/tmp/opencode/alpamayo-asius-20261114/cache.db')
  with sqlite3.connect(original.as_uri()+'?mode=ro',uri=True) as source,sqlite3.connect(cache) as target:
    source.backup(target)
  environment=dict(PYTHONPATH=f'{TG}:{SOURCE}:/tmp/opencode/openpilot-alpamayo:/home/john/asius/openpilot/opendbc_repo',
    CACHEDB=str(cache),TMPDIR=str(storage/'tmp'),XDG_CACHE_HOME=str(storage/'xdg'),TORCH_HOME=str(storage/'torch'),
    PYTHONDONTWRITEBYTECODE='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
  command=['python3','/tmp/opencode/run-alpamayo-speed-trial.py','agent-plan-bind-'+args.tag,
    '--program',str(EXP/'agent-plan-bind-run.py'),'--timeout-seconds','300',
    '--set','cache-db',str(cache),'--flag','execute-gpu']
  for key,value in environment.items(): command.extend(('--env',key,value))
  plan=dict(command=command,environment=environment,original_cache=str(original),cache_copy=str(cache),
    cache_copy_bytes=cache.stat().st_size,child_timeout_seconds=300,
    scope='One actual serial count1 metadata capture; no whole-model warmup, full64, persisted-plan execution, or timing claim')
  with (storage/'parent-command.json').open('x') as stream: json.dump(plan,stream,indent=2);stream.write('\n')
  print(json.dumps(plan,indent=2),flush=True)
  if args.execute_gpu: subprocess.run(command,check=True)


if __name__=='__main__': main()
