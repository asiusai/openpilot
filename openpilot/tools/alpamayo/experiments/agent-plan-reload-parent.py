"""WORK-only protocol freeze, CPU preflight, and optional bounded parent GPU submission."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

WORK=Path('/mnt/nvme-500/tmp/alpamayo-speed-20261116');EXP=Path(__file__).resolve().parent
SOURCE=WORK/'agent-paced-dynamic-source-cpu02/alpamayo'
TG=Path('/tmp/opencode/alpamayo-asius-20261114/tinygrad')
PY=Path('/tmp/opencode/alpamayo-asius-20261114/upstream/.venv/bin/python')
sys.dont_write_bytecode=True
sys.path[:0]=[str(TG),str(SOURCE),str(SOURCE/'experiments')]

def local(path):
  spec=importlib.util.spec_from_file_location(Path(path).stem.replace('-','_'),path)
  module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
  return module

def freeze(path):
  if path.exists() or not path.resolve().is_relative_to(WORK): raise ValueError('Use a new protocol under WORK')
  loader=local(EXP/'agent-plan-reload-loader.py');bind=local(EXP/'agent-plan-bind-run.py')
  for name,digest in bind.PINS.items():
    if loader.sha(TG/name)!=digest: raise ValueError('Qualified runtime boundary changed')
  files=[WORK/'agent-plan-bind-capture1/report.json',WORK/'agent-plan-bind-capture1/agent-plan-bind-capture/audit.json',
    WORK/'agent-startup-key-fix1-replay/report.json',WORK/'startup-cache-parent-source/agent-startup-spec-warm.py',
    SOURCE.parent/'pipeline-invocation.json',Path('/dev/shm/alpamayo-9060-fp8/index.json'),
    Path('/tmp/opencode/run-alpamayo-speed-trial.py')]
  files+=list(SOURCE.rglob('*.py'))
  files+=list(Path('/tmp/opencode/alpamayo-asius-20261114/model').glob('*.json'))
  qualified=json.loads(bind.QUALIFIED.read_text())
  for record in qualified['protocol']['files'].values():
    if isinstance(record,dict) and 'path' in record: files.append(Path(record['path']))
  protocol=dict(schema=1,scope='One exact count1 persistent plan; no whole-model or live adoption',
    tinygrad_sources=loader.runtime_sources(),reload_sources={p.name:loader.sha(p) for p in EXP.glob('agent-plan-reload-*.py')},
    bind_sources={p.name:loader.sha(p) for p in EXP.glob('agent-plan-bind-*.py')},
    files={str(p):loader.sha(p) for p in sorted(set(files))},full_target='USB+AMD:LLVM:gfx1200',child_timeout_seconds=300)
  path.write_text(json.dumps(protocol,indent=2)+'\n')
  print(json.dumps(dict(protocol=str(path),sha256=loader.sha(path),runtime_files=len(protocol['tinygrad_sources']))))

def main():
  parser=argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--tag',default='reload1');parser.add_argument('--execute-gpu',action='store_true')
  parser.add_argument('--protocol',type=Path,default=WORK/'agent-plan-reload-protocol2.json')
  parser.add_argument('--freeze-protocol',action='store_true');args=parser.parse_args()
  parent_tmp=WORK/'agent-plan-reload-parent-tmp';parent_tmp.mkdir(exist_ok=True)
  os.environ.update(TMPDIR=str(parent_tmp),CACHEDB=str(WORK/'agent-plan-reload-parent-cache.db'),PYTHONDONTWRITEBYTECODE='1')
  if args.freeze_protocol:
    if args.execute_gpu: raise ValueError('CPU protocol freeze cannot execute hardware')
    freeze(args.protocol);return
  if not re.fullmatch('[A-Za-z0-9_-]+',args.tag): raise ValueError('Invalid tag')
  loader=local(EXP/'agent-plan-reload-loader.py');protocol=json.loads(args.protocol.read_text());loader.source_gate(protocol)
  storage=WORK/('agent-plan-reload-'+args.tag+'-storage')
  if storage.exists(): raise FileExistsError('Use a new tag; earlier evidence is immutable')
  if shutil.disk_usage(WORK).free<5*1024**3: raise RuntimeError('WORK reserve is below five GiB')
  storage.mkdir()
  for name in ('tmp','xdg','torch'): (storage/name).mkdir()
  original=WORK/'agent-plan-bind-capture1-storage/compiler.db';cache=storage/'compiler.db'
  # Copy the completed old cache files without opening its SQLite database or
  # writing its WAL/SHM files. Reject a concurrent source change.
  copied={}
  for suffix in ('','-wal','-shm'):
    source=Path(str(original)+suffix)
    if not source.exists(): continue
    before=loader.sha(source);target=Path(str(cache)+suffix);shutil.copyfile(source,target)
    if loader.sha(source)!=before or loader.sha(target)!=before: raise ValueError('Source compiler cache changed during copy')
    copied[str(source)]=dict(sha256=before,bytes=target.stat().st_size)
  environment={k:v for k,v in json.loads((WORK/'agent-plan-bind-capture1/report.json').read_text())['protocol']['environment'].items() if v is not None}
  environment.update(PYTHONPATH=f'{TG}:{SOURCE}:/tmp/opencode/openpilot-alpamayo:/home/john/asius/openpilot/opendbc_repo',
    DEV='USB+AMD:LLVM',CACHEDB=str(cache),TMPDIR=str(storage/'tmp'),XDG_CACHE_HOME=str(storage/'xdg'),TORCH_HOME=str(storage/'torch'),
    PYTHONDONTWRITEBYTECODE='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
  preflight=[str(PY),str(EXP/'agent-plan-reload-run.py'),'--preflight','--output',str(storage/'preflight.json'),
    '--cache-db',str(cache),'--protocol',str(args.protocol)]
  subprocess.run(preflight,env={**os.environ,**environment},check=True,timeout=60)
  command=['python3','/tmp/opencode/run-alpamayo-speed-trial.py','agent-plan-reload-'+args.tag,
    '--program',str(EXP/'agent-plan-reload-run.py'),'--timeout-seconds','300',
    '--set','cache-db',str(cache),'--set','protocol',str(args.protocol),'--flag','execute-gpu']
  for key,value in environment.items(): command.extend(('--env',key,value))
  plan=dict(command=command,protocol=str(args.protocol),protocol_sha256=loader.sha(args.protocol),cache_source=copied,
    environment=environment,preflight=str(storage/'preflight.json'),child_timeout_seconds=300)
  (storage/'parent-command.json').write_text(json.dumps(plan,indent=2)+'\n');print(json.dumps(plan,indent=2),flush=True)
  if args.execute_gpu: subprocess.run(command,check=True)

if __name__=='__main__': main()
