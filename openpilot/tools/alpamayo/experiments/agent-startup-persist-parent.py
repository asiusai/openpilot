"""Normal serial ACL wrapper; two fresh bounded startup arms, no paced/full64 queue."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

EXP=Path(__file__).resolve().parent
WORK=Path('/mnt/nvme-500/tmp/alpamayo-speed-20261116')
TG=Path('/tmp/opencode/alpamayo-asius-20261114/tinygrad')
PY=Path('/tmp/opencode/alpamayo-asius-20261114/upstream/.venv/bin/python')
sys.dont_write_bytecode=True;sys.path.insert(0,str(TG))
spec=importlib.util.spec_from_file_location('startup_persist_parent_common',EXP/'agent-startup-persist-common.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

def environment(storage):
  prior=m.read(WORK/'agent-plan-bind-capture1/report.json')['protocol']['environment']
  result={k:str(v) for k,v in prior.items() if v is not None}
  result.update(CACHEDB=str(storage/'compiler.db'),TMPDIR=str(storage/'tmp'),XDG_CACHE_HOME=str(storage/'xdg'),
    TORCH_HOME=str(storage/'torch'),PYTHONDONTWRITEBYTECODE='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
    ALPAMAYO_FP8_CACHE=str(m.FP8),PYTHONPATH=f'{m.TG}:{m.ROOT}:/tmp/opencode/openpilot-alpamayo:/home/john/asius/openpilot/opendbc_repo')
  return result

def main():
  parser=argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--tag',default='trial1');parser.add_argument('--execute-gpu',action='store_true')
  parser.add_argument('--manifest',type=Path,default=WORK/'agent-startup-persist-protocol2.json')
  args=parser.parse_args()
  if not re.fullmatch('[A-Za-z0-9_-]+',args.tag): raise ValueError('Invalid fresh tag')
  manifest=m.read(args.manifest);m.verify(manifest)
  base=m.storage(WORK/('agent-startup-persist-'+args.tag+'-storage'))
  if base.exists(): raise FileExistsError('Immutable parent run directory exists')
  if shutil.disk_usage(WORK).free<10*1024**3: raise RuntimeError('Keep10GiB WORK reserve')
  base.mkdir();cache=m.cache_check();m.save(base/'fp8-verification.json',cache)
  origin=Path(manifest['compiler_cache_source']);copier=m.load(EXP/'agent-plan-mutation-v2-parent.py').copy_unchanged
  records=[];reports={}
  for role in ('ordinary','persisted'):
    store=base/role;store.mkdir()
    for sub in ('tmp','xdg','torch'): (store/sub).mkdir()
    copied={}
    for suffix in ('','-wal','-shm'):
      source=Path(str(origin)+suffix)
      if source.exists(): copied[str(source)]=copier(source,Path(str(store/'compiler.db')+suffix),m.sha)
    for source in sorted((origin.parent/'xdg').rglob('*')):
      if source.is_symlink(): raise ValueError('Unexpected compiler cache symlink')
      if source.is_file():
        target=store/'xdg'/source.relative_to(origin.parent/'xdg');target.parent.mkdir(parents=True,exist_ok=True)
        copied[str(source)]=copier(source,target,m.sha)
    env=environment(store)
    preflight=[str(PY),str(EXP/'agent-startup-persist-run.py'),'--preflight','--role',role,
      '--manifest',str(args.manifest),'--output',str(store/'preflight.json')]
    subprocess.run(preflight,env={**os.environ,**env},check=True,timeout=120)
    tag='agent-startup-persist-'+args.tag+'-'+role
    command=['python3','/tmp/opencode/run-alpamayo-speed-trial.py',tag,'--program',str(EXP/'agent-startup-persist-run.py'),
      '--timeout-seconds','450','--set','manifest',str(args.manifest),'--set','role',role,'--flag','execute-gpu']
    for key,value in env.items(): command.extend(('--env',key,value))
    record=dict(role=role,command=command,environment=env,copied_cache=copied,child_timeout_seconds=450,outer_timeout_seconds=0)
    records.append(record);m.save(store/'parent-command.json',record);reports[role]=WORK/tag/'report.json'
    print(json.dumps(dict(role=role,command=command,executing_gpu=args.execute_gpu,outer_timeout_seconds=0)),flush=True)
    if args.execute_gpu: subprocess.run(command,check=True)
  m.save(base/'pair-protocol.json',dict(manifest_sha256=m.sha(args.manifest),arms=records,reports={k:str(v) for k,v in reports.items()},
    maximum_child_total_seconds=900,automatic_full64_or_paced=False))
  if args.execute_gpu:
    comparison=m.compare(m.read(reports['ordinary']),m.read(reports['persisted']),reports['ordinary'],reports['persisted'])
    m.save(base/'comparison.json',comparison);m.verify(manifest);print(json.dumps(comparison,indent=2))

if __name__=='__main__': main()
