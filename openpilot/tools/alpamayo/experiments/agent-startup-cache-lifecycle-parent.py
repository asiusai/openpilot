"""Parent-only candidate retry with the completed ordinary exactness baseline."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

EXP=Path(__file__).resolve().parent;WORK=Path('/mnt/nvme-500/tmp/alpamayo-speed-20261116')
sys.dont_write_bytecode=True;sys.path.insert(0,'/tmp/opencode/alpamayo-asius-20261114/tinygrad')
spec=importlib.util.spec_from_file_location('lifecycle_parent_contract',EXP/'agent-startup-cache-lifecycle-contract.py')
c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c);m=c.m

def main():
  parser=argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--tag',default='retry1');parser.add_argument('--execute-gpu',action='store_true')
  parser.add_argument('--protocol',type=Path,default=WORK/'agent-startup-cache-lifecycle-protocol1.json');args=parser.parse_args()
  if not re.fullmatch('[A-Za-z0-9_-]+',args.tag): raise ValueError('Invalid new tag')
  protocol=m.read(args.protocol);c.gate(protocol)
  store=m.storage(WORK/('agent-startup-cache-lifecycle-'+args.tag+'-storage'))
  if store.exists(): raise FileExistsError('Fresh retry storage required')
  if shutil.disk_usage(WORK).free<10*1024**3: raise RuntimeError('Keep10GiB WORK reserve')
  store.mkdir()
  for name in ('tmp','xdg','torch'): (store/name).mkdir()
  m.save(store/'fp8-verification.json',m.cache_check())
  seed=m.read(WORK/'agent-startup-persist-trial1-storage/ordinary/parent-command.json')['copied_cache']
  origin=Path(m.read(c.BASE)['compiler_cache_source']);copier=c.local(EXP/'agent-plan-mutation-v2-parent.py').copy_unchanged
  copied={}
  for name,expected in seed.items():
    source=Path(name);m.storage(source)
    if source.is_symlink() or source.stat().st_size!=expected['bytes'] or m.sha(source)!=expected['sha256']:
      raise ValueError('Original ordinary compiler-cache seed changed')
    if name.startswith(str(origin)) and name[len(str(origin)):] in ('','-wal','-shm'):
      target=Path(str(store/'compiler.db')+name[len(str(origin)):])
    elif source.is_relative_to(origin.parent/'xdg'): target=store/'xdg'/source.relative_to(origin.parent/'xdg')
    else: raise ValueError('Unexpected seed path')
    target.parent.mkdir(parents=True,exist_ok=True);copied[name]=copier(source,target,m.sha)
  m.equal(copied,seed,'original ordinary cache seed')
  environment=c.local(EXP/'agent-startup-persist-parent.py').environment(store)
  command=[str(m.PY),str(EXP/'agent-startup-cache-lifecycle-run.py'),'--preflight','--protocol',str(args.protocol),'--output',str(store/'preflight.json')]
  subprocess.run(command,env={**os.environ,**environment},check=True,timeout=120)
  tag='agent-startup-cache-lifecycle-'+args.tag
  command=['python3','/tmp/opencode/run-alpamayo-speed-trial.py',tag,'--program',str(EXP/'agent-startup-cache-lifecycle-run.py'),
    '--timeout-seconds','450','--set','protocol',str(args.protocol),'--flag','execute-gpu']
  for k,v in environment.items(): command.extend(('--env',k,v))
  m.save(store/'parent-command.json',dict(command=command,environment=environment,copied_cache=copied,
    protocol_sha256=m.sha(args.protocol),ordinary_reference=c.reuse(),child_timeout_seconds=450,outer_timeout_seconds=0))
  print(json.dumps(dict(storage=str(store),protocol_sha256=m.sha(args.protocol),candidate_only=True,
    ordinary_sha256=c.PINS[str(c.REFERENCE)],executing_gpu=args.execute_gpu,child_seconds=450,outer_timeout_seconds=0)),flush=True)
  if args.execute_gpu: subprocess.run(command,check=True)

if __name__=='__main__': main()
