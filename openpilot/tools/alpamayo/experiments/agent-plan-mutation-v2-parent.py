"""Freeze the composed protocol, preflight on CPU, and optionally queue one parent trial."""
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
TG=Path('/tmp/opencode/alpamayo-asius-20261114/tinygrad')
SOURCE=WORK/'agent-paced-dynamic-source-cpu02/alpamayo'
PY=Path('/tmp/opencode/alpamayo-asius-20261114/upstream/.venv/bin/python')
PROTOCOL=WORK/'agent-plan-mutation-v2-protocol.json'
sys.dont_write_bytecode=True
sys.path[:0]=[str(TG),str(SOURCE),str(SOURCE/'experiments')]

def local(path):
  spec=importlib.util.spec_from_file_location('agent_plan_mutation_v2_'+Path(path).stem.replace('-','_'),path)
  module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
  return module

def freeze(path,evidence):
  if path.exists() or not path.resolve().is_relative_to(WORK): raise ValueError('Use a new protocol under WORK')
  adapter=local(EXP/'agent-plan-mutation-v2-adapter.py');replacement=adapter.verify_base()
  result=json.loads(evidence.read_text())
  if (not result['CPU_setup_passed'] or not result['actual_production_transport'] or result['simulated_transport'] or
      result['Buffer_monkeypatch'] or result['gpu_executed'] or len(result['cpu_setup_cases'])!=6):
    raise ValueError('Complete actual production-adapter CPU evidence is required')
  sources={name:adapter.sha(EXP/name) for name in adapter.SOURCES}
  if result['source_sha256']!=sources: raise ValueError('CPU evidence predates the final sources')
  files=[evidence,WORK/'agent-plan-reload-reload1/report.json',WORK/'agent-plan-reload-reload1/run.log',
         WORK/'agent-plan-mutation-protocol1.json',WORK/'agent-plan-mutation-cpu-v3.json']
  protocol=dict(schema=1,scope='Two pinned-API corrections only; original actual count1 numerical proof and all six GPU cases unchanged',
    base_protocol=dict(path=str(adapter.BASE),sha256=adapter.BASE_SHA),adapter_sources=sources,replacement=replacement,
    evidence={str(p):adapter.sha(p) for p in files},child_timeout_seconds=300,full_target='USB+AMD:LLVM:gfx1200')
  path.write_text(json.dumps(protocol,indent=2)+'\n')
  print(json.dumps(dict(protocol=str(path),sha256=adapter.sha(path),original_protocol2_unchanged=True)))

def copy_unchanged(source,target,sha):
  before=sha(source);shutil.copyfile(source,target)
  if sha(source)!=before or sha(target)!=before: raise ValueError('Cache changed while copying: '+str(source))
  return dict(sha256=before,bytes=target.stat().st_size)

def main():
  parser=argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--tag',default='retry1');parser.add_argument('--execute-gpu',action='store_true')
  parser.add_argument('--protocol',type=Path,default=PROTOCOL);parser.add_argument('--freeze-protocol',action='store_true')
  parser.add_argument('--cpu-evidence',type=Path,default=WORK/'agent-plan-mutation-v2-cpu-final.json');args=parser.parse_args()
  parent_tmp=WORK/'agent-plan-mutation-v2-parent-tmp';parent_tmp.mkdir(exist_ok=True)
  os.environ.update(TMPDIR=str(parent_tmp),CACHEDB=str(WORK/'agent-plan-mutation-v2-parent-cache.db'),PYTHONDONTWRITEBYTECODE='1')
  if args.freeze_protocol:
    if args.execute_gpu: raise ValueError('CPU freeze cannot execute hardware')
    freeze(args.protocol,args.cpu_evidence);return
  if not re.fullmatch('[A-Za-z0-9_-]+',args.tag): raise ValueError('Invalid trial tag')
  runner=local(EXP/'agent-plan-mutation-v2-run.py');runner.verify(json.loads(args.protocol.read_text()))
  storage=WORK/('agent-plan-mutation-'+args.tag+'-storage')
  if storage.exists(): raise FileExistsError('Use a new tag; previous artifacts remain intact')
  if shutil.disk_usage(WORK).free<5*1024**3: raise RuntimeError('WORK reserve is below five GiB')
  storage.mkdir()
  for name in ('tmp','xdg','torch'): (storage/name).mkdir()
  prior=json.loads((WORK/'agent-plan-reload-reload1/report.json').read_text())
  original=Path(prior['compiler_cache']);cache=storage/'compiler.db';copied={}
  if not original.resolve().is_relative_to(WORK): raise ValueError('Completed source cache must reside under WORK')
  for suffix in ('','-wal','-shm'):
    source=Path(str(original)+suffix)
    if source.exists(): copied[str(source)]=copy_unchanged(source,Path(str(cache)+suffix),runner.sha)
  # Reuse completed local firmware downloads; do not fetch dependencies.
  old_xdg=original.parent/'xdg'
  for source in sorted(old_xdg.rglob('*')):
    if source.is_symlink(): raise ValueError('Unexpected symlink in copied firmware cache')
    if not source.is_file(): continue
    target=storage/'xdg'/source.relative_to(old_xdg);target.parent.mkdir(parents=True,exist_ok=True)
    copied[str(source)]=copy_unchanged(source,target,runner.sha)
  environment={k:v for k,v in prior['protocol']['environment'].items() if v is not None}
  environment.update(PYTHONPATH=f'{TG}:{SOURCE}:/tmp/opencode/openpilot-alpamayo:/home/john/asius/openpilot/opendbc_repo',
    CACHEDB=str(cache),TMPDIR=str(storage/'tmp'),XDG_CACHE_HOME=str(storage/'xdg'),TORCH_HOME=str(storage/'torch'),
    PYTHONDONTWRITEBYTECODE='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
  preflight=[str(PY),str(EXP/'agent-plan-mutation-v2-run.py'),'--preflight','--output',str(storage/'preflight.json'),
    '--cache-db',str(cache),'--mutation-protocol',str(args.protocol)]
  subprocess.run(preflight,env={**os.environ,**environment},check=True,timeout=60)
  command=['python3','/tmp/opencode/run-alpamayo-speed-trial.py','agent-plan-mutation-'+args.tag,
    '--program',str(EXP/'agent-plan-mutation-v2-run.py'),'--timeout-seconds','300',
    '--set','cache-db',str(cache),'--set','mutation-protocol',str(args.protocol),'--flag','execute-gpu']
  for key,value in environment.items(): command.extend(('--env',key,value))
  plan=dict(command=command,protocol=str(args.protocol),protocol_sha256=runner.sha(args.protocol),copied_cache=copied,
    environment=environment,preflight=str(storage/'preflight.json'),child_timeout_seconds=300)
  (storage/'parent-command.json').write_text(json.dumps(plan,indent=2)+'\n')
  print(json.dumps(dict(storage=str(storage),preflight=plan['preflight'],command=command,executing_gpu=args.execute_gpu),indent=2),flush=True)
  if args.execute_gpu: subprocess.run(command,check=True)

if __name__=='__main__': main()
