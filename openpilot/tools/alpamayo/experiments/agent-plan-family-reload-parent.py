"""Freeze or launch one bounded family reload using the normal global serial ACL wrapper."""
import argparse
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

EXP=Path(__file__).resolve().parent;WORK=Path('/mnt/nvme-500/tmp/alpamayo-speed-20261116')
TG=Path('/tmp/opencode/alpamayo-asius-20261114/tinygrad');SOURCE=WORK/'agent-paced-dynamic-source-cpu02/alpamayo'
PY=Path('/tmp/opencode/alpamayo-asius-20261114/upstream/.venv/bin/python')
sys.dont_write_bytecode=True;sys.path[:0]=[str(TG),str(SOURCE),str(SOURCE/'experiments')]

def local(path):
  import importlib.util
  spec=importlib.util.spec_from_file_location('family_reload_parent_'+Path(path).stem.replace('-','_'),path)
  module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
  return module

def main():
  parser=argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--tag',default='trial1');parser.add_argument('--execute-gpu',action='store_true')
  parser.add_argument('--protocol',type=Path,default=WORK/'agent-plan-family-reload-protocol1.json')
  parser.add_argument('--freeze-protocol',action='store_true');parser.add_argument('--count6-only',action='store_true')
  parser.add_argument('--cpu-evidence',type=Path,default=WORK/'agent-plan-family-reload-cpu-final.json');args=parser.parse_args()
  tmp=WORK/'agent-plan-family-reload-parent-tmp';tmp.mkdir(exist_ok=True)
  os.environ.update(TMPDIR=str(tmp),CACHEDB=str(WORK/'agent-plan-family-reload-parent-cache.db'),XDG_CACHE_HOME=str(tmp),PYTHONDONTWRITEBYTECODE='1')
  contract=local(EXP/'agent-plan-family-reload-contract.py');family=contract.family()
  if args.freeze_protocol:
    if args.execute_gpu: raise ValueError('CPU freeze cannot execute hardware')
    family.fresh(args.protocol);evidence=json.loads(args.cpu_evidence.read_text())
    sources={name:contract.sha(EXP/name) for name in contract.SOURCES}
    if (evidence['sources']!=sources or not evidence['passed'] or evidence['gpu_executed'] or
        evidence['production_mutation_cases']!=30 or not evidence['all_five_proofs_recomputed_exact']):
      raise ValueError('Final full-family CPU evidence required')
    files,members=contract.evidence()
    protocol=dict(schema=1,base=dict(path=str(contract.BASE),sha256=contract.BASE_SHA),sources=sources,
      artifacts=files,members=members,cpu_evidence={str(args.cpu_evidence):contract.sha(args.cpu_evidence)},
      counts=[6] if args.count6_only else [2,3,4,5,6],limits=contract.LIMITS,target='USB+AMD:LLVM:gfx1200',
      scope='Fresh original-versus-persisted individual serial family plans; no whole-pipeline adoption')
    args.protocol.write_text(json.dumps(protocol,indent=2)+'\n');contract.gate(protocol)
    print(json.dumps(dict(protocol=str(args.protocol),sha256=contract.sha(args.protocol))));return
  if args.count6_only: raise ValueError('Count selection must be sealed into a separate protocol')
  if not re.fullmatch('[A-Za-z0-9_-]+',args.tag): raise ValueError('Invalid tag')
  protocol=json.loads(args.protocol.read_text());contract.gate(protocol)
  storage=family.fresh(WORK/('agent-plan-family-reload-'+args.tag+'-storage'))
  if shutil.disk_usage(WORK).free<contract.LIMITS['work_reserve_bytes']: raise RuntimeError('Insufficient WORK reserve')
  storage.mkdir()
  for name in ('tmp','xdg','torch'): (storage/name).mkdir()
  prior=json.loads(contract.CAPTURE.read_text());original=Path(prior['compiler_cache']);cache=storage/'compiler.db'
  if not original.resolve().is_relative_to(WORK): raise ValueError('Source compiler cache must remain under WORK')
  copy_unchanged=local(EXP/'agent-plan-mutation-v2-parent.py').copy_unchanged;copied={}
  for suffix in ('','-wal','-shm'):
    source=Path(str(original)+suffix)
    if source.exists(): copied[str(source)]=copy_unchanged(source,Path(str(cache)+suffix),contract.sha)
  for source in sorted((original.parent/'xdg').rglob('*')):
    if source.is_symlink(): raise ValueError('Unexpected cache symlink')
    if not source.is_file(): continue
    target=storage/'xdg'/source.relative_to(original.parent/'xdg');target.parent.mkdir(parents=True,exist_ok=True)
    copied[str(source)]=copy_unchanged(source,target,contract.sha)
  environment={k:v for k,v in prior['protocol']['environment'].items() if v is not None}
  environment.update(PYTHONPATH=f'{TG}:{SOURCE}:/tmp/opencode/openpilot-alpamayo:/home/john/asius/openpilot/opendbc_repo',
    CACHEDB=str(cache),TMPDIR=str(storage/'tmp'),XDG_CACHE_HOME=str(storage/'xdg'),TORCH_HOME=str(storage/'torch'),
    PYTHONDONTWRITEBYTECODE='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
  preflight=[str(PY),str(EXP/'agent-plan-family-reload-run.py'),'--preflight','--output',str(storage/'preflight.json'),
    '--cache-db',str(cache),'--reload-protocol',str(args.protocol)]
  subprocess.run(preflight,env={**os.environ,**environment},check=True,timeout=60)
  command=['python3','/tmp/opencode/run-alpamayo-speed-trial.py','agent-plan-family-reload-'+args.tag,
    '--program',str(EXP/'agent-plan-family-reload-run.py'),'--timeout-seconds','300',
    '--set','cache-db',str(cache),'--set','reload-protocol',str(args.protocol),'--flag','execute-gpu']
  for key,value in environment.items(): command.extend(('--env',key,value))
  record=dict(command=command,environment=environment,protocol=str(args.protocol),protocol_sha256=contract.sha(args.protocol),
    copied_cache=copied,preflight=str(storage/'preflight.json'),child_timeout_seconds=300,outer_timeout_seconds=0)
  (storage/'parent-command.json').write_text(json.dumps(record,indent=2)+'\n')
  print(json.dumps(dict(storage=str(storage),command=command,executing_gpu=args.execute_gpu,outer_timeout_seconds=0),indent=2),flush=True)
  if args.execute_gpu: subprocess.run(command,check=True)  # No outer timeout; wrapper owns the bounded child and ACL cleanup.

if __name__=='__main__': main()
