"""Freeze CPU-reviewed capture protocol, copy completed caches, then optionally queue the parent GPU capture."""
import argparse
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

EXP=Path(__file__).resolve().parent;WORK=Path('/mnt/nvme-500/tmp/alpamayo-speed-20261116')
TG=Path('/tmp/opencode/alpamayo-asius-20261114/tinygrad')
SOURCE=WORK/'agent-paced-dynamic-source-cpu02/alpamayo'
PY=Path('/tmp/opencode/alpamayo-asius-20261114/upstream/.venv/bin/python')
sys.dont_write_bytecode=True;sys.path[:0]=[str(TG),str(SOURCE),str(SOURCE/'experiments')]

def local(path):
  import importlib.util
  spec=importlib.util.spec_from_file_location('family_parent_'+Path(path).stem.replace('-','_'),path)
  module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
  return module

def main():
  parser=argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--tag',default='capture1');parser.add_argument('--execute-gpu',action='store_true')
  parser.add_argument('--protocol',type=Path,default=WORK/'agent-plan-family-protocol1.json')
  parser.add_argument('--freeze-protocol',action='store_true');parser.add_argument('--count6-only',action='store_true')
  parser.add_argument('--cpu-evidence',type=Path,default=WORK/'agent-plan-family-cpu-final.json');args=parser.parse_args()
  storage_env=WORK/'agent-plan-family-parent-tmp';storage_env.mkdir(exist_ok=True)
  os.environ.update(TMPDIR=str(storage_env),CACHEDB=str(WORK/'agent-plan-family-parent-cache.db'),
                    XDG_CACHE_HOME=str(storage_env),PYTHONDONTWRITEBYTECODE='1')
  contract=local(EXP/'agent-plan-family-contract.py');contract.source_gate()
  if args.freeze_protocol:
    if args.execute_gpu: raise ValueError('CPU freeze cannot execute hardware')
    contract.fresh(args.protocol);evidence=json.loads(args.cpu_evidence.read_text())
    sources={name:contract.sha(EXP/name) for name in contract.SOURCES}
    if evidence['sources']!=sources or not evidence['passed'] or evidence['gpu_executed']:
      raise ValueError('Final source-bound CPU evidence is required')
    files=[args.cpu_evidence,contract.SUCCESS,contract.QUALIFIED,Path('/tmp/opencode/run-alpamayo-speed-trial.py'),
           EXP/'agent-startup-key-prime.py',SOURCE/'experiments/block-decode.py',SOURCE/'model.py']
    protocol=dict(schema=1,scope='Original serial decoder lengths 2..6 metadata capture; separate offline proof and fresh reload required',
      base=dict(path=str(contract.BASE),sha256=contract.BASE_SHA),sources=sources,files={str(p):contract.sha(p) for p in files},
      counts=[6] if args.count6_only else list(contract.COUNTS),limits=contract.LIMITS,target='USB+AMD:LLVM:gfx1200',
      count1_gpu_reexecution=False,lookup_or_target_policy_changes=False)
    args.protocol.write_text(json.dumps(protocol,indent=2)+'\n')
    print(json.dumps(dict(protocol=str(args.protocol),sha256=contract.sha(args.protocol))));return
  if args.count6_only: raise ValueError('Count selection belongs in the immutable protocol')
  if not re.fullmatch('[A-Za-z0-9_-]+',args.tag): raise ValueError('Invalid tag')
  protocol=json.loads(args.protocol.read_text());contract.source_gate(protocol)
  storage=contract.fresh(WORK/('agent-plan-family-'+args.tag+'-storage'))
  if shutil.disk_usage(WORK).free<contract.LIMITS['work_reserve_bytes']: raise RuntimeError('Insufficient WORK reserve')
  storage.mkdir()
  for name in ('tmp','xdg','torch'): (storage/name).mkdir()
  prior=contract.validate_success();source_cache=Path(prior['compiler_cache']);cache=storage/'compiler.db'
  if not source_cache.resolve().is_relative_to(WORK): raise ValueError('Completed source cache must reside on WORK')
  copy_unchanged=local(EXP/'agent-plan-mutation-v2-parent.py').copy_unchanged;copied={}
  for suffix in ('','-wal','-shm'):
    source=Path(str(source_cache)+suffix)
    if source.exists(): copied[str(source)]=copy_unchanged(source,Path(str(cache)+suffix),contract.sha)
  for source in sorted((source_cache.parent/'xdg').rglob('*')):
    if source.is_symlink(): raise ValueError('Unexpected cache symlink')
    if not source.is_file(): continue
    target=storage/'xdg'/source.relative_to(source_cache.parent/'xdg');target.parent.mkdir(parents=True,exist_ok=True)
    copied[str(source)]=copy_unchanged(source,target,contract.sha)
  environment={k:v for k,v in prior['protocol']['environment'].items() if v is not None}
  environment.update(PYTHONPATH=f'{TG}:{SOURCE}:/tmp/opencode/openpilot-alpamayo:/home/john/asius/openpilot/opendbc_repo',
    CACHEDB=str(cache),TMPDIR=str(storage/'tmp'),XDG_CACHE_HOME=str(storage/'xdg'),TORCH_HOME=str(storage/'torch'),
    PYTHONDONTWRITEBYTECODE='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
  preflight=[str(PY),str(EXP/'agent-plan-family-run.py'),'--preflight','--output',str(storage/'preflight.json'),
    '--cache-db',str(cache),'--family-protocol',str(args.protocol)]
  subprocess.run(preflight,env={**os.environ,**environment},check=True,timeout=60)
  name='agent-plan-family-'+args.tag
  command=['python3','/tmp/opencode/run-alpamayo-speed-trial.py',name,'--program',str(EXP/'agent-plan-family-run.py'),
    '--timeout-seconds','300','--set','cache-db',str(cache),'--set','family-protocol',str(args.protocol),'--flag','execute-gpu']
  for key,value in environment.items(): command.extend(('--env',key,value))
  offline=[str(PY),str(EXP/'agent-plan-family-offline.py'),'--report',str(WORK/name/'report.json'),
           '--protocol',str(args.protocol),'--output',str(WORK/(name+'-offline.json'))]
  plan=dict(command=command,offline_command=offline,environment=environment,protocol=str(args.protocol),
    protocol_sha256=contract.sha(args.protocol),copied_cache=copied,preflight=str(storage/'preflight.json'),
    child_timeout_seconds=300,offline_proof_runs_separately=True,fresh_process_reload_included=False)
  (storage/'parent-command.json').write_text(json.dumps(plan,indent=2)+'\n')
  print(json.dumps(dict(storage=str(storage),command=command,offline_command=offline,executing_gpu=args.execute_gpu),indent=2),flush=True)
  if args.execute_gpu: subprocess.run(command,check=True)

if __name__=='__main__': main()
