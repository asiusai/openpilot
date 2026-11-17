"""Run the immutable count1 proof with only the approved mask and poison API repairs."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
from unittest.mock import patch

EXP=Path(__file__).resolve().parent;WORK=Path('/mnt/nvme-500/tmp/alpamayo-speed-20261116')
PROTOCOL=WORK/'agent-plan-mutation-v2-protocol.json'

def local(path):
  spec=importlib.util.spec_from_file_location('agent_plan_mutation_v2_'+Path(path).stem.replace('-','_'),path)
  module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
  return module

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def verify(protocol):
  adapter=local(EXP/'agent-plan-mutation-v2-adapter.py');change=adapter.verify_base()
  if protocol['base_protocol']!={'path':str(adapter.BASE),'sha256':adapter.BASE_SHA}: raise ValueError('Original protocol link changed')
  if protocol['adapter_sources']!={name:sha(EXP/name) for name in adapter.SOURCES}: raise ValueError('Reviewed adapter sources changed')
  if protocol['replacement']!=change or protocol['child_timeout_seconds']!=300: raise ValueError('Adapter or execution bound changed')
  for path,digest in protocol['evidence'].items():
    if sha(path)!=digest: raise ValueError('Frozen evidence changed')
  from tinygrad.device import Buffer
  if hasattr(Buffer,'copyin') or not callable(getattr(Buffer,'copy_from',None)):
    raise ValueError('Pinned Buffer API changed or has a compatibility monkeypatch')
  return adapter

def main():
  parser=argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--output',type=Path,required=True);parser.add_argument('--cache-db',type=Path,required=True)
  parser.add_argument('--mutation-protocol',type=Path,default=PROTOCOL)
  parser.add_argument('--execute-gpu',action='store_true');parser.add_argument('--preflight',action='store_true')
  args=parser.parse_args()
  if args.output.exists() or not args.output.resolve().is_relative_to(WORK): raise ValueError('Use a new WORK output')
  if not args.cache_db.resolve().is_relative_to(WORK): raise ValueError('Compiler cache must stay under WORK')
  if args.preflight and args.execute_gpu: raise ValueError('Preflight cannot execute hardware')
  for name in ('TMPDIR','CACHEDB'):
    if not Path(os.environ.get(name,'')).resolve().is_relative_to(WORK): raise ValueError('Writable storage must remain under WORK')
  if Path(os.environ['CACHEDB']).resolve()!=args.cache_db.resolve(): raise ValueError('Cache argument and environment differ')
  # Preserve the qualified numeric defaults before any tinygrad import.
  environment=json.loads((WORK/'agent-plan-bind-capture1/report.json').read_text())['protocol']['environment']
  for key,value in environment.items():
    if key in ('CACHEDB','TMPDIR','PYTHONDONTWRITEBYTECODE'): continue
    if value is None: os.environ.pop(key,None)
    else: os.environ[key]=value
  sys.dont_write_bytecode=True
  protocol=json.loads(args.mutation_protocol.read_text());adapter=verify(protocol)
  module=local(EXP/'agent-plan-reload-run.py')
  command=[str(module.__file__),'--output',str(args.output),'--cache-db',str(args.cache_db),'--protocol',str(adapter.BASE)]
  if args.preflight: command.append('--preflight')
  else:
    if not args.execute_gpu: raise ValueError('Explicit parent GPU execution required')
    command.append('--execute-gpu')
  try:
    with adapter.adapted(module),patch.object(sys,'argv',command): module.main()
  finally:
    if args.output.exists():
      report=json.loads(args.output.read_text())
      report['mutation_adapter']=dict(protocol_sha256=sha(args.mutation_protocol),base_protocol_sha256=adapter.BASE_SHA,
        replacement=protocol['replacement'],adapter_sources=protocol['adapter_sources'],
        transfer='Normal Buffer.copy_from with CPU staging retained through destination synchronization',
        Buffer_monkeypatch=False,numerical_PROGRAMs_modified=False)
      args.output.write_text(json.dumps(report,indent=2)+'\n')

if __name__=='__main__': main()
