"""Compose the immutable reload protocol with the minimal mutation-mask adapter."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
from unittest.mock import patch

EXP=Path(__file__).resolve().parent;WORK=Path('/mnt/nvme-500/tmp/alpamayo-speed-20261116')
PROTOCOL=WORK/'agent-plan-mutation-protocol1.json'

def local(path):
  spec=importlib.util.spec_from_file_location('agent_plan_mutation_'+Path(path).stem.replace('-','_'),path)
  module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
  return module

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def verify(protocol):
  adapter=local(EXP/'agent-plan-mutation-adapter.py');change=adapter.verify_base()
  if protocol['base_protocol']!={'path':str(adapter.BASE),'sha256':adapter.BASE_SHA}: raise ValueError('Original protocol link changed')
  if protocol['adapter_sources']!={p.name:sha(p) for p in EXP.glob('agent-plan-mutation-*.py')}: raise ValueError('Reviewed adapter sources changed')
  if protocol['replacement']!=change: raise ValueError('Mutation adapter changed')
  for path,digest in protocol['evidence'].items():
    if sha(path)!=digest: raise ValueError('Prior evidence changed')
  return adapter

def require_poison_API():
  from tinygrad.device import Buffer
  if not hasattr(Buffer,'copyin'):
    raise RuntimeError('Mask correction is valid, but frozen poison code requires absent Buffer.copyin. '
                       'A separate Buffer.copy_from correction needs approval under the mask-only scope. No GPU retry is ready.')

def main():
  parser=argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--output',type=Path,required=True);parser.add_argument('--cache-db',type=Path,required=True)
  parser.add_argument('--mutation-protocol',type=Path,default=PROTOCOL)
  parser.add_argument('--execute-gpu',action='store_true');parser.add_argument('--preflight',action='store_true')
  args=parser.parse_args()
  if args.output.exists() or not args.output.resolve().is_relative_to(WORK): raise ValueError('Use a new output under WORK')
  if not args.cache_db.resolve().is_relative_to(WORK): raise ValueError('Compiler cache must stay under WORK')
  if args.preflight and args.execute_gpu: raise ValueError('Preflight cannot execute hardware')
  protocol=json.loads(args.mutation_protocol.read_text());adapter=verify(protocol)
  module=local(EXP/'agent-plan-reload-run.py')
  command=[str(module.__file__),'--output',str(args.output),'--cache-db',str(args.cache_db),'--protocol',str(adapter.BASE)]
  if args.preflight: command.append('--preflight')
  else:
    if not args.execute_gpu: raise ValueError('Explicit parent GPU execution required')
    require_poison_API();command.append('--execute-gpu')
  try:
    with adapter.adapted(module) as change,patch.object(sys,'argv',command): module.main()
    if args.preflight: require_poison_API()
  finally:
    if args.output.exists():
      report=json.loads(args.output.read_text())
      report['mutation_adapter']=dict(protocol_sha256=sha(args.mutation_protocol),base_protocol_sha256=adapter.BASE_SHA,
        replacement=protocol['replacement'],adapter_sources=protocol['adapter_sources'],production_retry_ready=False)
      args.output.write_text(json.dumps(report,indent=2)+'\n')

if __name__=='__main__': main()
