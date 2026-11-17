"""Compose the observer repair with the frozen lifecycle candidate entrypoint."""
import argparse
import importlib.util
from pathlib import Path
import sys

EXP=Path(__file__).resolve().parent;WORK=Path('/mnt/nvme-500/tmp/alpamayo-speed-20261116')
BASE=WORK/'agent-startup-cache-lifecycle-protocol1.json'
BASE_SHA='1aebc25af9a9d2504c0b2e7fd493dc39fd44e749dd1187b777a921d9b21ab3fa'
PROTOCOL=WORK/'agent-startup-observer-protocol1.json'
NAMES=('agent-startup-observer.py','agent-startup-observer-run.py','agent-startup-observer-check.py','agent-startup-observer-parent.py')

def local(path):
  spec=importlib.util.spec_from_file_location('startup_observer_'+Path(path).stem.replace('-','_'),path)
  module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module);return module
c=local(EXP/'agent-startup-cache-lifecycle-contract.py');m=c.m
def sources(): return {name:m.sha(EXP/name) for name in NAMES}
def gate(protocol):
  if m.sha(BASE)!=BASE_SHA or protocol['base_sha256']!=BASE_SHA or protocol['sources']!=sources(): raise ValueError('Observer source/base changed')
  c.gate(m.read(BASE))
  if protocol['child_seconds']!=450: raise ValueError('Do not raise the retry bound')
  path=Path(protocol['cpu_evidence']);e=m.read(path)
  if m.sha(path)!=protocol['cpu_sha256'] or not e['passed'] or e['gpu_executed'] or e['sources']!=sources(): raise ValueError('Observer CPU qualification missing')

def main(argv=None):
  parser=argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--output',type=Path,required=True);parser.add_argument('--protocol',type=Path,default=BASE)
  parser.add_argument('--preflight',action='store_true');parser.add_argument('--execute-gpu',action='store_true');args=parser.parse_args(argv)
  if args.protocol.resolve()!=BASE: raise ValueError('Only the frozen lifecycle protocol is supported')
  protocol=m.read(PROTOCOL);gate(protocol);digest=m.sha(PROTOCOL)
  status=dict(completed=False,gpu_executed=False,observer_protocol_sha256=digest,sources=sources(),source_before_after_exact=False,hooks_restored=False)
  previous=importlib.util.spec_from_file_location;overlay=local(EXP/'agent-startup-observer.py')
  try:
    with overlay.imports(m) as state:
      command=['--protocol',str(BASE),'--output',str(args.output),'--preflight' if args.preflight else '--execute-gpu']
      if args.preflight==args.execute_gpu: raise ValueError('Choose preflight or parent-only GPU')
      local(EXP/'agent-startup-cache-lifecycle-run.py').main(command)
      status['observer']=overlay.summary(state)
      if state['modules']!=1 or (args.execute_gpu and not state['observers']): raise AssertionError('Actual production observer was not installed')
    status.update(completed=True,gpu_executed=args.execute_gpu)
  except BaseException as error: status['failure']=dict(type=type(error).__name__,message=str(error));raise
  finally:
    status['hooks_restored']=importlib.util.spec_from_file_location is previous
    try:
      gate(protocol)
      if m.sha(PROTOCOL)!=digest: raise ValueError('Observer protocol changed')
      status['source_before_after_exact']=True
    finally: m.save(args.output.with_suffix('.observer.json'),status)
    if not status['hooks_restored']: raise RuntimeError('Observer imports not restored')

if __name__=='__main__': main()
