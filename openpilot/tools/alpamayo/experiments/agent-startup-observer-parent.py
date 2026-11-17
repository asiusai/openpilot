"""Reuse the frozen parent protocol; change only entrypoint and fresh output prefix."""
import importlib.util
import inspect
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent;sys.dont_write_bytecode=True
sys.path.insert(0,'/tmp/opencode/alpamayo-asius-20261114/tinygrad')
spec=importlib.util.spec_from_file_location('startup_observer_parent_run',HERE/'agent-startup-observer-run.py')
run=importlib.util.module_from_spec(spec);spec.loader.exec_module(run)

def main():
  run.gate(run.m.read(run.PROTOCOL))
  parent=run.local(HERE/'agent-startup-cache-lifecycle-parent.py');source=inspect.getsource(parent.main)
  changes=[("str(EXP/'agent-startup-cache-lifecycle-run.py')","str(EXP/'agent-startup-observer-run.py')",2),
           ("'agent-startup-cache-lifecycle-'+args.tag","'agent-startup-observer-'+args.tag",2)]
  adapted=source
  for old,new,count in changes:
    if adapted.count(old)!=count: raise ValueError('Frozen parent anchor changed')
    adapted=adapted.replace(old,new)
  restored=adapted
  for old,new,_ in reversed(changes): restored=restored.replace(new,old)
  if restored!=source: raise AssertionError('Parent changes exceed observer entrypoint/output prefix')
  namespace=dict(vars(parent));exec(compile(adapted,str(__file__)+':parent','exec'),namespace)
  print('Observer protocol SHA256: '+run.m.sha(run.PROTOCOL),flush=True)
  namespace['main']()
  run.gate(run.m.read(run.PROTOCOL))

if __name__=='__main__': main()
