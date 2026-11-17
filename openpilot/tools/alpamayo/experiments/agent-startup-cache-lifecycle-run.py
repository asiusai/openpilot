"""Execute only the corrected candidate through the unchanged frozen runner."""
import argparse
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

HERE=Path(__file__).resolve().parent
def local(path):
  spec=importlib.util.spec_from_file_location('lifecycle_entry_'+Path(path).stem.replace('-','_'),path)
  module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module);return module

def main(argv=None):
  parser=argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--protocol',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
  parser.add_argument('--preflight',action='store_true');parser.add_argument('--execute-gpu',action='store_true');args=parser.parse_args(argv)
  c=local(HERE/'agent-startup-cache-lifecycle-contract.py');m=c.m;protocol=m.read(args.protocol);c.gate(protocol)
  m.storage(args.output);sidecar=args.output.with_suffix('.cache-lifecycle.json')
  if sidecar.exists(): raise FileExistsError('Fresh overlay output required')
  if args.preflight==args.execute_gpu: raise ValueError('Choose CPU preflight or parent-only execution')
  digest=m.sha(args.protocol);before=importlib.util.spec_from_file_location
  report=dict(completed=False,gpu_executed=False,protocol_sha256=digest,base_sha256=c.PINS[str(c.BASE)],
    sources=c.sources(),source_before_after_exact=False,hooks_restored=False,ready_for_live_use=False)
  try:
    overlay=local(HERE/'agent-startup-cache-lifecycle-overlay.py');frozen=c.local(HERE/'agent-startup-persist-run.py')
    with overlay.imports(m) as state:
      frozen.run(SimpleNamespace(manifest=c.BASE,output=args.output,role='persisted',preflight=args.preflight,execute_gpu=args.execute_gpu))
      report['lifecycle']=overlay.summary(state)
      if state['modules']!=1 or len(state['owners'])!=int(args.execute_gpu): raise AssertionError('Unexpected lifecycle installation count')
    report.update(completed=True,gpu_executed=args.execute_gpu)
  except BaseException as error:
    report['failure']=dict(type=type(error).__name__,message=str(error));raise
  finally:
    report['hooks_restored']=importlib.util.spec_from_file_location is before
    try:
      c.gate(protocol)
      if m.sha(args.protocol)!=digest: raise ValueError('Overlay protocol changed during execution')
      report['source_before_after_exact']=True
    finally: m.save(sidecar,report)
    if not report['hooks_restored']: raise RuntimeError('Lifecycle import hook was not restored')
  if args.execute_gpu:
    result=c.compare(protocol,args.output,digest);m.save(args.output.with_suffix('.cache-lifecycle-comparison.json'),result)
    print(__import__('json').dumps(result),flush=True)
  else: print(__import__('json').dumps(dict(preflight=True,devices=m.read(args.output)['devices'],output=str(args.output))),flush=True)

if __name__=='__main__': main()
