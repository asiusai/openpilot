"""One source-bound startup arm. CPU preflight never enters the model factory."""
import argparse
from contextlib import ExitStack
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
from unittest.mock import patch

HERE=Path(__file__).resolve().parent
def local(path):
  spec=importlib.util.spec_from_file_location('startup_persist_run_'+Path(path).stem.replace('-','_'),path)
  module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module);return module

def run(args):
  m=local(HERE/'agent-startup-persist-common.py');manifest=m.read(args.manifest);m.verify(manifest)
  if Path(sys.executable).resolve()!=m.PY.resolve(): raise ValueError('Use the pinned Python executable')
  m.storage(args.output)
  if any(args.output.parent.glob(args.output.stem+'*')): raise FileExistsError('Fresh output family required')
  if args.execute_gpu==args.preflight: raise ValueError('Choose CPU preflight or parent-only GPU')
  environment={k:os.environ[k] for k in ('CACHEDB','TMPDIR','XDG_CACHE_HOME','TORCH_HOME')}
  for value in environment.values(): m.storage(value)
  environment.update(ALPAMAYO_FP8_CACHE=str(m.FP8),PYTHONDONTWRITEBYTECODE='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
    PYTHONPATH=f'{m.TG}:{m.ROOT}:/tmp/opencode/openpilot-alpamayo:/home/john/asius/openpilot/opendbc_repo')
  status=dict(role=args.role,manifest_sha256=m.sha(args.manifest),completed=False,gpu_executed=False,
    source_before_after_exact=False,hooks_restored=False,ready_for_live_use=False,speed_claim=False)
  started=time.perf_counter();hooks=None;original=importlib.util.spec_from_file_location
  try:
    with ExitStack() as stack:
      stack.enter_context(patch.dict(os.environ));stack.enter_context(patch.object(sys,'path',list(sys.path)))
      for k,v in m.read(m.INVOCATION)['environment'].items():
        if v is None: os.environ.pop(k,None)
        else: os.environ[k]=str(v)
      os.environ.update(environment);sys.dont_write_bytecode=True
      sys.path[:0]=[str(m.TG),str(m.ROOT),str(m.ROOT/'experiments'),'/tmp/opencode/openpilot-alpamayo','/home/john/asius/openpilot/opendbc_repo']
      from tinygrad import Device
      if args.preflight:
        def forbid(*a,**kw): raise AssertionError('CPU preflight attempted device access')
        stack.enter_context(patch.object(type(Device),'__getitem__',forbid))
      installer=local(HERE/'agent-startup-persist-hooks.py');hooks=installer.Hooks(m,manifest,args.role,environment,preflight=args.preflight)
      with hooks:
        module=m.load(m.ROOT/'experiments/agent-live-replay.py')
        options=['--invocation',str(m.INVOCATION),'--mode','consumer','--decoder','speculative',
          '--prefill-attention','shuffle-fma','--execute-gpu','--output',str(args.output)]
        try: module.main(options)
        except installer.PreflightBoundary:
          if not args.preflight: raise
        if args.preflight:
          if hooks.factory_calls!=1 or not hooks.boundary.get('actual_runpy_factory_reached') or Device._opened_devices:
            raise AssertionError('Actual runpy preflight failed before the factory boundary')
          for k,v in environment.items():
            if hooks.boundary['environment'][k]!=v: raise AssertionError('Canonical launcher overwrote storage: '+k)
          status.update(preflight=hooks.boundary,transforms=hooks.transforms,devices=list(Device._opened_devices))
        else:
          status.update(hooks.coverage(),gpu_executed=True)
          m.equal(hooks.transforms,manifest['transforms'],'production adaptations')
      status['hooks_restored']=importlib.util.spec_from_file_location is original
      if not status['hooks_restored']: raise AssertionError('Import hook restoration failed')
    m.verify(manifest);status.update(source_before_after_exact=True,completed=True,whole_process_instrumented_ms=(time.perf_counter()-started)*1000)
    if args.preflight: m.save(args.output,status)
    else:
      actual=m.read(args.output);actual.update(startup_spec=hooks.startup,startup_persist=status,ready_for_live_use=False)
      args.output.write_text(json.dumps(actual,indent=2,allow_nan=False)+'\n')
  except BaseException as error:
    status['failure']=dict(type=type(error).__name__,message=str(error));raise
  finally:
    status['hooks_restored']=importlib.util.spec_from_file_location is original
    try: m.verify(manifest);status['source_before_after_exact']=True
    except BaseException as error:
      status.update(completed=False,source_before_after_exact=False,source_binding_error=repr(error))
    m.save(args.output.with_suffix('.startup-persist.json'),dict(status=status,startup={} if hooks is None else hooks.startup))
    if not status['source_before_after_exact']: raise RuntimeError('Startup source binding changed during execution')

if __name__=='__main__':
  parser=argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--manifest',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
  parser.add_argument('--role',choices=('ordinary','persisted'),required=True)
  parser.add_argument('--preflight',action='store_true');parser.add_argument('--execute-gpu',action='store_true')
  run(parser.parse_args())
