"""Admit the exact disk cache, then execute the unchanged frozen composition runner."""
import argparse
import importlib.util
import os
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('expert_combined_cache_run_common',HERE/'agent-expert-combined-cache-common.py')
N = importlib.util.module_from_spec(spec); spec.loader.exec_module(N)


def cpu_env():
  return {**os.environ,'DEV':'CPU','ALLOW_DEVICE_USAGE':'0','TC_OPT':'2','CCACHE':'0','PYTHONDONTWRITEBYTECODE':'1',
    'TMPDIR':str(N.ARTIFACTS/'tmp'),'CACHEDB':str(N.ARTIFACTS/'cpu-cache.db'),
    'PYTHONPATH':':'.join(map(str,(N.TINY,N.LOADER.parent,N.LOADER.parent.parent)))}


def preflight(data,stage,output):
  outer = N.read(data['stages'][stage]['invocation'])
  terminal = N.read(data['stages'][stage]['pipeline'])
  for invocation in (outer,terminal): assert invocation['environment']['ALPAMAYO_FP8_CACHE'] == str(N.CACHE)
  cmd = terminal['command']; checkpoint = cmd[cmd.index('--checkpoint')+1]
  target = output.with_suffix('.agent-expert-combined-cache-preflight.json')
  assert not target.exists()
  command = [N.PY,str(HERE/'agent-expert-combined-cache-preflight.py'),'--output',str(target),'--checkpoint',checkpoint]
  result = subprocess.run(command,env=cpu_env(),capture_output=True,text=True,timeout=90)
  target.parent.mkdir(parents=True,exist_ok=True)
  target.with_suffix('.log').write_text(result.stdout+result.stderr)
  result.check_returncode()
  check = N.read(target)
  assert check['complete'] and not check['gpu_executed']
  assert check['factory']['actual_factory_load_executed'] and check['factory']['stopped_before_conversion_or_upload']
  assert check['factory']['real_device_open_attempts'] == [] and check['factory']['constructor_restored']
  assert check['cache']['cache'] == str(N.CACHE)
  return {'path':str(target),'sha256':N.sha(target),'actual_canonical_factory_cache_admission_passed':True}


def main():
  p = argparse.ArgumentParser(add_help=False)
  p.add_argument('--output',type=Path,required=True)
  p.add_argument('--stage',choices=('quality','native'),required=True)
  p.add_argument('--protocol',type=Path,required=True)
  p.add_argument('--protocol-sha',required=True)
  args,_ = p.parse_known_args(); assert sys.flags.optimize == 0 and not args.output.exists()
  C,data,_ = N.verify(args.protocol,args.protocol_sha)
  from tinygrad import Device
  assert not Device._opened_devices, 'Cache admission must precede device opening'
  result = preflight(data,args.stage,args.output)
  assert not Device._opened_devices
  metadata = args.output.with_suffix('.agent-expert-combined-cache.json')
  status = {'cache_admitted':True,'preflight':result,'original_protocol_sha256':N.OLD_SHA,
    'retry_protocol_sha256':args.protocol_sha,'runner_sha256':N.sha(N.OLD/'agent-expert-combined-run.py'),
    'numerical_sources_unchanged':True,'original_runner_completed':False}
  N.write(metadata,status,new=True)
  runner = C.load(N.OLD/'agent-expert-combined-run.py','expert_combined_cache_unchanged_runner')
  try:
    runner.main()  # Original arguments, constructors, coverage, comparators, and cleanup.
    status['original_runner_completed'] = True
  except BaseException as exc:
    status['failure'] = {'type':type(exc).__name__,'message':str(exc)}; raise
  finally: N.write(metadata,status)


if __name__ == '__main__': main()
