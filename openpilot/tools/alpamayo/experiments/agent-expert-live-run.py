"""One optional PV Consumer parity run; paced execution requires a separate reviewed hash."""
import argparse
from contextlib import ExitStack
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('expert_live_run_common',HERE/'agent-expert-live-common.py')
N = importlib.util.module_from_spec(spec); spec.loader.exec_module(N)


def admit(output):
  path = output.parent/'agent-expert-live-cache-admission.json'
  assert not path.exists()
  env = {**os.environ,**N.storage(),'DEV':'CPU','ALLOW_DEVICE_USAGE':'0','TC_OPT':'2','CCACHE':'0','PYTHONDONTWRITEBYTECODE':'1',
    'PYTHONPATH':':'.join(map(str,(N.TINY,N.LIVE/'experiments',N.LIVE)))}
  result = subprocess.run([N.PY,str(HERE/'agent-expert-live-check.py'),'--admit','--output',str(path)],
    env=env,capture_output=True,text=True,timeout=120)
  path.with_suffix('.log').write_text(result.stdout+result.stderr); result.check_returncode()
  proof = N.read(path)
  assert proof['complete'] and not proof['gpu_executed'] and proof['factory']['actual_factory_load_executed']
  assert proof['factory']['real_device_open_attempts'] == [] and proof['actual_runpy_import_and_pickle']
  return {'path':str(path),'sha256':N.sha(path),'gpu_executed':False}


def run(args):
  assert sys.flags.optimize == 0
  storage = N.configure(); N.paths()
  data = N.verify(args.protocol,args.protocol_sha)
  m,manifest = N.live()
  from tinygrad import Device
  assert not Device._opened_devices
  assert args.output == N.WORK/f'agent-expert-live-{args.mode}/report.json'
  N.reserve(args.output.parent)
  assert not any(args.output.parent.glob('report*')), 'Fresh exclusive report paths required'
  assert Path(sys.executable).resolve() == Path(N.PY).resolve()
  if args.mode == 'paced':
    parity = N.WORK/'agent-expert-live-consumer/report.json'
    assert args.reviewed_parity_sha256 and N.sha(parity) == args.reviewed_parity_sha256
    N.qualify(N.read(parity),data,args.protocol_sha)
  else: assert args.reviewed_parity_sha256 is None
  args.output.parent.mkdir(parents=True,exist_ok=True)
  admission = admit(args.output)
  assert not Device._opened_devices
  status = {'mode':args.mode,'protocol_sha256':args.protocol_sha,'base_manifest_sha256':N.MANIFEST_SHA,
    'reference_sha256':N.REFERENCE_SHA,'saved_reference_admission':data['reference_admission'],
    'passed':False,'gpu_executed':False,'hooks_restored':False,'cache_admission':admission,
    'storage_reserve':storage,'automatic_paced':False,'speed_claim':False}
  sidecar = args.output.with_suffix('.agent-expert-live.json')
  N.write(sidecar,status,new=True)
  original_spec = importlib.util.spec_from_file_location
  hooks = None
  try:
    with ExitStack() as stack:
      stack.enter_context(patch.dict(os.environ))
      for key,value in m.read(m.INVOCATION)['environment'].items():
        if value is None: os.environ.pop(key,None)
        else: os.environ[key] = str(value)
      N.configure()
      stack.enter_context(patch.object(sys,'path',list(sys.path))); N.paths()
      installer = N.load(HERE/'agent-expert-live-install.py','expert_live_runtime_install')
      hooks = installer.factory(N)(args.mode)
      with hooks:
        if args.mode == 'consumer':
          module = m.load(m.ROOT/'experiments/agent-live-replay.py','expert_live_consumer_runner')
          options = ['--invocation',str(m.INVOCATION),'--mode','consumer','--decoder','speculative',
            '--prefill-attention','shuffle-fma','--execute-gpu','--output',str(args.output),'--compare-reference',str(N.REFERENCE)]
        else:
          module = m.load(m.ROOT/'experiments/agent-paced-replay.py','expert_live_paced_runner')
          options = ['--mode','run','--invocation',str(m.INVOCATION),
            '--consumer-parity-report',str(N.WORK/'agent-expert-live-consumer/report.json'),
            '--hz','10','--start-seconds','35','--seconds','8','--inference-budget-seconds','180',
            '--raw-frames',str(m.RGB),'--warp','native','--output',str(args.output)]
        module.main(options)
        expected = manifest['expected_transforms']
        if args.mode == 'consumer': expected = {k:v for k,v in expected.items() if k != 'paced'}
        m.equal(hooks.transforms,expected,'unchanged_full_startup_and_query_transforms')
        status.update(coverage=hooks.coverage(),transforms=hooks.transforms,storage_bindings=hooks.storage_bindings,gpu_executed=True)
        actual = N.read(args.output); actual['startup_spec'] = hooks.startup_report
        N.validate_startup(actual,paced=args.mode == 'paced')
        if args.mode == 'paced':
          assert actual['requested_hz'] == 10 and actual['source_window_s'] == 8 and actual['maximum_active_inference'] == 1
          assert not actual['queued_prediction_snapshots'] and actual['inference_budget_seconds'] == 180
          assert actual['rows'] and actual['rows'][0]['vision_hits'] == 0 and actual['rows'][0]['vision_misses'] == 10
      status['pv'] = hooks.composition.pv_handle.info
      status['hooks_restored'] = importlib.util.spec_from_file_location is original_spec and hooks.composition.session.stats['methods_restored']
      status['restorations'] = hooks.restorations+hooks.composition.restorations
      assert status['hooks_restored'] and all(r['restored'] for r in hooks.composition.restorations)
      N.validate_pv(status['pv'],True,5 if args.mode == 'consumer' else 3+len(actual['rows']),data['evidence']['pv'])
    N.verify(args.protocol,args.protocol_sha)
    status['passed'] = True
    actual['agent_expert_live'] = status
    if args.mode == 'consumer': status['comparison'] = N.qualify(actual,data,args.protocol_sha)
    N.write(args.output,actual)
  except BaseException as exc:
    status['passed'] = False; status['failure'] = {'type':type(exc).__name__,'message':str(exc)}
    raise
  finally:
    status['hooks_restored'] = importlib.util.spec_from_file_location is original_spec and (hooks is None or
      (hooks.composition.session is None or hooks.composition.session.stats['methods_restored']))
    if hooks is not None and hooks.composition.pv_handle is not None: status['pv'] = hooks.composition.pv_handle.info
    N.write(sidecar,status)
    if args.output.exists():
      actual = N.read(args.output); actual['agent_expert_live'] = status; N.write(args.output,actual)


if __name__ == '__main__':
  p = argparse.ArgumentParser(description=__doc__)
  p.add_argument('--mode',choices=('consumer','paced'),required=True)
  p.add_argument('--protocol',type=Path,required=True)
  p.add_argument('--protocol-sha',required=True)
  p.add_argument('--reviewed-parity-sha256')
  p.add_argument('--output',type=Path,required=True)
  run(p.parse_args())
