"""Parent-only fresh current/candidate Consumers; paced execution is separate."""
import argparse
from contextlib import ExitStack
import copy
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('storage_run_common',HERE/'agent-expert-live-storage-common.py')
S=importlib.util.module_from_spec(spec);spec.loader.exec_module(S);N=S.N


def run(args):
  assert not sys.flags.optimize
  S.configure();N.paths();data=S.verify(args.protocol,args.protocol_sha);m,manifest=N.live()
  from tinygrad import Device
  assert not Device._opened_devices
  assert args.output==S.WORK/f'agent-expert-live-storage-{args.arm}/report.json'
  assert not any(args.output.parent.glob('report*'))
  N.reserve(args.output.parent);args.output.parent.mkdir(exist_ok=True)
  if args.arm=='paced':
    current=S.read(S.WORK/'agent-expert-live-storage-current/report.json')
    path=S.WORK/'agent-expert-live-storage-candidate/report.json'
    assert args.reviewed_parity_sha256 and S.sha(path)==args.reviewed_parity_sha256
    S.paired(current,S.read(path),data,args.protocol_sha)
  else:assert args.reviewed_parity_sha256 is None
  admission=args.output.parent/'agent-expert-live-storage-admission.json'
  env={**os.environ,**N.storage(),'DEV':'CPU','ALLOW_DEVICE_USAGE':'0','TC_OPT':'2','CCACHE':'0','PYTHONDONTWRITEBYTECODE':'1'}
  child=subprocess.run([N.PY,str(HERE/'agent-expert-live-storage-check.py'),'--admit','--output',str(admission)],
    env=env,capture_output=True,text=True,timeout=120)
  admission.with_suffix('.log').write_text(child.stdout+child.stderr);child.check_returncode()
  proof=S.read(admission);assert proof['complete'] and not proof['gpu_executed'] and proof['factory']['real_device_open_attempts']==[]
  assert not Device._opened_devices
  status=dict(arm=args.arm,protocol_sha256=args.protocol_sha,base_protocol_sha256=S.BASE_SHA,passed=False,
    gpu_executed=False,hooks_restored=False,cache_admission=dict(path=str(admission),sha256=S.sha(admission)),
    automatic_paced=False,speed_claim=False)
  sidecar=args.output.with_suffix('.agent-expert-live-storage.json');S.write(sidecar,status,new=True)
  initial_spec=importlib.util.spec_from_file_location;hooks=None
  try:
    with ExitStack() as stack:
      stack.enter_context(patch.dict(os.environ));stack.enter_context(patch.object(sys,'path',sys.path[:]))
      for key,value in m.read(m.INVOCATION)['environment'].items():
        if value is None:os.environ.pop(key,None)
        else:os.environ[key]=str(value)
      S.configure();N.paths()
      installer=S.load(HERE/'agent-expert-live-storage-install.py','storage_runtime_install')
      hooks=installer.factory(S,pv_candidate=args.arm!='current')('paced' if args.arm=='paced' else 'consumer')
      with hooks:
        if args.arm=='paced':
          module=m.load(m.ROOT/'experiments/agent-paced-replay.py','storage_runtime_paced')
          options=['--mode','run','--invocation',str(m.INVOCATION),'--consumer-parity-report',str(path),
            '--hz','10','--start-seconds','35','--seconds','8','--inference-budget-seconds','180',
            '--raw-frames',str(m.RGB),'--warp','native','--output',str(args.output)]
        else:
          module=m.load(m.ROOT/'experiments/agent-live-replay.py','storage_runtime_consumer')
          options=['--invocation',str(m.INVOCATION),'--mode','consumer','--decoder','speculative',
            '--prefill-attention','shuffle-fma','--execute-gpu','--output',str(args.output),'--compare-reference',str(N.REFERENCE)]
        module.main(options)
        actual=S.read(args.output);status['gpu_executed']=actual.get('gpu_executed',args.arm=='paced')
        # Persist observations before any postrun guard can fail.
        status['storage_events']=copy.deepcopy(hooks.storage_events)
        status['transforms']=dict(hooks.transforms)
        status['startup_spec']=copy.deepcopy(hooks.startup_report)
        S.write(sidecar,status)
        expected=manifest['expected_transforms']
        if args.arm!='paced':expected={k:v for k,v in expected.items() if k!='paced'}
        m.equal(hooks.transforms,expected,'unaltered_original_full_startup_transforms')
        status['storage_proof']=S.check_storage(hooks.storage_events,paced=args.arm=='paced')
        status['coverage']=hooks.coverage()
        status['quantizer_idle']=hooks.composition.session.frame is None and not hooks.composition.session.state['prefill']
        assert status['quantizer_idle']
        actual['startup_spec']=hooks.startup_report;N.validate_startup(actual,paced=args.arm=='paced')
        if args.arm=='paced':
          assert actual['requested_hz']==10 and actual['source_window_s']==8 and actual['maximum_active_inference']==1
          assert actual['queued_prediction_snapshots']==0 and actual['inference_budget_seconds']==180
          assert actual['rows'][0]['vision_hits']==0 and actual['rows'][0]['vision_misses']==10
      status['pv']=hooks.composition.pv_handle.info
      status['composition_restorations']=hooks.composition.restorations
      status['constructor_restored']=hooks.composition.constructor_restored
      status['hooks_restored']=importlib.util.spec_from_file_location is initial_spec and hooks.composition.session.stats['methods_restored']
      assert status['constructor_restored'] and status['hooks_restored'] and all(x['restored'] for x in status['composition_restorations'])
      N.validate_pv(status['pv'],args.arm!='current',3+len(actual['rows']),data['evidence']['pv'])
    S.verify(args.protocol,args.protocol_sha)
    status['passed']=True;actual['agent_expert_live_storage']=status
    if args.arm!='paced':status['comparison']=S.qualify(actual,data,args.protocol_sha,args.arm)
    S.write(args.output,actual)
  except BaseException as error:
    status['passed']=False;status['failure']=dict(type=type(error).__name__,message=str(error));raise
  finally:
    status['hooks_restored']=importlib.util.spec_from_file_location is initial_spec and (hooks is None or
      hooks.composition.session is None or hooks.composition.session.stats['methods_restored'])
    if hooks is not None:
      status['storage_events']=copy.deepcopy(hooks.storage_events)
      status['startup_spec']=copy.deepcopy(hooks.startup_report)
      status['transforms']=dict(hooks.transforms)
      status['constructor_restored']=hooks.composition.constructor_restored
      status['composition_restorations']=hooks.composition.restorations
      if hooks.composition.pv_handle is not None:status['pv']=hooks.composition.pv_handle.info
    if args.output.exists():
      actual=S.read(args.output);status['gpu_executed']=actual.get('gpu_executed',status['gpu_executed'])
      actual['agent_expert_live_storage']=status;S.write(args.output,actual)
    S.write(sidecar,status)


if __name__=='__main__':
  p=argparse.ArgumentParser(description=__doc__)
  p.add_argument('--arm',choices=('current','candidate','paced'),required=True)
  p.add_argument('--protocol',type=Path,required=True);p.add_argument('--protocol-sha',required=True)
  p.add_argument('--reviewed-parity-sha256');p.add_argument('--output',type=Path,required=True)
  run(p.parse_args())
