"""Parent-only actual two-query Consumer, or separately requested bounded continuous replay."""
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
spec=importlib.util.spec_from_file_location('cpu_integrated_common',HERE/'agent-live-cpu-integrated-common.py')
C=importlib.util.module_from_spec(spec);spec.loader.exec_module(C)


def run(args):
  assert not sys.flags.optimize
  C.configure();data=C.verify(args.protocol,args.protocol_sha)
  assert args.output==C.output(args.arm) or (data.get('cpu_only') and args.output.is_relative_to(C.ARTIFACTS))
  assert not any(args.output.parent.glob(args.output.stem+'*')),'Keep all existing canonical artifacts'
  side=args.output.parent/'agent-live-cpu-integrated-status.json'
  assert not side.exists()
  C.N.reserve(args.output.parent);args.output.parent.mkdir(parents=True,exist_ok=True)
  m,manifest=C.N.live()
  if args.arm=='paced':
    if data.get('cpu_only'):
      # Exercise canonical paced setup using the existing qualified two-query gate.
      # Device use is disabled by the CPU protocol; this cannot admit a GPU run.
      path=C.REFERENCE;assert args.reviewed_parity_sha256==C.REFERENCE_SHA
    else:
      path=C.output('consumer')
      assert args.reviewed_parity_sha256 and C.sha(path)==args.reviewed_parity_sha256
      previous=C.read(path);status=previous['agent_live_cpu_integrated']
      assert status['protocol_sha256']==args.protocol_sha
      C.validate_status(previous,status,'consumer',path)
  else:assert args.reviewed_parity_sha256 is None
  admission=args.output.parent/'agent-live-cpu-integrated-cache-admission.json'
  env={**os.environ,**C.N.storage(),'DEV':'CPU','ALLOW_DEVICE_USAGE':'0','TC_OPT':'2','CCACHE':'0','PYTHONDONTWRITEBYTECODE':'1'}
  child=subprocess.run([C.N.PY,str(HERE/'agent-live-cpu-integrated-check.py'),'--admit','--output',str(admission)],
    env=env,capture_output=True,text=True,timeout=120)
  admission.with_suffix('.log').write_text(child.stdout+child.stderr);child.check_returncode()
  proof=C.read(admission);assert proof['complete'] and not proof['gpu_executed'] and proof['factory']['real_device_open_attempts']==[]
  status=dict(passed=False,gpu_executed=False,arm=args.arm,protocol_sha256=args.protocol_sha,
    reference_sha256=C.REFERENCE_SHA,cpu_report_sha256=C.CPU_SHA,hook_source_sha256=C.HOOK_SHA,
    cache_admission=dict(path=str(admission),sha256=C.sha(admission)),automatic_paced=False,hooks_restored=False)
  C.write(side,status,new=True)
  previous_spec=importlib.util.spec_from_file_location;hooks=None
  try:
    with ExitStack() as stack:
      stack.enter_context(patch.dict(os.environ));stack.enter_context(patch.object(sys,'path',sys.path[:]))
      for key,value in m.read(m.INVOCATION)['environment'].items():
        if value is None:os.environ.pop(key,None)
        else:os.environ[key]=str(value)
      C.configure()
      installer=C.load(HERE/'agent-live-cpu-integrated-install.py','cpu_integrated_installer')
      hooks=installer.factory(C)(args.arm)
      with hooks:
        if args.arm=='consumer':
          module=m.load(m.ROOT/'experiments/agent-live-replay.py','cpu_integrated_live_replay')
          options=['--invocation',str(m.INVOCATION),'--mode','consumer','--decoder','speculative',
            '--prefill-attention','shuffle-fma','--execute-gpu','--output',str(args.output),'--compare-reference',str(C.N.REFERENCE)]
        else:
          module=m.load(m.ROOT/'experiments/agent-paced-replay.py','cpu_integrated_paced_replay')
          options=['--mode','run','--invocation',str(m.INVOCATION),'--consumer-parity-report',str(path),
            '--hz','10','--start-seconds','35','--seconds','8','--inference-budget-seconds','180',
            '--raw-frames',str(m.RGB),'--warp','native','--output',str(args.output)]
        module.main(options)
        report=C.read(args.output);status['gpu_executed']=report.get('gpu_executed',args.arm=='paced')
        status.update(storage_events=copy.deepcopy(hooks.storage_events),transforms=dict(hooks.transforms),startup_spec=copy.deepcopy(hooks.startup_report))
        C.write(side,status)
        expected=manifest['expected_transforms']
        if args.arm=='consumer':expected={k:v for k,v in expected.items() if k!='paced'}
        m.equal(hooks.transforms,expected,'original_startup_and_policy_transforms')
        status['storage_proof']=C.S.check_storage(hooks.storage_events,paced=args.arm=='paced')
        status['coverage']=hooks.coverage()
        status['quantizer_idle']=hooks.composition.session.frame is None and not hooks.composition.session.state['prefill']
        assert status['quantizer_idle']
        report['startup_spec']=copy.deepcopy(hooks.startup_report)
      status.update(pv=hooks.composition.pv_handle.info,composition_restorations=hooks.composition.restorations,
        constructor_restored=hooks.composition.constructor_restored,cpu_hook=copy.deepcopy(hooks.cpu_info),
        hooks_restored=importlib.util.spec_from_file_location is previous_spec and hooks.composition.session.stats['methods_restored'])
    status['passed']=True
    status['comparison']=C.validate_status(report,status,args.arm,args.output)
    report['agent_live_cpu_integrated']=status;C.write(args.output,report)
  except BaseException as error:
    status['passed']=False;status['failure']=dict(type=type(error).__name__,message=str(error));raise
  finally:
    status['hooks_restored']=importlib.util.spec_from_file_location is previous_spec and (hooks is None or
      hooks.composition.session is None or hooks.composition.session.stats['methods_restored'])
    if hooks is not None:
      status.update(cpu_hook=copy.deepcopy(hooks.cpu_info),storage_events=copy.deepcopy(hooks.storage_events),
        constructor_restored=hooks.composition.constructor_restored,startup_spec=copy.deepcopy(hooks.startup_report))
    if args.output.exists():
      report=C.read(args.output);report['agent_live_cpu_integrated']=status;C.write(args.output,report)
    C.write(side,status)


if __name__=='__main__':
  p=argparse.ArgumentParser(description=__doc__)
  p.add_argument('--arm',choices=('consumer','paced'),required=True)
  p.add_argument('--protocol',type=Path,required=True);p.add_argument('--protocol-sha',required=True)
  p.add_argument('--reviewed-parity-sha256');p.add_argument('--output',type=Path,required=True)
  run(p.parse_args())
