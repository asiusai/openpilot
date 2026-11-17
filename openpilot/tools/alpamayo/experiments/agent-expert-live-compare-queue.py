"""CPU prepare/check; parent candidate only, followed by separately reviewed pacing."""
import argparse
import hashlib
import importlib.util
import os
from pathlib import Path
import subprocess

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('compare_queue_common',HERE/'agent-expert-live-compare-common.py')
V=importlib.util.module_from_spec(spec);spec.loader.exec_module(V);N=V.N


def prepare():
  assert not V.ROOT.exists() and not V.PROTOCOL.exists() and not V.ADMISSION.exists()
  N.ARTIFACTS=V.ARTIFACTS;N.reserve(V.ARTIFACTS);N.configure()
  target=V.ARTIFACTS/'hooks/agent-exact-combined-install.py';target.parent.mkdir(exist_ok=True)
  original=N.EXP/target.name
  assert V.sha(original)==V.read(N.MANIFEST)['files'][str(original)]
  if not target.exists():target.write_bytes(original.read_bytes());target.chmod(0o444)
  V.configure()
  sources={str(p):V.sha(p) for p in sorted(HERE.glob('agent-expert-live-compare-*.py'))}
  revision=hashlib.sha256(str(sorted(sources.items())).encode()).hexdigest()[:12]
  cpu=V.WORK/f'agent-expert-live-compare-cpu-{revision}.json';assert not cpu.exists()
  env={**os.environ,**N.storage(),'DEV':'CPU','ALLOW_DEVICE_USAGE':'0','TC_OPT':'2','CCACHE':'0','PYTHONDONTWRITEBYTECODE':'1','PYTHONPATH':str(N.TINY)}
  child=subprocess.run([N.PY,str(HERE/'agent-expert-live-compare-check.py'),'--output',str(cpu)],env=env,capture_output=True,text=True,timeout=300)
  cpu.with_suffix('.log').write_text(child.stdout+child.stderr);child.check_returncode()
  proof=V.read(cpu);assert proof['complete'] and not proof['gpu_executed'] and proof['source_sha256']==sources
  recovered=V.recover_current();assert recovered==proof['current_admission']
  V.write(V.ADMISSION,recovered,new=True);V.ADMISSION.chmod(0o444)
  files=dict(V.read(V.BASE)['files']);files[str(V.BASE)]=V.BASE_SHA
  for path in [*V.CURRENT.parent.glob('*.json'),V.CURRENT.parent/'run.log',V.FROZEN/'parity-parent.json',target,cpu,V.ADMISSION]:files[str(path)]=V.sha(path)
  for path in (V.CURRENT,N.CONSUMER):
    report=V.read(path);V.snapshot(report,path)
    for source in Path(report['consumer_provenance']['initializer']['factory']['source_snapshot']).iterdir():files[str(source)]=V.sha(source)
  V.ROOT.mkdir();helpers={}
  for path in sorted(HERE.glob('agent-expert-live-compare-*.py')):
    frozen=V.ROOT/path.name;frozen.write_bytes(path.read_bytes());frozen.chmod(0o444)
    files[str(path)]=files[str(frozen)]=helpers[frozen.name]=V.sha(path)
  data=dict(schema=1,storage_protocol_sha256=V.BASE_SHA,current_report_sha256=V.CURRENT_SHA,files=files,
    evidence=V.read(V.BASE)['evidence'],cpu_proof=dict(path=str(cpu),sha256=V.sha(cpu)),
    current_admission=dict(path=str(V.ADMISSION),sha256=V.sha(V.ADMISSION)),runtime_overlay=V.runtime_module()[2],
    overlay_helper_sha256=helpers,gpu_arms=['candidate'],automatic_paced=False,consumer_timeout_seconds=1200,paced_timeout_seconds=900,
    selected_indices=[0,16],initial_history='empty',storage=N.storage(),pace=V.read(V.BASE)['pace'])
  V.write(V.PROTOCOL,data,new=True);V.PROTOCOL.chmod(0o444)
  print(dict(prepared=True,gpu_executed=False,current_admitted=True,protocol_sha256=V.sha(V.PROTOCOL),gpu_arms=['candidate']))


def command(arm,digest,reviewed=None):
  result=[N.PY,str(N.WRAPPER),'agent-expert-live-compare-'+arm,'--program',str(V.ROOT/'agent-expert-live-compare-run.py'),
    '--timeout-seconds','1200' if arm=='candidate' else '900','--set','arm',arm,'--set','protocol',str(V.PROTOCOL),
    '--set','protocol-sha',digest,'--env','PYTHONDONTWRITEBYTECODE','1','--env','PYTHONPATH',str(N.TINY)]
  for k,v in N.storage().items():result+=['--env',k,v]
  if arm=='paced':result+=['--set','reviewed-parity-sha256',reviewed or '<REVIEWED_NEW_CANDIDATE_REPORT_SHA256>']
  return result


def main():
  p=argparse.ArgumentParser(description=__doc__);g=p.add_mutually_exclusive_group()
  for name in ('prepare','check','execute-candidate','execute-paced'):g.add_argument('--'+name,action='store_true')
  p.add_argument('--protocol-sha');p.add_argument('--reviewed-parity-sha256');args=p.parse_args()
  if args.prepare:prepare();return
  assert args.protocol_sha;V.configure();data=V.verify(V.PROTOCOL,args.protocol_sha)
  if not args.execute_candidate and not args.execute_paced:
    print(dict(cpu_verified=True,gpu_executed=False,candidate=command('candidate',args.protocol_sha),
      paced_separate=command('paced',args.protocol_sha),current_gpu_rerun=False));return
  arm='candidate' if args.execute_candidate else 'paced'
  directory=V.WORK/('agent-expert-live-compare-'+arm);assert not directory.exists(),'Exclusive new output required'
  statepath=V.ROOT/(arm+'-parent.json');assert not statepath.exists()
  if arm=='paced':
    assert args.reviewed_parity_sha256 and V.sha(V.CANDIDATE)==args.reviewed_parity_sha256
    parent=V.read(V.ROOT/'candidate-parent.json');assert parent['complete'] and parent['report_sha256']==args.reviewed_parity_sha256
    candidate=V.read(V.CANDIDATE);assert candidate['agent_expert_live_compare']['passed']
    V.paired(candidate,data,args.protocol_sha)
  else:assert args.reviewed_parity_sha256 is None
  state=dict(complete=False,protocol_sha256=args.protocol_sha,current_gpu_rerun=False,automatic_paced=False,
    arm=arm,command=command(arm,args.protocol_sha,args.reviewed_parity_sha256))
  V.write(statepath,state,new=True)
  try:
    child=subprocess.run(state['command'],env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1'})
    state['returncode']=child.returncode;V.write(statepath,state);child.check_returncode()
    V.verify(V.PROTOCOL,args.protocol_sha)
    report=V.read(directory/'report.json');overlay=report['agent_expert_live_compare'];status=report['agent_expert_live_storage']
    assert overlay==V.read(directory/'report.agent-expert-live-compare.json') and overlay['passed'] and overlay['gpu_executed']
    assert overlay['runtime_overlay']==data['runtime_overlay'] and overlay['overlay_helper_sha256']==data['overlay_helper_sha256']
    assert status==V.read(directory/'report.agent-expert-live-storage.json') and status['passed'] and status['hooks_restored']
    wall=V.read(directory/'wall.json');assert wall['returncode']==0 and not wall['timed_out'] and wall['timeout_seconds']==(1200 if arm=='candidate' else 900)
    if arm=='candidate':state['paired_exact']=V.paired(report,data,args.protocol_sha)
    else:
      _,runtime,_=V.runtime_module();runtime.check_storage(status['storage_events'],paced=True)
      N.validate_startup(report,paced=True);N.validate_pv(status['pv'],True,3+len(report['rows']),data['evidence']['pv'])
    state.update(complete=True,report_sha256=V.sha(directory/'report.json'))
  except BaseException as error:state['failure']=dict(type=type(error).__name__,message=str(error));raise
  finally:V.write(statepath,state)


if __name__=='__main__':main()
