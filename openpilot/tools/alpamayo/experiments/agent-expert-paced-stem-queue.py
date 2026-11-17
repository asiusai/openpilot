"""CPU preparation and one explicitly reviewed 900-second parent paced child."""
import argparse
import hashlib
import importlib.util
import os
from pathlib import Path
import subprocess

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('stem_queue_common',HERE/'agent-expert-paced-stem-common.py')
C=importlib.util.module_from_spec(spec);spec.loader.exec_module(C);N=C.N


def prepare():
  assert not C.ROOT.exists() and not C.PROTOCOL.exists()
  N.reserve(C.ARTIFACTS);N.configure()
  target=C.ARTIFACTS/'hooks/agent-exact-combined-install.py';target.parent.mkdir(exist_ok=True)
  original=N.EXP/target.name;assert C.sha(original)==C.read(N.MANIFEST)['files'][str(original)]
  if not target.exists():target.write_bytes(original.read_bytes());target.chmod(0o444)
  C.configure()
  sources={str(p):C.sha(p) for p in HERE.glob('agent-expert-paced-stem-*.py')}
  revision=hashlib.sha256(str(sorted(sources.items())).encode()).hexdigest()[:12]
  output=C.WORK/f'agent-expert-paced-stem-cpu-{revision}.json';assert not output.exists()
  env={**os.environ,**N.storage(),'DEV':'CPU','ALLOW_DEVICE_USAGE':'0','TC_OPT':'2','CCACHE':'0','PYTHONDONTWRITEBYTECODE':'1','PYTHONPATH':str(N.TINY)}
  with output.with_suffix('.log').open('x') as log:
    child=subprocess.run([N.PY,str(HERE/'agent-expert-paced-stem-check.py'),'--output',str(output)],
      env=env,stdout=log,stderr=subprocess.STDOUT,text=True,timeout=600)
  child.check_returncode()
  proof=C.read(output);assert proof['complete'] and not proof['gpu_executed'] and proof['source_sha256']==sources
  check=C.load(HERE/'agent-expert-paced-stem-check.py','stem_prepare_evidence')
  data=check.protocol(C.OUTPUT,check.evidence_files(),C.candidate_admission(),cpu=False)
  data['cpu_proof']=dict(path=str(output),sha256=C.sha(output));data['files'][str(output)]=C.sha(output)
  C.ROOT.mkdir()
  for source in sorted(HERE.glob('agent-expert-paced-stem-*.py')):
    dest=C.ROOT/source.name;dest.write_bytes(source.read_bytes());dest.chmod(0o444)
    data['files'][str(source)]=data['files'][str(dest)]=C.sha(source)
  C.write(C.PROTOCOL,data,new=True);C.PROTOCOL.chmod(0o444)
  print(dict(prepared=True,gpu_executed=False,protocol_sha256=C.sha(C.PROTOCOL),only_child='paced',parity_rerun=False))


def command(digest):
  result=[N.PY,str(N.WRAPPER),'agent-expert-paced-stem-run','--program',str(C.ROOT/'agent-expert-paced-stem-run.py'),
    '--timeout-seconds','900','--set','protocol',str(C.PROTOCOL),'--set','protocol-sha',digest,
    '--set','reviewed-parity-sha256',C.CANDIDATE_SHA,'--env','PYTHONDONTWRITEBYTECODE','1','--env','PYTHONPATH',str(N.TINY)]
  for k,v in N.storage().items():result+=['--env',k,v]
  return result


def main():
  p=argparse.ArgumentParser(description=__doc__);g=p.add_mutually_exclusive_group()
  for name in ('prepare','check','execute'):g.add_argument('--'+name,action='store_true')
  p.add_argument('--protocol-sha');p.add_argument('--reviewed-parity-sha256');args=p.parse_args()
  if args.prepare:prepare();return
  assert args.protocol_sha;C.configure();data=C.verify(C.PROTOCOL,args.protocol_sha)
  if not args.execute:
    print(dict(cpu_verified=True,gpu_executed=False,command=command(args.protocol_sha),parity_rerun=False));return
  assert args.reviewed_parity_sha256==C.CANDIDATE_SHA
  assert not C.OUTPUT.parent.exists(),'Never overwrite or resume a prior paced run'
  statepath=C.ROOT/'paced-parent.json';assert not statepath.exists()
  state=dict(complete=False,protocol_sha256=args.protocol_sha,candidate_sha256=C.CANDIDATE_SHA,
    command=command(args.protocol_sha),parity_rerun=False,automatic_paced=False)
  C.write(statepath,state,new=True)
  try:
    child=subprocess.run(state['command'],env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1'})
    state['returncode']=child.returncode;C.write(statepath,state);child.check_returncode()
    C.verify(C.PROTOCOL,args.protocol_sha)
    report=C.read(C.OUTPUT);paths=C.metadata(C.OUTPUT)
    current=C.read(paths['status']);assert current==report['agent_expert_paced_stem'] and current['passed'] and current['gpu_executed']
    assert current['artifact_overlay']==data['artifact_overlay'] and current['overlay_helper_sha256']==data['overlay_helper_sha256']
    status=C.read(paths['storage']);assert status==report['agent_expert_live_storage'] and status['passed'] and status['hooks_restored']
    assert C.read(paths['compare'])==report['agent_expert_live_compare'] and report['agent_expert_live_compare']['passed']
    _,runtime,_=C.build(C.OUTPUT);runtime.check_storage(status['storage_events'],paced=True)
    N.validate_startup(report,paced=True);N.validate_pv(status['pv'],True,3+len(report['rows']),C.read(C.QUALIFIED_PROTOCOL)['evidence']['pv'])
    wall=C.read(C.OUTPUT.parent/'wall.json');assert wall['returncode']==0 and not wall['timed_out'] and wall['timeout_seconds']==900
    state.update(complete=True,report_sha256=C.sha(C.OUTPUT))
  except BaseException as error:state['failure']=dict(type=type(error).__name__,message=str(error));raise
  finally:C.write(statepath,state)


if __name__=='__main__':main()
