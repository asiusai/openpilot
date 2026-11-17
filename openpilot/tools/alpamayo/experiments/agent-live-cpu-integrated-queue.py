"""Prepare CPU evidence; expose separate parent Consumer and continuous commands."""
import argparse
import hashlib
import importlib.util
import os
from pathlib import Path
import subprocess

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('cpu_integrated_queue_common',HERE/'agent-live-cpu-integrated-common.py')
C=importlib.util.module_from_spec(spec);spec.loader.exec_module(C)


def prepare():
  assert not C.ROOT.exists() and not C.PROTOCOL.exists()
  C.configure()
  sources={str(p):C.sha(p) for p in HERE.glob('agent-live-cpu-integrated-*.py')}
  revision=hashlib.sha256(str(sorted(sources.items())).encode()).hexdigest()[:12]
  proof=C.WORK/f'agent-live-cpu-integrated-cpu-{revision}.json';assert not proof.exists()
  env={**os.environ,**C.N.storage(),'DEV':'CPU','ALLOW_DEVICE_USAGE':'0','PYTHONDONTWRITEBYTECODE':'1','PYTHONPATH':str(C.N.TINY)}
  with proof.with_suffix('.log').open('x') as log:
    child=subprocess.run([C.N.PY,str(HERE/'agent-live-cpu-integrated-check.py'),'--output',str(proof)],
      env=env,stdout=log,stderr=subprocess.STDOUT,timeout=600)
  child.check_returncode();result=C.read(proof)
  assert result['complete'] and not result['gpu_executed']
  assert all(C.sha(path)==digest for path,digest in sources.items())
  data=dict(schema=1,cpu_only=False,files=C.files(),foundations=C.foundations(),arms=['consumer','paced'],automatic_paced=False,
    cpu_proof=dict(path=str(proof),sha256=C.sha(proof)),timeout_seconds={'consumer':1200,'paced':900})
  data['files'][str(proof)]=C.sha(proof)
  C.ROOT.mkdir()
  for path,digest in sources.items():
    target=C.ROOT/Path(path).name;target.write_bytes(Path(path).read_bytes());target.chmod(0o444)
    data['files'][str(target)]=digest
  C.write(C.PROTOCOL,data,new=True);C.PROTOCOL.chmod(0o444)
  print(dict(prepared=True,protocol_sha256=C.sha(C.PROTOCOL),direct_pins=len(data['files']),gpu_executed=False,automatic_paced=False))


def command(arm,digest,reviewed=None):
  args=[C.N.PY,str(C.N.WRAPPER),'agent-live-cpu-integrated-'+arm,'--program',str(C.ROOT/'agent-live-cpu-integrated-run.py'),
    '--timeout-seconds',str(1200 if arm=='consumer' else 900),'--set','arm',arm,
    '--set','protocol',str(C.PROTOCOL),'--set','protocol-sha',digest,'--env','PYTHONDONTWRITEBYTECODE','1',
    '--env','PYTHONPATH',str(C.N.TINY)]
  if reviewed:args+=['--set','reviewed-parity-sha256',reviewed]
  for k,v in C.N.storage().items():args+=['--env',k,v]
  return args


def main():
  p=argparse.ArgumentParser(description=__doc__);g=p.add_mutually_exclusive_group()
  g.add_argument('--prepare',action='store_true');g.add_argument('--consumer',action='store_true');g.add_argument('--paced',action='store_true')
  p.add_argument('--check',action='store_true');p.add_argument('--protocol-sha');p.add_argument('--reviewed-parity-sha256')
  args=p.parse_args()
  if args.prepare:prepare();return
  assert args.protocol_sha;C.configure();data=C.verify(C.PROTOCOL,args.protocol_sha)
  arm='paced' if args.paced else 'consumer'
  if arm=='paced' and not args.check:
    assert args.reviewed_parity_sha256 and C.sha(C.output('consumer'))==args.reviewed_parity_sha256
    report=C.read(C.output('consumer'));C.validate_status(report,report['agent_live_cpu_integrated'],'consumer',C.output('consumer'))
  if args.check:
    print(dict(cpu_verified=True,gpu_executed=False,consumer_command=command('consumer',args.protocol_sha),
      paced_requires_fresh_reviewed_candidate_sha=True,automatic_paced=False));return
  assert args.consumer or args.paced
  assert not C.output(arm).parent.exists(),'Use a new output; never overwrite or resume'
  statepath=C.ROOT/(arm+'-parent.json');assert not statepath.exists()
  state=dict(complete=False,arm=arm,protocol_sha256=args.protocol_sha,automatic_paced=False,
    command=command(arm,args.protocol_sha,args.reviewed_parity_sha256))
  C.write(statepath,state,new=True)
  try:
    child=subprocess.run(state['command'],env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1'})
    state['returncode']=child.returncode;C.write(statepath,state);child.check_returncode()
    C.verify(C.PROTOCOL,args.protocol_sha)
    report=C.read(C.output(arm));status=report['agent_live_cpu_integrated']
    assert status==C.read(C.output(arm).parent/'agent-live-cpu-integrated-status.json')
    result=C.validate_status(report,status,arm,C.output(arm))
    wall=C.read(C.output(arm).parent/'wall.json')
    assert wall['returncode']==0 and not wall['timed_out'] and wall['timeout_seconds']==data['timeout_seconds'][arm]
    state.update(complete=True,report_sha256=C.sha(C.output(arm)),validation=result)
  except BaseException as error:state['failure']=dict(type=type(error).__name__,message=str(error));raise
  finally:C.write(statepath,state)


if __name__=='__main__':main()
