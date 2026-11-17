"""CPU freeze/preflight and explicitly separate parent GPU commands."""
import argparse
import hashlib
import importlib.util
import os
from pathlib import Path
import subprocess

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('storage_queue_common',HERE/'agent-expert-live-storage-common.py')
S=importlib.util.module_from_spec(spec);spec.loader.exec_module(S);N=S.N


def prepare():
  assert not S.ROOT.exists() and not S.PROTOCOL.exists()
  N.reserve(S.ARTIFACTS);N.configure()
  target=S.ARTIFACTS/'hooks/agent-exact-combined-install.py';target.parent.mkdir(exist_ok=True)
  original=N.EXP/target.name
  assert N.sha(original)==N.read(N.MANIFEST)['files'][str(original)]
  if not target.exists():target.write_bytes(original.read_bytes());target.chmod(0o444)
  S.configure();N.paths()
  sources={str(p):S.sha(p) for p in sorted(HERE.glob('agent-expert-live-storage-*.py'))}
  revision=hashlib.sha256(str(sorted(sources.items())).encode()).hexdigest()[:12]
  proof=S.WORK/f'agent-expert-live-storage-cpu-{revision}.json'
  assert not proof.exists()
  env={**os.environ,**N.storage(),'DEV':'CPU','ALLOW_DEVICE_USAGE':'0','TC_OPT':'2','CCACHE':'0','PYTHONDONTWRITEBYTECODE':'1',
    'PYTHONPATH':str(N.TINY)}
  child=subprocess.run([N.PY,str(HERE/'agent-expert-live-storage-check.py'),'--output',str(proof)],
    env=env,capture_output=True,text=True,timeout=300)
  proof.with_suffix('.log').write_text(child.stdout+child.stderr);child.check_returncode()
  checked=S.read(proof);assert checked['complete'] and not checked['gpu_executed'] and checked['source_sha256']==sources
  files=dict(S.read(S.BASE)['files']);files[str(S.BASE)]=S.BASE_SHA
  for path in [*S.FAILED.glob('*.json'),S.FAILED/'run.log',S.OLD/'consumer-parent.json',target,proof]:files[str(path)]=S.sha(path)
  S.ROOT.mkdir()
  for path in sorted(HERE.glob('agent-expert-live-storage-*.py')):
    frozen=S.ROOT/path.name;frozen.write_bytes(path.read_bytes());frozen.chmod(0o444)
    files[str(path)]=files[str(frozen)]=S.sha(path)
  data=dict(schema=1,base_protocol_sha256=S.BASE_SHA,files=files,evidence=N.evidence(),failed_evidence=S.failed_evidence(),
    cpu_proof=dict(path=str(proof),sha256=S.sha(proof)),arms=['current','candidate'],queries_per_arm=2,selected_indices=[0,16],
    initial_lookup_history='empty',automatic_paced=False,consumer_timeout_seconds=1200,paced_timeout_seconds=900,
    pace=S.read(S.BASE)['pace'],storage=N.storage(),numerical_change='None versus qualified original-combined plus optional PV PROGRAM',
    repair='Install storage normalizer before transformed main captures globals; audit actual canonical cache opens and temporary directory',
    new_gpu_equality_established=False)
  S.write(S.PROTOCOL,data,new=True);S.PROTOCOL.chmod(0o444)
  print(dict(prepared=True,gpu_executed=False,protocol_sha256=S.sha(S.PROTOCOL),arms=data['arms']))


def command(arm,digest,reviewed=None):
  out=[N.PY,str(N.WRAPPER),'agent-expert-live-storage-'+arm,'--program',str(S.ROOT/'agent-expert-live-storage-run.py'),
    '--timeout-seconds','900' if arm=='paced' else '1200','--set','arm',arm,'--set','protocol',str(S.PROTOCOL),
    '--set','protocol-sha',digest,'--env','PYTHONDONTWRITEBYTECODE','1','--env','PYTHONPATH',str(N.TINY)]
  for key,value in N.storage().items():out+=['--env',key,value]
  if arm=='paced':out+=['--set','reviewed-parity-sha256',reviewed or '<REVIEWED_NEW_CANDIDATE_REPORT_SHA256>']
  return out


def main():
  p=argparse.ArgumentParser(description=__doc__);g=p.add_mutually_exclusive_group()
  for name in ('prepare','check','execute-parity','execute-paced'):g.add_argument('--'+name,action='store_true')
  p.add_argument('--protocol-sha');p.add_argument('--reviewed-parity-sha256');args=p.parse_args()
  if args.prepare:prepare();return
  assert args.protocol_sha
  S.configure();N.paths();data=S.verify(S.PROTOCOL,args.protocol_sha)
  if not args.execute_parity and not args.execute_paced:
    print(dict(cpu_verified=True,gpu_executed=False,current=command('current',args.protocol_sha),
      candidate=command('candidate',args.protocol_sha),paced_separate=command('paced',args.protocol_sha)));return
  arms=['paced'] if args.execute_paced else ['current','candidate']
  statepath=S.ROOT/('paced-parent.json' if args.execute_paced else 'parity-parent.json')
  assert not statepath.exists()
  for arm in arms:assert not (S.WORK/('agent-expert-live-storage-'+arm)).exists(),'No overwrite or resume'
  if args.execute_paced:
    candidate=S.WORK/'agent-expert-live-storage-candidate/report.json'
    assert args.reviewed_parity_sha256 and S.sha(candidate)==args.reviewed_parity_sha256
    S.paired(S.read(S.WORK/'agent-expert-live-storage-current/report.json'),S.read(candidate),data,args.protocol_sha)
    prior=S.read(S.ROOT/'parity-parent.json');assert prior['complete'] and prior['paired_exact']['fresh_matched_pair']
  else:assert args.reviewed_parity_sha256 is None
  state=dict(complete=False,protocol_sha256=args.protocol_sha,automatic_paced=False,children=[])
  S.write(statepath,state,new=True)
  try:
    for arm in arms:
      child=dict(arm=arm,command=command(arm,args.protocol_sha,args.reviewed_parity_sha256))
      state['children'].append(child);S.write(statepath,state)
      result=subprocess.run(child['command'],env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1'})
      child['returncode']=result.returncode;S.write(statepath,state);result.check_returncode()
      S.verify(S.PROTOCOL,args.protocol_sha)
      directory=S.WORK/('agent-expert-live-storage-'+arm);report=S.read(directory/'report.json')
      status=S.read(directory/'report.agent-expert-live-storage.json')
      assert status==report['agent_expert_live_storage'] and status['passed'] and status['gpu_executed'] and status['hooks_restored']
      wall=S.read(directory/'wall.json');assert wall['returncode']==0 and not wall['timed_out']
      assert wall['timeout_seconds']==(900 if arm=='paced' else 1200)
      if arm!='paced':child['qualification']=S.qualify(report,data,args.protocol_sha,arm)
      else:
        N.validate_startup(report,paced=True);S.check_storage(status['storage_events'],paced=True)
        N.validate_pv(status['pv'],True,3+len(report['rows']),data['evidence']['pv'])
      child['report_sha256']=S.sha(directory/'report.json')
    if args.execute_parity:state['paired_exact']=S.paired(S.read(S.WORK/'agent-expert-live-storage-current/report.json'),
      S.read(S.WORK/'agent-expert-live-storage-candidate/report.json'),data,args.protocol_sha)
    state['complete']=True
  except BaseException as error:state['failure']=dict(type=type(error).__name__,message=str(error));raise
  finally:S.write(statepath,state)


if __name__=='__main__':main()
