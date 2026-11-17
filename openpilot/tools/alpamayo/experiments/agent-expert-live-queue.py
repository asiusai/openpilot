#!/tmp/opencode/alpamayo-asius-20261114/upstream/.venv/bin/python
"""CPU freeze/check; optional Consumer parity and explicitly separate paced command."""
import argparse
import hashlib
import importlib.util
import os
from pathlib import Path
import subprocess

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('expert_live_queue_common',HERE/'agent-expert-live-common.py')
N=importlib.util.module_from_spec(spec);spec.loader.exec_module(N)


def cpu_env():
  return {**os.environ,**N.storage(),'DEV':'CPU','ALLOW_DEVICE_USAGE':'0','TC_OPT':'2','CCACHE':'0','PYTHONDONTWRITEBYTECODE':'1',
    'PYTHONPATH':':'.join(map(str,(N.TINY,N.LIVE/'experiments',N.LIVE)))}


def prepare():
  assert not N.ROOT.exists() and not N.PROTOCOL.exists()
  assert not (N.WORK/'agent-expert-live-consumer').exists() and not (N.WORK/'agent-expert-live-paced').exists()
  N.configure()
  assert N.sha(N.MANIFEST)==N.MANIFEST_SHA
  manifest=N.read(N.MANIFEST)
  source=N.EXP/'agent-exact-combined-install.py'
  assert N.sha(source)==manifest['files'][str(source)]
  destination=N.ARTIFACTS/'hooks'/source.name
  destination.parent.mkdir(exist_ok=True)
  if not destination.exists(): destination.write_bytes(source.read_bytes());destination.chmod(0o444)
  assert N.sha(destination)==N.sha(source)
  sources={str(p):N.sha(p) for p in HERE.glob('agent-expert-live-*.py')}
  revision=hashlib.sha256(str(sorted(sources.items())).encode()).hexdigest()[:12]
  proofs={}
  for mode in ('controls','admission'):
    output=N.WORK/f'agent-expert-live-{mode}-{revision}.json'
    if not output.exists():
      command=[N.PY,str(HERE/'agent-expert-live-check.py'),'--output',str(output)]
      if mode=='admission':command+=['--admit']
      child=subprocess.run(command,env=cpu_env(),capture_output=True,text=True,timeout=300)
      output.with_suffix('.log').write_text(child.stdout+child.stderr);child.check_returncode()
    proof=N.read(output)
    assert proof['complete'] and not proof['gpu_executed'] and proof['source_sha256']==sources
    proofs[mode]={'path':str(output),'sha256':N.sha(output)}
  N.paths(); evidence=N.evidence()
  N.ROOT.mkdir()
  files=dict(manifest['files'])
  files.update(N.read(N.CACHE_PROTOCOL)['files'])
  for path,digest in N.read(N.PV_PROTOCOL)['files'].items():files[path]=digest
  for path in [N.MANIFEST,N.CACHE_PROTOCOL,N.PV_PROTOCOL,N.REFERENCE,N.CONSUMER,N.PACED,N.WRAPPER,destination,
               *(Path(v['path']) for v in proofs.values())]:files[str(path)]=N.sha(path)
  for label in N.COMPOSED:
    directory=N.WORK/f'agent-expert-combined-cache-{label}'
    for path in directory.glob('*.json'):files[str(path)]=N.sha(path)
  for base in (N.REFERENCE,N.CONSUMER,N.PACED):
    for path in base.parent.glob('*.json'):files[str(path)]=N.sha(path)
  for path in sorted(HERE.glob('agent-expert-live-*.py')):
    dest=N.ROOT/path.name;dest.write_bytes(path.read_bytes());dest.chmod(0o444)
    files[str(path)]=N.sha(path);files[str(dest)]=N.sha(dest)
  data={'schema':1,'base_manifest_sha256':N.MANIFEST_SHA,'reference_sha256':N.REFERENCE_SHA,'files':files,
    'evidence':evidence,'reference_admission':N.read(proofs['controls']['path'])['reference_admission'],
    'cpu_proofs':proofs,'pace':manifest['pace'],'automatic_paced':False,'consumer_timeout_seconds':1200,
    'paced_timeout_seconds':900,'timeout_reason':'Retain original 1200s parity bound and complete 900s-bounded startup; paced active budget remains180s',
    'only_numeric_change':'Complete qualified WGP2x2 expert PV PROGRAM; original CU operation remains all noneligible dispatches',
    'storage':N.storage(),'new_gpu_runs_prepared':['consumer'],'gpu_executed_during_preparation':False}
  N.write(N.PROTOCOL,data,new=True);N.PROTOCOL.chmod(0o444)
  print({'prepared':True,'gpu_executed':False,'protocol_sha256':N.sha(N.PROTOCOL),'parity_runs':1})


def command(mode,digest,reviewed=None):
  result=[N.PY,str(N.WRAPPER),'agent-expert-live-'+mode,'--program',str(N.ROOT/'agent-expert-live-run.py'),
    '--timeout-seconds','1200' if mode=='consumer' else '900','--set','mode',mode,
    '--set','protocol',str(N.PROTOCOL),'--set','protocol-sha',digest,'--env','PYTHONDONTWRITEBYTECODE','1',
    '--env','PYTHONPATH',':'.join(map(str,(N.TINY,N.LIVE/'experiments',N.LIVE)))]
  for key,value in N.storage().items():result+=['--env',key,value]
  if mode=='paced':result+=['--set','reviewed-parity-sha256',reviewed or '<PARENT_INSPECTED_CONSUMER_REPORT_SHA256>']
  return result


def main():
  parser=argparse.ArgumentParser(description=__doc__)
  modes=parser.add_mutually_exclusive_group()
  for name in ('prepare','check','execute-parity','execute-paced'):modes.add_argument('--'+name,action='store_true')
  parser.add_argument('--protocol-sha');parser.add_argument('--reviewed-parity-sha256')
  args=parser.parse_args()
  if args.prepare:prepare();return
  assert args.protocol_sha
  N.configure();N.paths();data=N.verify(N.PROTOCOL,args.protocol_sha)
  if not args.execute_parity and not args.execute_paced:
    print({'cpu_verified':True,'gpu_executed':False,'parity':command('consumer',args.protocol_sha),
      'paced_separate':command('paced',args.protocol_sha),'automatic_paced':False});return
  mode='consumer' if args.execute_parity else 'paced'
  parity=N.WORK/'agent-expert-live-consumer/report.json'
  if mode=='paced':
    assert args.reviewed_parity_sha256 and N.sha(parity)==args.reviewed_parity_sha256
    N.qualify(N.read(parity),data,args.protocol_sha)
  else:assert args.reviewed_parity_sha256 is None
  directory=N.WORK/('agent-expert-live-'+mode);assert not directory.exists(),'Never overwrite or resume live evidence'
  target=N.ROOT/(mode+'-parent.json');assert not target.exists()
  state={'complete':False,'protocol_sha256':args.protocol_sha,'mode':mode,'automatic_paced':False,
    'command':command(mode,args.protocol_sha,args.reviewed_parity_sha256)}
  N.write(target,state,new=True)
  try:
    result=subprocess.run(state['command'],env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1'})
    state['returncode']=result.returncode;N.write(target,state);result.check_returncode()
    N.verify(N.PROTOCOL,args.protocol_sha)
    report=N.read(directory/'report.json');sidecar=N.read(directory/'report.agent-expert-live.json')
    assert report['agent_expert_live']==sidecar and sidecar['passed'] and sidecar['hooks_restored'] and sidecar['gpu_executed']
    wall=N.read(directory/'wall.json');assert wall['returncode']==0 and not wall['timed_out']
    assert wall['timeout_seconds']==(1200 if mode=='consumer' else 900)
    if mode=='consumer':state['exact_parity']=N.qualify(report,data,args.protocol_sha)
    else:
      N.validate_startup(report,paced=True)
      N.validate_pv(sidecar['pv'],True,3+len(report['rows']),data['evidence']['pv'])
    state.update(complete=True,report_sha256=N.sha(directory/'report.json'))
  except BaseException as error:
    state['failure']={'type':type(error).__name__,'message':str(error)};raise
  finally:N.write(target,state)


if __name__=='__main__':main()
