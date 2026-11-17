"""Private parent queue. The default only checks; --execute runs four gated children."""
import argparse
import importlib
import json
import subprocess

C=importlib.import_module('agent-compiler-speed-common')


def main():
  p=argparse.ArgumentParser(description=__doc__)
  p.add_argument('--protocol-sha',required=True);p.add_argument('--execute',action='store_true')
  args=p.parse_args();protocol=C.verify(C.ROOT/'protocol.json',args.protocol_sha)
  commands=C.read(C.ROOT/'commands.json')
  assert commands['protocol_sha256']==args.protocol_sha
  assert list(commands['commands'])==[s+'/'+a for s,a in protocol['run_order']]
  assert commands['commands']==C.commands(args.protocol_sha),'Command bytes differ from the source-bound recipe'
  if not args.execute:
    print(json.dumps({'checked':True,'gpu_executed':False,'commands':commands['shell']},indent=2));return
  assert all(not C.out(s,a).exists() for s,a in protocol['run_order']), 'Never overwrite an existing run'
  for stage,arm in protocol['run_order']:
    if stage=='native':C.quality_gate(args.protocol_sha,protocol)
    command=commands['commands'][stage+'/'+arm]
    assert command[command.index('--timeout-seconds')+1]=='900'
    subprocess.run(command,check=True)
    C.checked_result(C.metadata_path(stage,arm),args.protocol_sha)
    if arm=='consistent-wgp':
      result=C.validate_pair(stage,args.protocol_sha,protocol)
      C.write(C.WORK/f'agent-compiler-speed-{stage}-gate.json',result)
      print(json.dumps({'stage':stage,'exact':result['exact'],'native_throughput':result['native_throughput']},indent=2),flush=True)


if __name__=='__main__':main()
