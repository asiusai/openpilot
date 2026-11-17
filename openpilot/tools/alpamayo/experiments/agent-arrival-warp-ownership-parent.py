"""CPU preparation and command printing only. Parent executes the normal serial wrapper."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shlex
import subprocess

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('arrival_ownership_parent_common',HERE/'agent-arrival-warp-ownership-common.py')
A = importlib.util.module_from_spec(spec);spec.loader.exec_module(A)


def main():
  p = argparse.ArgumentParser(description=__doc__);p.add_argument('--prepare',action='store_true');p.add_argument('--protocol-sha')
  args = p.parse_args();A.configure()
  if args.prepare:
    assert not A.PROTOCOL.exists()
    files = A.files();revision = hashlib.sha256(json.dumps(files,sort_keys=True).encode()).hexdigest()[:12]
    proof = A.WORK/f'agent-arrival-warp-ownership-cpu-{revision}.json';assert not proof.exists()
    env = {**os.environ,**A.C.N.storage(),'DEV':'CPU','ALLOW_DEVICE_USAGE':'0','PYTHONDONTWRITEBYTECODE':'1','PYTHONPATH':str(A.C.N.TINY)}
    with proof.with_suffix('.log').open('x') as log:
      subprocess.run([A.C.N.PY,str(HERE/'agent-arrival-warp-ownership-check.py'),'--output',str(proof)],
        env=env,stdout=log,stderr=subprocess.STDOUT,timeout=300,check=True)
    assert A.files() == files
    result = A.read(proof);assert result['complete'] and not result['gpu_executed']
    data = dict(files=files,foundation=A.foundation(),cpu_only=False,window=[35,43],hz=10,timeout_seconds=900,
      cpu_proof=dict(path=str(proof),sha256=A.sha(proof)))
    data['files'][str(proof)] = A.sha(proof)
    A.write(A.PROTOCOL,data,new=True);A.PROTOCOL.chmod(0o444)
  digest = args.protocol_sha or A.sha(A.PROTOCOL);A.verify(A.PROTOCOL,digest)
  command = [A.C.N.PY,str(A.C.N.WRAPPER),'agent-arrival-warp-ownership-paced1','--program',str(HERE/'agent-arrival-warp-ownership-run.py'),
    '--timeout-seconds','900','--set','protocol',str(A.PROTOCOL),'--set','protocol-sha',digest,
    '--env','PYTHONDONTWRITEBYTECODE','1','--env','PYTHONPATH',str(A.C.N.TINY)]
  for k,v in A.C.N.storage().items():command += ['--env',k,v]
  print('protocol_sha256='+digest);print(shlex.join(command))


if __name__ == '__main__':main()
