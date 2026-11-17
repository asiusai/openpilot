"""Source-catalog repair around the unchanged qualified numerical/replay/timing runner."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

EXP=Path(__file__).parent
ROOT=Path('/mnt/nvme-500/tmp/alpamayo-speed-20261116/agent-ck-source-work')
PROTOCOL=ROOT/'prepared-final/protocol.json'
OLD_PROTOCOL=Path('/mnt/nvme-500/tmp/alpamayo-speed-20261116/agent-ck-qualified-work/prepared-final/protocol.json')
OLD_RUN_SHA='52f55ea50702c01157a1e7051341c6122a53868a7921a76c27c7acc655d1c566'
OLD_PROTOCOL_SHA='d84e251bd733ea279953c63fedb0c25c1a7c335562af379473073914ee3f4b2e'


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def load(name):
  spec=importlib.util.spec_from_file_location(name.replace('-','_'),EXP/(name+'.py'))
  mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);return mod


def configured():
  assert sha(EXP/'agent-ck-qualified-run.py')==OLD_RUN_SHA and sha(OLD_PROTOCOL)==OLD_PROTOCOL_SHA
  original=load('agent-ck-qualified-run');original.ROOT=ROOT;original.PROTOCOL=PROTOCOL
  old_load=original.load;guard=load('agent-ck-source-guard')
  def routed(name):
    if name=='agent-ck-output-run': return guard
    module=old_load(name)
    if name=='agent-ck-attention-v3-run': module.record_programs=guard.record_programs
    return module
  original.load=routed
  return original


def verify():
  original=configured();plan,cases=original.verify();old=json.loads(OLD_PROTOCOL.read_text())
  assert not sys.flags.optimize
  # Every predeclared numerical, mutation, finite, poison, validator, replay, I/O, timeout, and timing field is identical.
  for name,value in old.items():
    if name not in ('programs','source_sha256'): assert plan[name]==value,('Frozen protocol field changed',name)
  assert all(plan['source_sha256'][p]==s for p,s in old['source_sha256'].items())
  assert set(plan['programs'])==set(old['programs'])
  identity=load('agent-ck-source-identity')
  for aid,item in plan['associations'].items():
    assert identity.key(item)==aid and item['actual_mode']=='CU'
    assert aid in plan['programs'][item['binary_sha256']]['associations']
    assert sha(PROTOCOL.parent/(aid+'.source'))==item['source_sha256']
    assert sha(PROTOCOL.parent/(item['binary_sha256']+'.hsaco'))==item['binary_sha256']
  for digest,item in old['programs'].items():
    associations=[plan['associations'][a] for a in plan['programs'][digest]['associations']]
    assert any(a['source_sha256']==item['source_sha256'] for a in associations)
    assert all(a['descriptor']==item['descriptor'] for a in associations)
  for bid,rows in plan['batches'].items():
    assert identity.key(rows)==bid and all(r['association'] in plan['associations'] for r in rows)
  proof=json.loads((PROTOCOL.parent/'cpu-preflight.json').read_text())
  assert proof['gpu_executed'] is False and proof['opened_devices']==[]
  assert proof['real_HCQ_command_and_kernarg_construction_checked']
  assert all(a==b for a,b in proof['bank_hashes'].items())
  assert proof['io']['host_write_bytes']==plan['budget']['fixture_upload_bytes']
  assert len(plan['associations'])==30 and len(plan['programs'])==28 and len(plan['batches'])==19
  return original,plan,cases


def main():
  parser=argparse.ArgumentParser();parser.add_argument('--parent-run',action='store_true')
  parser.add_argument('--output',type=Path,default=ROOT/'parent-result.json');args=parser.parse_args()
  original,plan,cases=verify()
  if not args.parent_run:
    print(json.dumps({'cpu_preflight':'passed','gpu_executed':False,'protocol':str(PROTOCOL),'protocol_sha256':sha(PROTOCOL),
      'unchanged_runner_sha256':OLD_RUN_SHA,'binary_count':28,'exact_association_count':30,'exact_batch_count':19,
      'budget':plan['budget']}));return
  assert args.output.parent.resolve()==ROOT.resolve() and not args.output.exists() and not (ROOT/'parent-programs').exists()
  with args.output.open('x') as stream: stream.write('{"status":"starting","gpu_executed":false}\n')
  os.environ.update(DEV='USB+AMD:LLVM',CPU='0',AMD='0',ALLOW_DEVICE_USAGE='1',PARALLEL='0',PROFILE='0',BEAM='0',JITBEAM='0',JIT='1',CAPTURING='1',
    LLVMOPT='1',GMMU='0',TC_OPT='2',DEBUG='0',TMPDIR=str(ROOT),CACHEDB=str(ROOT/'cpu-cache.db'),PYTHONDONTWRITEBYTECODE='1',
    OPENBLAS_NUM_THREADS='4',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4')
  sys.path.insert(0,plan['tinygrad']);original.run(plan,cases,args.output)


if __name__=='__main__': main()
