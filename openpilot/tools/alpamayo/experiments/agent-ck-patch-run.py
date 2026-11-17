"""Parent entry for the transport repair. All original numerical and timing gates remain present."""
import argparse
import ast
import hashlib
import importlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

EXP=Path(__file__).parent
ROOT=Path('/mnt/nvme-500/tmp/alpamayo-speed-20261116/agent-ck-patch-work')
PROTOCOL=ROOT/'prepared-final/protocol.json'
OLD_PROTOCOL=Path('/mnt/nvme-500/tmp/alpamayo-speed-20261116/agent-ck-source-work/prepared-final/protocol.json')
OLD_RUN=EXP/'agent-ck-qualified-run.py'
OLD_RUN_SHA='52f55ea50702c01157a1e7051341c6122a53868a7921a76c27c7acc655d1c566'
OLD_PROTOCOL_SHA='dac1321b740f2787c33f79a4616ab6466dc6efbe2082cb7cddc33336116954c2'

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def replacement():
  old=OLD_RUN.read_text()
  before="""      b=tensor.uop.buffer
      assert offset>=0 and offset+len(data)<=b.nbytes and offset%4==len(data)%4==0 and b.host is not None
      b.host.view(offset=offset,size=len(data),fmt='B')[:]=data
      report['io']['host_write_bytes']+=len(data);report['io']['host_write_calls']+=1"""
  after="""      __import__('importlib').import_module('agent-ck-patch-transport').write(tensor,data,offset,device,report)"""
  marker="      report['gpu_executed']=True\n"
  insertion="      __import__('importlib').import_module('agent-ck-patch-transport').preflight(device,plan,report)\n"
  final_marker="        report['validator_qualified']=True\n"
  final_insertion="        __import__('importlib').import_module('agent-ck-patch-transport').finalize(report,plan)\n"
  assert old.count(before)==old.count(marker)==old.count(final_marker)==1
  return old.replace(before,after).replace(marker,marker+insertion).replace(final_marker,final_insertion+final_marker)


def configured():
  assert sha(OLD_RUN)==OLD_RUN_SHA and sha(OLD_PROTOCOL)==OLD_PROTOCOL_SHA
  source=PROTOCOL.parent/'qualified-transport-run.py'
  assert source.read_text()==replacement(), 'Only patch transport, early preflight, and transport accounting checks may change'
  spec=importlib.util.spec_from_file_location('ck_patch_qualified',source)
  module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
  module.EXP=EXP;module.ROOT=ROOT;module.PROTOCOL=PROTOCOL
  old_load=module.load
  sys.path.insert(0,str(EXP));guard=importlib.import_module('agent-ck-patch-guard')
  def routed(name):
    if name=='agent-ck-output-run': return guard
    result=old_load(name)
    if name=='agent-ck-attention-v3-run': result.record_programs=guard.record_programs
    return result
  module.load=routed
  return module


def verify():
  assert not sys.flags.optimize
  module=configured();plan,cases=module.verify();old=json.loads(OLD_PROTOCOL.read_text())
  for name,value in old.items():
    if name!='source_sha256': assert plan[name]==value,('Original protocol field changed',name)
  assert all(plan['source_sha256'][p]==h for p,h in old['source_sha256'].items())
  assert plan['patch_output_root']==str(ROOT)
  assert len(plan['copy_calls'])==1 and len(plan['copy_host_programs'])==8 and len(plan['copy_destinations'])==60
  identity=importlib.import_module('agent-ck-patch-identity')
  for pid,item in plan['copy_host_programs'].items():
    assert identity.key(item)==pid
    assert sha(PROTOCOL.parent/(pid+'.source'))==item['source_sha256']
    assert sha(PROTOCOL.parent/(pid+'.elf'))==item['binary_sha256']
    assert item['target']=={'device':'CPU','renderer':'LLVM','arch':'x86_64,native','interface':'','indices':''}
  for cid,item in plan['copy_calls'].items(): assert identity.key(item)==cid
  proof=json.loads((PROTOCOL.parent/'cpu-proof.json').read_text())
  assert proof['strict_runtime_guard_exercised'] and proof['device_copy_count']==120 and not proof['gpu_executed']
  assert proof['opened_devices']==[] and all(x['full_byte_match'] for x in proof['cases'])
  assert all(x['rejected'] and x['all_target_bytes_unchanged'] for x in proof['negative_controls'])
  assert proof['patch_transport_preflight']['completed'] and not proof['patch_transport_preflight']['copy_executed']
  prior=json.loads(Path(plan['patch_repair']['prior_report']).read_text())
  assert len(prior['full_actual_shape_checks'])==7 and prior['causal_gqa_mutations_passed']
  assert all(x['complete'] for x in prior['full_actual_shape_checks'])
  assert not prior['completed'] and prior['failure']['phase']=='validator_all_bit_negative_controls_A_normalized_candidate'
  return module,plan,cases


def main():
  parser=argparse.ArgumentParser();parser.add_argument('--parent-run',action='store_true')
  parser.add_argument('--output',type=Path,default=ROOT/'parent-result.json');args=parser.parse_args()
  module,plan,cases=verify()
  if not args.parent_run:
    print(json.dumps({'cpu_preflight':'passed','gpu_executed':False,'protocol':str(PROTOCOL),'protocol_sha256':sha(PROTOCOL),
      'numerical_fixtures_rerun':7,'ordered_mutable_replays':40,'copy_host_programs':8,'bounded_copy_ranges':60,
      'original_gpu_binaries':len(plan['programs']),'timeout_seconds':300}));return
  assert args.output.parent.resolve()==ROOT.resolve() and not args.output.exists() and not (ROOT/'parent-programs').exists()
  with args.output.open('x') as stream: stream.write('{"status":"starting","gpu_executed":false}\n')
  os.environ.update(DEV='USB+AMD:LLVM',CPU='0',AMD='0',ALLOW_DEVICE_USAGE='1',PARALLEL='0',PROFILE='0',BEAM='0',JITBEAM='0',JIT='1',CAPTURING='1',
    LLVMOPT='1',GMMU='0',TC_OPT='2',DEBUG='0',TMPDIR=str(ROOT),CACHEDB=str(ROOT/'cpu-cache.db'),PYTHONDONTWRITEBYTECODE='1',
    OPENBLAS_NUM_THREADS='4',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4')
  sys.path.insert(0,plan['tinygrad']);module.run(plan,cases,args.output)


if __name__=='__main__': main()
