"""Fresh paths and protocol for the one-helper lifetime repair. No new kernel catalog."""
import argparse
import importlib.util
import json
from pathlib import Path
import shutil
import sqlite3

EXP=Path(__file__).parent
spec=importlib.util.spec_from_file_location('ck_lifetime_overlay',EXP/'agent-ck-lifetime-overlay.py')
O=importlib.util.module_from_spec(spec);spec.loader.exec_module(O)

def write(path,value):
  with path.open('x') as stream: stream.write(json.dumps(value,indent=2,allow_nan=False)+'\n')

def stage():
  assert all(O.sha(p)==s for p,s in O.PINS.items())
  O.ROOT.mkdir(exist_ok=False);O.PREPARED.mkdir();(O.ROOT/'tmp').mkdir()
  for name in ('catalog.json','producer-catalog.json'): shutil.copyfile(O.OLD_ROOT/'prepared'/name,O.PREPARED/name)
  write(O.PREPARED/'invocation.json',O.invocation())
  source=O.ROOT/'cache-source';source.mkdir()
  paths=[p for name in ('cache.db','cache.db-wal','cache.db-shm') if (p:=O.OLD_ROOT/name).exists()]
  before={str(p):O.sha(p) for p in paths}
  for path in paths: shutil.copyfile(path,source/path.name)
  assert before=={str(p):O.sha(p) for p in paths}
  with sqlite3.connect('file:'+str(source/'cache.db')+'?mode=ro',uri=True) as src,sqlite3.connect(O.ROOT/'cache.db') as dst: src.backup(dst)
  assert before=={str(p):O.sha(p) for p in paths}
  print(str(O.ROOT))

def freeze(proof):
  plan=O.plan();test=json.loads(proof.read_text())
  assert not test['gpu_executed'] and not test['native_GPU_arithmetic_executed'] and test['original_duplicate_module_failure_reproduced']
  assert len(test['runs'])==2 and test['runs'][0]['runner_reached_final_success'] and not test['runs'][1]['runner_reached_final_success']
  assert all(row['scope_restored'] and row['hooks_restored'] for row in test['runs'])
  for path,digest in {**plan['source_sha256'],**test['source_sha256']}.items(): assert O.sha(path)==digest
  paths=[*O.PINS,proof,*O.PREPARED.iterdir(),*[p for p in EXP.glob('agent-ck-lifetime-*.py') if p.name!='agent-ck-lifetime-parent.py']]
  plan['source_sha256'].update({str(p):O.sha(p) for p in paths if p.is_file()})
  plan['lifetime_overlay']={'repair':'One output helper per runner; its cached adapter owns both abi_scope and candidate attention.',
    'ACTIVE_bypassed':False,'scope_lifetime_changed':False,'native_source_binary_ABI_catalog_changed':False,
    'cpu_mock_proof':str(proof),'cpu_mock_proof_sha256':O.sha(proof),'old_protocol':str(O.OLD_PROTOCOL),
    'old_protocol_sha256':O.sha(O.OLD_PROTOCOL),'failed_report_sha256':O.PINS[O.WORK/'agent-ck-storage-parent/report.json']}
  write(O.PROTOCOL,plan);O.verify()
  print(json.dumps({'protocol':str(O.PROTOCOL),'sha256':O.sha(O.PROTOCOL)}))

if __name__=='__main__':
  parser=argparse.ArgumentParser();parser.add_argument('--stage',action='store_true');parser.add_argument('--proof',type=Path)
  args=parser.parse_args();assert args.stage!=bool(args.proof);stage() if args.stage else freeze(args.proof)
