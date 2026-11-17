"""Create only fresh storage-overlay artifacts and a source-bound retry protocol."""
import argparse
import importlib.util
import json
from pathlib import Path
import shutil
import sqlite3

EXP=Path(__file__).parent
spec=importlib.util.spec_from_file_location('ck_storage_overlay',EXP/'agent-ck-storage-overlay.py')
O=importlib.util.module_from_spec(spec);spec.loader.exec_module(O)

def write(path,value):
  with path.open('x') as stream: stream.write(json.dumps(value,indent=2,allow_nan=False)+'\n')

def stage():
  assert all(O.sha(p)==s for p,s in O.PINS.items())
  O.ROOT.mkdir(exist_ok=False);O.PREPARED.mkdir();(O.ROOT/'tmp').mkdir()
  for name in ('catalog.json','producer-catalog.json'):
    with (O.PREPARED/name).open('xb') as stream: stream.write((O.ORIGINAL_ROOT/'prepared-final'/name).read_bytes())
  write(O.PREPARED/'invocation.json',O.derived_invocation())
  old_cache=O.ORIGINAL_ROOT/'cache.db'
  snapshot=O.ROOT/'cache-source';snapshot.mkdir()
  cache_files=[p for p in (old_cache,old_cache.with_name('cache.db-wal'),old_cache.with_name('cache.db-shm')) if p.exists()]
  before={str(p):O.sha(p) for p in cache_files}
  for path in cache_files: shutil.copyfile(path,snapshot/path.name)
  assert before=={str(p):O.sha(p) for p in cache_files}, 'Cache changed during source copy'
  # SQLite touches only the new private copies, including its WAL/shared-memory files.
  with sqlite3.connect('file:'+str(snapshot/'cache.db')+'?mode=ro',uri=True) as source,sqlite3.connect(O.ROOT/'cache.db') as dest:
    source.backup(dest)
  assert before=={str(p):O.sha(p) for p in cache_files}
  write(O.PREPARED/'cache-snapshot.json',{'source':str(old_cache),'original_files_sha256':before,
    'SQLite_opened_original_files':False,'destination':str(O.ROOT/'cache.db'),'snapshot_bytes':(O.ROOT/'cache.db').stat().st_size,
    'runtime_cache_is_private_and_mutable':True,'cache_is_not_numerical_attestation':True})
  print(str(O.ROOT))

def freeze(proof):
  assert all(O.sha(p)==s for p,s in O.PINS.items())
  plan=O.derived_plan()
  for path,digest in plan['source_sha256'].items(): assert O.sha(path)==digest, f'Changed frozen source: {path}'
  evidence=json.loads(proof.read_text())
  assert evidence['gpu_executed'] is False and evidence['recorded_failure_reproduced']
  assert evidence['recorded_real_storage_rows_pass_corrected_gate'] and len(evidence['positive_lifecycles'])==8
  assert len(evidence['negative_controls'])==14 and evidence['frozen_runner_runpy_and_overlay_binding']
  for path,digest in evidence['source_sha256'].items(): assert O.sha(path)==digest
  paths=[*O.PINS,proof,*O.PREPARED.iterdir(),proof.parent/(proof.stem+'-actual-lifecycle.py'),
    *[p for p in EXP.glob('agent-ck-storage-*.py') if p.name!='agent-ck-storage-parent.py']]
  plan['source_sha256'].update({str(p):O.sha(p) for p in paths if p.is_file()})
  plan['storage_overlay']={'gate_delta':O.check_delta(),'original_protocol':str(O.ORIGINAL_PROTOCOL),
    'original_protocol_sha256':O.sha(O.ORIGINAL_PROTOCOL),'frozen_runner_loaded_via_runpy':str(O.ORIGINAL_RUNNER),
    'frozen_runner_sha256':O.sha(O.ORIGINAL_RUNNER),'overlay_source':str(EXP/'agent-ck-storage-overlay.py'),
    'overlay_sha256':O.sha(EXP/'agent-ck-storage-overlay.py'),'cpu_proof':str(proof),'cpu_proof_sha256':O.sha(proof),
    'failed_report':str(O.WORK/'agent-ck-model-parent/report.json'),'failed_report_sha256':O.PINS[O.WORK/'agent-ck-model-parent/report.json'],
    'numeric_source_binary_ABI_dependency_catalog_changes':False,
    'path_changes_only':['output root','invocation path','private cache','temporary directory']}
  write(O.PROTOCOL,plan);O.verify()
  print(json.dumps({'protocol':str(O.PROTOCOL),'sha256':O.sha(O.PROTOCOL),'bound_files':len(plan['source_sha256'])}))

if __name__=='__main__':
  parser=argparse.ArgumentParser();parser.add_argument('--stage',action='store_true');parser.add_argument('--proof',type=Path)
  args=parser.parse_args();assert args.stage != bool(args.proof)
  stage() if args.stage else freeze(args.proof)
