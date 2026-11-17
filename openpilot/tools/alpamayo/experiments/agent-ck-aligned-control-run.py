"""Prepare or run one fresh candidate; reuse only the exact completed baseline."""
import argparse
import difflib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sqlite3
import sys

EXP=Path(__file__).parent
s=importlib.util.spec_from_file_location('ck_aligned_overlay',EXP/'agent-ck-aligned-control-overlay.py')
N=importlib.util.module_from_spec(s);s.loader.exec_module(N)

def stage():
  N.O.verify();N.PREPARED.mkdir(parents=True,exist_ok=False);(N.ROOT/'tmp').mkdir()
  for name in ('samples.json','inputs.json','ck-identity.json'): shutil.copyfile(N.O.PREPARED/name,N.PREPARED/name)
  for arm in ('baseline','candidate'):
    inv=json.loads((N.O.PREPARED/f'{arm}-invocation.json').read_text())
    inv['command'][inv['command'].index('--output')+1]=str(N.ROOT/arm/'pipeline.json')
    inv['environment'].update(CACHEDB=str(N.ROOT/f'{arm}-cache.db'),TMPDIR=str(N.ROOT/'tmp'))
    N.C.write(N.PREPARED/f'{arm}-invocation.json',inv)
  # Copy the frozen cache before opening SQLite. No connection touches the old database or its WAL/SHM.
  private=N.ROOT/'cache-source';private.mkdir();source=N.O.ROOT/'candidate-cache.db';pins={}
  for suffix in ('','-wal','-shm'):
    p=Path(str(source)+suffix)
    if p.exists():
      pins[str(p)]=N.C.sha(p);shutil.copyfile(p,private/('cache.db'+suffix));assert N.C.sha(p)==pins[str(p)]
  for name in ('candidate','cpu'):
    with sqlite3.connect(f'file:{private}/cache.db?mode=ro',uri=True) as src,sqlite3.connect(N.ROOT/f'{name}-cache.db') as dst: src.backup(dst)
  N.C.write(N.PREPARED/'cache-copy.json',pins)
  old=(EXP/'agent-ck-integrated-controls.py').read_text();new=N.controls_source()
  (N.PREPARED/'effective-controls.py').write_text(new)
  (N.PREPARED/'repair.diff').write_text(''.join(difflib.unified_diff(old.splitlines(True),new.splitlines(True),
    fromfile='frozen/agent-ck-integrated-controls.py',tofile='in-memory/agent-ck-integrated-controls.py')))
  (N.PREPARED/'effective-compare.py').write_text(N.comparator_source())
  N.C.write(N.PREPARED/'baseline-lineage.json',N.lineage())
  print(json.dumps({'staged':str(N.PREPARED),'gpu_executed':False}))

def freeze(proof):
  p=json.loads(proof.read_text());assert p['completed_mock_runner'] and not p['gpu_executed']
  assert p['aligned_control']['strict_USB_reader_used_for_complete_runner'] and p['aligned_control']['old_six_byte_read_rejected']
  paths=[*EXP.glob('agent-ck-aligned-control*.py'),*N.PREPARED.iterdir(),proof,
    N.C.TINY/'tinygrad/runtime/support/usb.py',N.BASELINE,N.FAILED,N.FAILED.with_name('run.log')]
  N.C.write(N.PREPARED/'protocol.json',{'scope':'candidate_diagnostic_neighbor_read_alignment_only',
    'timeout_seconds':600,'internal_timeout_seconds':580,'baseline_lineage':N.lineage(),
    'adoption_allowed':False,'gpu_executed_by_agent':False,
    'pins':{str(p):N.C.sha(p) for p in sorted(set(paths))}})
  print(json.dumps({'protocol_sha256':N.C.sha(N.PREPARED/'protocol.json')}))

def parent(output,verify_only):
  assert not sys.flags.optimize
  plan=N.C.verify();folder=N.ROOT/'candidate';original=folder/'report.json'
  assert output.parent.is_dir() and not output.exists() and output.resolve()!=original.resolve()
  assert not folder.exists(),'This candidate was already attempted'
  if verify_only:
    print(json.dumps({'verified':True,'gpu_executed':False,'baseline_reused_by_hash':N.BASELINE_SHA}));return
  folder.mkdir();N.R.configure('candidate');os.environ['ALLOW_DEVICE_USAGE']='1'
  try: N.R.run(plan,'candidate',folder)
  finally:
    N.C.verify()
    if original.exists():
      r=json.loads(original.read_text());r['baseline_lineage']=N.lineage()
      r['parent_bridge']={'original_report':str(original),'original_sha256':N.C.sha(original),'bridge_sha256':N.C.sha(__file__)}
      N.C.write(output,r)

def main():
  p=argparse.ArgumentParser();g=p.add_mutually_exclusive_group(required=True)
  g.add_argument('--stage',action='store_true');g.add_argument('--freeze',type=Path)
  g.add_argument('--parent-run',action='store_true');g.add_argument('--verify-only',action='store_true');g.add_argument('--compare',type=Path)
  p.add_argument('--output',type=Path);a=p.parse_args()
  if a.stage: stage()
  elif a.freeze: freeze(a.freeze)
  else:
    assert a.output is not None
    if a.compare:
      if not N.compare(a.compare,a.output): raise SystemExit(1)
    else: parent(a.output,a.verify_only)

if __name__=='__main__': main()
