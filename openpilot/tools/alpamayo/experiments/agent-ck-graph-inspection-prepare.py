"""Copy only private caches and invocation paths. Reuse the original source verification."""
import argparse
import difflib
import importlib.util
import json
from pathlib import Path
import shutil
import sqlite3

EXP=Path(__file__).parent
s=importlib.util.spec_from_file_location('ck_graph_overlay',EXP/'agent-ck-graph-inspection-overlay.py')
O=importlib.util.module_from_spec(s);s.loader.exec_module(O)

def stage():
  O.OLD.verify()
  O.PREPARED.mkdir(parents=True,exist_ok=False);(O.ROOT/'tmp').mkdir()
  for name in ('samples.json','inputs.json','ck-identity.json'):
    shutil.copyfile(O.OLD.PREPARED/name,O.PREPARED/name)
  for arm in ('baseline','candidate'):
    value=json.loads((O.OLD.PREPARED/f'{arm}-invocation.json').read_text())
    value['command'][value['command'].index('--output')+1]=str(O.ROOT/arm/'pipeline.json')
    value['environment'].update(CACHEDB=str(O.ROOT/f'{arm}-cache.db'),TMPDIR=str(O.ROOT/'tmp'))
    O.C.write(O.PREPARED/f'{arm}-invocation.json',value)
  # Copy first, including WAL. Opening SQLite may touch SHM even with mode=ro.
  private=O.ROOT/'cache-source';private.mkdir()
  pins={};source=O.OLD.ROOT/'baseline-cache.db'
  for suffix in ('','-wal','-shm'):
    path=Path(str(source)+suffix)
    if path.exists():
      pins[str(path)]=O.C.sha(path);shutil.copyfile(path,private/('cache.db'+suffix))
      assert pins[str(path)]==O.C.sha(path)
  for arm in ('baseline','candidate','cpu'):
    with sqlite3.connect(f'file:{private}/cache.db?mode=ro',uri=True) as src,sqlite3.connect(O.ROOT/f'{arm}-cache.db') as dst: src.backup(dst)
  O.C.write(O.PREPARED/'cache-copy.json',pins)
  old=(EXP/'agent-ck-integrated-hook.py').read_text();new=O.hook_source()
  (O.PREPARED/'effective-hook.py').write_text(new)
  (O.PREPARED/'repair.diff').write_text(''.join(difflib.unified_diff(old.splitlines(True),new.splitlines(True),
    fromfile='frozen/agent-ck-integrated-hook.py',tofile='in-memory/agent-ck-integrated-hook.py')))
  print(json.dumps({'staged':str(O.PREPARED),'gpu_executed':False}))

def freeze(proof):
  row=json.loads(proof.read_text());assert row['completed_mock_runner'] and not row['gpu_executed']
  assert row['graph_shape_checks']['old_failure_reproduced'] and row['graph_shape_checks']['both_second_calls_checked']
  paths=list(EXP.glob('agent-ck-graph-inspection*.py'))+[proof,*O.PREPARED.iterdir()]
  paths += [O.OLD.WORK/'agent-ck-integrated-parent-baseline'/x for x in ('run.log','report.json')]
  paths += [O.OLD.TINY/'tinygrad/uop/ops.py']
  O.C.write(O.PREPARED/'protocol.json',{'previous_protocol_sha256':O.OLD_PROTOCOL_SHA,
    'scope':'only_HCQ_graph_inspection_retry','timeout_seconds_per_arm':600,'internal_timeout_seconds':580,
    'indices':O.C.INDICES,'adoption_allowed':False,'gpu_executed_by_agent':False,
    'effective_hook_sha256':O.C.sha(O.PREPARED/'effective-hook.py'),
    'unchanged':'Original runner, all28 audits, numerical/storage/ABI gates, five replays, GQA, poison, tails, screen8 and comparison.',
    'pins':{str(p):O.C.sha(p) for p in sorted(set(paths))}})
  print(json.dumps({'protocol':str(O.PREPARED/'protocol.json'),'sha256':O.C.sha(O.PREPARED/'protocol.json')}))

if __name__=='__main__':
  p=argparse.ArgumentParser();p.add_argument('--stage',action='store_true');p.add_argument('--freeze',type=Path);a=p.parse_args()
  assert bool(a.stage)!=bool(a.freeze)
  if a.stage: stage()
  else: freeze(a.freeze)
