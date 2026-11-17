"""Prepare the lean pair. It remains locked until a completed full64 review is supplied."""
import argparse
import importlib.util
import json
from pathlib import Path
import shutil
import sqlite3

EXP=Path(__file__).parent
s=importlib.util.spec_from_file_location('ck_timing_common',EXP/'agent-ck-quality64-timing-common.py')
T=importlib.util.module_from_spec(s);s.loader.exec_module(T)

def stage():
  T.Q.verify();T.PREPARED.mkdir(parents=True,exist_ok=False);(T.ROOT/'tmp').mkdir()
  for name in ('samples.json','inputs.json','ck-identity.json'): shutil.copyfile(T.Q.PREPARED/name,T.PREPARED/name)
  for arm in ('baseline','candidate'):
    inv=json.loads((T.Q.PREPARED/f'{arm}-invocation.json').read_text());inv['command'][inv['command'].index('--output')+1]=str(T.ROOT/arm/'pipeline.json')
    inv['environment'].update(CACHEDB=str(T.ROOT/f'{arm}-cache.db'),TMPDIR=str(T.ROOT/'tmp'))
    T.Q.write(T.PREPARED/f'{arm}-invocation.json',inv)
  private=T.ROOT/'cache-source';private.mkdir();source=T.Q.WORK/'agent-ck-aligned-control-work/candidate-cache.db';pins={}
  for suffix in ('','-wal','-shm'):
    p=Path(str(source)+suffix)
    if p.exists():
      pins[str(p)]=T.Q.sha(p);shutil.copyfile(p,private/('cache.db'+suffix));assert T.Q.sha(p)==pins[str(p)]
  for arm in ('baseline','candidate','cpu'):
    with sqlite3.connect(f'file:{private}/cache.db?mode=ro',uri=True) as src,sqlite3.connect(T.ROOT/f'{arm}-cache.db') as dst: src.backup(dst)
  T.Q.write(T.PREPARED/'cache-copy.json',pins)
  (T.PREPARED/'effective-hook.py').write_text(T.hook_source());(T.PREPARED/'effective-runner.py').write_text(T.runner_source())
  T.Q.write(T.PREPARED/'review-template.json',{'comparison':'FILL_AFTER_FULL64_REVIEW','comparison_sha256':'FILL',
    'baseline_report':'FILL','candidate_report':'FILL','reviewed_query_indices':[],
    'text_speed_slowing_turns_reviewed':False,'decision':'UNREVIEWED'})
  print(json.dumps({'staged':str(T.PREPARED),'timing_launch_allowed':False,'gpu_executed':False}))

def freeze(proof):
  r=json.loads(proof.read_text());assert r['completed_mock_runner'] and r['lean_checks']['no_captured_poison'] and not r['gpu_executed']
  paths=[*EXP.glob('agent-ck-quality64-timing-*.py'),*T.PREPARED.iterdir(),proof]
  T.Q.write(T.PREPARED/'protocol.json',{'quality_protocol_sha256':T.QUALITY_PROTOCOL_SHA,'timeout_seconds':600,
    'requires_completed_full64_review':True,'adoption_allowed':False,'audit_excluded_query_indices':list(T.AUDIT_INDICES),
    'warm_rule':'Common indices2..62 with zero new runtime or compilation entries in both arms; require at least16.',
    'arithmetic_gate':'Every query must match its same-arm quality64 xyz/rot bits, tokens, reasoning, and decoder history exactly.',
    'pins':{str(p):T.Q.sha(p) for p in sorted(set(paths))}})
  print(json.dumps({'timing_protocol_sha256':T.Q.sha(T.PREPARED/'protocol.json')}))

if __name__=='__main__':
  p=argparse.ArgumentParser();p.add_argument('--stage',action='store_true');p.add_argument('--freeze',type=Path);a=p.parse_args()
  assert bool(a.stage)!=bool(a.freeze)
  if a.stage: stage()
  else: freeze(a.freeze)
