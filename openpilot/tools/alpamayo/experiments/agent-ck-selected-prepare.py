"""Small direct manifest and new caches. No GPU or new source-tree snapshots."""
import argparse
import ast
import difflib
import importlib.util
from pathlib import Path
import shutil
import sqlite3

s=importlib.util.spec_from_file_location('ck_selected_common',Path(__file__).with_name('agent-ck-selected-common.py'))
S=importlib.util.module_from_spec(s);s.loader.exec_module(S)

def stage():
  assert S.sha(S.QUALITY)==S.QUALITY_SHA and S.sha(S.PACED)==S.PACED_SHA
  S.PREPARED.mkdir(parents=True,exist_ok=False)
  for n in ('samples.json','inputs.json','ck-identity.json'):shutil.copyfile(S.Q.PREPARED/n,S.PREPARED/n)
  template=S.read(S.T.PREPARED/'candidate-invocation.json')
  A,C,N=S.selected();N.reserve(S.ROOT)
  for arm in ('baseline','candidate'):
    inv=S.read(S.T.PREPARED/'candidate-invocation.json')
    inv['command']=[x.replace(str(S.Q.BASE),str(S.BASE)) for x in inv['command']]
    inv['command'][inv['command'].index('--output')+1]=str(S.ROOT/arm/'pipeline.json')
    inv['environment'].update(N.storage(),PYTHONPATH=f'{S.TINY}:{S.BASE}',PARALLEL='32')
    S.write(S.PREPARED/f'{arm}-invocation.json',inv)
  for key,value in N.storage().items():
    if key not in ('CACHEDB','ALPAMAYO_FP8_CACHE'):Path(value).mkdir(parents=True,exist_ok=True)
  source=S.T.ROOT/'candidate-cache.db';dest=Path(N.storage()['CACHEDB'])
  # Snapshot before opening SQLite. Never modify the frozen source DB or its journals.
  private=S.ROOT/'cache-input';private.mkdir()
  for suffix in ('','-wal','-shm'):
    p=Path(str(source)+suffix)
    if p.exists():shutil.copyfile(p,private/('cache.db'+suffix))
  with sqlite3.connect(f'file:{private}/cache.db?mode=ro',uri=True) as src,sqlite3.connect(dest) as dst:src.backup(dst)
  (S.PREPARED/'quality-hook.py').write_text(S.hook_source())
  (S.PREPARED/'live-hook.py').write_text(S.hook_source(True))
  (S.PREPARED/'quality-runner.py').write_text(S.runner_source())
  old=S.Q.BASE/'experiments/benchmark-pipeline.py';new=S.BASE/'experiments/benchmark-pipeline.py'
  (S.PREPARED/'factory.diff').write_text(''.join(difflib.unified_diff(old.read_text().splitlines(True),new.read_text().splitlines(True),fromfile=str(old),tofile=str(new))))
  equal=[]
  for rel in ('model.py','amd_attention.py','amd_gemm.py',*('experiments/'+n for n in ('benchmark-fast.py','expert-kernels.py','quantize-packed.py',
      'batched-vision.py','vision-cache.py','lookup-decode.py','draft-decode.py','block-decode.py','padded-prefill.py','agent-speculation-run.py',
      'agent-speculation-prefill.py','compact-inputs.py','agent-attention-rowsoftmax.py','wgp_compiler.py','coc_vocab.json'))):
    assert S.sha(S.Q.BASE/rel)==S.sha(S.BASE/rel),rel
    equal.append({'old':str(S.Q.BASE/rel),'current':str(S.BASE/rel),'sha256':S.sha(S.BASE/rel)})
  S.write(S.PREPARED/'source-equality.json',equal)
  print({'staged':str(S.PREPARED),'gpu_executed':False})

def freeze(proof):
  assert not (S.PREPARED/'protocol.json').exists(),'The selected protocol is immutable after freeze'
  cpu=S.read(proof);assert cpu['complete'] and not cpu['gpu_executed']
  A,C,N=S.selected()
  paths=[*S.EXP.glob('agent-ck-selected-*.py'),*S.PREPARED.iterdir(),proof,S.QUALITY,S.TIMING,S.PACED,
    S.Q.PREPARED/'protocol.json',S.T.PREPARED/'protocol.json',Path(S.read(S.QUALITY)['pipeline']),
    Path(S.read(S.QUALITY)['pipeline']).with_suffix('.speculation.json'),S.REVIEW,Path(S.read(S.REVIEW)['comparison']),
    C.PROTOCOL,N.MANIFEST,N.PV_PROTOCOL,C.HOOK,C.OLD/'agent-expert-live-storage-install.py',
    S.WORK/'agent-live-cpu-integrated-source/agent-live-cpu-integrated-install.py']
  paths += [S.EXP/n for n in ('agent-arrival-warp-ownership-install.py','agent-arrival-warp-integrated-install.py',
    'agent-arrival-warp-integrated-run.py','agent-exact-live-install.py','agent-exact-combined-install.py','agent-arrival-warp-cache.py')]
  paths += [S.BASE/'experiments/benchmark-pipeline.py',S.Q.BASE/'experiments/benchmark-pipeline.py']
  # Existing manifests bind implementation dependencies. Runtime installers keep their own exact source and PROGRAM checks.
  S.write(S.PREPARED/'protocol.json',{'schema':1,'files':{str(p):S.sha(p) for p in sorted(set(paths)) if p!=S.PREPARED/'protocol.json'},
    'cpu_proof':{'path':str(proof),'sha256':S.sha(proof),'complete':True,'gpu_executed':False},
    'quality_reference_sha256':S.QUALITY_SHA,'continuous_reference_sha256':S.PACED_SHA,'timeout_seconds':600,
    'automatic_paced':False,'vision_group_limit':2,'new_numerical_kernel':False,'default_adoption':False})
  print({'protocol_sha256':S.sha(S.PREPARED/'protocol.json')})

def refresh():
  assert not (S.PREPARED/'protocol.json').exists()
  for arm in ('baseline','candidate'):
    p=S.PREPARED/f'{arm}-invocation.json';inv=S.read(p);inv['environment']['PARALLEL']='32';S.write(p,inv)
  for name,text in (('quality-hook.py',S.hook_source()),('live-hook.py',S.hook_source(True)),('quality-runner.py',S.runner_source())):
    (S.PREPARED/name).write_text(text)

def proof(output):
  assert not output.exists()
  parts={name:S.ROOT/file for name,file in {'quality':'cpu-quality-2.json','composition':'cpu-composition-4.json',
    'paced':'cpu-paced-7.json','admission':'cpu-admission-2.json'}.items()}
  reports={k:S.read(p) for k,p in parts.items()}
  assert all(not r['gpu_executed'] for r in reports.values())
  assert reports['quality']['completed_mock_runner'] and all(r['prefills']==64 for r in reports['quality']['arms'])
  assert reports['composition']['complete'] and reports['composition']['ck_layers']==list(range(28))
  assert reports['paced']['complete'] and reports['paced']['actual']['all_actual_super_coverage_executed']
  assert reports['admission']['complete'] and reports['admission']['no_device_opened']
  assert reports['admission']['device_lookup_blocked']==['AMD']
  S.write(output,{'complete':True,'gpu_executed':False,'native_arithmetic_mocked':True,
    'parts':{k:{'path':str(p),'sha256':S.sha(p)} for k,p in parts.items()},
    'scope':'Complete64 CK TinyJit/ABI simulation, real selected numerical installer routing with fake tensors, actual factory/Consumer/arrival lifecycle with neural doubles, and strict parity rejection.'})

if __name__=='__main__':
  p=argparse.ArgumentParser();p.add_argument('--stage',action='store_true');p.add_argument('--freeze',type=Path);p.add_argument('--refresh',action='store_true');p.add_argument('--proof',type=Path);a=p.parse_args()
  assert sum(map(bool,(a.stage,a.freeze,a.refresh,a.proof)))==1
  if a.stage:stage()
  elif a.refresh:refresh()
  elif a.proof:proof(a.proof)
  else:freeze(a.freeze)
