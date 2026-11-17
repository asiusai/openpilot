"""Stage direct source/input pins and private caches. Never opens a GPU."""
import argparse
import importlib.util
import json
from pathlib import Path
import shutil
import sqlite3

EXP=Path(__file__).parent
s=importlib.util.spec_from_file_location('ck_quality64_common',EXP/'agent-ck-quality64-common.py')
C=importlib.util.module_from_spec(s);s.loader.exec_module(C)

def stage():
  assert C.sha(C.CONTROL)==C.CONTROL_SHA and C.sha(C.SCREEN)==C.SCREEN_SHA
  C.PREPARED.mkdir(parents=True,exist_ok=False);(C.ROOT/'tmp').mkdir()
  cmd=json.loads(C.SOURCE64.read_text())['command'];samples=cmd[cmd.index('--samples')+1:cmd.index('--initial-noise')]
  assert len(samples)==len(set(samples))==64
  stamps=[int(Path(x).name.removeprefix('sample-')) for x in samples];assert all(a<b for a,b in zip(stamps,stamps[1:]))
  old=C.WORK/'agent-ck-aligned-control-work/prepared'
  assert [samples[i] for i in range(0,64,8)]==json.loads((old/'samples.json').read_text())
  manifest=json.loads(C.TRUTH.read_text());byname={Path(r['sample']).name:r for r in manifest['schedule']}
  inputs=[]
  for i,sample in enumerate(samples):
    row=byname[Path(sample).name];assert row['query_ns']==stamps[i]*1000
    files={str(p):C.sha(p) for p in sorted(Path(sample).glob('*.npy'))}
    assert {Path(p).name for p in files}=={'absolute_timestamps.npy','camera_indices.npy','ego_history_rot.npy','ego_history_xyz.npy','image_frames.npy','nav_text.npy'}
    inputs.append({'original64_index':i,'sample':sample,'query_ns':row['query_ns'],'files':files,
      'recorded_motion_future_xyz':row['recorded_motion_future_xyz']})
  C.write(C.PREPARED/'samples.json',samples);C.write(C.PREPARED/'inputs.json',inputs)
  shutil.copyfile(old/'ck-identity.json',C.PREPARED/'ck-identity.json')
  template=json.loads((old/'candidate-invocation.json').read_text())
  for arm in ('baseline','candidate'):
    inv=json.loads(json.dumps(template));cmd=inv['command'];cmd[cmd.index('--runs')+1]='64'
    cmd[cmd.index('--output')+1]=str(C.ROOT/arm/'pipeline.json')
    inv['environment'].update(CACHEDB=str(C.ROOT/f'{arm}-cache.db'),TMPDIR=str(C.ROOT/'tmp'))
    C.write(C.PREPARED/f'{arm}-invocation.json',inv)
  source=C.WORK/'agent-ck-aligned-control-work/candidate-cache.db';private=C.ROOT/'cache-source';private.mkdir();pins={}
  for suffix in ('','-wal','-shm'):
    p=Path(str(source)+suffix)
    if p.exists():
      pins[str(p)]=C.sha(p);shutil.copyfile(p,private/('cache.db'+suffix));assert C.sha(p)==pins[str(p)]
  for arm in ('baseline','candidate','cpu'):
    with sqlite3.connect(f'file:{private}/cache.db?mode=ro',uri=True) as src,sqlite3.connect(C.ROOT/f'{arm}-cache.db') as dst: src.backup(dst)
  C.write(C.PREPARED/'cache-copy.json',pins)
  (C.PREPARED/'effective-hook.py').write_text(C.hook_source());(C.PREPARED/'effective-runner.py').write_text(C.runner_source())
  # Observed service times exclude queued-input age. Retain a hard bound; this is an estimate, not a completed full64 run.
  reports=[json.loads((C.WORK/p).read_text()) for p in ('agent-ck-graph-inspection-parent-baseline-v1/report.json','agent-ck-aligned-control-parent-v1/report.json')]
  budget=[]
  for r in reports:
    pipe=json.loads(Path(r['pipeline']).read_text());warm=max(x['service_ms'] for x in pipe['runs'][5:])/1000
    budget.append({'arm':r['arm'],'observed_eight_wall_s':r['elapsed_seconds'],'observed_warm_max_service_s':warm,
      'conservative_projection_s':r['elapsed_seconds']+56*max(1,warm)*3+120,
      'assumption':'Retain the entire observed eight-query wall time, add 3 seconds for each additional query and 120 seconds for unseen compilation.'})
  assert all(r['conservative_projection_s']<580 for r in budget)
  C.write(C.PREPARED/'budget.json',{'rows':budget,'actual64_duration_verified':False,'outer_limit_s':600,'internal_limit_s':580})
  print(json.dumps({'staged':str(C.PREPARED),'gpu_executed':False}))

def freeze(proof):
  r=json.loads(proof.read_text());assert r['completed_mock_runner'] and not r['gpu_executed']
  assert all(x['prefills']==64 for x in r['arms'])
  original=json.loads((C.OLD.PREPARED/'protocol.json').read_text())
  paths=[]
  # Reuse direct production sources, not prior protocol/verifier chains or old failed test inventories.
  for name,digest in original['pins'].items():
    p=Path(name)
    if p.is_relative_to(C.BASE) or p.is_relative_to(C.TINY):
      assert C.sha(p)==digest;paths.append(p)
  helpers=('agent-ck-integrated-common','agent-ck-integrated-run','agent-ck-integrated-compare','agent-ck-integrated-check',
    'agent-ck-graph-inspection','agent-ck-model-hook','agent-ck-output-common','agent-ck-attention-v2-adapter',
    'agent-ck-attention-v3-run','agent-ck-qualified-common','agent-ck-source-identity','agent-ck-aligned-control-check')
  paths += [EXP/(n+'.py') for n in helpers]
  paths += [EXP/'agent-ck-attention-v2/native-overlay'/n for n in ('source.cpp','kernel.hsaco')]
  paths += [p for p in EXP.glob('agent-ck-quality64-*.py') if '-timing-' not in p.name]
  paths += [*C.PREPARED.iterdir(),C.HOOK,C.SOURCE64,C.TRUTH,C.CONTROL,C.SCREEN,C.LIVE,C.MICRO,proof,C.ROOT/'metrics-check.json']
  paths += [C.TINY/'tinygrad/runtime/support/usb.py',C.TINY/'tinygrad/uop/ops.py']
  inv=json.loads((C.PREPARED/'candidate-invocation.json').read_text())
  paths += [Path(inv['command'][inv['command'].index(flag)+1]) for flag in ('--initial-noise','--prefill-calibration','--reduced-vocab')]
  paths += [Path(inv['environment']['ALPAMAYO_FP8_CACHE'])/'index.json']
  C.write(C.PREPARED/'protocol.json',{'indices':C.INDICES,'timeout_seconds':600,'internal_timeout_seconds':580,'adoption_allowed':False,
    'tinygrad_commit':'89846dcfb383a98ee01cfec3bba440ae7dcf9b1e','fresh_history_per_arm':True,'timings_valid_for_throughput':False,
    'accuracy_gate':{'aggregate_ADE_ratio':1.05,'aggregate_FDE_ratio':1.05,'each_original_16_query_window_same_limits':True},
    'all28_audits':'Every query, including initial and capture. Candidate reads 1792 flag bytes per query; baseline reads 112.',
    'reused_controls_report_sha256':C.CONTROL_SHA,'pins':{str(p):C.sha(p) for p in sorted(set(paths))}})
  print(json.dumps({'protocol_sha256':C.sha(C.PREPARED/'protocol.json')}))

if __name__=='__main__':
  p=argparse.ArgumentParser();p.add_argument('--stage',action='store_true');p.add_argument('--freeze',type=Path);a=p.parse_args()
  assert bool(a.stage)!=bool(a.freeze)
  if a.stage: stage()
  else: freeze(a.freeze)
