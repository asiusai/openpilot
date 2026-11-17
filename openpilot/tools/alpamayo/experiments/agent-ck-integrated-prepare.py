"""CPU-only preparation. All substantial artifacts stay on the 500 GB workspace."""
import argparse
import importlib.util
import json
from pathlib import Path
import shutil
import sqlite3

spec=importlib.util.spec_from_file_location('ck_integrated_common',Path(__file__).with_name('agent-ck-integrated-common.py'))
C=importlib.util.module_from_spec(spec);spec.loader.exec_module(C)

def stage():
  assert C.sha(C.LIVE)==C.LIVE_SHA and C.sha(C.MICRO)==C.MICRO_SHA
  C.PREPARED.mkdir(parents=True,exist_ok=True);assert not list(C.PREPARED.iterdir());(C.ROOT/'tmp').mkdir(exist_ok=True)
  original=C.WORK/'agent-ck-model-work/prepared-final'
  inputs=json.loads((original/'screen8-inputs.json').read_text())
  assert [x['original64_index'] for x in inputs]==C.INDICES
  manifest=C.WORK/'agent-expert-pv-route-source/inputs/5100176f7e48eaee-manifest.json'
  truth=json.loads(manifest.read_text());assert truth['future_times_s']==[(i+1)*.1 for i in range(64)]
  by_name={Path(r['sample']).name:r for r in truth['schedule']}
  for record in inputs:
    record['files']=record.pop('file_sha256')
    row=by_name[Path(record['sample']).name]
    assert row['query_ns']==int(Path(record['sample']).name.removeprefix('sample-'))*1000
    record.update(query_ns=row['query_ns'],recorded_motion_future_xyz=row['recorded_motion_future_xyz'])
  C.write(C.PREPARED/'inputs.json',inputs);C.write(C.PREPARED/'samples.json',[r['sample'] for r in inputs])
  cat=json.loads((original/'catalog.json').read_text())
  native=[x for x in cat['associations'].values() if x['name']=='alpamayo_ck_d128'];assert len(native)==1
  C.write(C.PREPARED/'ck-identity.json',native[0])
  # Copy the frozen SQLite files first. Never open the original database, even in read-only mode.
  cache_copy=C.ROOT/'cache-source';cache_copy.mkdir()
  source=C.WORK/'agent-ck-lifetime-work/cache.db';cache_pins={}
  for suffix in ('','-wal','-shm'):
    path=Path(str(source)+suffix)
    if path.exists():
      digest=C.sha(path);shutil.copyfile(path,cache_copy/path.name);assert C.sha(path)==digest
      cache_pins[str(path)]=digest
  for name in ('baseline','candidate','cpu'):
    with sqlite3.connect(f'file:{cache_copy}/cache.db?mode=ro',uri=True) as src,sqlite3.connect(C.ROOT/f'{name}-cache.db') as dst: src.backup(dst)
  C.write(C.PREPARED/'cache-copy.json',cache_pins)
  for arm in ('baseline','candidate'):
    inv=json.loads((original/'invocation.json').read_text())
    for flag,value in (('--runs','8'),('--output',str(C.ROOT/arm/'pipeline.json'))): inv['command'][inv['command'].index(flag)+1]=value
    inv['environment'].update(CACHEDB=str(C.ROOT/f'{arm}-cache.db'),TMPDIR=str(C.ROOT/'tmp'))
    C.write(C.PREPARED/f'{arm}-invocation.json',inv)
  print(json.dumps({'staged':str(C.PREPARED),'gpu_executed':False}))

def freeze(cpu_report):
  assert not (C.PREPARED/'protocol.json').exists()
  proof=json.loads(cpu_report.read_text());assert proof['completed_mock_runner'] and not proof['gpu_executed']
  paths=list(C.EXP.glob('agent-ck-integrated-*.py'))
  paths += [C.EXP/(n+'.py') for n in ('agent-ck-output-common','agent-ck-attention-v2-adapter','agent-ck-attention-v3-run',
    'agent-ck-qualified-common','agent-ck-model-hook','agent-ck-source-identity')]
  paths += [C.EXP/'agent-ck-attention-v2/native-overlay'/n for n in ('source.cpp','kernel.hsaco')]
  names=('agent-speculation-run','agent-speculation-prefill','agent-speculation-policy','benchmark-pipeline','benchmark-fast',
    'quantize-packed','agent-attention-rowsoftmax','online-attention','padded-prefill','lookup-decode','draft-decode',
    'block-decode','compact-inputs','fast-inputs','batched-vision','fused-norm','expert-kernels','wgp_compiler',
    'restore-precision','prefill-kernels','quantize-reduction','vision-cache','projector-kernels','bf16-linear','fast-quantize')
  paths += [C.BASE/'experiments'/(n+'.py') for n in names]
  paths += [C.BASE/'model.py',C.BASE/'benchmark.py',C.LIVE,C.MICRO,cpu_report,
    C.WORK/'agent-expert-pv-route-source/inputs/5100176f7e48eaee-manifest.json']
  paths += [C.TINY/'tinygrad'/name for name in ('tensor.py','device.py','engine/jit.py','engine/realize.py','runtime/ops_amd.py',
    'runtime/support/hcq2.py','runtime/support/compiler_amd.py','runtime/support/compiler_llvm.py','renderer/llvmir.py')]
  paths += list(C.PREPARED.glob('*.json'))
  inv=json.loads((C.PREPARED/'candidate-invocation.json').read_text())
  for flag in ('--prefill-calibration','--initial-noise','--reduced-vocab'): paths.append(Path(inv['command'][inv['command'].index(flag)+1]))
  cache=Path(inv['environment']['ALPAMAYO_FP8_CACHE'])
  assert C.sha(cache/'index.json')=='c4366c5b7e92b4cd5e2d826881b99b9a80361ec171ebd4a50d51b0f718dffa3e'
  assert C.sha(cache/'weights.bin')=='dd6b716a2ade3646eff71d874bef676926422d9aca39358fd3f8b4fe75755052'
  paths += [cache/'index.json']
  C.write(C.PREPARED/'protocol.json',{'scope':'opt_in_all28_fresh_paired_screen8','adoption_allowed':False,'gpu_executed_by_agent':False,
    'indices':C.INDICES,'timeout_seconds_per_arm':600,'internal_timeout_seconds':580,'pins':{str(p):C.sha(p) for p in sorted(set(paths))},
    'tinygrad_commit':'89846dcfb383a98ee01cfec3bba440ae7dcf9b1e','numerical_variant':True,
    'FP8_weights_sha256':'dd6b716a2ade3646eff71d874bef676926422d9aca39358fd3f8b4fe75755052',
    'accuracy_gate':{'aggregate_ADE_ratio_max':1.05,'aggregate_FDE_ratio_max':1.05,'each_of_four_windows_same_limits':True},
    'manual_review':['all eight texts and tokens','speed','slowing','turn shape'],
    'source_binding':'Direct fixed-shape source/AST/ELF/full USB target/CU/ABI identity; normal model compiler remains unchanged.',
    'boundary':'Existing full pipeline and unchanged input recipe; CK Q/K/V materialization, O/LSE, conversions, and audits remain inside it.',
    'timings_valid_for_throughput':False,'full64_automatically_authorized':False,
    'budget':'Existing 23.5s weight load and 16.4s cold vision; allow language capture/compilation, eight fresh samples, and five bounded replays. No seven-fixture rerun.'})
  print(json.dumps({'frozen_protocol':str(C.PREPARED/'protocol.json'),'sha256':C.sha(C.PREPARED/'protocol.json')}))

if __name__=='__main__':
  p=argparse.ArgumentParser();p.add_argument('--stage',action='store_true');p.add_argument('--freeze',type=Path);a=p.parse_args()
  assert bool(a.stage)!=bool(a.freeze)
  if a.stage: stage()
  else: freeze(a.freeze)
