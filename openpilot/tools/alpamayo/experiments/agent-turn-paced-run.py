"""One parent-only turn replay; selected numerical recipe and arrival ownership stay original."""
import argparse
from contextlib import contextmanager
import copy
from functools import wraps
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace
from unittest.mock import patch

HERE=Path(__file__).resolve().parent
WORK=Path('/mnt/nvme-500/tmp/alpamayo-speed-20261116')
EXP=Path('/tmp/opencode/openpilot-alpamayo/openpilot/tools/alpamayo/experiments')
ARTIFACTS=WORK/'agent-turn-paced-artifacts'
PROTOCOL=WORK/'agent-turn-paced-protocol.json'
FROZEN=WORK/'agent-turn-paced-source'
OUTPUT=WORK/'agent-turn-paced-111-119/report.json'
BASE_PROTOCOL=WORK/'agent-arrival-warp-ownership-protocol.json'
BASE_SHA='6a192909d866aa84437d2318422c3883feca26987278fa97737d6ef79457b756'
SELECTED=WORK/'agent-arrival-warp-ownership-paced1/report.json'
SELECTED_SHA='bbdfdb419f6a3b13dea4a347351233a3e7c0af61b94013850c8f4d8b0099cab2'
CACHE=WORK/'agent-turn-paced-cache-111-119'
INVENTORY_SHA='64092bf9590de322c5dda515a6f48579ff78cf4edc4aabce65c92f020b19c019'
DECLARATION=WORK/'agent-turn-paced-interval.json'

def load(path,name):
  spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec)
  sys.modules[name]=module;spec.loader.exec_module(module);return module
def read(path):return json.loads(Path(path).read_text())
def sha(path):
  with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def write(path,value):
  with Path(path).open('w') as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n')

def context():
  A=load(EXP/'agent-arrival-warp-ownership-common.py','turn_selected_ownership_common');C=A.C
  legacy=A.OLD.ARTIFACTS
  C.N.ARTIFACTS=ARTIFACTS;C.N.configure()  # Before tinygrad import; all new caches on 500GB.
  original_verify=A.verify
  def base_verify():
    previous=C.N.ARTIFACTS
    C.N.ARTIFACTS=legacy
    try:return original_verify(BASE_PROTOCOL,BASE_SHA)
    finally:C.N.ARTIFACTS=previous
  base_verify()
  assert sha(SELECTED)==SELECTED_SHA
  selected=read(SELECTED);assert selected['agent_arrival_warp_integrated']['passed']
  own=read(SELECTED.parent/'arrival-warp-ownership-status.json')
  assert own['complete'] and own['gpu_executed'] and own['report_sha256']==SELECTED_SHA
  assert own['ownership'][0]['finally_restored'] and own['ownership'][0]['all_original_coverage_retained']
  C.ARTIFACTS=C.S.ARTIFACTS=C.N.ARTIFACTS=ARTIFACTS
  def configure():
    C.configure();m,_=C.N.live();m.HERE=ARTIFACTS/'hooks'
  A.configure=configure;A.ARTIFACTS=ARTIFACTS;A.turn_base_verify=base_verify
  A.turn_calls=[];A.turn_histories=[];A.turn_cleanup=[]
  def verify(path,digest,*,cpu_only=False):
    assert sha(path)==digest;data=read(path);base_verify()
    assert data['base_protocol_sha256']==BASE_SHA and data['selected_report_sha256']==SELECTED_SHA
    assert data['window']==[111,119] and data['hz']==10 and data['timeout_seconds']==900
    assert data['cpu_only'] is cpu_only and not data['automatic_gpu']
    assert data['inventory_sha256']==INVENTORY_SHA==sha(CACHE/'paced-cache.json')
    for name,expected in data['files'].items():assert sha(name)==expected,name
    declaration=read(DECLARATION)
    assert declaration['window']==[111,119] and declaration['prime_window']==[107.7,111]
    if cpu_only:assert os.environ['DEV']=='CPU' and os.environ['ALLOW_DEVICE_USAGE']=='0'
    else:
      proof=read(data['cpu_proof']['path']);assert sha(data['cpu_proof']['path'])==data['cpu_proof']['sha256']
      assert proof['complete'] and not proof['gpu_executed'] and proof['devices']==[]
      assert proof['production']['all_actual_super_coverage_executed'] and proof['actual_turn_history_captured']
    return data
  A.verify=verify
  return A

def adapt_main(A,module):
  original=module.main
  @wraps(original)
  def main(argv=None):
    assert isinstance(argv,list);options=list(argv)
    wanted={'--mode':'run','--hz':'10','--start-seconds':'35','--seconds':'8','--inference-budget-seconds':'180','--warp':'native'}
    for flag,value in wanted.items():assert options.count(flag)==1 and options[options.index(flag)+1]==value
    assert options.count('--raw-frames')==1
    before=list(options);options[options.index('--start-seconds')+1]='111';options[options.index('--raw-frames')+1]=str(CACHE/'rgb')
    changed=[i for i,(a,b) in enumerate(zip(before,options,strict=True)) if a!=b]
    assert changed==[options.index('--start-seconds')+1,options.index('--raw-frames')+1]
    A.turn_calls.append(dict(original_arguments=before,actual_arguments=options,only_interval_and_rgb_path_changed=True))
    prepare=module.bridge.Preparer.prepare
    def observed(preparer,selection):
      arrays,detail=prepare(preparer,selection)
      import numpy as np
      start=time.perf_counter_ns()
      assert arrays['image_frames'].shape==(10,1,3,432,768) and arrays['image_frames'].dtype==np.uint8
      assert arrays['camera_indices'].tolist()==[1]*5+[6]*5
      capture=dict(query_ns=selection.query_ns,history_sample_ns=(selection.query_ns+np.arange(-19,1,dtype=np.int64)*100_000_000).tolist(),
        absolute_timestamps_us=arrays['absolute_timestamps'].tolist(),camera_indices=arrays['camera_indices'].tolist(),
        image_shape=list(arrays['image_frames'].shape),full_images=10,patches_per_image=720)
      for key in ('ego_history_xyz','ego_history_rot'):
        value=arrays[key];assert value.dtype==np.float32 and np.isfinite(value).all()
        raw=value.tobytes();capture[key]=dict(shape=list(value.shape),dtype=value.dtype.str,sha256=hashlib.sha256(raw).hexdigest(),float_bits_hex=raw.hex(),values=value.tolist())
      capture['audit_ms']=(time.perf_counter_ns()-start)/1e6
      assert len(A.turn_histories)<160;A.turn_histories.append(capture)
      detail['turn_model_ready_history']=capture
      return arrays,detail
    try:
      with patch.object(module.bridge.Preparer,'prepare',observed):return original(options)
    finally:
      A.turn_cleanup.append(module.bridge.Preparer.prepare is prepare)
      assert A.turn_cleanup[-1]
  module.main=main
  return module

def runtime(A):
  C=A.C;A.configure();m,_=C.N.live();original_load=m.load
  def importing(path,name):
    module=original_load(path,name)
    if Path(path)==m.ROOT/'experiments/agent-paced-replay.py':adapt_main(A,module)
    return module
  m.load=importing
  base_load=A.original_load
  def owned_load(path,name):
    module=base_load(path,name)
    if Path(path)==EXP/'agent-arrival-warp-integrated-run.py':
      def admission(command,**kw):
        assert command[1:3]==[str(EXP/'agent-arrival-warp-integrated-check.py'),'--admit']
        return subprocess.run([command[0],str(HERE/'agent-turn-paced-run.py'),*command[2:]],**kw)
      module.subprocess=SimpleNamespace(run=admission)
    return module
  A.original_load=owned_load
  R=base_load(EXP/'agent-arrival-warp-ownership-run.py','turn_original_ownership_runner');R.A=A
  return R

def breakdown(cases):
  import numpy as np
  def stat(values):return dict(median=float(np.median(values)),p95=float(np.percentile(values,95)),maximum=max(values))
  groups={}
  for key,predicate in [('nominal_accepted',lambda r:r['nominal_accepted'] is True),
      ('nominal_rejected',lambda r:r['nominal_eligible'] is True and r['nominal_accepted'] is False),
      ('nominal_ineligible',lambda r:r['nominal_eligible'] is False),('longer_than_six_tokens',lambda r:r['generated_tokens']>6)]:
    selected=[r for r in cases if predicate(r)];groups[key]=dict(count=len(selected),token_lengths=[r['generated_tokens'] for r in selected])
    if selected:groups[key].update(consumer_ms=stat([r['consumer_ms'] for r in selected]),road_sof_to_result_ms=stat([r['road_sof_to_result_ms'] for r in selected]))
  return groups

def summarize(output):
  import numpy as np
  report=read(output);rows=[]
  with output.with_suffix('.jsonl').open() as f:
    for line in f:
      row=json.loads(line)
      if row.get('kind')=='model_completion':rows.append(row)
  assert len(rows)==report['completed_queries'] and 0<len(rows)<=160
  def stat(values):return dict(median=float(np.median(values)),p95=float(np.percentile(values,95)),maximum=max(values))
  cases=[]
  for row in rows:
    spec=row['paced_consumer']['speculative_record'];prediction=row['prediction'];meta=row['input']
    history=meta['turn_model_ready_history'];assert history['query_ns']==meta['query_ns']
    road=next(c for c in row['timing']['camera'] if c['view']==6)
    cases.append(dict(query_ns=meta['query_ns'],route_offset_s=(meta['query_ns']-read(DECLARATION)['origin_ns'])/1e9,
      navigation=meta['navigation_text'],reasoning=prediction.get('reasoning'),generated_tokens=len(prediction['generated_token_ids']),
      nominal_eligible=spec.get('eligible'),nominal_accepted=spec.get('accepted'),speculative_record=spec,
      consumer_ms=row['timing']['consumer_and_completion_ms'],road_sof_to_result_ms=road['sof_to_output_ms'],
      history_sha256={k:history[k]['sha256'] for k in ('ego_history_xyz','ego_history_rot')}))
  groups=breakdown(cases)
  return dict(evaluation_only=True,route='0000008e--321b2ddc03',window=[111,119],canonical_report_sha256=sha(output),
    completed_predictions=len(rows),elapsed_with_last_completion_s=report['observation_with_final_completion_s'],
    actual_completion_hz=report['recorded_replay_completion_hz'],consumer_ms=stat([r['consumer_ms'] for r in cases]),
    road_sof_to_result_ms=stat([r['road_sof_to_result_ms'] for r in cases]),breakdown=groups,queries=cases,
    genuine_fallback_observed=groups['nominal_rejected']['count']>0,long_reasoning_observed=groups['longer_than_six_tokens']['count']>0,
    scene_limit='Original model already misses part of the recorded turn. This is cadence/reasoning evaluation, not a driving-quality approval.',
    receipt_limit='Recorded publication ordering and measured PC replay arrival; original subscriber receipt times were not logged.',
    extra_history_audit_ms=stat([r['input']['turn_model_ready_history']['audit_ms'] for r in rows]))

def run(args):
  A=context();R=runtime(A)
  assert args.cpu_preflight or args.output==OUTPUT
  side=args.output.parent/'agent-turn-paced-status.json';assert not side.exists()
  status=dict(complete=False,gpu_executed=False,cpu_only=args.cpu_preflight,protocol_sha256=args.protocol_sha,
    selected_report_sha256=SELECTED_SHA,model_recipe_unchanged=True,ssse3_packing_installed=False)
  try:
    R.run(args)
    report=read(args.output);assert report['source_start_ns']==read(DECLARATION)['source_start_ns'] and report['source_end_ns']==read(DECLARATION)['source_end_ns']
    assert len(A.turn_calls)==1 and A.turn_cleanup==[True]
    analysis=summarize(args.output);summary=args.output.parent/'turn-benchmark-summary.json';assert not summary.exists();write(summary,analysis)
    status.update(complete=True,gpu_executed=True,report_sha256=sha(args.output),summary_sha256=sha(summary))
  except BaseException as error:status['failure']=dict(type=type(error).__name__,message=str(error));raise
  finally:
    status.update(invocation_adapter=A.turn_calls,history_count=len(A.turn_histories),history_capture=A.turn_histories,
      preparer_restored=A.turn_cleanup,arrival_ownership=A.ownership)
    if args.output.parent.exists():write(side,status)

def files():
  paths=[BASE_PROTOCOL,SELECTED,SELECTED.parent/'arrival-warp-ownership-status.json',DECLARATION,CACHE/'paced-cache.json',CACHE/'trace-metadata.json']
  paths+=list(HERE.glob('agent-turn-paced-*.py'))
  return {str(p):sha(p) for p in paths}

def prepare():
  assert not PROTOCOL.exists() and not FROZEN.exists()
  A=context();A.configure();revision=hashlib.sha256(str(sorted(files().items())).encode()).hexdigest()[:12]
  proof=WORK/f'agent-turn-paced-cpu-{revision}.json';assert not proof.exists()
  data=dict(files=files(),base_protocol_sha256=BASE_SHA,selected_report_sha256=SELECTED_SHA,inventory_sha256=INVENTORY_SHA,
    window=[111,119],hz=10,timeout_seconds=900,cpu_only=True,automatic_gpu=False)
  provisional=ARTIFACTS/f'protocol-{revision}.json';write(provisional,data)
  env={**os.environ,**A.C.N.storage(),'DEV':'CPU','ALLOW_DEVICE_USAGE':'0','PYTHONDONTWRITEBYTECODE':'1','PYTHONPATH':str(A.C.N.TINY)}
  with proof.with_suffix('.log').open('x') as log:
    subprocess.run([A.C.N.PY,str(HERE/'agent-turn-paced-check.py'),'--protocol',str(provisional),'--output',str(proof)],
      env=env,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=300)
  result=read(proof);assert result['complete'] and not result['gpu_executed'] and result['production']['all_actual_super_coverage_executed']
  assert data['files']==files();FROZEN.mkdir()
  for p in HERE.glob('agent-turn-paced-*.py'):
    target=FROZEN/p.name;target.write_bytes(p.read_bytes());target.chmod(0o444);data['files'][str(target)]=sha(target)
  data.update(cpu_only=False,cpu_proof=dict(path=str(proof),sha256=sha(proof)));write(PROTOCOL,data);PROTOCOL.chmod(0o444)
  print(json.dumps(dict(ready=True,gpu_executed=False,protocol_sha256=sha(PROTOCOL),command=command(sha(PROTOCOL)))),flush=True)

def command(digest):
  A=context();result=[A.C.N.PY,str(A.C.N.WRAPPER),'agent-turn-paced-111-119','--program',str(FROZEN/'agent-turn-paced-run.py'),
    '--timeout-seconds','900','--set','protocol',str(PROTOCOL),'--set','protocol-sha',digest,
    '--env','PYTHONDONTWRITEBYTECODE','1','--env','PYTHONPATH',str(A.C.N.TINY)]
  for key,value in A.C.N.storage().items():result+=['--env',key,value]
  return result

def main():
  p=argparse.ArgumentParser(description=__doc__);p.add_argument('--protocol',type=Path);p.add_argument('--protocol-sha');p.add_argument('--output',type=Path)
  p.add_argument('--prepare',action='store_true');p.add_argument('--plan',action='store_true');p.add_argument('--admit',action='store_true');p.add_argument('--cpu-preflight',action='store_true')
  args=p.parse_args()
  if args.prepare:prepare()
  elif args.admit:
    A=context();A.configure();K=A.original_load(A.C.EXP/'agent-live-cpu-integrated-check.py','turn_cache_admission');K.C=A.C
    assert not args.output.exists();write(args.output,K.admit())
  elif args.plan:
    A=context();A.configure();A.verify(args.protocol,args.protocol_sha);print(json.dumps(command(args.protocol_sha)),flush=True)
  else:
    assert args.output is not None and args.protocol is not None and args.protocol_sha;run(args)

if __name__=='__main__':main()
