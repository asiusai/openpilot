"""Shape-derived coverage and independent admission of observed reference outputs."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import re
from types import MethodType,SimpleNamespace
import numpy as np

HERE=Path(__file__).resolve().parent
WORK=Path('/mnt/nvme-500/tmp/alpamayo-speed-20261116')
OLD=HERE/'agent-adaptive-prefill-run.py'
OLD_SHA='fcac561c51107f86fd7e7612edf36038cea18ed24c60da86330cf85a58255bc5'
PLAN=WORK/'agent-adaptive-prefill-plan.json'
PLAN_SHA='219f9c84cc9d28093658794cf4415e4a26e1a407d7ebcfe9e87db505544438af'
REFERENCE=WORK/'agent-adaptive-prefill-quality-reference/report.json'
REFERENCE_SHA='fbda536c6c9a541f2f27d0dc790d5628be3930b8c61356b6d0c02ba83a91f0b5'
FAILED_STATUS_SHA='ec9c1e900ee158bb73b679d0d40b0f57d4b63e3b702c884b1f51ee7a5a85b059'
RECEIPT=WORK/'agent-adaptive-coverage-reference-receipt.json'
PAIR=((1,720,768),(1,720,1152))


def sha(path):
  with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def read(path):return json.loads(Path(path).read_text())


def write(path,value,new=False):
  path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
  with path.open('x' if new else 'w') as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n')


def load(path,name):
  spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def frozen():
  assert sha(OLD)==OLD_SHA and sha(PLAN)==PLAN_SHA
  R=load(OLD,'adaptive_coverage_original_runner');plan=R.verify_plan()
  N,C,data,evidence=R.base()
  return R,plan,N,C,evidence


def exact(a,b,path='root'):
  if type(a) is not type(b):raise AssertionError('Type differs: '+path)
  if isinstance(a,dict):
    assert a.keys()==b.keys(),path
    for k in a:exact(a[k],b[k],path+'/'+str(k))
  elif isinstance(a,list):
    assert len(a)==len(b),path
    for i,(x,y) in enumerate(zip(a,b,strict=True)):exact(x,y,path+'/'+str(i))
  elif isinstance(a,float):
    assert np.isfinite(a) and np.isfinite(b) and np.float64(a).tobytes()==np.float64(b).tobytes(),path
  else:assert a==b,path


def clean_prediction(p):return {k:v for k,v in p.items() if k not in ('timings_ms','stage_timing_kind')}


def snapshot_files(report):
  root=Path(report['source_snapshot']);expected=report['source_sha256']
  assert {p.name for p in root.iterdir()}==set(expected)
  for name,digest in expected.items():assert sha(root/name)==digest,name
  return {'files':len(expected),'sha256':hashlib.sha256(json.dumps(expected,sort_keys=True).encode()).hexdigest()}


def reference_receipt(C,write_result=False):
  assert sha(REFERENCE)==REFERENCE_SHA and sha(REFERENCE.parent/'adaptive-prefill-status.json')==FAILED_STATUS_SHA
  failed=read(REFERENCE.parent/'adaptive-prefill-status.json');report=read(REFERENCE)
  assert failed['complete'] is False and failed['gpu_executed'] is True
  assert failed['failure']=={'type':'AssertionError','message':''}
  assert failed['import_hook_restored'] and failed['adaptive_hook_restored']
  assert 'coverage' not in failed and 'pv' not in failed
  log=(REFERENCE.parent/'run.log').read_text()
  assert 'Iteration 17:' in log and 'agent-expert-combined-install.py", line 77, in coverage' in log
  assert log.rstrip().endswith('AssertionError')
  rows=report['warmup']+report['runs'];plan=read(PLAN)
  assert len(report['warmup'])==2 and len(report['runs'])==16 and [r['sample'] for r in rows]==plan['samples']
  source_identity=snapshot_files(report)
  speculation=read(REFERENCE.with_suffix('.speculation.json'))
  assert len(speculation['records'])==18
  comparisons=[]
  for title in ('agent-exact-combined-quality-candidate','agent-expert-pv-qualified-quality-candidate','agent-expert-pv-qualified-quality-reference'):
    path=WORK/title/'report.json';assert sha(path)==C.PRIOR[title]
    old=read(path);old_rows=old['warmup']+old['runs'];old_spec=read(path.with_suffix('.speculation.json'))
    assert len(old_rows)==len(old_spec['records'])==64
    for k in ('initial_noise_sha256','prefill_calibration_sha256','reduced_vocab','pad_prompt_to',
              'trajectory_solver','pixel_scale','history_pixel_scale','history_feature_scale','vision_calibration_sha256'):
      exact(report[k],old[k],title+'/'+k)
    if title=='agent-exact-combined-quality-candidate':exact(report['settings'],old['settings'],title+'/settings')
    for index,(actual,prior,record,old_record) in enumerate(zip(rows,old_rows[14:32],speculation['records'],old_spec['records'][14:32],strict=True)):
      assert actual['sample']==prior['sample'] and actual['input_latest_timestamp_us']==prior['input_latest_timestamp_us']
      exact(clean_prediction(actual['prediction']),clean_prediction(prior['prediction']),title+'/prediction/'+str(index))
      exact(record,old_record,title+'/target_verification/'+str(index))
    comparisons.append({'report':str(path),'sha256':sha(path),'all_18_nontiming_prediction_and_speculation_fields_exact':True,
      'primary_same_settings_reference':title=='agent-exact-combined-quality-candidate',
      'settings_differences':{k:[report['settings'].get(k),old['settings'].get(k)] for k in report['settings']|old['settings']
        if report['settings'].get(k)!=old['settings'].get(k)}})
  # Establish the first-query history difference explicitly, then prove the
  # original all-target-accepted nominal path makes both histories identical.
  policy=load(C.MLP/'experiments/agent-speculation-policy.py','coverage_nominal_policy')
  original=load(C.MLP/'experiments/agent-speculation-prefill.py','coverage_nominal_transition')
  nominal=list(policy.CHECKPOINT_DRAFT_IDS);first=speculation['records'][0]
  assert first['accepted'] and first['generated']==first['targets']==first['draft']==nominal and first['all_targets_finite']
  prior_records=read(WORK/'agent-exact-combined-quality-candidate/report.speculation.json')['records']
  assert all(r['accepted'] and r['generated']==nominal for r in prior_records[:14])
  endings=[]
  for history in ([],[nominal]):
    def forbidden(*a):raise AssertionError('First accepted nominal query must not invoke historical lookup')
    decoder=SimpleNamespace(generate=forbidden,history=copy.deepcopy(history),statistics=[],read_tokens=lambda x:x)
    original.install_decoder(decoder,nominal,nominal[-1])
    model=SimpleNamespace(padded_prefill=SimpleNamespace(last_prompt=first['actual_prompt'],
      last_targets=np.asarray([*nominal,*([1]*6)],dtype=np.int32),prefix_audit=None))
    result=decoder.generate(model,SimpleNamespace(),None,None,None,None,None,first['actual_prompt'])
    assert result==nominal;endings.append(decoder.history)
  assert endings==[[nominal],[nominal]]
  # Histories subsequently use the same completed tokens and bounded deduplication.
  left,right=endings
  for record in speculation['records'][1:]:
    generated=record['generated'];left=[x for x in left if x!=generated][-63:]+[generated[:]]
    right=[x for x in right if x!=generated][-63:]+[generated[:]];assert left==right
  existing=read(WORK/'agent-exact-combined-quality-candidate/report.json')['agent_exact_combined']
  assert existing['passed'] and existing['hooks_restored'] and existing['paired_exact']['exact']
  shapes={tuple(map(tuple,r['shape'])) for r in existing['coverage']['vision_graphs']}
  assert shapes=={PAIR,PAIR*2}
  result={'reference_predictions_admitted':True,'original_status_preserved':failed,
    'reference':str(REFERENCE),'reference_sha256':REFERENCE_SHA,'source_snapshot_identity':source_identity,
    'comparison_receipts':comparisons,'queries':18,'numerical_reference_only':True,
    'initial_lookup_history_equal':False,'initial_history_difference':'empty versus one previously completed NOMINAL',
    'original_nominal_transition_executed':True,'first_query_historical_lookup_calls':0,
    'lookup_history_equal_after_first_completed_query':True,
    'vehicle_history_and_all_inputs':'Same immutable full sample files, checked against the original plan before execution',
    'runtime_capture_inventory_recovered':False,'runtime_pv_metadata_recovered':False,
    'unavailable_fields':['current run raw vision trace counters','current run captured program inventory','current run PV info and final restoration lists'],
    'use':'Admit already observed predictions for numerical comparison only. Do not restamp failed run or use its timing as qualified.',
    'existing_vision_batch_coverage':[1,2],'plan_sha256':PLAN_SHA}
  if write_result:write(RECEIPT,result,new=True)
  return result


def expected_chunks(cache,batches):
  assert cache.grouped and cache.views and cache.batch_encoder is not None and cache.group_limit==2
  assert len(cache.keys)==len(batches)==10 and all(tuple(key[0])==(1,20,36) for key in cache.keys)
  assert all((end-begin,count)==(720,1) for begin,end,count in batches)
  missing={}
  for key,batch in zip(cache.keys,batches,strict=True):
    if key not in cache.cache:missing.setdefault(key,batch)
  count=len(missing)
  return [2]*(count//2)+([1] if count%2 else []),count


def inventory(comp):
  def graph(jit):return {'calls':int(getattr(jit,'cnt',0)),'captured':getattr(jit,'captured',None) is not None,'names':comp.names(jit)}
  return {'language':[dict(name=n,**graph(j)) for n,j in comp.hook.prefill_jits(comp.model)] if comp.model is not None else [],
    'vision':[dict(shape=[list(x) for x in shape],**graph(j)) for shape,j in comp.cache.encoders.items()] if comp.cache is not None else [],
    'vision_hook':copy.deepcopy(comp.vision_info),'language_stats':copy.deepcopy(comp.session.stats) if comp.session else None,
    'frame_idle':comp.session is not None and comp.session.frame is None and not comp.session.state['prefill'],
    'vision_calls':copy.deepcopy(getattr(comp,'coverage_vision_calls',[])),
    'pv':copy.deepcopy(comp.pv_handle.info) if comp.pv_handle is not None else None,
    'restorations':copy.deepcopy(comp.restorations),'events':comp.events[:],'lookup_initial':copy.deepcopy(comp.lookup_initial)}


def validate_shapes(comp):
  value=inventory(comp)
  assert comp.arm_calls==1 and len(comp.lookup_initial)==1 and comp.pv_handle is not None
  assert comp.lookup_initial==[{'history':[],'vision_cache_empty':True,'vision_encoders_empty':True,'vision_stitchers_empty':True}]
  stats=value['language_stats'];prefix=comp.hook.PREFIX
  assert set(stats['layer_hits'])=={prefix+str(i) for i in range(28)} and len(set(stats['layer_hits'].values()))==1
  assert stats['python_trace_fc1']==stats['python_trace_fc2']==stats['python_trace_fused_quantize']==stats['layer_completions']>0
  assert not stats['failed_text_layers'] and set(stats['shape_hits'])=={'2048'}
  assert any(sum(n.startswith('agent_packed_epilogue') for n in r['names'])==28 for r in value['language'])
  observed=set()
  for row in value['vision']:
    shape=tuple(map(tuple,row['shape']));assert shape in (PAIR,PAIR*2), 'Unqualified image graph shape: '+str(shape)
    batch=len(shape)//2;observed.add(batch)
    names=[n for n in row['names'] if n.startswith('agent_vision_tiles_')]
    assert row['captured'] and names.count(f'agent_vision_tiles_3x3_b{batch}')==25 and names.count(f'agent_vision_tiles_3x6_b{batch}')==25
    assert len(names)==50, 'Unexpected vision program'
  calls=value['vision_calls'];assert len(calls)==18 and all(c['completed'] for c in calls)
  scheduled={batch for row in calls for batch in row['chunks']}
  assert observed==scheduled and observed, 'Captured graph set differs from actual cache grouping'
  traces=value['vision_hook']['trace_calls']
  required={f'vlm.model.visual.encoder.layers.{i}.mlp.fc{fc}/batch{batch}' for batch in observed for i in range(2,27) for fc in (1,2)}
  assert {n for n,v in traces.items() if v>0}==required
  for batch in observed:
    assert len({traces[f'vlm.model.visual.encoder.layers.{i}.mlp.fc{fc}/batch{batch}'] for i in range(2,27) for fc in (1,2)})==1
  assert value['frame_idle'] and comp.stage=='quality' and comp.sampler_calls==0 and comp.sampler is None
  value.update(validated=True,vision_batches_exercised=sorted(observed),all_28_mlp_and_50_per_batch_vision_projections=True)
  return value


def factory(C):
  old=load(WORK/'agent-expert-combined-source/agent-expert-combined-install.py','adaptive_coverage_frozen_composition')
  Base=old.factory(C)
  class Composition(Base):
    def __init__(self,*args,**kwargs):super().__init__(*args,**kwargs);self.coverage_vision_calls=[]
    def after_load(self,module,path):
      super().after_load(module,path)
      if path.name=='vision-cache.py':
        original=module.VisionCache.encode;owner=self
        def encode(cache,pixels,positions,ids,batches,spans):
          assert cache is owner.cache
          chunks,misses=expected_chunks(cache,batches)
          row={'chunks':chunks,'expected_misses':misses,'completed':False};owner.coverage_vision_calls.append(row)
          try:
            result=original(cache,pixels,positions,ids,batches,spans)
            assert cache.misses==misses
            row.update(completed=True,hits=cache.hits,misses=cache.misses)
            return result
          except BaseException as exc:row['failure']={'type':type(exc).__name__,'message':str(exc)};raise
        self.patched(module.VisionCache,'encode',encode)
    def coverage(self):return validate_shapes(self)
  return Composition
