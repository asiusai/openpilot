"""Full original chronology; unchanged numerical hooks; small source-bound harness."""
import ast
import copy
import hashlib
import importlib.util
import inspect
import json
from pathlib import Path
import textwrap
import numpy as np

HERE=Path(__file__).resolve().parent
WORK=Path('/mnt/nvme-500/tmp/alpamayo-speed-20261116')
ART=WORK/'agent-adaptive-quality64-artifacts'
PLAN=WORK/'agent-adaptive-quality64-plan.json'
TRUTH=WORK/'agent-ck-quality64-work/prepared/inputs.json'
TRUTH_SHA='55bd2fedd2ec7f48aa4526f2207e86431891fa198b435cc1e811e5825a5c6c93'
PRIOR=WORK/'agent-adaptive-coverage-quality-candidate'
PINS={HERE/'agent-adaptive-coverage-common.py':'3e116e2102f6caa532b01eaee624d05a7df0663c2ca025b07b45c7568c42b3fb',
 HERE/'agent-adaptive-coverage-run.py':'fe51ef1f847589f344b939612bfb6f6d44211246e8abe5c435fa91c75cea10c4',
 HERE/'agent-adaptive-coverage-check.py':'d1f67d1d29e006f4ff5138f1acd8b3745fcc6c0252096981b3fbff8d05d92db6',
 HERE/'agent-adaptive-prefill-hook.py':'7cd0d8ce7c5c2ff1d1bac4fec0ddc4dd2d4d0cb37b14455fba3deca8be4a967a',
 HERE/'agent-adaptive-motion-review.py':'50c08878be138c7aa4496666b4a6dd4e123435a25cbbc423583dc39977d531f9',
 PRIOR/'report.json':'d94949b9261a354b06a71f1e4a074d10f375f3edabf64ad18185a786519395c6',
 PRIOR/'adaptive-coverage-status.json':'717950a58cfbbde8284c77aa70d2ef54f548d1bd7132dc97fce8b079e8ced7ee',TRUTH:TRUTH_SHA}

def sha(path):
  with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def read(path):return json.loads(Path(path).read_text())
def write(path,value,new=True):
  p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
  with p.open('x' if new else 'w') as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n')
def load(path,name):
  spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def base():
  for path,digest in PINS.items():assert sha(path)==digest,str(path)
  K=load(HERE/'agent-adaptive-coverage-common.py','quality64_prior_common')
  R,old,N,C,evidence=K.frozen()
  status=read(PRIOR/'adaptive-coverage-status.json')
  assert status['complete'] and status['gpu_executed'] and status['import_hook_restored'] and status['adaptive_hook_restored']
  assert status['checks']['shape_coverage']['validated'] and status['checks']['pv_complete']
  assert status['checks']['adaptive']['prefix_audit_count']==1
  return K,R,old,N,C,evidence

def verify_inputs():
  assert sha(TRUTH)==TRUTH_SHA
  items=read(TRUTH);assert [x['original64_index'] for x in items]==list(range(64))
  assert all(a['query_ns']<b['query_ns'] for a,b in zip(items,items[1:]))
  for item in items:
    assert set(item['files'])=={str(p) for p in Path(item['sample']).glob('*.npy')}
    for name,digest in item['files'].items():assert sha(name)==digest,name
    xyz=np.asarray(item['recorded_motion_future_xyz']);assert xyz.shape==(64,3) and np.isfinite(xyz).all()
  return items

def counted_coverage(K,count):
  """Change only the fixture length assertion; keep every qualified shape gate."""
  tree=ast.parse(textwrap.dedent(inspect.getsource(K.validate_shapes)));original=copy.deepcopy(tree)
  found=[]
  for node in ast.walk(tree):
    if isinstance(node,ast.Compare) and ast.dump(node.left)==ast.dump(ast.parse('len(calls)',mode='eval').body):
      assert len(node.comparators)==1 and isinstance(node.comparators[0],ast.Constant) and node.comparators[0].value==18
      node.comparators[0].value=count;found.append(node)
  assert len(found)==1
  inverse=copy.deepcopy(tree)
  for node in ast.walk(inverse):
    if isinstance(node,ast.Compare) and ast.dump(node.left)==ast.dump(ast.parse('len(calls)',mode='eval').body):node.comparators[0].value=18
  assert ast.dump(inverse)==ast.dump(original)
  ns=dict(K.validate_shapes.__globals__);exec(compile(ast.fix_missing_locations(tree),str(HERE/'agent-adaptive-coverage-common.py')+':count-only','exec'),ns)
  return ns['validate_shapes']

def composition(K,C,count):
  Base=K.factory(C);validate=counted_coverage(K,count)
  class Composition(Base):
    def coverage(self):return validate(self)
  return Composition

def output(stage,arm):return WORK/f'agent-adaptive-quality64-{stage}-{arm}'/'report.json'
def status_path(path):return path.parent/'adaptive-quality64-status.json'
def rows(report):return report['warmup']+report['runs']

def prepare():
  K,R,old,N,C,evidence=base();R.reserve();items=verify_inputs()
  for name in ('tmp','cache'):(ART/name).mkdir(parents=True,exist_ok=True)
  pipeline=read(old['pipeline']);cmd=pipeline['command'];begin=cmd.index('--samples')+1;end=cmd.index('--initial-noise')
  cmd[begin:end]=[x['sample'] for x in items]
  for key,value in (('warmup',2),('runs',62),('output',ART/'unused.json')):R.option(cmd,key,value)
  pipeline['environment'].update(CACHEDB=str(ART/'gpu-cache.db'),TMPDIR=str(ART/'tmp'),
    XDG_CACHE_HOME=str(ART/'cache'),HF_HOME=str(ART/'cache/huggingface'),TORCH_HOME=str(ART/'cache/torch'))
  path=WORK/'agent-adaptive-quality64-pipeline.json';write(path,pipeline)
  write(PLAN,{'samples':[x['sample'] for x in items],'indices':list(range(64)),'pipeline':str(path),'pipeline_sha256':sha(path),
    'pins':{str(p):sha(p) for p in [*PINS,*HERE.glob('agent-adaptive-quality64-*.py')]},
    'fresh_matched_reference_required':True,'baseline_reuse_rejected':{
      'agent-exact-combined-quality-candidate':'64 same settings/source, but no qualified PV overlay',
      'agent-expert-pv-qualified-quality-candidate':'64 PV only; different source and rotary-cache/coarse-timing settings',
      'agent-expert-combined-cache-quality-candidate':'Exact selected combination, but only four queries',
      'agent-ck-quality64-parent-baseline-v1':'Different padding-source and projection implementation'},
    'quality':{'warmup':2,'runs':62,'candidate_prefix_audits':1,'numerical_hook_changed':False},
    'timing':{'warmup':64,'runs':64,'empty_history_each_pass':True,'audit':False,'pairwise_exclude_uncaptured_or_compiled_queries':True},
    'automatic_timing':False,'automatic_continuous':False,'adoption_allowed':False})

def verify():
  data=read(PLAN)
  assert data['indices']==list(range(64)) and data['fresh_matched_reference_required']
  for path,digest in data['pins'].items():assert sha(path)==digest,path
  assert sha(data['pipeline'])==data['pipeline_sha256']
  assert data['samples']==[x['sample'] for x in verify_inputs()]
  return data

def check_report(K,report,plan,count):
  result=rows(report);assert len(result)==count
  assert [r['sample'] for r in result]==plan['samples']*(count//64)
  prior=read(PRIOR/'report.json')
  for key in ('source_sha256','settings','runtime_environment','initial_noise_sha256','prefill_calibration_sha256','precision_restoration',
    'reduced_vocab','pixel_scale','history_pixel_scale','history_feature_scale','vision_calibration_sha256','pad_prompt_to',
    'trajectory_solver','vision_group_limit','batched_vision'):
    K.exact(report[key],prior[key],key)
  K.snapshot_files(report)
  for i,row in enumerate(result):
    assert row['input_latest_timestamp_us']*1000==read(TRUTH)[i%64]['query_ns']
    p=row['prediction'];assert p['prefill_tokens']==2048 and p['generated_tokens']==len(p['generated_token_ids'])
    assert np.isfinite(np.asarray(p['xyz'])).all() and np.isfinite(np.asarray(p['rot'])).all()
  return {'all_inputs_order_and_recipe_checked':True,'count':count}

def same_arm(K,quality,actual):
  expected=rows(quality);observed=rows(actual);assert len(expected)==64 and len(observed)==128
  for i,row in enumerate(observed):
    old=expected[i%64]
    for key in ('sample','input_latest_timestamp_us','vision_hits','vision_misses','uploaded_pixel_bytes'):K.exact(old[key],row[key],f'{i}/{key}')
    K.exact(K.clean_prediction(old['prediction']),K.clean_prediction(row['prediction']),f'{i}/prediction')
    # The single quality audit and host-time diagnostic are the only omitted draft fields.
    def draft(value):
      if isinstance(value,dict):return {k:draft(v) for k,v in value.items() if k not in ('audit_prefix_once','prefix_audit','draft_build_and_prefill_host_ms')}
      if isinstance(value,list):return [draft(v) for v in value]
      return value
    K.exact(draft(old.get('draft')),draft(row.get('draft')),f'{i}/target_history_record')
  return {'warm64_and_measured64_exact_to_own_quality':True,'all_prediction_float_bits':True,'both_histories_start_empty':True}

def review(reference,candidate):
  a,b=map(read,(reference,candidate));items=verify_inputs();aa,bb=rows(a),rows(b);assert len(aa)==len(bb)==64
  M=load(HERE/'agent-adaptive-motion-review.py','quality64_original_motion_formula')
  result=[]
  for ar,br,item in zip(aa,bb,items,strict=True):
    assert ar['sample']==br['sample']==item['sample']
    value={'original64_index':item['original64_index'],'sample':item['sample'],'query_ns':item['query_ns'],
      'nav':str(np.load(Path(item['sample'])/'nav_text.npy',allow_pickle=False).item()),
      'baseline_text':ar['prediction']['reasoning'],'candidate_text':br['prediction']['reasoning'],
      'baseline_tokens':ar['prediction']['generated_token_ids'],'candidate_tokens':br['prediction']['generated_token_ids']}
    for name,xyz in (('baseline',ar['prediction']['xyz']),('candidate',br['prediction']['xyz']),('recorded',item['recorded_motion_future_xyz'])):
      points=np.asarray(xyz,np.float64).reshape(64,3);assert np.isfinite(points).all()
      value[name+'_motion']=M.motion(points)
      if name!='recorded':
        error=np.linalg.norm((points-np.asarray(item['recorded_motion_future_xyz']))[:,:2],axis=1)
        value[name+'_ADE']=float(error.mean());value[name+'_FDE']=float(error[-1])
    value['individual_FDE_over_five_percent']=value['candidate_FDE']>1.05*value['baseline_FDE']
    for field in ('speed_mps','acceleration_mps2','heading_change_rad'):
      value['maximum_change_'+field]=float(np.max(np.abs(np.asarray(value['candidate_motion'][field])-value['baseline_motion'][field])))
    value['motion_summary']={name:{'initial_speed_mps':value[name+'_motion']['speed_mps'][0],
      'final_speed_mps':value[name+'_motion']['speed_mps'][-1],
      'speed_change_mps':value[name+'_motion']['speed_mps'][-1]-value[name+'_motion']['speed_mps'][0],
      'final_heading_change_rad':value[name+'_motion']['heading_change_rad'][-1]} for name in ('baseline','candidate','recorded')}
    result.append(value)
  aggregate=M.metrics(result);windows={f'{lo}-{lo+15}':M.metrics(result[lo:lo+16]) for lo in range(0,64,16)}
  return {'complete':True,'numerical_candidate':True,'reference_sha256':sha(reference),'candidate_sha256':sha(candidate),
    'truth_sha256':TRUTH_SHA,'aggregate':aggregate,'four_windows':windows,'rows':result,
    'five_percent_screen_passed':aggregate['within_five_percent'] and all(x['within_five_percent'] for x in windows.values()),
    'manual_review_complete':False,'adoption_allowed':False,'timing_allowed_automatically':False,
    'limits':'Overlapping recorded route samples. Index24 shared missed-turn failure remains a baseline limitation; parent reviews every text and motion.'}
