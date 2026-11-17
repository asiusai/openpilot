"""CPU-only contract, real native mini-gradient, fixed160 entrypoint, and resume checks."""
import argparse
from copy import deepcopy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace
from unittest.mock import patch

sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
BASE=Path('/tmp/opencode/alpamayo-asius-20261114')
WORK=Path('/mnt/nvme-500/tmp/alpamayo-speed-20261116')
ROOT=WORK/'agent-distill-vision-cohort-fit-artifacts'


def load(name,path):
  spec=importlib.util.spec_from_file_location(name,path);value=importlib.util.module_from_spec(spec)
  sys.modules[name]=value;spec.loader.exec_module(value);return value


def main():
  p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
  root=args.output.resolve().parent
  if not root.is_relative_to(ROOT) or root.exists():raise ValueError('Need fresh owned CPU-check directory')
  root.mkdir(parents=True);(root/'tmp').mkdir()
  os.environ.update(DEV='CPU',TC='0',ALLOW_TF32='0',BEAM='0',JITBEAM='0',PARALLEL='2',OPTIM_DTYPE='float32',FUSE_OPTIM='0',CONST_LR='0',
    OPENBLAS_NUM_THREADS='4',OMP_NUM_THREADS='4',CUDA_VISIBLE_DEVICES='',HIP_VISIBLE_DEVICES='',ROCR_VISIBLE_DEVICES='',
    CACHEDB=str(root/'cpu-cache.db'),TMPDIR=str(root/'tmp'),PYTHONDONTWRITEBYTECODE='1')
  def audit(event,values):
    if event!='open' or not isinstance(values[0],(str,bytes)):return
    path=os.fsdecode(values[0])
    if path.startswith(('/dev/kfd','/dev/dri','/dev/nvidia','/dev/bus/usb')):raise RuntimeError('CPU check attempted device access')
    if 'agent-distill-cohort-artifacts' in path:raise RuntimeError('Pending producer output access')
    if any(x in path for x in ('0000008e--321b2ddc03','00000090--1cf3f7d926')):raise RuntimeError('Excluded route access')
  sys.addaudithook(audit);sys.path.insert(0,str(BASE/'tinygrad'))
  import numpy as np
  from tinygrad import Tensor,Device
  from tinygrad.helpers import Context
  from tinygrad.nn.optim import AdamW
  T=load('vision_cohort_fit_check_train',HERE/'agent-distill-vision-cohort-fit-train.py');C=T.C;S=T.S
  C.source_pins();CAP=C.capture_adapter();legacy=C.read(C.LEGACY);old=legacy['rows'][0];negatives=[]
  def reject(name,fn):
    try:fn()
    except (ValueError,AssertionError,KeyError):negatives.append(name)
    else:raise AssertionError('Accepted '+name)

  # Clearly marked metadata fixtures. No pending files or decoded frames are used.
  fixture=root/'synthetic-contract';fixture.mkdir();anchors=C.read(CAP.COVERAGE)['anchors']
  producer=dict(synthetic_cpu_fixture=True,admissible_for_internal_layer_capture=True,admissible_for_driving_quality_validation=False,
    completed_native_frames=240,completed_conditioned_inputs=24,independent_drives=2,protected_and_sealed_drives_opened=False,
    protocol=dict(coverage=dict(path=str(CAP.COVERAGE),sha256=CAP.COVERAGE_SHA)),samples=[])
  for a in anchors:
    row={k:a[k] for k in ('id','split','route_id','drive_id','query_ns','conditioning')}
    row.update(command_index=0,actual_prompt_tokens=2027,complete_input=True,official_selected_tokens_and_image_normalization_exact=True,
      sample=str(WORK/'agent-distill-cohort-artifacts/inputs-v1/samples'/row['id']),input_file_sha256=old['input_file_sha256'])
    producer['samples'].append(row)
  producer_path=fixture/'producer.json';C.write(producer_path,producer)
  reports=[]
  for job in range(12):
    directory=fixture/f'job{job}';directory.mkdir();report=deepcopy(legacy)
    report.update(synthetic_cpu_fixture=True,legacy_control=None,inputs_manifest_sha256=C.sha(producer_path),rows=[])
    for wanted in producer['samples'][job*2:(job+1)*2]:
      row=deepcopy(old);row.pop('legacy_comparison',None)
      for key in ('id','split','route_id','query_ns','input_file_sha256','actual_prompt_tokens'):row[key]=wanted[key]
      for i in range(10):
        digest=hashlib.sha256((row['id']+str(i)).encode()).hexdigest()
        row['vision_input_keys'][i]['pixel_sha256']=digest;row['images'][i]['pixel_sha256']=digest
      for name,info in row['targets']['tensors'].items():info['path']=str(directory/row['id']/(name+'.npy'))
      report['rows'].append(row)
    path=directory/'report.json';C.write(path,report);reports.append(dict(path=str(path),sha256=C.sha(path)))
  bundle=fixture/'capture-set.json'
  C.write(bundle,dict(schema=1,synthetic_cpu_fixture=True,inputs_manifest=dict(path=str(producer_path),sha256=C.sha(producer_path)),reports=reports))
  reject('fixture_not_admissible_in_production',lambda:C.admit(bundle,C.sha(bundle)))
  original_read=C.read;original_real=C.real_report;verified=[]
  def mock_read(path):
    value=original_read(path)
    if Path(path)==bundle:value.pop('synthetic_cpu_fixture')
    return value
  def mock_real(report):
    value=dict(report);assert value.pop('synthetic_cpu_fixture') is True;original_real(value)
  def mock_array(info,path,scene,name):
    assert Path(info['path'])==Path(path).parent/scene/(name+'.npy')
    verified.append((scene,name))
  def admission(mutate=None):
    def reader(path):
      value=mock_read(path)
      return mutate(Path(path),value) if mutate else value
    with patch.object(C,'read',reader),patch.object(C,'real_report',mock_real),patch.object(C,'verify_tensor',mock_array):
      return C.admit(bundle,C.sha(bundle))
  dataset=admission();assert len(verified)==1920 and len(dataset['train'])==160 and len(dataset['validation'])==80
  assert [(r['scene'],r['image_index']) for r in dataset['train']]==[(f'train-scene-{s:02d}',i) for i in range(10) for s in range(16)]
  first=Path(reports[0]['path']);last=Path(reports[-1]['path'])
  def change(path,field,value):
    def mutate(p,obj):
      if p==path:obj[field]=value
      return obj
    return mutate
  reject('incomplete_capture',lambda:admission(change(first,'complete',False)))
  reject('GPU_mock_capture',lambda:admission(change(first,'cpu_integration_mock',True)))
  reject('legacy_control_rejected',lambda:admission(change(first,'legacy_control','train-command-0')))
  reject('changed_source_settings',lambda:admission(change(first,'settings',{})))
  reject('missing_report',lambda:admission(change(bundle,'reports',reports[:-1])))
  def row_change(p,obj):
    if p==first:obj['rows'][1]=deepcopy(obj['rows'][0])
    return obj
  reject('duplicate_scene',lambda:admission(row_change))
  def split_change(p,obj):
    if p==first:obj['rows'][0]['split']='validation'
    return obj
  reject('changed_split',lambda:admission(split_change))
  def mapping_change(p,obj):
    if p==first:obj['rows'][0]['images'][0]['member']=1
    return obj
  reject('wrong_logical_mapping',lambda:admission(mapping_change))
  def leakage(p,obj):
    if p==last:
      digest=hashlib.sha256(b'train-scene-000').hexdigest()
      obj['rows'][0]['vision_input_keys'][0]['pixel_sha256']=digest;obj['rows'][0]['images'][0]['pixel_sha256']=digest
    return obj
  reject('cross_split_identical_pixel',lambda:admission(leakage))
  def duplicates(p,obj):
    if p==first:
      row=obj['rows'][0];row['vision_input_keys'][9]=deepcopy(row['vision_input_keys'][0])
      row['images'][9].update(batch=0,member=0,projector=0,pixel_sha256=row['images'][0]['pixel_sha256'])
      row['targets']['vision_batches'][4]['images']=1;row['targets']['projector_calls']=9
      del row['targets']['tensors']['projector9.features']
      for name,info in row['targets']['tensors'].items():
        if name.startswith('vision4.'):info['shape'][0]=1
    return obj
  dedup=admission(duplicates)
  assert dedup['train'][0]['member']==dedup['train'][144]['member']==0 and dedup['train'][144]['batch']==0
  def reverse_reports(p,obj):
    if p==bundle:obj['reports'].reverse()
    return obj
  assert admission(reverse_reports)['train']==dataset['train']
  reject('legacy_wrong_path',lambda:C.verify_tensor(old['targets']['tensors']['vision0.group12.x'],first,'train-scene-00','vision0.group12.x'))

  # Read only one completed legacy group12 pair. It cannot enter the actual cohort.
  actual=dict(member=0,x=old['targets']['tensors']['vision0.group12.x'],y=old['targets']['tensors']['vision0.group12.y'])
  arrays=C.image_arrays(actual);assert arrays['x'].shape==arrays['y'].shape==(1,720,1152)
  B=C.original().graph();seed,_=C.original().seeds();block=B.Block(seed);opt=AdamW(list(block.weights.values()),lr=3.5e-5,b1=.9,b2=.999,eps=1e-8,weight_decay=.01,fused=False)
  initial=block.weights['mlp.fc2.bias'].numpy().copy()
  with Context(TRAINING=1):
    x=Tensor(arrays['x'][:,:3].copy()).is_param_(False);y=Tensor(arrays['y'][:,:3].copy()).is_param_(False)
    loss=B.mse(block(x),y);gradients=loss.gradient(*block.weights.values());Tensor.realize(*gradients)
    before=float(loss.item());norms={}
    for (name,parameter),gradient in zip(block.weights.items(),gradients,strict=True):
      values=gradient.numpy();assert np.isfinite(values).all();norms[name]=float(np.linalg.norm(values));parameter.grad=gradient
    assert norms['mlp.fc2.weight']>0 and norms['self_attn.q_proj.weight']>0 and norms['self_attn.k_proj.weight']>0
    opt.step()
  assert np.any(block.weights['mlp.fc2.bias'].numpy()!=initial)
  native_state=S.snapshot(block.weights,opt);S.validate_arrays(native_state,B.shapes(),1)
  mini=dict(shape=[1,3,1152],parameter_count=sum(v.size for v in seed.values()),before_mse=before,
    gradient_norms=norms,first_bias_delta=float(np.max(np.abs(block.weights['mlp.fc2.bias'].numpy()-initial))),actual_native_FP32_graph=True)
  del block,opt,gradients,native_state,seed,x,y,loss;__import__('gc').collect()

  class Ledger:
    def __init__(self,prior=0):self.report=dict(requested_bytes=prior,live_bytes=0,peak_live_bytes=0,phases=[])
    def phase(self,name):self.report['phases'].append(dict(name=name))
  class Engine:
    cpu_mock=True
    def __init__(self,prior=0):
      self.weights={'w':Tensor(np.array([[.2,-.1],[.3,.4]],np.float32)).is_param_(True).realize()};self.shapes={'w':(2,2)}
      self.optimizer=AdamW(list(self.weights.values()),lr=3.5e-5,b1=.9,b2=.999,eps=1e-8,weight_decay=.01,fused=False)
      self.identity={'fixture':'CPU arithmetic; no GPU identity assertion'};self.ledger=Ledger(prior);self.visits=[]
    def objective(self,row):
      factor=1+row['image_index']/100
      return (self.weights['w']*factor-Tensor([[1.,2.],[3.,4.]])).square().mean()
    def evaluate(self,row):self.visits.append(('eval',row['id']));self.ledger.report['requested_bytes']+=8;return float(self.objective(row).item())
    def update(self,row):
      self.visits.append(('update',row['id']));self.ledger.report['requested_bytes']+=32
      with Context(TRAINING=1):
        self.optimizer.zero_grad();loss=self.objective(row);grads=loss.gradient(*self.weights.values());Tensor.realize(*grads)
        value=float(loss.item())
        for weight,gradient in zip(self.weights.values(),grads):weight.grad=gradient
        self.optimizer.step();self.optimizer.zero_grad()
      return value
  fake_plan=dict(dataset=dataset['receipt'],parameter_count=4)
  def options(directory):return SimpleNamespace(output_dir=directory,plan_sha='c'*64,mode='fit',resume_from=None)

  qualified=root/'mock-qualify';qualified.mkdir();qualifier=Engine();qualifier.started=time.monotonic()-2
  qa=options(qualified);qa.mode='qualify';qualified_report=T.run_fit(qa,fake_plan,dataset,qualifier)
  assert qualified_report['committed_step']==qualified_report['real_optimizer_updates']==1
  assert qualified_report['measurements']['setup_seconds']>=2 and qualified_report['selection']=='qualification has no checkpoint selection'
  assert [kind for kind,_ in qualifier.visits]==['eval','update'] and not (qualified/'best.npz').exists()

  # Actual CLI preflight and child entrypoint; only GPU boundaries and admission IO are mocked.
  preflight_args=['fit','--plan',str(root/'mock-plan.json'),'--plan-sha','c'*64]
  with patch.object(sys,'argv',preflight_args),patch.object(C,'verify_plan',return_value=(fake_plan,dataset)):T.main()
  full=root/'mock-full160';full.mkdir();C.write(full/'parent-invocation.json',dict(synthetic_cpu_fixture=True,previous_total_wall_seconds=0))
  engine=Engine();available=dict(vram_free_bytes=64*1024**3)
  K=SimpleNamespace(passive_availability=lambda:available,require_no_observed_conflict=lambda value:None)
  cli=preflight_args+['--mode','fit','--execute-gpu','--child','--output-dir',str(full)]
  with patch.object(sys,'argv',cli),patch.dict(os.environ,AGENT_DISTILL_VISION_COHORT_FIT_PARENT_PID=str(os.getppid())),\
       patch.object(C,'verify_plan',return_value=(fake_plan,dataset)),patch.object(T,'availability',return_value=K),\
       patch.object(T,'qualification'),patch.object(T,'GPUEngine',return_value=engine):T.main()
  report=C.read(full/'report.json');assert report['complete'] and report['cpu_integration_mock'] and not report['gpu_executed']
  assert report['real_optimizer_updates']==report['committed_step']==160 and report['best_step']==160
  assert [kind for kind,_ in engine.visits]==['eval']*80+['update']*160+['eval']*80
  assert [name for kind,name in engine.visits if kind=='update']==[r['id'] for r in dataset['train']]
  expected_state,metadata=S.load(full,T.context(options(full)),engine.shapes);S.check_best(full,metadata,engine.shapes)
  assert metadata['validation'][1]['mean']<metadata['validation'][0]['mean']

  interrupted=root/'mock-stop73';interrupted.mkdir();other=Engine()
  def stop(step,report):
    if step==73:raise RuntimeError('CPU fixture stops after a clean committed step')
  try:T.run_fit(options(interrupted),fake_plan,dataset,other,after_commit=stop)
  except RuntimeError as error:assert str(error)=='CPU fixture stops after a clean committed step'
  C.write(interrupted/'wall.json',dict(seconds=10.,returncode=1,timed_out=False))
  C.write(interrupted/'parent-invocation.json',dict(previous_total_wall_seconds=12.))
  restored,metadata,carry=T.resume_carry(interrupted,options(interrupted),other.shapes,dataset)
  assert carry['previous_total_wall_seconds']==22. and carry['requested_bytes']>0
  resumed=root/'mock-resume160';resumed.mkdir();restarted=Engine(carry['requested_bytes'])
  resumed_report=T.run_fit(options(resumed),fake_plan,dataset,restarted,resume=(interrupted,restored,metadata))
  resumed_state,_=S.load(resumed,T.context(options(resumed)),restarted.shapes)
  assert resumed_report['starting_checkpoint_step']==73 and resumed_report['real_optimizer_updates']==87
  assert resumed_report['allocation_budget']['requested_bytes']==report['allocation_budget']['requested_bytes']
  for name in expected_state:np.testing.assert_array_equal(expected_state[name],resumed_state[name])
  assert [kind for kind,_ in restarted.visits]==['update']*87+['eval']*80
  assert C.sha(full/'best.npz')==C.sha(resumed/'best.npz')
  tied=deepcopy(metadata);tied['step']=160;tied['last_training_image']=dataset['train'][-1]['id']
  tied['validation']=[deepcopy(metadata['validation'][0])]*2;tied['validation'][1]=deepcopy(tied['validation'][1]);tied['validation'][1]['step']=160
  tied['best_step']=0;tied['best_validation']=tied['validation'][0]['mean'];T.validate_metadata(tied,dataset)
  tied['best_step']=160;reject('tie_must_keep_seed',lambda:T.validate_metadata(tied,dataset))
  bad={k:v.copy() for k,v in resumed_state.items()};bad['learning_rate'][:]=3.5e-6
  reject('wrong_learning_rate',lambda:S.validate_arrays(bad,restarted.shapes,160))
  bad={k:v.copy() for k,v in resumed_state.items()};bad['beta1_power'][:]=1
  reject('wrong_beta_power',lambda:S.validate_arrays(bad,restarted.shapes,160))
  bad={k:v.copy() for k,v in resumed_state.items()};bad['moment2/w'][0,0]=-1
  reject('negative_second_moment',lambda:S.validate_arrays(bad,restarted.shapes,160))
  S.intent(interrupted,73);reject('dirty_journal_no_replay',lambda:S.load(interrupted,T.context(options(interrupted)),other.shapes))
  qa=SimpleNamespace(qualification=full/'report.json',qualification_sha=C.sha(full/'report.json'),plan_sha='c'*64)
  C.write(full/'wall.json',dict(seconds=1.,returncode=0,timed_out=False))
  reject('CPU_mock_is_not_GPU_qualification',lambda:T.qualification(qa))

  # Exercise the reused complete ledger with reused handles and owned USERPTR unmaps.
  L=C.module('vision_cohort_fit_check_ledger',C.LEDGER);owner=123
  passive=lambda:dict(selected_node=dict(gpu_id=27211,pci='0000:0c:00.0'),kfd_proc_root_read=True,kfd_processes=[],
    observed_selected_kfd_owners=[],vram_free_bytes=64*1024**3)
  ledger=L.Ledger(C.LIMITS,passive,own_pid=owner);iface=SimpleNamespace(gpu_id=27211,dev_sysfs_path='/synthetic/0000:0c:00.0')
  alloc=ledger.alloc(lambda iface,size:SimpleNamespace(meta=SimpleNamespace(handle=7,size=size)))
  free=ledger.free(lambda iface,value:None);unmap=ledger.unmap(lambda iface,value:None)
  value=alloc(iface,128);free(iface,value);alloc(iface,128);unmap(iface,SimpleNamespace(meta=(7,True)))
  assert ledger.report['requested_bytes']==256 and ledger.report['live_bytes']==0 and ledger.report['owned_unmap_calls']==1
  reject('double_release',lambda:free(iface,value))
  reject('oversized_allocation',lambda:alloc(iface,C.LIMITS['max_single_buffer']+1))
  ledger.report['requested_bytes']=C.LIMITS['cumulative_allocation_limit']-4
  reject('cumulative_limit_retained',lambda:alloc(iface,8))
  C.no_devices();assert all(name.startswith(('CPU','NPY','PYTHON','DISK')) for name in Device._opened_devices)
  result=dict(complete=True,gpu_executed=False,pending_cohort_outputs_read=False,actual_cohort_admitted=False,
    synthetic_complete24_scene_admission_checked=True,synthetic_array_io_mocked=True,actual_legacy_group12_mini=mini,
    actual_train_cli_with_GPU_boundaries_mocked=True,one_update_qualification_setup_and_projection_checked=True,
    fixed160_schedule_exact=True,validation_only_at0_and160=True,
    balanced8scene_validation=True,strict_tie_keeps_seed=True,resume73_to160_all_arrays_bitexact=True,
    resumed_time_and_cumulative_allocation_carried=True,complete_owned_unmap_ledger_checked=True,negative_controls=negatives,
    opened_devices=sorted(Device._opened_devices),sources=C.source_pins())
  args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
  print(json.dumps({k:v for k,v in result.items() if k not in ('actual_legacy_group12_mini','sources')},indent=2))


if __name__=='__main__':main()
