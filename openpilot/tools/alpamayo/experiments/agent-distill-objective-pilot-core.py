"""Fixed 25-update connected pilot, independent-drive validation at 0/25, complete recovery."""
import copy
import gc
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('joint_pilot_qualified_core',HERE/'agent-distill-objective-qualify-core.py')
Q=importlib.util.module_from_spec(spec);sys.modules[spec.name]=Q;spec.loader.exec_module(Q)
module,sha,read=Q.module,Q.sha,Q.read
WORK,BASE,SEEDS,S,O,CAPTURE=Q.WORK,Q.BASE,Q.SEEDS,Q.S,Q.O,Q.C
ROOT=WORK/'agent-distill-objective-pilot-artifacts'
KIND='connected-joint-feasibility-pilot-v1'
UPDATES=25
BOUNDS=dict(Q.BOUNDS,max_single_buffer=320*1024**2)
HYPER=dict(Q.HYPER,max_committed_updates=UPDATES,validation_steps=[0,UPDATES],
  selection='Strictly lower independent-drive joint validation total; ties retain step zero')
QUALIFIED=WORK/'agent-distill-objective-scratch-artifacts/live-v2/report.json'
QUALIFIED_SHA='4191e66f44ab63729813181d531e97b33560b0beb593bbc55f7740f4451ad71d'
TRAIN_SHA='3b6fa87b7efca80e065dea4a1b6ee8120f1a7b115de03629461ba0a397d0e9fe'
VALIDATION_SHA='ac855d82293cf1a06f9b4607895b18adba450783810b7e0a99f751850c1e4eba'
A=module('joint_pilot_selection',HERE/'agent-distill-continue-admit.py')
seed_arrays=Q.seed_arrays


def estimate():
  if sha(QUALIFIED)!=QUALIFIED_SHA:raise ValueError('Changed completed full-shape resource qualification')
  report=read(QUALIFIED);wall=read(QUALIFIED.parent/'wall.json')['seconds']
  assert report['complete'] and report['recovery_arrays']==51 and report['journal']==dict(committed=1,intended=None)
  assert report['bounds']==BOUNDS and report['masked_context_gradient_exactly_zero']
  p={item['name']:item for item in report['allocation_budget']['phases']}
  begin=p['one-connected-joint-forward-backward'];adam=p['one-native-adamw-update'];end=p['fresh-optimizer-recovery']
  step_bytes=end['requested_bytes_at_start']-begin['requested_bytes_at_start']
  step_seconds=end['seconds_since_start']-begin['seconds_since_start']
  graph_bytes=adam['requested_bytes_at_start']-begin['requested_bytes_at_start']
  graph_seconds=adam['seconds_since_start']-begin['seconds_since_start']
  # Keep the whole measured qualification as overhead. Bound three extra evaluations at twice its cold forward/backward cost each.
  projected_bytes=report['allocation_budget']['requested_bytes']+UPDATES*step_bytes+6*graph_bytes+2*229658624
  projected_seconds=wall+UPDATES*step_seconds+6*graph_seconds+10+120
  assert projected_bytes<BOUNDS['cumulative_allocation_limit'] and projected_seconds<BOUNDS['timeout_seconds']
  return dict(qualified_report_sha256=QUALIFIED_SHA,updates=UPDATES,measured_iteration_with_checkpoint_bytes=step_bytes,
    measured_iteration_with_checkpoint_seconds=step_seconds,measured_graph_bytes=graph_bytes,measured_graph_seconds=graph_seconds,
    projected_cumulative_bytes=projected_bytes,projected_seconds=projected_seconds,
    projection_not_a_runtime_guarantee=True,extra_evaluations=3,evaluation_multiplier=2,time_margin_seconds=120,
    measured_peak_live_bytes=report['allocation_budget']['peak_live_bytes'],measured_largest_request_bytes=279969792,
    snapshots_each_update=True,workspace_free_required_bytes=6*1024**3,workspace_reserve_bytes=4*1024**3,
    final_checkpoint_payload_bytes=688975884,selected_weights_payload_bytes=229658624,
    peak_checkpoint_and_selection_payload_bytes=2*688975884+2*229658624,
    additional_resident_validation_targets_approximately_bytes=60000000)


def target(path,digest,split):
  if path is None or not digest:raise ValueError('Explicit completed train and independent validation reports with SHA256 are required')
  path=Path(path).resolve()
  if not path.is_relative_to(WORK) or any(word in str(path) for word in CAPTURE.FORBIDDEN):raise ValueError('Unapproved aligned target path')
  if sha(path)!=digest:raise ValueError('Aligned report checksum changed')
  report=read(path)
  if not report.get('complete') or not report.get('gpu_executed') or report.get('synthetic_fixture'):raise ValueError('Completed actual aligned capture required')
  row=report['record'];targets=row['targets']
  if row.get('synthetic_fixture') or row['split']!=split or row['id']!=split+'-command-0' or row['route_id']!=CAPTURE.ROUTES[split]:
    raise ValueError('Only the two approved independent legacy anchors are admitted')
  if report['input_manifest']['sha256']!='83da7c2dc65a50185d052de07c30698ade2b1818103d90d69ab394f8f600f3b6':raise ValueError('Changed legacy inputs')
  if not targets['complete'] or targets['capacity']!=2304 or not 0<targets['prompt_length']<targets['valid_length']<=2048 or len(targets['expert'])!=3:
    raise ValueError('Preserve the qualified full2048/full2304/three-Euler geometry')
  for name,expected in [('agent-distill-objective-core.py','3ad258bf1236b31b397314b7caab9fb0b941e60d8afe3db3750939a6dec39a84'),
      ('agent-distill-objective-capture.py','7b045707be2ebec71ead5b4d6f881405e24939ed884532c38a8e36960eb213a6'),
      ('agent-distill-objective-position-run.py','248e493b935b2965b81afedf29ea0ca3b45353fce1e2a8ea692a5baf87c1e520')]:
    if sha(HERE/name)!=expected or report['source_sha256'].get(str(HERE/name))!=expected:raise ValueError('Changed objective/capture source')
  return row


def control(train_path,train_sha,validation_path,validation_sha):
  if validation_path is None or not validation_sha:raise ValueError('Training blocked until independent validation capture is complete and hash-bound')
  if train_sha!=TRAIN_SHA:raise ValueError('Use the completed qualified train anchor')
  if validation_sha!=VALIDATION_SHA:raise ValueError('Use the completed independent validation anchor')
  records=dict(train=target(train_path,train_sha,'train'),validation=target(validation_path,validation_sha,'validation'))
  if records['train']['route_id']==records['validation']['route_id']:raise ValueError('Validation requires a distinct drive')
  context=dict(kind=KIND,aligned_report_sha256=train_sha,aligned_target_sha256=dict(train=train_sha,validation=validation_sha),
    hyperparameters=HYPER,objective_sha256=sha(HERE/'agent-distill-objective-core.py'),seeds={k:v[1] for k,v in SEEDS.items()},
    independent_anchors_only=True,broader_training_cohort_sample=False,quality_qualified=False)
  return records,context


def load_pair(records,device):return {split:CAPTURE.load_record(row,device) for split,row in records.items()}
C=SimpleNamespace(load_record=load_pair)


def graph(weights,data,*,fixture=False):
  l={k.removeprefix('language.'):v for k,v in weights.items() if k.startswith('language.')}
  e={k.removeprefix('expert.'):v for k,v in weights.items() if k.startswith('expert.')}
  options=dict(head_dim=4,minimum_prefill=8,query_chunk=128) if fixture else dict(query_chunk=128)
  return O.objective(l,e,data,**options)


def component_values(losses,out):
  import numpy as np
  values={name:float(value.item()) for name,value in losses.items()}
  if not np.isfinite(list(values.values())).all() or min(values.values())<0:raise ValueError('Nonfinite or negative joint objective')
  for tensor in (*out['language'],out['context_k'],out['context_v'],*out['expert']):
    if not bool(tensor.isfinite().all().item()):raise ValueError('Nonfinite connected forward output')
  return values


def evaluate(weights,data,step,*,fixture=False):
  from tinygrad import Tensor
  from tinygrad.helpers import Context
  with Context(TRAINING=0):
    losses,out=graph(weights,data,fixture=fixture)
    Tensor.realize(*losses.values(),*out['language'],*out['expert'])
    values=component_values(losses,out)
  return dict(step=step,mean=values['total'],components=values,samples=['validation-command-0'])


def update(weights,opt,data,root,step,device,phase,*,fixture=False):
  import numpy as np
  from tinygrad import Tensor,Device
  from tinygrad.helpers import Context
  phase(f'update-{step:02d}-connected-forward-backward')
  with Context(TRAINING=1):
    opt.zero_grad();losses,out=graph(weights,data,fixture=fixture)
    if not fixture and out['language_extent']!=2048:raise ValueError('Changed full language extent')
    parameters=list(weights.values());grads=losses['total'].gradient(*parameters,out['context_k'],out['context_v'])
    Tensor.realize(*losses.values(),*out['language'],*out['expert'],*grads)
    values=component_values(losses,out);norms={}
    for name,weight,gradient in zip(weights,parameters,grads[:len(weights)],strict=True):
      if not bool(gradient.isfinite().all().item()):raise ValueError('Nonfinite parameter gradient')
      norms[name]=float(gradient.square().sum().sqrt().item());weight.grad=gradient
      if not np.isfinite(norms[name]):raise ValueError('Nonfinite parameter gradient norm')
    for role in ('language','expert'):
      if not sum(v for k,v in norms.items() if k.startswith(role+'.'))>0:raise ValueError('Gradient did not reach '+role)
    context_norms=[]
    for gradient in grads[-2:]:
      if not bool(gradient.isfinite().all().item()):raise ValueError('Nonfinite connected context gradient')
      if float(gradient[:,:,data['valid_length']:].abs().max().item())!=0:raise ValueError('Masked context received gradient')
      norm=float(gradient[:,:,:data['valid_length']].square().sum().sqrt().item())
      if not np.isfinite(norm) or not norm>0:raise ValueError('Expert context gradient missing or nonfinite')
      context_norms.append(norm)
    phase(f'update-{step:02d}-native-adamw');S.intent(root,step-1);opt.step();Device[device].synchronize();opt.zero_grad()
  return dict(step=step,pre_update_components=values,parameter_gradient_norms=norms,
    expert_to_student_context_gradient_norms=context_norms,masked_context_gradient_exactly_zero=True)


def validate_metadata(metadata):
  import numpy as np
  step=metadata['step'];assert type(step) is int and 0<=step<=UPDATES and metadata['kind']==KIND
  expected=[0,UPDATES] if step==UPDATES else [0]
  assert [v['step'] for v in metadata['validation']]==expected
  for value in metadata['validation']:
    assert value['samples']==['validation-command-0'] and value['mean']==value['components']['total']
    components=value['components']
    assert set(components)=={'prompt','reasoning','expert','total'} and np.isfinite(list(components.values())).all() and min(components.values())>=0
    assert np.isclose(components['total'],.5*(components['prompt']+components['reasoning'])+components['expert'],rtol=2e-6,atol=1e-7)
  winner=min(metadata['validation'],key=lambda v:v['mean'])
  assert metadata['best_step']==winner['step'] and metadata['best_validation']==winner['mean']
  assert [v['step'] for v in metadata['training']]==list(range(1,step+1))


def run_step(values,loader,root,context,device,phase,*,resume=None,fixture=False,stop_after=UPDATES):
  import numpy as np
  if stop_after!=UPDATES and not fixture:raise ValueError('The real pilot has exactly25 planned updates')
  assert 1<=stop_after<=UPDATES
  root=Path(root);weights=Q.construct(values,device);del values;opt=Q.optimizer(weights)
  shapes={k:v.shape for k,v in weights.items()}
  if not fixture and (len(weights)!=16 or sum(v.numel() for v in weights.values())!=57414656):raise ValueError('Changed complete student geometry')
  phase('load-both-independent-aligned-targets');data=loader(device)
  if resume:
    arrays,metadata=S.load(Path(resume),context,shapes);metadata=copy.deepcopy(metadata);validate_metadata(metadata)
    A.read_best(resume,metadata,shapes);S.restore(weights,opt,arrays);del arrays;gc.collect()
    A.copy_best(resume,root,metadata,S)
  else:
    phase('validation-at-zero');initial=evaluate(weights,data['validation'],0,fixture=fixture)
    metadata=dict(step=0,kind=KIND,best_step=0,best_validation=initial['mean'],validation=[initial],training=[],
      independent_anchors_only=True,quality_qualified=False)
    A.commit_best(root,weights,metadata,S)
  phase('initial-complete-checkpoint');Q.checkpoint_space(root)
  S.save(root,S.snapshot(weights,opt),metadata,context,shapes,initial_recovery=bool(resume));start=metadata['step']
  for step in range(start+1,stop_after+1):
    item=update(weights,opt,data['train'],root,step,device,phase,fixture=fixture)
    metadata['step']=step;metadata['training'].append(item);gc.collect()
    if step==UPDATES:
      phase('validation-at-25');validation=evaluate(weights,data['validation'],step,fixture=fixture)
      metadata['validation'].append(validation)
      if validation['mean']<metadata['best_validation']:
        metadata.update(best_step=step,best_validation=validation['mean']);A.commit_best(root,weights,metadata,S)
    validate_metadata(metadata);phase(f'update-{step:02d}-complete-checkpoint');Q.checkpoint_space(root)
    S.save(root,S.snapshot(weights,opt),metadata,context,shapes)
    S.atomic_json(root/'progress.json',dict(committed_step=step,best_step=metadata['best_step'],validation=metadata['validation']))
  final_train=None
  if metadata['step']==UPDATES:
    phase('train-at-25-diagnostic');final_train=evaluate(weights,data['train'],UPDATES,fixture=fixture)
    final_train['samples']=['train-command-0'];final_train['used_for_selection']=False
  del data,weights,opt;gc.collect();phase('complete-final-recovery')
  arrays,committed=S.load(root,context,shapes);validate_metadata(committed)
  recovered=Q.construct({name:np.zeros(shape,np.float32) for name,shape in shapes.items()},device)
  recovered_opt=Q.optimizer(recovered);S.restore(recovered,recovered_opt,arrays);checked=0
  for index,(name,weight) in enumerate(recovered.items()):
    for prefix,tensor in (('weight',weight),('moment1',recovered_opt.m[index]),('moment2',recovered_opt.v[index])):
      np.testing.assert_array_equal(tensor.numpy().view(np.uint32),arrays[prefix+'/'+name].view(np.uint32));checked+=1
  for name,tensor in (('beta1_power',recovered_opt.b1_t),('beta2_power',recovered_opt.b2_t),('learning_rate',recovered_opt.lr)):
    np.testing.assert_array_equal(tensor.numpy().view(np.uint32),arrays[name].view(np.uint32));checked+=1
  assert checked==51;A.read_best(root,committed,shapes)
  return dict(complete=committed['step']==UPDATES,starting_step=start,committed_step=committed['step'],updates_this_run=committed['step']-start,
    metadata=committed,validation=committed['validation'],training_losses=committed['training'],final_train_diagnostic=final_train,
    best_step=committed['best_step'],best_sha256=sha(root/'best.npz'),strict_validation_improvement=committed['best_step']==UPDATES,
    checkpoint_sha256=sha(root/f"checkpoint-{committed['step']:04d}.npz"),journal=read(root/'journal.json'),
    recovery_arrays=checked,recovery_bit_exact=True,additional_updates_after_recovery=0,
    checkpoint_selection_allowed=True,quality_qualified=False,training_cohort_sample=False,independent_anchor_feasibility_only=True)
