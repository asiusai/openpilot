"""One connected joint update and exact recovery. Qualification artifacts never select a model."""
import gc
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys

HERE=Path(__file__).resolve().parent
WORK=Path('/mnt/nvme-500/tmp/alpamayo-speed-20261116')
ROOT=WORK/'agent-distill-objective-qualify-artifacts'
BASE=Path('/tmp/opencode/alpamayo-asius-20261114')
SEEDS={'language':(WORK/'agent-distill-continue-artifacts/pilot-100-v1/best.npz','cf5762ab5cfdb3b622ff8d12d2d6e7f20c3d6f75f7ed772dcca97653d85504fd'),
       'expert':(WORK/'agent-distill-expert-artifacts/pilot-v2/best.npz','0206e0e9f3458173da1468a99c6aa2fa7942481ac00639c179fd0e0fd4c5fc3d')}
BOUNDS=dict(timeout_seconds=900,live_allocation_limit=12*1024**3,cumulative_allocation_limit=256*1024**3,
            max_single_buffer=256*1024**2,free_reserve=5*1024**3)
HYPER=dict(optimizer='tinygrad.AdamW',learning_rate=3.5e-6,beta1=.9,beta2=.999,epsilon=1e-8,weight_decay=.01,
           fused=False,max_committed_updates=1,query_chunk=128,objective='.5*prompt + .5*reasoning + mean(three connected expert residuals)')
KIND='joint-resource-qualification-v1'


def module(name,path):
  spec=importlib.util.spec_from_file_location(name,path);value=importlib.util.module_from_spec(spec)
  sys.modules[name]=value;spec.loader.exec_module(value);return value
def sha(path):
  with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
def read(path):return json.loads(Path(path).read_text())
S=module('objective_qualification_state',HERE/'agent-distill-fit-state.py')
O=module('objective_qualification_math',HERE/'agent-distill-objective-core.py')
C=module('objective_qualification_targets',HERE/'agent-distill-objective-capture.py')


def control(path,digest):
  path=Path(path).resolve()
  if not path.is_relative_to(WORK) or any(w in str(path) for w in C.FORBIDDEN):raise ValueError('Unapproved aligned control path')
  if sha(path)!=digest:raise ValueError('Aligned report checksum changed')
  report=read(path)
  if not report.get('complete') or not report.get('gpu_executed') or report.get('synthetic_fixture'):raise ValueError('Completed actual aligned capture required')
  row=report['record'];target=row['targets']
  if row.get('synthetic_fixture'):raise ValueError('Synthetic record is not an actual aligned capture')
  if row['id']!='train-command-0' or row['split']!='train' or row['route_id']!=C.ROUTES['train']:raise ValueError('This qualification uses only the legacy control')
  if report['input_manifest']['sha256']!='83da7c2dc65a50185d052de07c30698ade2b1818103d90d69ab394f8f600f3b6':raise ValueError('Control is not the approved legacy input')
  for filename,expected in [('agent-distill-objective-core.py','3ad258bf1236b31b397314b7caab9fb0b941e60d8afe3db3750939a6dec39a84'),
      ('agent-distill-objective-capture.py','7b045707be2ebec71ead5b4d6f881405e24939ed884532c38a8e36960eb213a6'),
      ('agent-distill-objective-position-run.py','248e493b935b2965b81afedf29ea0ca3b45353fce1e2a8ea692a5baf87c1e520')]:
    if sha(HERE/filename)!=expected or report['source_sha256'].get(str(HERE/filename))!=expected:raise ValueError('Changed reviewed objective/capture')
  if not target['complete'] or target['capacity']!=2304 or not 0<target['prompt_length']<target['valid_length']<=2048:
    raise ValueError('This first qualification requires the complete 2048-row extent')
  if len(target['expert'])!=3:raise ValueError('Missing actual Euler targets')
  context=dict(kind=KIND,aligned_report_sha256=digest,hyperparameters=HYPER,
    objective_sha256=sha(HERE/'agent-distill-objective-core.py'),seeds={name:value[1] for name,value in SEEDS.items()},
    training_cohort_sample=False,checkpoint_selection_allowed=False)
  return row,context


def seed_arrays():
  import numpy as np
  result={}
  for role,(path,digest) in SEEDS.items():
    if sha(path)!=digest:raise ValueError('Selected starting checkpoint changed')
    with np.load(path,allow_pickle=False) as archive:values={name:archive[name].copy() for name in archive.files}
    expected=50335744 if role=='language' else 7078912
    if len(values)!=8 or sum(v.size for v in values.values())!=expected:raise ValueError('Wrong student geometry')
    if any(v.dtype!=np.float32 or not np.isfinite(v).all() for v in values.values()):raise ValueError('Invalid FP32 starting masters')
    result.update({role+'.'+name:value for name,value in sorted(values.items())})
  return result


def optimizer(weights):
  from tinygrad.nn.optim import AdamW
  return AdamW(list(weights.values()),lr=3.5e-6,b1=.9,b2=.999,eps=1e-8,weight_decay=.01,fused=False)


def construct(values,device):
  from tinygrad import Tensor
  return {name:Tensor(value.copy(),device=device).is_param_(True).realize() for name,value in sorted(values.items())}


def checkpoint_space(root):
  if shutil.disk_usage(root).free<4*1024**3+700*1024**2:raise ValueError('Keep checkpoint workspace reserve')


def run_step(values,loader,root,context,device,phase,*,resume=None,fixture=False):
  import numpy as np
  from tinygrad import Tensor,Device
  from tinygrad.helpers import Context
  root=Path(root);weights=construct(values,device);del values
  shapes={k:v.shape for k,v in weights.items()};opt=optimizer(weights)
  if not fixture and (len(weights)!=16 or sum(v.numel() for v in weights.values())!=57414656):raise ValueError('Wrong complete joint student')
  metadata=dict(step=0,kind=KIND,training_cohort_sample=False,checkpoint_selection_allowed=False)
  if resume:
    arrays,metadata=S.load(Path(resume),context,shapes)
    if metadata['step'] not in (0,1) or metadata['kind']!=KIND:raise ValueError('Qualification update budget exhausted')
    S.restore(weights,opt,arrays);del arrays;gc.collect()
  phase('initial-complete-checkpoint')
  checkpoint_space(root)
  S.save(root,S.snapshot(weights,opt),metadata,context,shapes,initial_recovery=bool(resume))
  start=metadata['step'];result=dict(updates_this_run=0,starting_step=start,resource_qualification_only=True,
    training_cohort_sample=False,checkpoint_selection_allowed=False,quality_qualified=False)
  if start==0:
    seed=root/'seed-state';seed.mkdir()
    os.link(root/'checkpoint-0000.npz',seed/'checkpoint-0000.npz')
    S.atomic_json(seed/'latest.json',read(root/'latest.json'));S.atomic_json(seed/'journal.json',read(root/'journal.json'))
    result['preserved_initial_state']=str(seed)
    phase('aligned-inputs-and-teacher-kv-diagnostic');data=loader(device)
    l={k.removeprefix('language.'):v for k,v in weights.items() if k.startswith('language.')}
    e={k.removeprefix('expert.'):v for k,v in weights.items() if k.startswith('expert.')}
    options=dict(head_dim=4,minimum_prefill=8,query_chunk=128) if fixture else dict(query_chunk=128)
    diagnostic=[]
    with Context(TRAINING=0):
      for step in data['expert']:
        y=O.expert(e,step['x'],step['cos'],step['sin'],data['language']['k'].detach(),data['language']['v'].detach(),step['mask'],
                   head_dim=options.get('head_dim',128))
        diagnostic.append(float((y-step['y']).square().mean().item()))
      del y
    if not np.isfinite(diagnostic).all():raise ValueError('Nonfinite teacher-KV diagnostic')
    result['expert_detached_teacher_kv']=dict(mean=float(np.mean(diagnostic)),euler_losses=diagnostic,in_optimized_loss=False,
      description='Same pre-update expert weights and Euler inputs, original teacher KV; separate forward-only diagnostic')
    phase('one-connected-joint-forward-backward')
    with Context(TRAINING=1):
      opt.zero_grad();losses,out=O.objective(l,e,data,**options)
      if not fixture and out['language_extent']!=2048:raise ValueError('Changed full language extent')
      parameters=list(weights.values());grads=losses['total'].gradient(*parameters,out['context_k'],out['context_v'])
      Tensor.realize(*losses.values(),*out['language'],*out['expert'],*grads)
      result['components']={name:float(value.item()) for name,value in losses.items()}
      if not np.isfinite(list(result['components'].values())).all():raise ValueError('Nonfinite objective')
      for tensor in (*out['language'],out['context_k'],out['context_v'],*out['expert']):
        if not bool(tensor.isfinite().all().item()):raise ValueError('Nonfinite joint forward')
      norms={}
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
      result.update(parameter_gradient_norms=norms,expert_to_student_context_gradient_norms=context_norms,
                    masked_context_gradient_exactly_zero=True,connected_joint_forward_count=1,connected_joint_backward_count=1)
      phase('one-native-adamw-update');S.intent(root,0);opt.step();Device[device].synchronize()
      result['updates_this_run']=1;metadata=dict(metadata,step=1)
      opt.zero_grad()
    del grads,losses,out,parameters,l,e,data,gradient,weight,tensor,step;gc.collect()
    phase('complete-step-one-checkpoint');checkpoint_space(root);S.save(root,S.snapshot(weights,opt),metadata,context,shapes)
  phase('fresh-optimizer-recovery')
  arrays,committed=S.load(root,context,shapes)
  del weights,opt;gc.collect()
  recovered=construct({name:np.zeros(shape,np.float32) for name,shape in shapes.items()},device)
  recovered_opt=optimizer(recovered);S.restore(recovered,recovered_opt,arrays)
  checked=0
  for index,(name,weight) in enumerate(recovered.items()):
    for prefix,tensor in (('weight',weight),('moment1',recovered_opt.m[index]),('moment2',recovered_opt.v[index])):
      actual=tensor.numpy();np.testing.assert_array_equal(actual.view(np.uint32),arrays[prefix+'/'+name].view(np.uint32));checked+=1
  for name,tensor in (('beta1_power',recovered_opt.b1_t),('beta2_power',recovered_opt.b2_t),('learning_rate',recovered_opt.lr)):
    np.testing.assert_array_equal(tensor.numpy().view(np.uint32),arrays[name].view(np.uint32));checked+=1
  assert checked==3*len(shapes)+3==51 and committed['step']==1
  result.update(complete=True,committed_step=1,recovery_arrays=checked,recovery_bit_exact=True,
    checkpoint_sha256=sha(root/'checkpoint-0001.npz'),checkpoint_bytes=(root/'checkpoint-0001.npz').stat().st_size,
    journal=read(root/'journal.json'),metadata=committed,additional_updates_after_recovery=0)
  return result


def estimate():
  value=O.memory_assessment(2047);snapshot=value['checkpoint_weights_and_two_moments_bytes']
  value.update(recovery_tensor_count=51,parameter_tensor_count=16,shared_optimizer_scalars=3,
    checkpoint_peak_payload_bytes=2*snapshot,checkpoint_final_payload_bytes=2*snapshot,
    host_snapshot_payload_bytes=snapshot,largest_streamed_recovery_array_bytes=value['largest_parameter_bytes'],
    checkpoint_device_to_host_transfer_bytes=3*snapshot,checkpoint_host_to_device_transfer_bytes=snapshot,
    compulsory_checkpoint_transfer_bytes=4*snapshot,optional_resume_host_to_device_transfer_bytes=snapshot,
    seed_and_fresh_recovery_weight_upload_bytes=2*value['master_bytes'],
    cumulative_qualification_note='Two snapshots and final verification download three states; recovery uploads one state. Optional resume uploads another. These logical transfer bytes are not allocator counts. Seed/input transfer, optimizer and graph allocations add work. Actual live/cumulative counts come only from the unchanged KFD ledger; no cumulative reset.',
    workspace_free_required_bytes=6*1024**3,workspace_reserve_bytes=4*1024**3)
  return value
