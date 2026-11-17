"""Bounded 25-update expert pilot. Default is CPU preflight; no targets are selected implicitly."""
import argparse
import copy
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('agent_distill_expert_block_train',HERE/'agent-distill-expert-block.py')
B=importlib.util.module_from_spec(spec);sys.modules[spec.name]=B;spec.loader.exec_module(B)
S=B.module('agent_distill_expert_recovery',HERE/'agent-distill-fit-state.py')
A=B.module('agent_distill_expert_selection',HERE/'agent-distill-continue-admit.py')


def optimizer(block):
  from tinygrad.nn.optim import AdamW
  return AdamW(list(block.weights.values()),lr=3.5e-6,b1=.9,b2=.999,eps=1e-8,weight_decay=.01,fused=False)


def fit(block,records,loader,root,context,device,*,resume=None,stop_after=25):
  """Reusable loop. CPU controls may stop early; real pilot always stops at update25."""
  import numpy as np
  from tinygrad import Tensor,Device
  from tinygrad.helpers import Context
  assert 1<=stop_after<=25
  root=Path(root);root.mkdir(exist_ok=True)
  opt=optimizer(block);shapes={k:v.shape for k,v in block.weights.items()}
  train=[r for r in records if r['split']=='train'];validation=[r for r in records if r['split']=='validation']
  assert train and validation and {r['route_id'] for r in train}.isdisjoint(r['route_id'] for r in validation)
  def objective(row):
    data=B.inputs(loader(row),device)
    return B.loss(block,data),data
  def validate(step):
    losses=[]
    with Context(TRAINING=0):
      for row in validation:
        objective_value,data=objective(row);value=float(objective_value.item())
        assert np.isfinite(value) and value>=0;losses.append(value)
        assert all(value.grad is None and not value.is_param for value in data.values())
    return dict(step=step,mean=float(np.mean(losses)),samples=[r['id'] for r in validation],losses=losses)
  if resume:
    arrays,metadata=S.load(Path(resume),context,shapes);metadata=copy.deepcopy(metadata)
    assert metadata['step']<stop_after and metadata['best_step']==0
    assert [item['step'] for item in metadata['validation']]==[0]
    _,best=A.read_best(resume,metadata,shapes)
    S.restore(block.weights,opt,arrays)
    A.copy_best(resume,root,metadata,S)
  else:
    initial=validate(0)
    metadata=dict(step=0,best_step=0,best_validation=initial['mean'],validation=[initial])
    A.commit_best(root,block.weights,metadata,S)
  S.save(root,S.snapshot(block.weights,opt),metadata,context,shapes,initial_recovery=bool(resume))
  start=metadata['step'];losses=[]
  for step in range(start+1,stop_after+1):
    row=train[(step-1)%len(train)]
    S.intent(root,step-1)
    with Context(TRAINING=1):
      opt.zero_grad();objective_value,data=objective(row)
      gradients=objective_value.gradient(*block.weights.values());Tensor.realize(*gradients)
      value=float(objective_value.item());assert np.isfinite(value)
      for weight,gradient in zip(block.weights.values(),gradients,strict=True):
        assert bool(gradient.isfinite().all().item());weight.grad=gradient
      assert all(v.grad is None and not v.is_param for v in data.values())
      opt.step();Device[device].synchronize()
    metadata['step']=step
    losses.append(dict(step=step,sample=row['id'],loss=value))
    if step%25==0:
      result=validate(step);metadata['validation'].append(result)
      if result['mean']<metadata['best_validation']:
        metadata.update(best_step=step,best_validation=result['mean']);A.commit_best(root,block.weights,metadata,S)
    S.save(root,S.snapshot(block.weights,opt),metadata,context,shapes)
    S.atomic_json(root/'progress.json',dict(committed_step=step,best_step=metadata['best_step'],validation=metadata['validation']))
  return dict(complete=metadata['step']==25,starting_step=start,committed_step=metadata['step'],
              updates_this_run=metadata['step']-start,metadata=metadata,training_losses=losses,
              best_sha256=B.sha(root/'best.npz'),conditioning='full detached teacher KV',quality_qualified=False)


def receipt(args,manifest):
  if args.admission is None or args.reviewed_admission_sha is None:raise ValueError('Parent KFD1 admission is required')
  if B.sha(args.admission)!=args.reviewed_admission_sha:raise ValueError('Admission hash changed')
  value=B.read(args.admission)
  assert value['parent_reviewed'] is True and value['no_reset_or_owner_displacement'] is True
  assert value['target']=='KFD:1+AMD:LLVM:gfx1100' and value['selected_pci']=='0000:0c:00.0' and value['selected_gpu_id']==27211
  assert value['manifest_sha256']==manifest['manifest_sha256'] and value['no_memory_planner'] in (0,1)
  checked=value['cpu_check'];assert B.sha(checked['path'])==checked['sha256']
  cpu=B.read(checked['path']);assert cpu['all_passed'] is True and cpu['gpu_executed'] is False
  for path in (HERE/'agent-distill-expert-block.py',HERE/'agent-distill-expert-train.py',HERE/'agent-distill-fit-state.py'):
    assert cpu['source_sha256'][str(path)]==B.sha(path),path
  availability=value['availability_review'];assert B.sha(availability['path'])==availability['sha256']
  reviewed=B.read(availability['path'])
  assert 0<=time.time_ns()-reviewed['captured_unix_ns']<=60_000_000_000
  assert reviewed['selected_node']['pci']==value['selected_pci'] and reviewed['selected_node']['gpu_id']==value['selected_gpu_id']
  assert reviewed['kfd_pid_coverage_complete'] and not reviewed['observed_selected_kfd_owners']
  assert reviewed['vram_free_bytes']>=B.BOUNDS['live_allocation_limit']+B.BOUNDS['free_reserve']
  return value


def child(args,records,manifest,admission):
  # Same KFD identity, availability and allocation guard as the previous language trainer.
  K=B.module('agent_distill_expert_kfd_guard',HERE/'agent-distill-kfd1-common.py')
  K.no_gpu_fds();availability=K.passive_availability();K.require_no_observed_conflict(availability)
  assert availability['vram_free_bytes']>=B.BOUNDS['live_allocation_limit']+B.BOUNDS['free_reserve']
  root=args.output_dir.resolve();S.atomic_json(root/'availability-before-open.json',availability)
  def guard(event,values):
    if event=='open' and isinstance(values[0],(str,bytes)):
      path=os.fsdecode(values[0])
      if path.startswith(('/dev/dri/','/dev/kfd','/dev/nvidia','/dev/bus/usb/')):
        assert path in {'/dev/kfd','/dev/dri/renderD129'}
  sys.addaudithook(guard)
  from tinygrad import Device
  from tinygrad.helpers import DEV,MAX_BUFFER_SIZE,PROFILE
  from tinygrad.runtime.ops_amd import KFDIface,PMC,SQTT
  assert not Device._opened_devices and repr(DEV)==admission['target'] and DEV.target('AMD').indices=='1'
  assert not PMC.value and not SQTT.value and not PROFILE.value and MAX_BUFFER_SIZE.value==B.BOUNDS['max_single_buffer']
  old_alloc,old_free=KFDIface.alloc,KFDIface.free
  live={};budget=dict(live=0,peak=0,cumulative=0,calls=0)
  def alloc(self,size,*args,**kwargs):
    assert self.gpu_id==27211 and Path(self.dev_sysfs_path).resolve().name=='0000:0c:00.0'
    assert 0<size<=B.BOUNDS['max_single_buffer'] and budget['live']+size<=B.BOUNDS['live_allocation_limit']
    assert budget['cumulative']+size<=B.BOUNDS['cumulative_allocation_limit']
    value=old_alloc(self,size,*args,**kwargs);key=int(value.meta.handle);assert key not in live
    live[key]=size;budget['live']+=size;budget['cumulative']+=size;budget['calls']+=1;budget['peak']=max(budget['peak'],budget['live'])
    return value
  def free(self,value):
    old_free(self,value);key=int(value.meta.handle)
    if key in live:budget['live']-=live.pop(key)
  report=dict(complete=False,gpu_executed=False,allocation_budget=budget,quality_qualified=False,
              manifest=manifest,hyperparameters=B.HYPER,bounds=B.BOUNDS,
              source_sha256={str(p):B.sha(p) for p in HERE.glob('agent-distill-expert-*.py')})
  KFDIface.alloc,KFDIface.free=alloc,free
  try:
    dev=Device['AMD'];selected=K.selected_node(K.passive_topology())
    identity=dict(requested_kfd_index=1,logical_device_id=dev.device_id,filtered_kfd_nodes=list(KFDIface.gpus),
      interface=type(dev.iface).__name__,gpu_id=dev.iface.gpu_id,pci=Path(dev.iface.dev_sysfs_path).resolve().name,
      render_minor=dev.iface.props['drm_render_minor'],arch=dev.arch,renderer_target=repr(dev.renderer.target),
      is_usb=dev.is_usb,native_pci=dev.is_am(),can_recover=dev.can_recover,allocator=type(dev.allocator).__name__)
    K.validate_opened_identity(identity,selected);report.update(gpu_executed=True,actual_device=identity)
    weights=B.seed();seed_hash=__import__('hashlib').sha256()
    for name,value in sorted(weights.items()):seed_hash.update(name.encode());seed_hash.update(value.tobytes())
    block=B.Block(weights,'AMD')
    context=dict(manifest_sha256=manifest['manifest_sha256'],hyperparameters=B.HYPER,seed_sha256=seed_hash.hexdigest(),
                 block_sha256=B.sha(HERE/'agent-distill-expert-block.py'),synthetic_only=False)
    report.update(fit(block,records,B.load,root,context,'AMD',resume=args.resume_from))
  except BaseException as error:
    report['failure']=dict(type=type(error).__name__,message=str(error));raise
  finally:
    KFDIface.alloc,KFDIface.free=old_alloc,old_free
    report['allocator_hooks_restored']=KFDIface.alloc is old_alloc and KFDIface.free is old_free
    S.atomic_json(root/'report.json',report)


def main():
  parser=argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--train-report',type=Path,action='append',default=[])
  parser.add_argument('--validation-report',type=Path,action='append',default=[])
  parser.add_argument('--output-dir',type=Path)
  parser.add_argument('--admission',type=Path);parser.add_argument('--reviewed-admission-sha')
  parser.add_argument('--resume-from',type=Path)
  parser.add_argument('--execute-training',action='store_true');parser.add_argument('--child',action='store_true')
  args=parser.parse_args()
  if not args.train_report or not args.validation_report:
    if args.execute_training:raise ValueError('Training blocked: independent TRAIN and VAL targets are required')
    print(json.dumps(dict(ready=False,gpu_executed=False,blocked=['Explicit completed TRAIN and independent VAL target reports',
      'Parent-selected guarded KFD1 launch'],hyperparameters=B.HYPER,bounds=B.BOUNDS),indent=2));return
  records,manifest=B.target_headers(args.train_report,args.validation_report);B.validate_targets(records)
  if not args.execute_training:
    print(json.dumps(dict(ready=True,gpu_executed=False,teacher_euler_examples=len(records),manifest=manifest),indent=2));return
  admission=receipt(args,manifest)
  assert args.output_dir is not None and args.output_dir.resolve().is_relative_to(B.ROOT)
  if args.resume_from:assert args.resume_from.resolve().is_relative_to(B.ROOT) and args.resume_from.resolve()!=args.output_dir.resolve()
  if args.child:
    assert os.environ.get('AGENT_DISTILL_EXPERT_PARENT_PID')==str(os.getppid())
    return child(args,records,manifest,admission)
  if args.output_dir.exists():raise ValueError('Use a new output directory')
  root=args.output_dir.resolve()
  with Path('/tmp/opencode/alpamayo-chestnut-speed.lock').open('a') as lock:
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    root.mkdir(parents=True);(root/'tmp').mkdir()
    # Reuse the previously qualified environment builder; no driver/dependency changes.
    old=B.module('agent_distill_expert_old_training_environment',HERE/'agent-distill-fit-train.py')
    p=dict(B.BOUNDS,target=admission['target'])
    env=old.environment(root,p,admission);env['AGENT_DISTILL_EXPERT_PARENT_PID']=str(os.getpid())
    command=[str(B.BASE/'upstream/.venv/bin/python'),str(Path(__file__).resolve()),*sys.argv[1:],'--child']
    S.atomic_json(root/'parent-invocation.json',dict(command=command,manifest=manifest,admission_sha256=args.reviewed_admission_sha))
    started=time.monotonic();timed_out=False
    with (root/'run.log').open('w') as stream:
      process=subprocess.Popen(command,env=env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
      try:code=process.wait(timeout=900)
      except subprocess.TimeoutExpired:
        timed_out=True;os.killpg(process.pid,signal.SIGTERM)
        try:code=process.wait(timeout=5)
        except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);code=process.wait()
    S.atomic_json(root/'wall.json',dict(seconds=time.monotonic()-started,returncode=code,timed_out=timed_out))
    assert code==0 and not timed_out and B.read(root/'report.json')['complete']


if __name__=='__main__':main()
