"""Fixed160 original-seed group12 fit. CPU preflight is the default; parent owns GPU execution."""
import argparse
import fcntl
import gc
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
spec=importlib.util.spec_from_file_location('vision_cohort_fit_common_train',HERE/'agent-distill-vision-cohort-fit-common.py')
C=importlib.util.module_from_spec(spec);sys.modules[spec.name]=C;spec.loader.exec_module(C)
S=C.module('vision_cohort_fit_state_train',HERE/'agent-distill-vision-cohort-fit-state.py')


def context(args):return dict(plan_sha256=args.plan_sha,hyperparameters=C.HYPER,mode=args.mode)


def validation(engine,rows,step):
  import numpy as np
  engine.ledger.phase('validation-'+str(step));losses=[];scenes={}
  for row in rows:
    value=engine.evaluate(row)
    if not np.isfinite(value):raise ValueError('Nonfinite validation loss')
    losses.append(value);scenes.setdefault(row['scene'],[]).append(value)
  if len(scenes)!=8 or any(len(v)!=10 for v in scenes.values()):raise ValueError('Validation requires8 complete scenes')
  per_scene={key:float(np.mean(value)) for key,value in scenes.items()}
  return dict(step=step,mean=float(np.mean(list(per_scene.values()))),per_scene=per_scene,
    per_image=losses,images=[r['id'] for r in rows])


def validate_metadata(metadata,dataset):
  import numpy as np
  step=metadata['step'];assert 0<=step<=160
  expected_steps=[0,160] if step==160 else [0]
  assert [item['step'] for item in metadata['validation']]==expected_steps
  for item in metadata['validation']:
    assert item['images']==[r['id'] for r in dataset['validation']] and len(item['per_image'])==80
    assert np.isfinite(item['per_image']).all()
    per_scene={f'validation-scene-{i:02d}':float(np.mean(item['per_image'][i*10:(i+1)*10])) for i in range(8)}
    assert item['per_scene']==per_scene and item['mean']==float(np.mean(list(per_scene.values())))
  best=metadata['validation'][0]
  if step==160 and metadata['validation'][1]['mean']<best['mean']:best=metadata['validation'][1]
  assert metadata['best_step']==best['step'] and metadata['best_validation']==best['mean']
  if step:assert metadata['last_training_image']==dataset['train'][step-1]['id']


def resume_carry(root,args,shapes,dataset):
  arrays,metadata=S.load(root,context(args),shapes);validate_metadata(metadata,dataset)
  assert 0<=metadata['step']<160 and metadata['best_step']==0
  S.check_best(root,metadata,shapes)
  report=C.read(root/'report.json');wall=C.read(root/'wall.json');parent=C.read(root/'parent-invocation.json')
  assert report['execution_finished'] and not wall['timed_out'] and report['committed_step']==metadata['step']
  assert report['plan_sha256']==args.plan_sha and report['mode']=='fit'
  cumulative=report['allocation_budget']['requested_bytes']
  assert cumulative>=metadata['allocated_bytes_at_commit']
  seconds=parent.get('previous_total_wall_seconds',0)+wall['seconds']
  assert cumulative<C.LIMITS['cumulative_allocation_limit'] and seconds<900
  return arrays,metadata,dict(requested_bytes=cumulative,previous_total_wall_seconds=seconds)


def run_fit(args,plan,dataset,engine,*,resume=None,after_commit=None):
  """Actual training entrypoint. CPU tests inject an explicitly marked arithmetic engine; CLI never does."""
  import numpy as np
  root=args.output_dir;started=getattr(engine,'started',time.monotonic());shapes=engine.shapes;ctx=context(args)
  report=dict(complete=False,execution_finished=False,gpu_executed=False,cpu_integration_mock=engine.cpu_mock,
    plan_sha256=args.plan_sha,mode=args.mode,patches=720,real_optimizer_updates=0,committed_step=0,
    allocation_budget=engine.ledger.report,quality_qualified=False,throughput_qualified=False,
    driving_model_modified=False,dataset=plan['dataset'],actual_device=engine.identity,
    allocator_guards_retained_through_finalizers=True,hyperparameters=C.HYPER)
  try:
    forward_start=time.monotonic();forward_bytes=engine.ledger.report['requested_bytes']
    if args.mode=='qualify':
      loss=engine.evaluate(dataset['train'][0]);assert np.isfinite(loss)
      metadata=dict(step=0,qualification_train_mse=loss)
    elif resume:
      old,arrays,metadata=resume;S.restore(engine.weights,engine.optimizer,arrays)
      previous=S.check_best(old,metadata,shapes)
      shutil.copyfile(old/'best.npz',root/'best.npz');S.atomic_json(root/'best.json',previous)
    else:
      item=validation(engine,dataset['validation'],0)
      metadata=dict(step=0,best_validation=item['mean'],best_step=0,validation=[item]);S.save_best(root,engine.weights,metadata)
    report['gpu_executed']=not engine.cpu_mock
    forward_seconds=time.monotonic()-forward_start;forward_bytes=engine.ledger.report['requested_bytes']-forward_bytes
    def commit(initial=False):
      engine.ledger.phase('checkpoint-'+str(metadata['step']));C.storage(root)
      arrays=S.snapshot(engine.weights,engine.optimizer);metadata['allocated_bytes_at_commit']=engine.ledger.report['requested_bytes']
      S.save(root,arrays,metadata,ctx,shapes,initial_recovery=initial and bool(resume))
      report['committed_step']=metadata['step']
    commit(initial=True);starting=metadata['step'];report['starting_checkpoint_step']=starting
    initial_seconds=time.monotonic()-started
    final=1 if args.mode=='qualify' else 160
    for step in range(starting+1,final+1):
      update_start=time.monotonic();update_bytes=engine.ledger.report['requested_bytes'];engine.ledger.phase('update-'+str(step))
      S.intent(root,step-1);row=dataset['train'][step-1]
      assert row['scene']==f'train-scene-{(step-1)%16:02d}' and row['image_index']==(step-1)//16
      loss=engine.update(row);assert np.isfinite(loss)
      report['real_optimizer_updates']+=1;report['gpu_executed']=not engine.cpu_mock
      metadata.update(step=step,last_training_image=row['id'])
      if args.mode=='fit' and step==160:
        item=validation(engine,dataset['validation'],160);metadata['validation'].append(item)
        if item['mean']<metadata['best_validation']:
          metadata['best_validation']=item['mean'];metadata['best_step']=160;S.save_best(root,engine.weights,metadata)
      commit();report.update(last_training_image=row['id'],last_training_mse=loss)
      if args.mode=='qualify':
        update_seconds=time.monotonic()-update_start;update_bytes=engine.ledger.report['requested_bytes']-update_bytes
        report['measurements']=dict(setup_seconds=initial_seconds,forward_seconds=forward_seconds,forward_bytes=forward_bytes,
          update_checkpoint_seconds=update_seconds,update_checkpoint_bytes=update_bytes)
        report['projected_fit']=dict(cumulative_bytes=engine.ledger.report['requested_bytes']+160*update_bytes+160*forward_bytes+16*plan['parameter_count'],
          seconds=initial_seconds+160*update_seconds+160*forward_seconds+45,
          basis='Cold full720 update/checkpoint;160 validation forwards; two best snapshots and45seconds margin')
      else:
        validate_metadata(metadata,dataset);report.update(best_step=metadata['best_step'],validation=metadata['validation'])
      S.atomic_json(root/'report.json',report)
      if after_commit:after_commit(step,report)  # CPU fixture only; absent from production CLI.
    assert report['real_optimizer_updates']+starting==final
    report.update(complete=True,selection='qualification has no checkpoint selection' if args.mode=='qualify' else
      'strict lower balanced validation loss at step160; ties keep original step0')
  finally:
    report.update(seconds=time.monotonic()-started,execution_finished=True)
    S.atomic_json(root/'report.json',report)
  return report


class GPUEngine:
  cpu_mock=False
  def __init__(self,K,carry):
    self.started=time.monotonic()
    import numpy as np
    from tinygrad import Tensor,Device
    from tinygrad.helpers import DEV,MAX_BUFFER_SIZE,PROFILE,USE_TC,ALLOW_TF32,NO_MEMORY_PLANNER,LRU
    from tinygrad.runtime.ops_amd import KFDIface,PMC,SQTT
    from tinygrad.nn.optim import AdamW
    assert not Device._opened_devices and repr(DEV)==C.TARGET and DEV.target('AMD').indices=='1'
    assert not PMC.value and not SQTT.value and not PROFILE.value and MAX_BUFFER_SIZE.value==C.LIMITS['max_single_buffer']
    assert USE_TC.value==ALLOW_TF32.value==0 and NO_MEMORY_PLANNER.value==1 and LRU.value==0
    L=C.module('vision_cohort_fit_complete_ledger',C.LEDGER);self.ledger=L.Ledger(C.LIMITS,K.passive_availability)
    self.ledger.report['requested_bytes']=carry.get('requested_bytes',0)
    self.ledger.report['resumed_cumulative_bytes']=self.ledger.report['requested_bytes']
    KFDIface.alloc=self.ledger.alloc(KFDIface.alloc);KFDIface.free=self.ledger.free(KFDIface.free);KFDIface.unmap=self.ledger.unmap(KFDIface.unmap)
    self.ledger.phase('device-and-original-seed');self.device=Device['AMD'];dev=self.device
    self.identity=dict(requested_kfd_index=1,logical_device_id=dev.device_id,filtered_kfd_nodes=list(KFDIface.gpus),
      interface=type(dev.iface).__name__,gpu_id=dev.iface.gpu_id,pci=Path(dev.iface.dev_sysfs_path).resolve().name,
      render_minor=dev.iface.props['drm_render_minor'],arch=dev.arch,renderer_target=repr(dev.renderer.target),
      is_usb=dev.is_usb,native_pci=dev.is_am(),can_recover=dev.can_recover,allocator=type(dev.allocator).__name__)
    K.validate_opened_identity(self.identity,K.selected_node(K.passive_topology()))
    self.B=C.original().graph();arrays,_=C.original().seeds();self.block=self.B.Block(arrays,'AMD')
    self.weights=self.block.weights;self.shapes=self.B.shapes()
    self.optimizer=AdamW(list(self.weights.values()),lr=3.5e-5,b1=.9,b2=.999,eps=1e-8,weight_decay=.01,fused=False)

  def objective(self,row):
    from tinygrad import Tensor
    values=C.image_arrays(row)
    data={k:Tensor(v,device='AMD').is_param_(False).realize() for k,v in values.items()}
    return self.B.mse(self.block(data['x']),data['y']),data

  def evaluate(self,row):
    from tinygrad.helpers import Context
    with Context(TRAINING=0):
      loss,data=self.objective(row);value=float(loss.item())
      assert all(v.grad is None and not v.is_param for v in data.values())
    self.device.synchronize();return value

  def update(self,row):
    import numpy as np
    from tinygrad import Tensor
    from tinygrad.helpers import Context
    with Context(TRAINING=1):
      self.optimizer.zero_grad();loss,data=self.objective(row)
      grads=loss.gradient(*self.weights.values());Tensor.realize(*grads)
      value=float(loss.item());assert np.isfinite(value)
      for parameter,gradient in zip(self.weights.values(),grads,strict=True):
        assert bool(gradient.isfinite().all().item());parameter.grad=gradient
      assert float(self.weights['mlp.fc2.weight'].grad.square().sum().item())>0
      assert all(v.grad is None and not v.is_param for v in data.values())
      self.optimizer.step();self.device.synchronize()
    del grads,gradient,parameter,loss,data;self.optimizer.zero_grad();gc.collect();return value


def availability(args):
  assert args.availability_snapshot and args.reviewed_availability_sha
  assert args.availability_snapshot.resolve().is_relative_to(C.ROOT)
  assert C.sha(args.availability_snapshot)==args.reviewed_availability_sha
  a=C.read(args.availability_snapshot);assert 0<=time.time_ns()-a['captured_unix_ns']<=60_000_000_000
  K=C.module('vision_cohort_fit_kfd',C.KFD);K.require_no_observed_conflict(a)
  assert a['vram_free_bytes']>=C.LIMITS['live_allocation_limit']+C.LIMITS['free_reserve'];return K


def qualification(args):
  assert args.qualification and args.qualification_sha and C.sha(args.qualification)==args.qualification_sha
  q=C.read(args.qualification);wall=C.read(args.qualification.parent/'wall.json')
  assert q['complete'] and q['execution_finished'] and q['gpu_executed'] and not q['cpu_integration_mock']
  assert q['mode']=='qualify' and q['plan_sha256']==args.plan_sha and q['patches']==720
  assert q['real_optimizer_updates']==q['committed_step']==1 and wall['returncode']==0 and not wall['timed_out']
  assert q['allocator_guards_retained_through_finalizers'] and q['projected_fit']['seconds']<900
  assert q['projected_fit']['cumulative_bytes']<C.LIMITS['cumulative_allocation_limit']


def child(args,plan,dataset):
  K=availability(args);fresh=K.passive_availability();K.require_no_observed_conflict(fresh)
  assert fresh['vram_free_bytes']>=C.LIMITS['live_allocation_limit']+C.LIMITS['free_reserve']
  C.write(args.output_dir/'availability-before-open.json',fresh)
  resume=None;carry={}
  if args.resume_from:
    arrays,metadata,carry=resume_carry(args.resume_from,args,C.original().graph().shapes(),dataset)
    resume=(args.resume_from,arrays,metadata)
  def hardware_guard(event,values):
    if event=='open' and isinstance(values[0],(str,bytes)):
      path=os.fsdecode(values[0])
      if path.startswith(('/dev/dri','/dev/kfd','/dev/nvidia','/dev/bus/usb')):assert path in ('/dev/kfd','/dev/dri/renderD129'),path
  sys.addaudithook(hardware_guard)
  return run_fit(args,plan,dataset,GPUEngine(K,carry),resume=resume)


def main():
  parser=argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--prepare',action='store_true');parser.add_argument('--capture-set',type=Path);parser.add_argument('--capture-set-sha')
  parser.add_argument('--plan',type=Path,required=True);parser.add_argument('--plan-sha')
  parser.add_argument('--mode',choices=('qualify','fit'),default='qualify');parser.add_argument('--execute-gpu',action='store_true')
  parser.add_argument('--child',action='store_true');parser.add_argument('--output-dir',type=Path);parser.add_argument('--resume-from',type=Path)
  parser.add_argument('--snapshot-availability',action='store_true');parser.add_argument('--availability-snapshot',type=Path)
  parser.add_argument('--reviewed-availability-sha');parser.add_argument('--qualification',type=Path);parser.add_argument('--qualification-sha')
  args=parser.parse_args()
  if sys.flags.optimize:raise ValueError('Assertions are required')
  if not args.child:os.environ.update(DEV='CPU',CUDA_VISIBLE_DEVICES='',HIP_VISIBLE_DEVICES='',ROCR_VISIBLE_DEVICES='')
  if args.prepare:
    assert not args.execute_gpu and not args.child and args.capture_set and args.capture_set_sha
    C.prepare(args.capture_set,args.capture_set_sha,args.plan)
    print(json.dumps(dict(plan=str(args.plan),sha256=C.sha(args.plan),gpu_executed=False)));return
  assert args.plan_sha;C.no_devices();plan,dataset=C.verify_plan(args.plan,args.plan_sha,all_arrays=not args.execute_gpu)
  if args.snapshot_availability:
    assert not args.execute_gpu and not args.child and args.availability_snapshot;C.storage(args.availability_snapshot)
    K=C.module('vision_cohort_fit_snapshot',C.KFD);C.write(args.availability_snapshot,K.passive_availability());C.no_devices()
    print(json.dumps(dict(snapshot=str(args.availability_snapshot),sha256=C.sha(args.availability_snapshot))));return
  if not args.execute_gpu:
    assert not args.child;C.no_devices()
    print(json.dumps(dict(preflight_passed=True,gpu_executed=False,train_images=160,validation_images=80,scenes=24)));return
  assert args.output_dir;args.output_dir=args.output_dir.resolve();C.storage(args.output_dir)
  if args.resume_from:
    args.resume_from=args.resume_from.resolve();assert args.mode=='fit' and args.resume_from.is_relative_to(C.ROOT) and args.resume_from!=args.output_dir
  if args.mode=='fit':qualification(args)
  availability(args)
  if args.child:
    assert os.environ.get('AGENT_DISTILL_VISION_COHORT_FIT_PARENT_PID')==str(os.getppid())
    assert (args.output_dir/'parent-invocation.json').is_file();return child(args,plan,dataset)
  assert not args.output_dir.exists();carry={}
  if args.resume_from:
    arrays,metadata,carry=resume_carry(args.resume_from,args,C.original().graph().shapes(),dataset);del arrays,metadata
  with Path('/tmp/opencode/alpamayo-chestnut-speed.lock').open('a') as lock:
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);args.output_dir.mkdir();(args.output_dir/'tmp').mkdir()
    timeout=300 if args.mode=='qualify' else 900-carry.get('previous_total_wall_seconds',0)
    command=[str(C.PY),str(Path(__file__).resolve()),*sys.argv[1:],'--child']
    C.write(args.output_dir/'parent-invocation.json',dict(command=command,timeout_seconds=timeout,plan_sha256=args.plan_sha,
      previous_total_wall_seconds=carry.get('previous_total_wall_seconds',0)))
    old=C.module('vision_cohort_fit_original_environment',HERE/'agent-distill-vision-train.py');env=old.environment(args.output_dir)
    env['AGENT_DISTILL_VISION_COHORT_FIT_PARENT_PID']=str(os.getpid())
    started=time.monotonic();timed_out=False
    with (args.output_dir/'run.log').open('w') as stream:
      process=subprocess.Popen(command,env=env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
      try:code=process.wait(timeout=timeout)
      except subprocess.TimeoutExpired:
        timed_out=True;os.killpg(process.pid,signal.SIGTERM)
        try:code=process.wait(timeout=5)
        except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);code=process.wait()
    C.write(args.output_dir/'wall.json',dict(seconds=time.monotonic()-started,returncode=code,timed_out=timed_out))
    assert code==0 and not timed_out and C.read(args.output_dir/'report.json')['complete']


if __name__=='__main__':main()
