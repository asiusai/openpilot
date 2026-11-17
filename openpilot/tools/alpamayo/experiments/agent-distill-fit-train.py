"""Fixed 25-update pilot. Default OFF; reviewed full-length/resource and fresh ownership evidence are mandatory."""
import argparse
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import shutil
import signal
import struct
import subprocess
import sys
import time

sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('agent_distill_fit_common_train',HERE/'agent-distill-fit-common.py')
C=importlib.util.module_from_spec(spec);sys.modules[spec.name]=C;spec.loader.exec_module(C)

def seed_arrays(p):
  import numpy as np
  index=C.read(C.BASE/'model/model.safetensors.index.json')['weight_map'];result={}
  for name,item in p['master_weights'].items():
    full='vlm.model.language_model.layers.12.'+name;path=C.BASE/'model'/index[full]
    with path.open('rb') as stream:
      size=struct.unpack('<Q',stream.read(8))[0];assert size<16*1024**2
      header=json.loads(stream.read(size));entry=header[full]
      assert entry['dtype']=='BF16' and entry['shape']==item['shape']
      begin,end=entry['data_offsets'];stream.seek(8+size+begin);raw=stream.read(end-begin)
    assert __import__('hashlib').sha256(raw).hexdigest()==item['bf16_sha256']
    result['w:'+name]=C.bf16(np.frombuffer(raw,dtype=np.uint16).reshape(item['shape']))
  assert sum(v.size for v in result.values())==50335744
  return result

def environment(root,p,receipt):
  env=os.environ.copy()
  for key in list(env):
    if key.startswith(('AM_','AMD_','HCQ_')) or key in ('CPU_CC','CPU_LLVM','AMD_LLVM','AMD_CC'):env.pop(key,None)
  env.update(DEV=p['target'],TC='0',ALLOW_TF32='0',TRAINING='0',PARALLEL='2',BEAM='0',JITBEAM='0',
    PROFILE='0',PMC='',SQTT='',VIZ='0',LRU='0',MAX_BUFFER_SIZE=str(p['max_single_buffer']),ALLOW_DEVICE_USAGE='1',
    OPTIM_DTYPE='float32',FUSE_OPTIM='0',CONST_LR='0',NO_MEMORY_PLANNER=str(receipt['no_memory_planner']),
    PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=str(C.BASE/'tinygrad'),CACHEDB=str(root/'compiler-cache.db'),TMPDIR=str(root/'tmp'),
    OPENBLAS_NUM_THREADS='4',OMP_NUM_THREADS='4',CUDA_VISIBLE_DEVICES='',HIP_VISIBLE_DEVICES='',ROCR_VISIBLE_DEVICES='')
  return env

def preflight(args):
  assert args.protocol.resolve().is_relative_to(C.ROOT) and C.sha(args.protocol)==args.protocol_sha
  p=C.protocol(args.protocol);p['_sha256']=args.protocol_sha
  rows=C.records()
  for row in rows:C.load_target(row,rows)  # Validation only: no student forward/loss/fit here.
  if args.execute_training or args.child:
    receipt=C.qualification(p,args.admission,args.reviewed_admission_sha)
    assert receipt['no_memory_planner'] in (0,1)
    assert receipt['availability_review']['path'].startswith(str(C.ROOT)+'/')
    assert C.sha(receipt['availability_review']['path'])==receipt['availability_review']['sha256']
    availability=C.read(receipt['availability_review']['path'])
    assert 0<=time.time_ns()-availability['captured_unix_ns']<=60_000_000_000
    assert availability['selected_node']['pci']==p['selected_pci'] and availability['selected_node']['gpu_id']==p['selected_gpu_id']
    assert availability['kfd_pid_coverage_complete'] and not availability['observed_selected_kfd_owners']
    assert availability['vram_free_bytes']>=p['live_allocation_limit']+p['free_reserve']
    return p,rows,receipt
  C.no_devices();return p,rows,None

def child(args,p,rows,receipt):
  import numpy as np
  C.no_devices();root=args.output_dir.resolve()
  assert root.is_relative_to(C.ROOT) and (root/'parent-invocation.json').is_file()
  K=C.module('agent_distill_fit_kfd',C.KFD)
  availability=K.passive_availability();K.require_no_observed_conflict(availability)
  assert availability['vram_free_bytes']>=p['live_allocation_limit']+p['free_reserve']
  C.write(root/'availability-before-open.json',availability)
  permitted={'/dev/kfd','/dev/dri/renderD129'}
  def hardware_guard(event,values):
    if event=='open' and isinstance(values[0],(str,bytes)):
      path=os.fsdecode(values[0])
      if path.startswith(('/dev/dri/','/dev/kfd','/dev/nvidia','/dev/bus/usb/')):assert path in permitted
  sys.addaudithook(hardware_guard)
  from tinygrad import Tensor,Device,dtypes
  from tinygrad.helpers import Context,DEV,MAX_BUFFER_SIZE,PROFILE
  from tinygrad.runtime.ops_amd import KFDIface,PMC,SQTT
  from tinygrad.nn.optim import AdamW
  assert not Device._opened_devices and repr(DEV)==p['target'] and DEV.target('AMD').indices=='1'
  assert not PMC.value and not SQTT.value and not PROFILE.value and MAX_BUFFER_SIZE.value==p['max_single_buffer']
  original_alloc,original_free=KFDIface.alloc,KFDIface.free
  live={};budget=dict(live=0,peak=0,cumulative=0,calls=0)
  def alloc(self,size,*a,**kw):
    assert self.gpu_id==p['selected_gpu_id'] and Path(self.dev_sysfs_path).resolve().name==p['selected_pci']
    assert 0<size<=p['max_single_buffer'] and budget['live']+size<=p['live_allocation_limit']
    assert budget['cumulative']+size<=p['cumulative_allocation_limit']
    value=original_alloc(self,size,*a,**kw);key=int(value.meta.handle);assert key not in live
    live[key]=size;budget['live']+=size;budget['cumulative']+=size;budget['calls']+=1;budget['peak']=max(budget['peak'],budget['live'])
    return value
  def free(self,value):
    original_free(self,value);key=int(value.meta.handle)
    if key in live:budget['live']-=live.pop(key)
  KFDIface.alloc,KFDIface.free=alloc,free
  report=dict(complete=False,gpu_executed=False,real_optimizer_updates=0,protocol_sha256=args.protocol_sha,
    sealed_targets_opened=False,driving_model_modified=False,allocation_budget=budget,quality_qualified=False)
  S=C.module('agent_distill_fit_state_child',HERE/'agent-distill-fit-state.py')
  try:
    dev=Device['AMD'];selected=K.selected_node(K.passive_topology())
    identity=dict(requested_kfd_index=1,logical_device_id=dev.device_id,filtered_kfd_nodes=list(KFDIface.gpus),
      interface=type(dev.iface).__name__,gpu_id=dev.iface.gpu_id,pci=Path(dev.iface.dev_sysfs_path).resolve().name,
      render_minor=dev.iface.props['drm_render_minor'],arch=dev.arch,renderer_target=repr(dev.renderer.target),
      is_usb=dev.is_usb,native_pci=dev.is_am(),can_recover=dev.can_recover,allocator=type(dev.allocator).__name__)
    K.validate_opened_identity(identity,selected);report.update(gpu_executed=True,actual_device=identity)
    values=seed_arrays(p);B=C.graph();block=B.Block(values,p['block_config'],'AMD')
    block.weights={name:block.weights[name].is_param_(True).realize() for name in sorted(block.weights)}
    opt=AdamW(list(block.weights.values()),lr=3.5e-6,b1=.9,b2=.999,eps=1e-8,weight_decay=.01,fused=False)
    shapes={k:tuple(v['shape']) for k,v in p['master_weights'].items()}
    context=dict(protocol_sha256=args.protocol_sha,manifest_sha256=C.MANIFEST_SHA,hyperparameters=C.HYPER)
    training=[r for r in rows if r['split']=='train'];validation=[r for r in rows if r['split']=='validation']
    # Target tensors are immutable, unparameterized constants. No teacher forward executes.
    def tensors(row):
      arrays=C.load_target(row,rows)
      return {k:Tensor(v,device='AMD').is_param_(False).realize() for k,v in arrays.items()}
    def objective(row):
      data=tensors(row);outputs,_=block(data['x'],data['cos'],data['sin'])
      return C.loss(B,outputs,(data['y'],data['k'],data['v']),row['actual_prompt_tokens']),data
    def validate(step):
      losses=[]
      with Context(TRAINING=0):
        for row in validation:
          loss,data=objective(row);value=float(loss.item());assert np.isfinite(value);losses.append(value)
          assert all(v.grad is None and not v.is_param for v in data.values())
      return dict(step=step,mean=float(np.mean(losses)),samples=[r['id'] for r in validation])
    if args.resume_from:
      arrays,metadata=S.load(args.resume_from.resolve(),context,shapes);S.restore(block.weights,opt,arrays)
      assert metadata['step']<25 and metadata['best_step'] in (0,25)
      # The previous best artifact is separately hash-bound; never select by training loss.
      previous=C.read(args.resume_from.resolve()/'best.json');assert C.sha(args.resume_from.resolve()/previous['file'])==previous['sha256']
      assert previous['file']=='best.npz' and previous['selected_step']==metadata['best_step'] and previous['validation_mean']==metadata['best_validation']
      shutil.copyfile(args.resume_from.resolve()/previous['file'],root/'best.npz');S.atomic_json(root/'best.json',dict(previous,file='best.npz'))
    else:
      initial=validate(0);metadata=dict(step=0,best_validation=initial['mean'],best_step=0,validation=[initial])
      with (root/'best.npz').open('xb') as f:np.savez(f,**{k:v.numpy() for k,v in block.weights.items()})
      S.atomic_json(root/'best.json',dict(file='best.npz',sha256=C.sha(root/'best.npz'),selected_step=0,validation_mean=initial['mean']))
    S.save(root,S.snapshot(block.weights,opt),metadata,context,shapes,initial_recovery=bool(args.resume_from))
    start=metadata['step'];report['starting_checkpoint_step']=start
    for step in range(start+1,26):
      row=training[(step-1)%3];S.intent(root,step-1)
      with Context(TRAINING=1):
        opt.zero_grad();loss,data=objective(row)
        grads=loss.gradient(*block.weights.values());Tensor.realize(*grads)
        assert np.isfinite(float(loss.item()))
        for parameter,gradient in zip(block.weights.values(),grads,strict=True):
          assert bool(gradient.isfinite().all().item());parameter.grad=gradient
        assert all(v.grad is None and not v.is_param for v in data.values())
        opt.step();dev.synchronize()
      report['real_optimizer_updates']+=1
      metadata['step']=step
      if step%25==0:
        item=validate(step);metadata['validation'].append(item)
        if item['mean']<metadata['best_validation']:
          metadata['best_validation']=item['mean'];metadata['best_step']=step
          temporary=root/'best-next.npz'
          with temporary.open('xb') as f:np.savez(f,**{k:v.numpy() for k,v in block.weights.items()})
          os.replace(temporary,root/'best.npz')
          S.atomic_json(root/'best.json',dict(file='best.npz',sha256=C.sha(root/'best.npz'),selected_step=step,validation_mean=item['mean']))
      S.save(root,S.snapshot(block.weights,opt),metadata,context,shapes)
      report.update(committed_step=step,best_step=metadata['best_step'],validation=metadata['validation'])
      S.atomic_json(root/'report.json',report)
    assert report['real_optimizer_updates']+start==25<=200
    report.update(complete=True,selection='validation mean only; strict improvement; ties retain earlier checkpoint')
  finally:
    KFDIface.alloc,KFDIface.free=original_alloc,original_free
    report['allocator_hooks_restored']=KFDIface.alloc is original_alloc and KFDIface.free is original_free
    S.atomic_json(root/'report.json',report)

def main():
  parser=argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--protocol',type=Path,default=C.ROOT/'agent-distill-fit-protocol.json')
  parser.add_argument('--protocol-sha',required=True)
  parser.add_argument('--admission',type=Path)
  parser.add_argument('--reviewed-admission-sha')
  parser.add_argument('--output-dir',type=Path,default=C.ROOT/'pilot-v1')
  parser.add_argument('--resume-from',type=Path)
  parser.add_argument('--execute-training',action='store_true',default=False)
  parser.add_argument('--child',action='store_true')
  args=parser.parse_args()
  if args.child:assert args.execute_training and os.environ.get('AGENT_DISTILL_FIT_PARENT_PID')==str(os.getppid())
  p,rows,receipt=preflight(args)
  if not args.execute_training and not args.child:
    print(json.dumps(dict(data_preflight_passed=True,training_executed=False,optimizer_updates=0,
      blocked=['Parent-reviewed successful full L2048 zero-update qualification.','Reviewed full-parameter AdamW dummy-update resource/time envelope.','Fresh parent-reviewed KFD1 no-reset availability.']),indent=2));return
  if args.child:return child(args,p,rows,receipt)
  C.no_devices();root=args.output_dir.resolve();assert root.is_relative_to(C.ROOT) and root.name.startswith('pilot-') and not root.exists()
  if args.resume_from:assert args.resume_from.resolve().is_relative_to(C.ROOT) and args.resume_from.resolve()!=root
  with Path('/tmp/opencode/alpamayo-chestnut-speed.lock').open('a') as lock:
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    root.mkdir();(root/'tmp').mkdir()
    command=[str(C.BASE/'upstream/.venv/bin/python'),str(Path(__file__).resolve()),*sys.argv[1:],'--child']
    C.write(root/'parent-invocation.json',dict(command=command,timeout_seconds=900,protocol_sha256=args.protocol_sha,admission_sha256=args.reviewed_admission_sha))
    started=time.monotonic();timed_out=False
    with (root/'run.log').open('w') as stream:
      child_env=environment(root,p,receipt);child_env['AGENT_DISTILL_FIT_PARENT_PID']=str(os.getpid())
      process=subprocess.Popen(command,env=child_env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
      try:code=process.wait(timeout=900)
      except subprocess.TimeoutExpired:
        timed_out=True;os.killpg(process.pid,signal.SIGTERM)
        try:code=process.wait(timeout=5)
        except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);code=process.wait()
    C.write(root/'wall.json',dict(seconds=time.monotonic()-started,returncode=code,timed_out=timed_out))
    assert not timed_out and code==0 and C.read(root/'report.json')['complete']

if __name__=='__main__':main()
