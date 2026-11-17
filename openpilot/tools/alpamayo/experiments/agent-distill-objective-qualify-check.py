"""CPU synthetic one-update/recovery/resume and full parent/child entrypoint with mocked KFD."""
import argparse
from copy import deepcopy
import gc
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import runpy
import sys
import subprocess
from types import SimpleNamespace
from unittest.mock import patch

HERE=Path(__file__).resolve().parent
BASE=Path('/tmp/opencode/alpamayo-asius-20261114');WORK=Path('/mnt/nvme-500/tmp/alpamayo-speed-20261116')
os.environ.update(DEV='CPU',TC='0',ALLOW_TF32='0',NO_MEMORY_PLANNER='1',LRU='0',PROFILE='0',
  MAX_BUFFER_SIZE=str(256*1024**2),PYTHONDONTWRITEBYTECODE='1',OPENBLAS_NUM_THREADS='4',OMP_NUM_THREADS='4')
sys.dont_write_bytecode=True;sys.path.insert(0,str(BASE/'tinygrad'))
import numpy as np
import tinygrad
from tinygrad import Tensor,Device
import tinygrad.helpers as helpers
import tinygrad.runtime.ops_amd as amd
R=runpy.run_path(str(HERE/'agent-distill-objective-qualify-run.py'));Q=R['Q']


def no_devices(event,args):
  if event=='open' and isinstance(args[0],(str,bytes)):
    path=os.fsdecode(args[0])
    if path.startswith(('/dev/dri','/dev/kfd','/dev/nvidia','/dev/bus/usb')):raise RuntimeError('Qualification check is CPU-only')
sys.addaudithook(no_devices)


def fixture():
  rng=np.random.default_rng(2711);result={}
  for role,width,hidden in [('language',8,16),('expert',4,8)]:
    shapes={'input_layernorm.weight':(width,),'post_attention_layernorm.weight':(width,),
      'self_attn.q_proj.weight':(8,width),'self_attn.k_proj.weight':(4,width),'self_attn.v_proj.weight':(4,width),
      'self_attn.o_proj.weight':(width,8),'mlp.fc1.weight':(hidden,width),'mlp.fc2.weight':(width,hidden)}
    result.update({role+'.'+k:np.ones(s,np.float32) if 'layernorm' in k else rng.normal(0,.1,s).astype(np.float32) for k,s in shapes.items()})
  array=lambda shape:rng.normal(0,.2,shape).astype(np.float32)
  angles=array((1,1,256,4));lang=dict(x=array((1,256,8)),y=array((1,256,8)),k=array((1,1,256,4)),v=array((1,1,256,4)),
    cos=np.cos(angles),sin=np.sin(angles))
  mask=np.full(260,-np.inf,np.float32);mask[:9]=0;mask[256:]=0
  steps=[]
  for _ in range(3):
    angles=array((1,1,4,4));steps.append(dict(x=array((1,4,4)),y=array((1,4,4)),cos=np.cos(angles),sin=np.sin(angles),mask=mask.copy()))
  def load(device):
    return dict(prompt_length=5,valid_length=9,capacity=256,language={k:Tensor(v.copy(),device=device) for k,v in lang.items()},
      expert=[{k:Tensor(v.copy(),device=device) for k,v in step.items()} for step in steps])
  return result,load


def recovery(root):
  weights,load=fixture();context=dict(kind=Q.KIND,synthetic_fixture=True,hyperparameters=Q.HYPER)
  first=root/'uninterrupted';first.mkdir();result=Q.run_step(deepcopy(weights),load,first,context,'CPU',lambda name:None,fixture=True)
  replay=root/'resume-zero';replay.mkdir()
  resumed=Q.run_step(deepcopy(weights),load,replay,context,'CPU',lambda name:None,resume=first/'seed-state',fixture=True)
  shapes={k:v.shape for k,v in sorted(weights.items())};a,_=Q.S.load(first,context,shapes);b,_=Q.S.load(replay,context,shapes)
  for name in a:np.testing.assert_array_equal(a[name].view(np.uint32),b[name].view(np.uint32))
  done=root/'resume-one';done.mkdir();verified=Q.run_step(deepcopy(weights),load,done,context,'CPU',lambda name:None,resume=first,fixture=True)
  assert result['updates_this_run']==resumed['updates_this_run']==1 and verified['updates_this_run']==0
  assert result['recovery_arrays']==resumed['recovery_arrays']==verified['recovery_arrays']==51
  negative=[]
  def rejected(name,call):
    try:call()
    except (AssertionError,ValueError):negative.append(name)
    else:raise AssertionError('Accepted '+name)
  journal=Q.read(done/'journal.json');Q.S.atomic_json(done/'journal.json',dict(committed=1,intended=2))
  rejected('dirty optimizer intent',lambda:Q.S.load(done,context,shapes));Q.S.atomic_json(done/'journal.json',journal)
  pointer=Q.read(done/'latest.json');wrong=deepcopy(pointer);wrong['sha256']='0'*64;Q.S.atomic_json(done/'latest.json',wrong)
  rejected('changed recovery checksum',lambda:Q.S.load(done,context,shapes));Q.S.atomic_json(done/'latest.json',pointer)
  return dict(synthetic_only=True,uninterrupted_update=result,resumed_step_zero_updates=resumed['updates_this_run'],
    resumed_step_one_updates=verified['updates_this_run'],all51_resumed_arrays_bit_exact=True,negative_controls=negative)


def control_headers(root):
  path=root/'header-only-fixture.json'
  value=dict(complete=True,gpu_executed=True,input_manifest=dict(sha256='83da7c2dc65a50185d052de07c30698ade2b1818103d90d69ab394f8f600f3b6'),
    source_sha256={str(HERE/name):Q.sha(HERE/name) for name in ('agent-distill-objective-core.py',
      'agent-distill-objective-capture.py','agent-distill-objective-position-run.py')},
    record=dict(id='train-command-0',split='train',route_id=Q.C.ROUTES['train'],
      targets=dict(complete=True,capacity=2304,prompt_length=2027,valid_length=2047,expert=[{}, {}, {}])))
  def invoke(header):
    path.write_text(json.dumps(header));return Q.control(path,Q.sha(path))
  _,context=invoke(value)
  assert context['training_cohort_sample'] is False and context['checkpoint_selection_allowed'] is False
  rejected=[]
  for name,edit in (
      ('incomplete capture',lambda v:v.update(complete=False)),
      ('synthetic report',lambda v:v.update(synthetic_fixture=True)),
      ('synthetic record',lambda v:v['record'].update(synthetic_fixture=True)),
      ('unapproved cohort ID',lambda v:v['record'].update(id='train-scene-00')),
      ('wrong route',lambda v:v['record'].update(route_id='unapproved-metadata-only-route')),
      ('changed input manifest',lambda v:v['input_manifest'].update(sha256='0'*64)),
      ('incomplete targets',lambda v:v['record']['targets'].update(complete=False)),
      ('different full extent',lambda v:v['record']['targets'].update(valid_length=2049)),
      ('missing Euler step',lambda v:v['record']['targets'].update(expert=[{},{}])),
      ('changed direct source',lambda v:v['source_sha256'].update({str(HERE/'agent-distill-objective-core.py'):'0'*64}))):
    bad=deepcopy(value);edit(bad)
    try:invoke(bad)
    except ValueError:rejected.append(name)
    else:raise AssertionError('Accepted '+name)
  invoke(value)
  try:Q.control(path,'0'*64)
  except ValueError:rejected.append('changed completed report checksum')
  else:raise AssertionError('Accepted changed checksum')
  path.unlink()
  return dict(metadata_only=True,target_arrays_read=False,not_a_training_cohort_sample=True,rejected=rejected)


def entrypoint(root):
  # Reuse prior qualified ledger negative evidence. Exercise native release paths again only where this entrypoint depends on them.
  prior=WORK/'agent-distill-expert-artifacts/ledger-cpu-v2/agent-distill-expert-ledger-cpu.json'
  assert Q.sha(prior)=='16fa67bc49e4fc8898cec5e06b3a44a98491f06cd1055e2327dc7b2baef030d8'
  old=Q.read(prior);assert old['all_passed'] and old['qualified_ledger_sha256']==R['LEDGER_SHA']
  real_devices=tinygrad.Device;events=[];live={};reusable=[];sequence=[20];owner=[];late=[];box=[]
  selected=dict(node='2',pci='0000:0c:00.0',gpu_id=27211,render_minor=129,gfx_target_version=110000,
    vendor='0x1002',device='0x744c',driver='amdgpu',render_rw_access=True)
  def availability():return dict(selected_kfd_index=1,selected_node=selected,kfd_proc_root_read=True,kfd_pid_coverage_complete=True,
    kfd_processes=[dict(pid=os.getpid(),complete=True)],observed_selected_kfd_owners=owner[:],vram_free_bytes=24*1024**3)
  def alloc(self,size,*a,**kw):
    if reusable:handle=reusable.pop()
    else:sequence[0]+=1;handle=sequence[0]
    assert handle not in live;live[handle]=size;return amd.BufferStorage(0x1000,SimpleNamespace(handle=handle,size=size))
  def release(fd,handle):live.pop(handle);reusable.append(handle)
  def initialize(self):self.gpu_id=27211;self.dev_sysfs_path='/synthetic/0000:0c:00.0';self.props={'drm_render_minor':129};self.kfd=None
  Iface=type('KFDIface',(),dict(__init__=initialize,alloc=alloc,free=amd.KFDIface.free,unmap=amd.KFDIface.unmap,map=amd.KFDIface.map,
    _map_handle=lambda *a:None,_unmap_handle=lambda *a:None,gpus=['2']))
  class Target:
    def __repr__(self):return R['TARGET']
    def target(self,name):return SimpleNamespace(indices='1')
  class Devices:
    _opened_devices=[];DEFAULT='CPU'
    def __getitem__(self,key):
      if key!='AMD':return real_devices[key]
      events.append('device-construction');owner[:]=[os.getpid()];iface=Iface();box.append(iface)
      host=iface.map(SimpleNamespace(device='CPU',_buf=0x3000,nbytes=4096));iface.unmap(host)
      return SimpleNamespace(device_id=0,iface=iface,arch='gfx1100',renderer=SimpleNamespace(target=Target()),
        is_usb=False,is_am=lambda:False,can_recover=False,allocator=type('AMDAllocator',(),{})())
  class Delayed:
    def __init__(self,iface):self.iface=iface;self.storage=iface.alloc(4096)
    def __del__(self):self.iface.free(self.storage);events.append('late-finalizer')
  K=Q.module('qualify_mock_identity',HERE/'agent-distill-kfd1-common.py')
  K.passive_availability=availability;K.passive_topology=lambda:dict(nodes=[{'gpu_id':1},selected],kfd_rw_access=True)
  original_spec=importlib.util.spec_from_file_location
  weights,load=fixture();context=dict(kind=Q.KIND,synthetic_fixture=True,hyperparameters=Q.HYPER,aligned_report_sha256='synthetic-control')
  def configure(core):
    real_module=core.module;real_work=core.run_step
    core.module=lambda name,path:K if Path(path).name=='agent-distill-kfd1-common.py' else real_module(name,path)
    core.control=lambda *a:({'synthetic_fixture':True},context)
    core.seed_arrays=lambda:deepcopy(weights)
    def work(values,loader,directory,ctx,device,phase,**kwargs):
      assert device=='AMD';events.append('first-weight-upload')
      host=box[0].map(SimpleNamespace(device='CPU',_buf=0x5000,nbytes=4096));box[0].unmap(host);late.append(Delayed(box[0]))
      result=real_work(values,load,directory,ctx,'CPU',phase,fixture=True,**kwargs)
      result.update(cpu_mock=True,gpu_executed=False);events.append('one-update-and-recovery');return result
    core.run_step=work
  def hook(name,path,*a,**kw):
    spec=original_spec(name,path,*a,**kw)
    if Path(path).resolve()!=HERE/'agent-distill-objective-qualify-core.py':return spec
    original=spec.loader
    class Loader:
      def create_module(self,spec):return original.create_module(spec)
      def exec_module(self,value):original.exec_module(value);configure(value)
    spec.loader=Loader();return spec
  def invoke(command):
    old_argv=sys.argv[:];sys.argv=command[1:]
    try:
      namespace=runpy.run_path(command[1]);main=namespace['main']
      main.__globals__['admission']=lambda *a:dict(target=R['TARGET'],no_memory_planner=1)
      main()
    finally:sys.argv=old_argv
  class Process:
    pid=987654
    def __init__(self,command,**kwargs):
      assert kwargs['start_new_session'];events.append('bounded-parent-child-dispatch')
      with patch.dict(os.environ,kwargs['env']),patch.object(os,'getppid',lambda:os.getpid()):invoke(command)
    def wait(self,timeout):assert timeout==900;return 0
  real_popen=subprocess.Popen
  def popen(command,**kwargs):
    if isinstance(command,list) and len(command)>1 and command[1]==str(HERE/'agent-distill-objective-qualify-run.py'):return Process(command,**kwargs)
    return real_popen(command,**kwargs)
  args=[str(BASE/'upstream/.venv/bin/python'),str(HERE/'agent-distill-objective-qualify-run.py'),
        '--aligned-report',str(root/'synthetic-input.json'),'--aligned-report-sha','synthetic-control',
        '--output-dir',str(root/'entrypoint'),'--execute-qualification']
  with patch.object(importlib.util,'spec_from_file_location',hook),patch.object(tinygrad,'Device',Devices()),patch.object(helpers,'DEV',Target()),\
       patch.object(amd,'KFDIface',Iface),patch.object(amd.kfd,'AMDKFD_IOC_FREE_MEMORY_OF_GPU',release),\
       patch.object(amd.FileIOInterface,'munmap',lambda *a:None),patch('subprocess.Popen',popen),\
       patch.object(fcntl,'flock',lambda fd,mode:events.append('mock-global-lock')):
    invoke(args)
    report=Q.read(root/'entrypoint/report.json')
    assert report['complete'] and report['cpu_mock'] and not report['gpu_executed']
    assert report['allocator_guards_retained_through_finalizers'] and report['allocation_budget']['live_bytes']==4096
    cumulative=report['allocation_budget']['requested_bytes'];late.clear();gc.collect();assert not live
    assert Iface.alloc is not alloc and events[-1]=='late-finalizer'
  return dict(actual_runpy_cli_parent_and_child=True,admission_and_hardware_mocked=True,numerical_step_executed_on_cpu=True,
    events=events,native_owned_unmap_handle_reuse=True,late_finalizer_preserved=True,
    qualified_negative_allocator_checks_reused=str(prior),mock_requested_bytes=cumulative)


def main():
  parser=argparse.ArgumentParser();parser.add_argument('--output-dir',type=Path,required=True);args=parser.parse_args()
  root=args.output_dir.resolve()
  if root.exists() or not root.is_relative_to(Q.ROOT):raise ValueError('Fresh owned CPU output required')
  root.mkdir(parents=True)
  result=dict(gpu_executed=False,actual_data_optimizer_updates=0,control_headers=control_headers(root),recovery=recovery(root),entrypoint=entrypoint(root))
  assert all(name.startswith(('CPU','NPY','PYTHON')) for name in Device._opened_devices)
  result.update(all_passed=True,opened_devices=list(Device._opened_devices),source_sha256={str(p):Q.sha(p) for p in
    (*HERE.glob('agent-distill-objective-qualify-*.py'),HERE/'agent-distill-objective-core.py')})
  (root/'report.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
