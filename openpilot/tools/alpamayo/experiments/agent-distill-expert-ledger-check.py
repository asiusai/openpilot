"""CPU tests of the qualified ledger and complete corrected child entrypoint. No optimizer execution."""
import argparse
import copy
import gc
import importlib.util
import json
import os
from pathlib import Path
import runpy
import sys
from types import SimpleNamespace
from unittest.mock import patch

HERE=Path(__file__).resolve().parent
WORK=Path('/mnt/nvme-500/tmp/alpamayo-speed-20261116')
BASE=Path('/tmp/opencode/alpamayo-asius-20261114')
sys.dont_write_bytecode=True


def main():
  p=argparse.ArgumentParser();p.add_argument('--output-dir',type=Path,required=True);args=p.parse_args()
  root=args.output_dir.resolve();assert root.is_relative_to(WORK/'agent-distill-expert-artifacts')
  root.mkdir(parents=True,exist_ok=False)
  os.environ.update(DEV='CPU:LLVM',PROFILE='0',TC='0',ALLOW_TF32='0',NO_MEMORY_PLANNER='1',LRU='0',
                    MAX_BUFFER_SIZE=str(256*1024**2),CUDA_VISIBLE_DEVICES='',HIP_VISIBLE_DEVICES='',ROCR_VISIBLE_DEVICES='')
  def deny(event,values):
    if event=='open' and isinstance(values[0],(str,bytes)):
      assert not os.fsdecode(values[0]).startswith(('/dev/dri','/dev/kfd','/dev/nvidia','/dev/bus/usb/'))
  sys.addaudithook(deny);sys.path.insert(0,str(BASE/'tinygrad'))
  import numpy as np
  import tinygrad
  import tinygrad.helpers as helpers
  import tinygrad.runtime.ops_amd as amd
  E=runpy.run_path(str(HERE/'agent-distill-expert-ledger-run.py'))
  T=E['load']();B=T.B
  L=B.module('expert_ledger_exact_cpu',E['QUALIFIED'])
  real_free,real_unmap,real_map=amd.KFDIface.free,amd.KFDIface.unmap,amd.KFDIface.map
  events=[];live_native={};reusable=[];sequence=[10];owners=[]
  selected=dict(node='2',pci='0000:0c:00.0',gpu_id=27211,render_minor=129,gfx_target_version=110000,
                vendor='0x1002',device='0x744c',driver='amdgpu',render_rw_access=True)
  def availability():
    return dict(selected_kfd_index=1,selected_node=selected,kfd_proc_root_read=True,kfd_pid_coverage_complete=True,
      kfd_processes=[dict(pid=os.getpid(),complete=True)],observed_selected_kfd_owners=owners[:],vram_free_bytes=24*1024**3)
  def native_alloc(self,size,*args,**kwargs):
    if reusable:handle=reusable.pop(0)
    else:sequence[0]+=1;handle=sequence[0]
    assert handle not in live_native;live_native[handle]=size
    events.append(('alloc',handle,size,kwargs.get('host',False)))
    return amd.BufferStorage(0x1000,SimpleNamespace(handle=handle,size=size))
  def free_ioctl(fd,handle):
    assert handle in live_native;live_native.pop(handle);reusable.insert(0,handle);events.append(('native_free',handle))
  def unmap_handle(self,handle):events.append(('native_unmap',handle))
  def initialize(self):
    self.gpu_id=27211;self.dev_sysfs_path='/synthetic/0000:0c:00.0';self.props={'drm_render_minor':129};self.kfd=None
  Iface=type('KFDIface',(),dict(__init__=initialize,alloc=native_alloc,free=real_free,unmap=real_unmap,map=real_map,
    _unmap_handle=unmap_handle,_map_handle=lambda self,h:None,gpus=['2']))
  parameters=dict(B.BOUNDS,selected_gpu_id=27211,selected_pci='0000:0c:00.0')
  def install():
    ledger=L.Ledger(parameters,availability)
    Iface.alloc=ledger.alloc(native_alloc);Iface.free=ledger.free(real_free);Iface.unmap=ledger.unmap(real_unmap)
    return ledger
  failures=[]
  def reject(name,fn):
    try:fn()
    except (AssertionError,KeyError):failures.append(name)
    else:raise AssertionError('Negative control accepted: '+name)
  with patch.object(amd.kfd,'AMDKFD_IOC_FREE_MEMORY_OF_GPU',free_ioctl),patch.object(amd.FileIOInterface,'munmap',lambda *a:None):
    ledger=install();iface=Iface()
    plain=iface.alloc(4096);handle=plain.meta.handle
    iface.unmap(amd.BufferStorage(plain.buf,(handle,False)))
    assert ledger.report['live_bytes']==4096 and handle in ledger.handles
    iface.free(plain);assert ledger.report['live_bytes']==0
    host=iface.map(SimpleNamespace(device='CPU',_buf=0x2000,nbytes=8192))
    assert host.meta==(handle,True)
    iface.unmap(host);assert ledger.report['live_bytes']==0 and not ledger.handles
    again=iface.alloc(4096);assert again.meta.handle==handle;iface.free(again)
    assert ledger.report['requested_bytes']==16384 and ledger.report['calls']==3
    assert ledger.report['free_calls']==2 and ledger.report['owned_unmap_calls']==1
    reject('unknown owned unmap',lambda:iface.unmap(amd.BufferStorage(0,(9999,True))))
    reject('double free',lambda:iface.free(again))
    saved=ledger.report['requested_bytes'];ledger.phase('after-release')
    assert ledger.report['requested_bytes']==saved
    original_limit=parameters['cumulative_allocation_limit'];parameters['cumulative_allocation_limit']=saved
    reject('cumulative budget never resets',lambda:iface.alloc(4096));parameters['cumulative_allocation_limit']=original_limit
    basic=copy.deepcopy(ledger.report)
    # Run the actual overlay as __main__, with the actual trainer parser and child dispatch.
    # Only device construction and numerical work are mocked; real free/map/unmap remain active.
    Iface.alloc,Iface.free,Iface.unmap=native_alloc,real_free,real_unmap
    reports=[];late=[];calls=[];device_box=[]
    class LateBuffer:
      def __init__(self,iface,storage):self.iface,self.storage=iface,storage
      def __del__(self):self.iface.free(self.storage);events.append(('late_finalizer',))
    class Target:
      def __repr__(self):return 'KFD:1+AMD:LLVM:gfx1100'
      def target(self,name):return SimpleNamespace(indices='1')
    class FakeDevices:
      _opened_devices=[]
      def __getitem__(self,key):
        assert key=='AMD';calls.append('device-construction');owners[:]=[os.getpid()]
        iface=Iface()
        host=iface.map(SimpleNamespace(device='CPU',_buf=0x4000,nbytes=4096));iface.unmap(host)
        device=SimpleNamespace(device_id=0,iface=iface,arch='gfx1100',renderer=SimpleNamespace(target=Target()),
          is_usb=False,is_am=lambda:False,can_recover=False,allocator=type('AMDAllocator',(),{})())
        device_box.append(device);return device
    K=B.module('expert_ledger_cpu_identity',HERE/'agent-distill-kfd1-common.py')
    K.passive_availability=availability;K.passive_topology=lambda:dict(nodes=[{'gpu_id':1},selected],kfd_rw_access=True)
    original_spec=importlib.util.spec_from_file_location
    def configure(trainer):
      old_module=trainer.B.module
      trainer.B.module=lambda name,path:K if Path(path).name=='agent-distill-kfd1-common.py' else old_module(name,path)
      trainer.B.target_headers=lambda *a:([dict(split='train'),dict(split='validation')],dict(manifest_sha256='cpu-fixture'))
      trainer.B.validate_targets=lambda rows:True
      trainer.B.seed=lambda:{'first':np.ones(1,np.float32)}
      def block(weights,device):
        calls.append('first-weight-construction');assert device=='AMD' and weights['first'][0]==1
        iface=device_box[0].iface
        host=iface.map(SimpleNamespace(device='CPU',_buf=0x6000,nbytes=4096));iface.unmap(host)
        storage=iface.alloc(4096);late.append(LateBuffer(iface,storage))
        return SimpleNamespace(weights=weights)
      trainer.B.Block=block
      def fit(*args,**kwargs):
        calls.append('unchanged-fit-call-boundary');assert args[5]=='AMD'
        assert kwargs==dict(resume=None)
        return dict(complete=False,updates_this_run=0,cpu_mock=True)
      trainer.fit=fit
      trainer.receipt=lambda *args:dict(target='KFD:1+AMD:LLVM:gfx1100',no_memory_planner=1)
      trainer.S.atomic_json=lambda path,value:reports.append((Path(path).name,value))
    def hooked_spec(name,location,*args,**kwargs):
      spec=original_spec(name,location,*args,**kwargs)
      if Path(location).resolve()!=HERE/'agent-distill-expert-train.py':return spec
      original=spec.loader
      class Loader:
        def create_module(self,spec):return original.create_module(spec)
        def exec_module(self,value):original.exec_module(value);configure(value)
      spec.loader=Loader();return spec
    old_argv=sys.argv[:]
    try:
      sys.argv=[str(HERE/'agent-distill-expert-ledger-run.py'),'--execute-training','--child',
        '--train-report','synthetic-train','--validation-report','synthetic-validation','--output-dir',str(root)]
      with patch.dict(os.environ,{'AGENT_DISTILL_EXPERT_PARENT_PID':str(os.getppid())}),\
           patch.object(tinygrad,'Device',FakeDevices()),patch.object(helpers,'DEV',Target()),\
           patch.object(amd,'KFDIface',Iface),patch.object(importlib.util,'spec_from_file_location',hooked_spec):
        runpy.run_path(str(HERE/'agent-distill-expert-ledger-run.py'),run_name='__main__')
        report=[value for name,value in reports if name=='report.json'][0]
        assert report['allocator_guards_retained_through_finalizers'] and report['allocation_snapshot_before_native_finalizers']
        assert Iface.alloc is not native_alloc and Iface.free is not real_free and Iface.unmap is not real_unmap
        assert report['allocation_budget']['live_bytes']==4096
        cumulative=report['allocation_budget']['requested_bytes']
        late.clear();gc.collect()
        assert report['allocation_budget']['live_bytes']==0 and report['allocation_budget']['requested_bytes']==cumulative
        assert not live_native
    finally:sys.argv=old_argv
  assert calls==['device-construction','first-weight-construction','unchanged-fit-call-boundary']
  result=dict(all_passed=True,gpu_executed=False,optimizer_updates=0,numerical_code_changed=False,
    actual_native_free_map_unmap_methods_used=True,owned_host_handle_reuse_passed=True,unowned_unmap_preserves_allocation=True,
    qualified_ledger_sha256=E['QUALIFIED_SHA'],unchanged_trainer_sha256=E['TRAINER_SHA'],basic_ledger=basic,
    actual_runpy_child_entrypoint=True,first_weight_path=calls,guards_retained_through_late_finalizer=True,
    final_live_bytes=report['allocation_budget']['live_bytes'],cumulative_never_reset=True,negative_controls=failures,
    source_sha256={str(path):B.sha(path) for path in (HERE/'agent-distill-expert-ledger-run.py',Path(__file__),E['QUALIFIED'])})
  (root/'agent-distill-expert-ledger-cpu.json').write_text(json.dumps(result,indent=2)+'\n')
  print(json.dumps(result,indent=2))


if __name__=='__main__':main()
