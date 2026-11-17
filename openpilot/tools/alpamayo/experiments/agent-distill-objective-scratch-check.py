"""CPU native scratch-link/cap checks and the complete overlaid parent/child lifecycle."""
import argparse
import gc
import inspect
import json
import os
from pathlib import Path
import runpy
import sys
from types import SimpleNamespace
from unittest.mock import patch

sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
BASE=Path('/tmp/opencode/alpamayo-asius-20261114')
os.environ.update(DEV='CPU',TC='0',ALLOW_TF32='0',NO_MEMORY_PLANNER='1',LRU='0',PROFILE='0',
  MAX_BUFFER_SIZE=str(320*1024**2),OPENBLAS_NUM_THREADS='4',OMP_NUM_THREADS='4')
sys.path.insert(0,str(BASE/'tinygrad'))


def no_gpu(event,args):
  if event=='open' and isinstance(args[0],(str,bytes)) and os.fsdecode(args[0]).startswith(('/dev/dri','/dev/kfd','/dev/nvidia','/dev/bus/usb')):
    raise RuntimeError('Scratch checks are CPU-only')
sys.addaudithook(no_gpu)
R=runpy.run_path(str(HERE/'agent-distill-objective-scratch-run.py'));Q=R['Q']


def scratch_link():
  import tinygrad.device as device
  from tinygrad import Device,dtypes
  from tinygrad.helpers import Context
  from tinygrad.uop.ops import UOp
  from tinygrad.runtime import ops_amd as amd
  from tinygrad.runtime.support import hcq2
  ledger_impl=Q.module('scratch_check_qualified_ledger',R['R'].LEDGER)
  selected=dict(gpu_id=27211,pci='0000:0c:00.0')
  availability=lambda:dict(selected_node=selected,kfd_proc_root_read=True,kfd_processes=[],observed_selected_kfd_owners=[],vram_free_bytes=24*1024**3)
  ledger=ledger_impl.Ledger(dict(Q.BOUNDS,selected_gpu_id=27211,selected_pci='0000:0c:00.0'),availability)
  live={};events=[];next_handle=[0]
  def alloc(iface,size,*args,**kwargs):
    next_handle[0]+=1;handle=next_handle[0];live[handle]=size;events.append(dict(event='mock-native-allocation',bytes=size))
    return amd.BufferStorage(0x1000+handle*0x1000,SimpleNamespace(handle=handle,size=size))
  def release(fd,handle):live.pop(handle)
  iface=SimpleNamespace(gpu_id=27211,dev_sysfs_path='/synthetic/0000:0c:00.0',kfd=None,props={'max_slots_scratch_cu':32},
    _unmap_handle=lambda handle:None)
  guarded_alloc=ledger.alloc(alloc);guarded_free=ledger.free(amd.KFDIface.free)
  allocator=SimpleNamespace(alloc=lambda size,options:guarded_alloc(iface,size),free=lambda storage,size,options:guarded_free(iface,storage))
  dev=SimpleNamespace(device='AMD',target=(11,0,0),cu_cnt=96,xccs=1,max_private_segment_size=0,iface=iface,allocator=allocator)
  class Devices:
    canonicalize=staticmethod(Device.canonicalize)
    def __getitem__(self,key):return dev if key=='AMD' else Device[key]
  dev.scratch_buffer=lambda size:amd.AMDDevice.scratch_buffer(dev,size)
  def linked(private):return hcq2.bufferize_buf(hcq2.LinkCtx([]),UOp.alloc((private,),dtypes.uint8,0,device='AMD',name='scratch'))
  rejected=[]
  def fail(name,call,exception):
    try:call()
    except exception:rejected.append(name)
    else:raise AssertionError('Accepted '+name)
  with patch.object(device,'Device',Devices()),patch.object(hcq2,'Device',Devices()),\
       patch.dict(device.Compiled.link_bufs,{hcq2.to_name('AMD','scratch'):lambda b:dev.scratch_buffer(b.max_numel())}),\
       patch.object(amd.AMDDevice,'max_scratch_psize',0),patch.object(amd.kfd,'AMDKFD_IOC_FREE_MEMORY_OF_GPU',release),\
       patch.object(amd.FileIOInterface,'munmap',lambda *args:None):
    with Context(MAX_BUFFER_SIZE=256*1024**2):fail('original native Buffer cap',lambda:linked(1424),RuntimeError)
    assert ledger.report['calls']==0
    with Context(MAX_BUFFER_SIZE=320*1024**2):
      first=linked(1424);assert first.buffer.nbytes==279969792 and ledger.report['calls']==1
      second=linked(176);assert second.buffer is first.buffer and ledger.report['calls']==1
      fail('larger native scratch exceeds new cap',lambda:linked(1708),RuntimeError)
      fail('native Buffer cap plus one',lambda:device.Buffer('AMD',320*1024**2+1,preallocate=True),RuntimeError)
      assert ledger.report['calls']==1
      boundary=guarded_alloc(iface,320*1024**2);guarded_free(iface,boundary)
      fail('qualified ledger cap plus one',lambda:guarded_alloc(iface,320*1024**2+1),AssertionError)
      assert ledger.report['calls']==2
      first.buffer.deallocate();del first,second;del dev.scratch;gc.collect()
    assert not live and not ledger.handles and ledger.report['live_bytes']==0
  return dict(all_passed=True,native_hcq_bufferize=True,native_amd_scratch_formula=True,native_Buffer_allocate=True,
    actual_scratch_bytes=279969792,physical_storage_mocked=True,kernels_executed=0,
    existing_scratch_reused=True,rejected=rejected,allocation_events=events,ledger=ledger.report,
    unchanged_live_cumulative_reserve_guards=True)


def zero_state():
  import numpy as np
  context=Q.read(R['FAILED']/'report.json')['context']
  transition=R['R'].scratch_transition(context)
  values=Q.seed_arrays();shapes={name:value.shape for name,value in sorted(values.items())}
  arrays,metadata=Q.S.load(R['FAILED'],context,shapes)
  assert len(arrays)==51 and metadata['step']==0
  for name,value in values.items():
    np.testing.assert_array_equal(arrays['weight/'+name].view(np.uint32),value.view(np.uint32))
    assert np.count_nonzero(arrays['moment1/'+name])==np.count_nonzero(arrays['moment2/'+name])==0
  assert arrays['beta1_power'].item()==arrays['beta2_power'].item()==1
  assert Q.read(R['FAILED']/'seed-state/latest.json')['sha256']==R['STATE_SHA']
  assert Q.sha(R['FAILED']/'seed-state/checkpoint-0000.npz')==R['STATE_SHA']
  bad=dict(context,kind='changed-context')
  try:R['R'].scratch_transition(bad)
  except ValueError:pass
  else:raise AssertionError('Accepted restamped context')
  return dict(actual_completed_zero_state_checked=True,arrays=51,weights_equal_both_selected_seeds=True,
    moments_zero=True,beta_powers_one=True,clean_journal=True,seed_state_same_checksum=True,
    changed_context_rejected=True,optimizer_updates=0,transition=transition)


def entrypoint(root):
  from tinygrad import Device
  from tinygrad.helpers import Context
  Device['CPU'] # Import the actual CPU backend before the KFD identity mock replaces DEV.
  prior=Q.module('scratch_reused_entry_check',HERE/'agent-distill-objective-qualify-check.py')
  prior.Q=Q;prior.R=R
  source=inspect.getsource(prior.entrypoint)
  source=source.replace('agent-distill-objective-qualify-run.py','agent-distill-objective-scratch-run.py')
  before="main.__globals__['admission']=lambda *a:dict(target=R['TARGET'],no_memory_planner=1)"
  assert source.count(before)==1
  source=source.replace(before,before+"\n      main.__globals__['scratch_transition']=lambda context:dict(synthetic_mock=True)")
  exec(compile(source,str(Path(__file__))+':reused-entrypoint','exec'),prior.__dict__)
  with Context(MAX_BUFFER_SIZE=320*1024**2):result=prior.entrypoint(root)
  report=Q.read(root/'entrypoint/report.json')
  assert report['bounds']['max_single_buffer']==report['allocation_budget']['single_buffer_limit']==320*1024**2
  assert report['bounds']==Q.BOUNDS and report['resource_transition']['synthetic_mock']
  assert report['recovery_arrays']==51 and report['components']['total']==0.2020806074142456
  assert report['masked_context_gradient_exactly_zero'] and all(v>0 for v in report['expert_to_student_context_gradient_norms'])
  result.update(new_cap_in_actual_child_and_ledger=True,resource_transition_written=True,
    synthetic_connected_loss=report['components']['total'],all51_recovered=True,
    gradients_reach_both_students=True,masked_context_gradients_zero=True)
  return result


def main():
  parser=argparse.ArgumentParser();parser.add_argument('--output-dir',type=Path,required=True);args=parser.parse_args()
  root=args.output_dir.resolve()
  if root.exists() or not root.is_relative_to(R['ROOT']):raise ValueError('Fresh owned CPU check directory required')
  root.mkdir(parents=True)
  result=dict(gpu_executed=False,actual_data_optimizer_updates=0,scratch_link=scratch_link(),zero_state=zero_state(),entrypoint=entrypoint(root))
  from tinygrad import Device
  assert all(name.startswith(('CPU','NPY','PYTHON')) for name in Device._opened_devices)
  result.update(all_passed=True,new_bounds=Q.BOUNDS,opened_devices=list(Device._opened_devices),
    source_sha256={str(p):Q.sha(p) for p in (*HERE.glob('agent-distill-objective-scratch-*.py'),
      HERE/'agent-distill-objective-qualify-core.py',HERE/'agent-distill-objective-qualify-run.py',HERE/'agent-distill-objective-core.py')})
  (root/'report.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
