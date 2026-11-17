"""Source-bound, nonblocking, 300-second KFD1 L2048 parent diagnostic. Default is read-only preflight."""
import argparse
import fcntl
import importlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

C=importlib.import_module('agent-distill-full-common')
PROTOCOL=C.ROOT/'protocol.json'
OUTPUT=C.ROOT/'parent-gfx1100-kfd1'

def hardware_open_guard(permitted):
  def guard(event,args):
    if event=='open' and isinstance(args[0],(str,bytes)):
      path=os.fsdecode(args[0])
      if path.startswith(('/dev/dri/','/dev/kfd','/dev/nvidia','/dev/bus/usb/')):
        assert path in permitted,f'Unselected hardware open blocked: {path}'
  return guard

def require_available(snapshot,own_pid=None,upcoming=0,initial=False):
  assert snapshot['selected_kfd_index']==C.K.SELECTED_INDEX
  assert snapshot['selected_node']['gpu_id']==C.K.SELECTED_GPU_ID and snapshot['selected_node']['pci']==C.K.SELECTED_PCI
  assert snapshot['kfd_proc_root_read']
  assert all(p['complete'] for p in snapshot['kfd_processes'] if p['pid']!=own_pid),'Incomplete other-process KFD coverage'
  assert not (set(snapshot['observed_selected_kfd_owners'])-{own_pid}),'Observed selected-card KFD owner; preserve workload'
  required=C.FREE_RESERVE+(C.LIVE_LIMIT if initial else upcoming)
  assert snapshot['vram_free_bytes']>=required,'Selected-card free reserve would be violated'

class AllocationBudget:
  def __init__(self,selected,availability):
    self.selected=selected;self.availability=availability;self.handles={}
    self.report={'calls':0,'requested_bytes':0,'live_bytes':0,'peak_live_bytes':0,'maximum_request_bytes':0,
      'free_calls':0,'owned_unmap_calls':0,'live_limit':C.LIVE_LIMIT,'cumulative_limit':C.TOTAL_LIMIT,
      'single_buffer_limit':C.MAX_BUFFER,'selected_free_reserve':C.FREE_RESERVE,
      'accounting':'All KFD allocations, including host mappings; tracked by actual handle through free and owned unmap'}
  def validate(self,iface):
    assert iface.gpu_id==self.selected['gpu_id']==C.K.SELECTED_GPU_ID
    assert Path(iface.dev_sysfs_path).resolve().name==self.selected['pci']==C.K.SELECTED_PCI
  def wrap_alloc(self,original):
    def bounded(iface,size,*args,**kwargs):
      self.validate(iface);r=self.report
      assert 0<size<=C.MAX_BUFFER,'Individual KFD buffer cap exceeded'
      assert r['requested_bytes']+size<=C.TOTAL_LIMIT,'Cumulative KFD allocation cap exceeded'
      assert r['live_bytes']+size<=C.LIVE_LIMIT,'Live KFD allocation cap exceeded'
      require_available(self.availability(),os.getpid(),upcoming=size)
      r['requested_bytes']+=size;r['calls']+=1;r['maximum_request_bytes']=max(size,r['maximum_request_bytes'])
      storage=original(iface,size,*args,**kwargs);handle=int(storage.meta.handle)
      assert handle not in self.handles and int(storage.meta.size)==size
      self.handles[handle]=size;r['live_bytes']+=size;r['peak_live_bytes']=max(r['peak_live_bytes'],r['live_bytes'])
      return storage
    return bounded
  def release(self,handle):
    self.report['live_bytes']-=self.handles.pop(int(handle));assert self.report['live_bytes']>=0
  def wrap_free(self,original):
    def bounded(iface,storage):
      self.validate(iface);handle=int(storage.meta.handle);assert handle in self.handles
      result=original(iface,storage);self.release(handle);self.report['free_calls']+=1;return result
    return bounded
  def wrap_unmap(self,original):
    def bounded(iface,mapping):
      self.validate(iface);handle,owned=mapping.meta
      if owned:assert int(handle) in self.handles
      result=original(iface,mapping)
      if owned:self.release(handle);self.report['owned_unmap_calls']+=1
      return result
    return bounded

def verify(digest):
  assert not sys.flags.optimize and C.file_sha(PROTOCOL)==digest
  p=C.read(PROTOCOL)
  assert p['target']==C.TARGET and p['length']==C.LENGTH and p['valid_rows']==C.VALID_ROWS
  assert p['timeout_seconds']==300 and p['tolerances']==C.TOLERANCES and p['optimizer_updates']==0
  assert p['limits']=={'live':C.LIVE_LIMIT,'cumulative':C.TOTAL_LIMIT,'single_buffer':C.MAX_BUFFER,'free_reserve':C.FREE_RESERVE}
  for path,digest in p['source_sha256'].items():assert C.file_sha(path)==digest,path
  assert C.K.passive_topology()==p['passive_topology'],'KFD topology or access changed'
  C.K.selected_node(p['passive_topology'])
  oracle=C.read(C.ROOT/'oracle.json');assert oracle['complete'] and not oracle['gpu_executed'] and oracle['optimizer_updates']==0
  assert oracle['length']==2048 and oracle['tolerances']==C.TOLERANCES
  assert oracle['student_bf16_master_sha256']==C.read(C.OLD/'oracle.json')['student_bf16_master_sha256']
  return p

def environment():
  env=os.environ.copy()
  for key in list(env):
    if key.startswith(('AM_','AMD_','HCQ_')) or key in ('CPU_CC','CPU_LLVM','AMD_LLVM','AMD_CC'):env.pop(key,None)
  env.update(DEV=C.TARGET,TC='0',ALLOW_TF32='0',TRAINING='0',NO_MEMORY_PLANNER='1',PARALLEL='2',BEAM='0',JITBEAM='0',
    PROFILE='0',PMC='',SQTT='',VIZ='0',LRU='0',MAX_BUFFER_SIZE=str(C.MAX_BUFFER),ALLOW_DEVICE_USAGE='1',
    PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=':'.join(map(str,(C.ROOT/'frozen',C.TINY))),
    CACHEDB=str(C.ROOT/'cache/gpu-cache.db'),TMPDIR=str(C.ROOT/'tmp'),
    OPENBLAS_NUM_THREADS='4',OMP_NUM_THREADS='4',CUDA_VISIBLE_DEVICES='',HIP_VISIBLE_DEVICES='',ROCR_VISIBLE_DEVICES='')
  return env

def child(digest,output):
  p=verify(digest);C.no_gpu_fds();started=time.monotonic();selected=C.K.selected_node(p['passive_topology'])
  sys.addaudithook(hardware_open_guard({'/dev/kfd',f'/dev/dri/renderD{C.K.SELECTED_RENDER_MINOR}'}))
  availability=C.K.passive_availability();C.write(output/'availability-before-open.json',availability)
  require_available(availability,initial=True)
  from tinygrad import Device
  from tinygrad.helpers import DEV,PROFILE,MAX_BUFFER_SIZE,USE_TC,ALLOW_TF32,TRAINING,NO_MEMORY_PLANNER,LRU
  from tinygrad.runtime.ops_amd import KFDIface,PMC,SQTT
  from tinygrad.codegen import to_program_cache
  check=importlib.import_module('agent-distill-full-check');assert check.C is C
  assert repr(DEV)==C.TARGET and DEV.target('AMD').indices=='1' and not Device._opened_devices
  assert PROFILE.value==PMC.value==SQTT.value==0
  assert MAX_BUFFER_SIZE.value==C.MAX_BUFFER and NO_MEMORY_PLANNER.value==1 and LRU.value==0
  assert USE_TC.value==ALLOW_TF32.value==TRAINING.value==0
  budget=AllocationBudget(selected,C.K.passive_availability)
  originals=(KFDIface.alloc,KFDIface.free,KFDIface.unmap)
  KFDIface.alloc=budget.wrap_alloc(originals[0]);KFDIface.free=budget.wrap_free(originals[1]);KFDIface.unmap=budget.wrap_unmap(originals[2])
  report={'complete':False,'gpu_executed':False,'optimizer_updates':0,'protocol_sha256':digest,
    'target':C.TARGET,'length':C.LENGTH,'allocation_budget':budget.report,'availability_before_open':availability,'stages':[]}
  def stage(name):
    snapshot=C.K.passive_availability();require_available(snapshot,os.getpid())
    record={'stage':name,'seconds':time.monotonic()-started,'availability':snapshot,'allocation':dict(budget.report)}
    report['stages'].append(record);C.write(output/'progress.json',report)
  try:
    dev=Device['AMD']
    assert type(dev.iface) is KFDIface
    actual={'requested_kfd_index':1,'logical_device_id':dev.device_id,'filtered_kfd_nodes':list(KFDIface.gpus),
      'interface':type(dev.iface).__name__,'arch':dev.arch,'gpu_id':dev.iface.gpu_id,
      'pci':Path(dev.iface.dev_sysfs_path).resolve().name,'render_minor':dev.iface.props['drm_render_minor'],
      'renderer_target':repr(dev.renderer.target),'allocator':type(dev.allocator).__name__,'can_recover':dev.can_recover,
      'is_usb':dev.is_usb,'native_pci':dev.is_am()}
    C.K.validate_opened_identity(actual,selected);report['actual_device']=actual;report['gpu_executed']=True;stage('device-identity-validated')
    report['numerical']=check.execute('AMD',stage);assert report['numerical']['all_passed']
    assert set(Device._opened_devices)<={'AMD','CPU','NPY'}
    report['compiled_programs']=[{'input_ast':key[0].hex(),'renderer':key[1].__module__+'.'+key[1].__name__,
      'target':repr(key[2]),'program_key':program.key.hex(),'source_sha256':C.B.sha(program.src[2].arg.encode()),
      'binary_sha256':C.B.sha(program.src[3].arg)} for key,program in to_program_cache.items() if key[2].device=='AMD']
    assert report['compiled_programs'] and all(x['target']==C.TARGET for x in report['compiled_programs'])
    report['complete']=True
  except BaseException as exc:
    report['failure']={'type':type(exc).__name__,'message':str(exc)};raise
  finally:
    report['seconds']=time.monotonic()-started;C.write(output/'report.json',report)
    # Keep wrappers active through buffer finalizers. Parent process exit releases its own KFD context.

def main():
  parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--protocol-sha',required=True)
  group=parser.add_mutually_exclusive_group();group.add_argument('--execute-reviewed',action='store_true');group.add_argument('--child',action='store_true')
  parser.add_argument('--output',type=Path,default=OUTPUT);args=parser.parse_args();verify(args.protocol_sha)
  assert args.output.is_relative_to(C.ROOT) and args.output!=C.ROOT
  if args.child:return child(args.protocol_sha,args.output)
  assert args.output==OUTPUT,'Only the CPU-blocked child preflight can redirect output'
  C.no_gpu_fds();snapshot=C.K.passive_availability();require_available(snapshot,initial=True)
  if not args.execute_reviewed:
    print(json.dumps({'preflight_passed':True,'gpu_executed':False,'availability':snapshot,'limits':C.read(PROTOCOL)['limits']},indent=2));return
  qualification=C.read(C.ROOT/'cpu-check.json')
  assert qualification['complete'] and qualification['protocol_sha256']==args.protocol_sha and not qualification['gpu_executed']
  assert not OUTPUT.exists(),'Never overwrite evidence'
  with Path('/tmp/opencode/alpamayo-chestnut-speed.lock').open('a') as lock:
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    verify(args.protocol_sha);snapshot=C.K.passive_availability();require_available(snapshot,initial=True)
    OUTPUT.mkdir();C.write(OUTPUT/'availability-before-launch.json',snapshot)
    command=[str(C.PY),str(Path(__file__).resolve()),'--protocol-sha',args.protocol_sha,'--child']
    C.write(OUTPUT/'invocation.json',{'command':command,'timeout_seconds':300,'optimizer_updates':0,'parent_latest_availability_review_required':True})
    begin=time.monotonic();timed_out=False
    with (OUTPUT/'run.log').open('w') as log:
      process=subprocess.Popen(command,env=environment(),stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
      try:code=process.wait(timeout=300)
      except subprocess.TimeoutExpired:
        timed_out=True;os.killpg(process.pid,signal.SIGTERM)
        try:code=process.wait(timeout=5)
        except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);code=process.wait()
    C.write(OUTPUT/'wall.json',{'seconds':time.monotonic()-begin,'returncode':code,'timed_out':timed_out})
    assert not timed_out and code==0
    report=C.read(OUTPUT/'report.json');assert report['complete'] and report['numerical']['all_passed']
    print(json.dumps({'complete':True,'report':str(OUTPUT/'report.json'),'optimizer_updates':0}))

if __name__=='__main__':main()
