"""KFD1 parent diagnostic with corrected pinned PMC/SQTT imports. Default: passive preflight."""
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

C=importlib.import_module('agent-distill-kfd1-import-common')
PROTOCOL=C.ROOT/'protocol.json'
OUTPUT=C.ROOT/'parent-gfx1100-kfd1'

def allocation_guard(original,selected,budget):
  def bounded(self,size,*args,**kwargs):
    assert self.gpu_id==selected['gpu_id']==C.SELECTED_GPU_ID
    assert Path(self.dev_sysfs_path).resolve().name==selected['pci']==C.SELECTED_PCI
    assert size>0 and budget['requested_bytes']+size<=C.LIMIT,'Cumulative KFD allocation budget exceeded'
    budget['requested_bytes']+=size;budget['calls']+=1
    return original(self,size,*args,**kwargs)
  return bounded

def hardware_open_guard(permitted):
  def guard(event,args):
    if event=='open' and isinstance(args[0],(str,bytes)):
      path=os.fsdecode(args[0])
      if path.startswith(('/dev/dri/','/dev/kfd','/dev/nvidia','/dev/bus/usb/')):
        assert path in permitted,f'Unselected hardware open blocked: {path}'
  return guard

def verify(digest):
  assert not sys.flags.optimize and C.file_sha(PROTOCOL)==digest
  p=C.read(PROTOCOL)
  assert p['target']==C.TARGET and p['selected_kfd_index']==C.SELECTED_INDEX
  assert p['selected_pci']==C.SELECTED_PCI and p['selected_gpu_id']==C.SELECTED_GPU_ID
  assert p['selected_render_minor']==C.SELECTED_RENDER_MINOR and p['selected_node']==C.SELECTED_NODE
  assert p['timeout_seconds']==300 and p['cumulative_kfd_allocation_limit']==C.LIMIT
  assert p['max_single_buffer_bytes']==128*1024**2 and p['tolerances']==C.TOLERANCES
  for path,expected in p['source_sha256'].items():assert C.file_sha(path)==expected,path
  assert C.file_sha(C.OLD/'protocol.json')==C.OLD_PROTOCOL_SHA==p['original_protocol_sha256']
  assert C.file_sha(C.OLD/'oracle.json')==p['oracle_sha256']
  assert C.file_sha(C.OLD/'cpu-check-matched.json')==p['cpu_evidence_sha256']
  cpu=C.read(C.OLD/'cpu-check-matched.json');assert cpu['complete'] and cpu['all_passed'] and not cpu['gpu_executed']
  current=C.passive_topology()
  assert current==p['passive_topology'],'KFD enumeration, PCI identity, capacity, driver, or access changed'
  C.selected_node(current)
  return p

def environment():
  env=os.environ.copy()
  for key in list(env):
    if key.startswith(('AM_','AMD_','HCQ_')) or key in ('CPU_CC','CPU_LLVM','AMD_LLVM','AMD_CC'):env.pop(key,None)
  env.update(DEV=C.TARGET,TC='0',ALLOW_TF32='0',TRAINING='0',PARALLEL='2',BEAM='0',JITBEAM='0',
    PROFILE='0',PMC='0',SQTT='0',LRU='0',MAX_BUFFER_SIZE=str(128*1024**2),ALLOW_DEVICE_USAGE='1',
    PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=':'.join(map(str,(C.ROOT/'frozen',C.NUMERIC,C.TINY))),
    CACHEDB=str(C.ROOT/'gpu-cache.db'),TMPDIR=str(C.ROOT/'tmp'),
    OPENBLAS_NUM_THREADS='4',OMP_NUM_THREADS='4',CUDA_VISIBLE_DEVICES='',HIP_VISIBLE_DEVICES='',ROCR_VISIBLE_DEVICES='')
  return env

def child(digest):
  p=verify(digest);C.no_gpu_fds();started=time.monotonic()
  selected=C.selected_node(p['passive_topology'])
  permitted={'/dev/kfd',f'/dev/dri/renderD{C.SELECTED_RENDER_MINOR}'}
  sys.addaudithook(hardware_open_guard(permitted))
  availability=C.passive_availability();C.write(OUTPUT/'availability-before-open.json',availability)
  C.require_no_observed_conflict(availability)
  from tinygrad import Device
  from tinygrad.helpers import DEV,PROFILE,MAX_BUFFER_SIZE
  from tinygrad.runtime.ops_amd import KFDIface,PMC,SQTT
  assert repr(DEV)==C.TARGET and DEV.target('AMD').indices==str(C.SELECTED_INDEX) and not Device._opened_devices
  assert PROFILE.value==PMC.value==SQTT.value==0 and MAX_BUFFER_SIZE.value==128*1024**2
  original_alloc=KFDIface.alloc;budget={'requested_bytes':0,'calls':0,'limit_bytes':C.LIMIT}
  KFDIface.alloc=allocation_guard(original_alloc,selected,budget)
  report={'complete':False,'gpu_executed':False,'optimizer_updates':0,'protocol_sha256':digest,
    'selected_pci':C.SELECTED_PCI,'selected_kfd_index':C.SELECTED_INDEX,'target':C.TARGET,'allocation_budget':budget,
    'availability_before_open':availability,'original_protocol_sha256':C.OLD_PROTOCOL_SHA}
  try:
    dev=Device['AMD']
    assert type(dev.iface) is KFDIface
    actual={'requested_kfd_index':C.SELECTED_INDEX,'logical_device_id':dev.device_id,
      'filtered_kfd_nodes':list(KFDIface.gpus),'interface':type(dev.iface).__name__,'arch':dev.arch,
      'gpu_id':dev.iface.gpu_id,'pci':Path(dev.iface.dev_sysfs_path).resolve().name,
      'render_minor':dev.iface.props['drm_render_minor'],'renderer_target':repr(dev.renderer.target),
      'allocator':type(dev.allocator).__name__,'can_recover':dev.can_recover,'is_usb':dev.is_usb,'native_pci':dev.is_am()}
    C.validate_opened_identity(actual,selected);report['actual_device']=actual
    report['gpu_executed']=True
    check=C.import_numeric('agent-distill-tinygrad-check',C.NUMERIC_CHECK_SHA)
    assert check.C is C.B
    report['numerical']=check.execute('AMD',controls=False)
    assert report['numerical']['all_passed'] and report['numerical']['optimizer_updates']==0
    assert set(Device._opened_devices)<= {'AMD','CPU','NPY'}
    from tinygrad.codegen import to_program_cache
    report['compiled_programs']=[{'input_ast':key[0].hex(),'renderer':key[1].__module__+'.'+key[1].__name__,
      'target':repr(key[2]),'program_key':program.key.hex(),'source_sha256':C.B.sha(program.src[2].arg.encode()),
      'binary_sha256':C.B.sha(program.src[3].arg)} for key,program in to_program_cache.items() if key[2].device=='AMD']
    assert report['compiled_programs'] and all(x['target']==C.TARGET for x in report['compiled_programs'])
    report['complete']=True
  except BaseException as exc:
    report['failure']={'type':type(exc).__name__,'message':str(exc)};raise
  finally:
    report['seconds']=time.monotonic()-started;C.write(OUTPUT/'report.json',report)

def main():
  parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--protocol-sha',required=True)
  group=parser.add_mutually_exclusive_group();group.add_argument('--execute-reviewed',action='store_true');group.add_argument('--child',action='store_true')
  args=parser.parse_args();verify(args.protocol_sha)
  if args.child:return child(args.protocol_sha)
  C.no_gpu_fds();availability=C.passive_availability()
  if not args.execute_reviewed:
    print(json.dumps({'identity_preflight_passed':True,'gpu_executed':False,'target':C.TARGET,
      'availability':availability,'timeout_seconds':300,'cumulative_kfd_allocation_limit':C.LIMIT},indent=2));return
  assert not OUTPUT.exists(),'Never overwrite prior evidence'
  C.require_no_observed_conflict(availability)
  # Preserve shared Chestnut ownership; do not wait for, poll, or alter its current owner.
  with Path('/tmp/opencode/alpamayo-chestnut-speed.lock').open('a') as lock:
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    verify(args.protocol_sha);availability=C.passive_availability();C.require_no_observed_conflict(availability)
    OUTPUT.mkdir();C.write(OUTPUT/'availability-before-launch.json',availability)
    command=[str(C.PY),str(Path(__file__).resolve()),'--protocol-sha',args.protocol_sha,'--child']
    C.write(OUTPUT/'invocation.json',{'command':command,'target':C.TARGET,'timeout_seconds':300,
      'parent_availability_review_required':True,'optimizer_updates':0})
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
