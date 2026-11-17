"""Parent-only single KFD FP32 forward/backward diagnostic; defaults to passive preflight."""
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

C=importlib.import_module('agent-distill-tinygrad-common')
PROTOCOL=C.ROOT/'protocol.json'
OUTPUT=C.ROOT/'parent-gfx1100'
LIMIT=2*1024**3

def allocation_guard(original,selected,budget):
  def bounded(self,size,*args,**kwargs):
    assert self.gpu_id==selected['gpu_id'] and Path(self.dev_sysfs_path).resolve().name==C.SELECTED_PCI
    assert size>0 and budget['requested_bytes']+size<=LIMIT,'Cumulative KFD allocation budget exceeded'
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
  assert p['target']==C.TARGET and p['selected_pci']==C.SELECTED_PCI and p['timeout_seconds']==300
  assert p['cumulative_kfd_allocation_limit']==LIMIT and p['tolerances']==C.TOLERANCES
  for path,expected in p['source_sha256'].items():assert C.file_sha(path)==expected,path
  assert C.file_sha(C.ROOT/'oracle.json')==p['oracle_sha256']
  assert C.file_sha(C.ROOT/'cpu-check-matched.json')==p['cpu_evidence_sha256']
  cpu=C.read(C.ROOT/'cpu-check-matched.json');assert cpu['complete'] and cpu['all_passed'] and not cpu['gpu_executed']
  current=C.passive_topology()
  assert current==p['passive_topology'],'KFD enumeration, PCI identity, capacity, driver, or access changed'
  selected=current['nodes'][0]
  assert selected['pci']==C.SELECTED_PCI and selected['gfx_target_version']==110000
  assert selected['vendor']=='0x1002' and selected['device']=='0x744c' and selected['driver']=='amdgpu'
  assert selected['render_rw_access'] and current['kfd_rw_access']
  return p

def environment():
  env=os.environ.copy()
  # Explicit interface selection prevents native-PCI and USB fallback. No power/profiling controls.
  for key in list(env):
    if key.startswith(('AM_','AMD_','HCQ_')) or key in ('CPU_CC','CPU_LLVM','AMD_LLVM','AMD_CC'):
      env.pop(key,None)
  env.update(DEV=C.TARGET,TC='0',ALLOW_TF32='0',TRAINING='0',PARALLEL='2',BEAM='0',JITBEAM='0',
    PROFILE='0',PMC='0',SQTT='0',LRU='0',MAX_BUFFER_SIZE=str(128*1024**2),ALLOW_DEVICE_USAGE='1',
    PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=str(C.ROOT/'frozen')+':'+str(C.TINY),
    CACHEDB=str(C.ROOT/'gpu-cache.db'),TMPDIR=str(C.ROOT/'tmp'),
    OPENBLAS_NUM_THREADS='4',OMP_NUM_THREADS='4',CUDA_VISIBLE_DEVICES='',HIP_VISIBLE_DEVICES='',ROCR_VISIBLE_DEVICES='')
  return env

def child(digest):
  p=verify(digest);C.no_gpu_fds();started=time.monotonic()
  selected=p['passive_topology']['nodes'][0]
  permitted={'/dev/kfd',f'/dev/dri/renderD{selected["render_minor"]}'}
  sys.addaudithook(hardware_open_guard(permitted))
  from tinygrad import Device
  from tinygrad.helpers import DEV,PROFILE,PMC,SQTT,MAX_BUFFER_SIZE
  from tinygrad.runtime.ops_amd import KFDIface
  assert repr(DEV)==C.TARGET and not Device._opened_devices
  assert PROFILE.value==PMC.value==SQTT.value==0 and MAX_BUFFER_SIZE.value==128*1024**2
  # Bound every KFD allocation, including host mappings and queues. Count cumulative
  # requested bytes without subtracting frees, so live requested storage cannot exceed it.
  original_alloc=KFDIface.alloc;budget={'requested_bytes':0,'calls':0,'limit_bytes':LIMIT}
  KFDIface.alloc=allocation_guard(original_alloc,selected,budget)
  report={'complete':False,'gpu_executed':False,'optimizer_updates':0,'protocol_sha256':digest,
    'selected_pci':C.SELECTED_PCI,'target':C.TARGET,'allocation_budget':budget}
  try:
    dev=Device['AMD']
    assert type(dev.iface) is KFDIface and not dev.is_usb and not dev.is_am() and not dev.can_recover
    assert dev.arch=='gfx1100' and dev.iface.gpu_id==selected['gpu_id']
    assert Path(dev.iface.dev_sysfs_path).resolve().name==C.SELECTED_PCI
    report['actual_device']={'interface':type(dev.iface).__name__,'arch':dev.arch,'gpu_id':dev.iface.gpu_id,
      'pci':Path(dev.iface.dev_sysfs_path).resolve().name,'renderer_target':repr(dev.renderer.target),
      'allocator':type(dev.allocator).__name__,'can_recover':dev.can_recover,'is_usb':dev.is_usb}
    # One student forward and one explicit reverse-mode gradient calculation; no controls replayed on GPU.
    report['gpu_executed']=True
    check=importlib.import_module('agent-distill-tinygrad-check')
    report['numerical']=check.execute('AMD',controls=False)
    assert report['numerical']['all_passed']
    assert set(Device._opened_devices)<= {'AMD','CPU','NPY'}
    from tinygrad.codegen import to_program_cache
    report['compiled_programs']=[{'input_ast':key[0].hex(),'renderer':key[1].__module__+'.'+key[1].__name__,
      'target':repr(key[2]),'program_key':program.key.hex(),'source_sha256':C.sha(program.src[2].arg.encode()),
      'binary_sha256':C.sha(program.src[3].arg)} for key,program in to_program_cache.items() if key[2].device=='AMD']
    assert report['compiled_programs'] and all(x['target']==C.TARGET for x in report['compiled_programs'])
    report['complete']=True
  except BaseException as exc:
    report['failure']={'type':type(exc).__name__,'message':str(exc)}
    raise
  finally:
    # Retain the bound until process exit/finalization. It does not reset or alter drivers.
    report['seconds']=time.monotonic()-started
    C.write(OUTPUT/'report.json',report)

def main():
  parser=argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--protocol-sha',required=True)
  group=parser.add_mutually_exclusive_group();group.add_argument('--execute-reviewed',action='store_true');group.add_argument('--child',action='store_true')
  args=parser.parse_args();p=verify(args.protocol_sha)
  if args.child:return child(args.protocol_sha)
  C.no_gpu_fds()
  if not args.execute_reviewed:
    print(json.dumps({'preflight_passed':True,'gpu_executed':False,'selected_pci':C.SELECTED_PCI,'target':C.TARGET,
      'parent_must_verify':'Exclusive availability and safe allocator access before --execute-reviewed; capacity/access bits alone do not establish this',
      'timeout_seconds':300,'cumulative_kfd_allocation_limit':LIMIT},indent=2));return
  assert not OUTPUT.exists(),'Never overwrite prior evidence'
  # Reuse existing Chestnut ownership even though explicit KFD and open guards exclude USB.
  with Path('/tmp/opencode/alpamayo-chestnut-speed.lock').open('a') as lock:
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    verify(args.protocol_sha);OUTPUT.mkdir()
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
