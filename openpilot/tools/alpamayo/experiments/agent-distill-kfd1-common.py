"""KFD ordinal-one derivative. Numeric code and oracle remain the exact frozen originals."""
import importlib.util
import os
from pathlib import Path
import sys
import time

EXP=Path('/tmp/opencode/openpilot-alpamayo/openpilot/tools/alpamayo/experiments')
WORK=Path('/mnt/nvme-500/tmp/alpamayo-speed-20261116')
OLD=WORK/'agent-distill-tinygrad-artifacts'
ROOT=WORK/'agent-distill-kfd1-artifacts'
NUMERIC=OLD/'frozen'
TARGET='KFD:1+AMD:LLVM:gfx1100'
SELECTED_INDEX=1
SELECTED_PCI='0000:0c:00.0'
SELECTED_NODE='2'
SELECTED_GPU_ID=27211
SELECTED_RENDER_MINOR=129
LIMIT=2*1024**3
OLD_PROTOCOL_SHA='87f08a0d0ec7eaee93d41681ecba58e6e89d18c093edf5be6d8085ae36c5310c'
NUMERIC_COMMON_SHA='4ed60b0ffc6d40ccf6bdf4cc55965465249aae40bb34878be95b3b3a138fbee0'
NUMERIC_CHECK_SHA='a30efa2db60dd0f4ce9e727261c88ecb816572a527b21c1f8ce820d75e51d5a6'

def import_numeric(name,digest):
  import hashlib
  path=NUMERIC/(name+'.py')
  assert hashlib.sha256(path.read_bytes()).hexdigest()==digest
  if name in sys.modules:
    module=sys.modules[name];assert Path(module.__file__).resolve()==path
  else:
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec)
    sys.modules[name]=module;spec.loader.exec_module(module)
  return module

B=import_numeric('agent-distill-tinygrad-common',NUMERIC_COMMON_SHA)
file_sha,read,write,no_gpu_fds,passive_topology=B.file_sha,B.read,B.write,B.no_gpu_fds,B.passive_topology
TINY,PY,TOLERANCES=B.TINY,B.PY,B.TOLERANCES

def selected_node(topology):
  row=topology['nodes'][SELECTED_INDEX]
  assert row['node']==SELECTED_NODE and row['pci']==SELECTED_PCI and row['gpu_id']==SELECTED_GPU_ID
  assert row['render_minor']==SELECTED_RENDER_MINOR and row['gfx_target_version']==110000
  assert row['vendor']=='0x1002' and row['device']=='0x744c' and row['driver']=='amdgpu'
  assert row['render_rw_access'] and topology['kfd_rw_access']
  assert sum(r['gpu_id']==SELECTED_GPU_ID for r in topology['nodes'])==1
  return row

def validate_opened_identity(actual,selected):
  assert actual['requested_kfd_index']==SELECTED_INDEX and actual['logical_device_id']==0
  assert actual['filtered_kfd_nodes']==[SELECTED_NODE] and actual['interface']=='KFDIface'
  assert actual['gpu_id']==selected['gpu_id']==SELECTED_GPU_ID
  assert actual['pci']==selected['pci']==SELECTED_PCI
  assert actual['render_minor']==selected['render_minor']==SELECTED_RENDER_MINOR
  assert actual['arch']=='gfx1100' and actual['renderer_target']==TARGET
  assert not actual['is_usb'] and not actual['native_pci'] and not actual['can_recover']
  assert actual['allocator']=='AMDAllocator'

def passive_availability():
  """Read sysfs only. Does not open KFD, query queues by ioctl, or inspect other jobs' logs."""
  topology=passive_topology();selected=selected_node(topology)
  dev=Path('/sys/bus/pci/devices')/SELECTED_PCI
  total=int((dev/'mem_info_vram_total').read_text());used=int((dev/'mem_info_vram_used').read_text())
  assert 0<=used<=total
  errors=[]
  try:busy=int((dev/'gpu_busy_percent').read_text())
  except OSError as exc:busy=None;errors.append({'path':str(dev/'gpu_busy_percent'),'error':type(exc).__name__})
  root=Path('/sys/class/kfd/kfd/proc');processes=[];root_read=True
  try:pids=sorted(root.iterdir(),key=lambda p:int(p.name))
  except OSError as exc:pids=[];root_read=False;errors.append({'path':str(root),'error':type(exc).__name__})
  for pid in pids:
    row={'pid':int(pid.name),'vram_by_gpu_id':{},'queues':[],'complete':True}
    try:
      for gpu in topology['nodes']:
        row['vram_by_gpu_id'][str(gpu['gpu_id'])]=int((pid/f'vram_{gpu["gpu_id"]}').read_text())
      for queue in sorted((pid/'queues').iterdir(),key=lambda p:int(p.name)):
        row['queues'].append({'queue':queue.name,'gpu_id':int((queue/'gpuid').read_text()),
                             'type':int((queue/'type').read_text()),'size':int((queue/'size').read_text())})
    except (OSError,ValueError) as exc:
      row['complete']=False;errors.append({'path':str(pid),'error':type(exc).__name__})
    processes.append(row)
  owners=[p['pid'] for p in processes if p['vram_by_gpu_id'].get(str(SELECTED_GPU_ID),0)>0 or
          any(q['gpu_id']==SELECTED_GPU_ID for q in p['queues'])]
  return {'captured_unix_ns':time.time_ns(),'selected_kfd_index':SELECTED_INDEX,'selected_node':selected,
    'vram_total_bytes':total,'vram_used_bytes':used,'vram_free_bytes':total-used,'gpu_busy_percent_single_sample':busy,
    'kfd_proc_path':str(root),'kfd_proc_root_read':root_read,'kfd_processes':processes,
    'kfd_pid_coverage_complete':root_read and all(p['complete'] for p in processes),
    'observed_selected_kfd_owners':owners,'errors':errors,'proc_fd_scan_performed':False,
    'availability_established':False,'no_device_opened':True,
    'coverage_limit':'One sysfs snapshot. KFD records exclude non-KFD clients; no universal process-FD visibility or exclusive ownership is proved.',
    'parent_context':'Parent previously found 465 unreadable proc/fd directories. This scan does not refresh that count.',
    'parent_review':'Parent must review latest availability before execution. Free VRAM and one busy sample do not prove idle.'}

def require_no_observed_conflict(snapshot):
  assert snapshot['selected_kfd_index']==SELECTED_INDEX
  assert snapshot['selected_node']['gpu_id']==SELECTED_GPU_ID and snapshot['selected_node']['pci']==SELECTED_PCI
  assert snapshot['kfd_pid_coverage_complete'],'Incomplete KFD process/queue coverage'
  assert not snapshot['observed_selected_kfd_owners'],'Selected card has an observed KFD owner; preserve the workload'
  assert snapshot['vram_free_bytes']>=LIMIT,'Insufficient observed free VRAM for the requested allocation cap'
  # The busy sample stays visible for parent review; neither zero nor nonzero proves ownership.
