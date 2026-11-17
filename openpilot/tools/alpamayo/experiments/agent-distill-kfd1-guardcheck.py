"""CPU-only KFD index1 binding, immutable numeric identity, and independent rejection controls."""
import argparse
import ast
import copy
import importlib
import inspect
import json
from pathlib import Path
import sys
from types import SimpleNamespace

C=importlib.import_module('agent-distill-kfd1-common')
P=importlib.import_module('agent-distill-kfd1-parent')

def reject(f):
  try:f()
  except AssertionError:return
  raise AssertionError('Negative control was accepted')

def main():
  parser=argparse.ArgumentParser();parser.add_argument('--protocol-sha',required=True);args=parser.parse_args()
  C.no_gpu_fds();protocol=P.verify(args.protocol_sha);topology=C.passive_topology();selected=C.selected_node(topology)
  # Prohibit all hardware opens during this check, including the selected card.
  sys.addaudithook(P.hardware_open_guard(set()))
  from tinygrad.helpers import Target,Context,select_by_name
  from tinygrad.runtime.support.system import filter_visible_devices
  from tinygrad import Device
  target=Target.parse(C.TARGET)
  assert target.device=='AMD' and target.renderer=='LLVM' and target.arch=='gfx1100'
  assert target.interface=='KFD' and target.indices=='1' and C.SELECTED_INDEX==1
  candidates=[SimpleNamespace(name=n) for n in ('KFD','PCI','USB')]
  assert [x.name for x in select_by_name(candidates,lambda x:x.name,target.interface,'failure')]==['KFD']
  def bound_filter():
    assert filter_visible_devices([r['node'] for r in topology['nodes']],'AMD')==[C.SELECTED_NODE]
  with Context(DEV=C.TARGET):bound_filter()
  with Context(DEV='KFD:0+AMD:LLVM:gfx1100'):reject(bound_filter)
  actual={'requested_kfd_index':1,'logical_device_id':0,'filtered_kfd_nodes':['2'],'interface':'KFDIface',
    'gpu_id':27211,'pci':'0000:0c:00.0','render_minor':129,'arch':'gfx1100','renderer_target':C.TARGET,
    'is_usb':False,'native_pci':False,'can_recover':False,'allocator':'AMDAllocator'}
  C.validate_opened_identity(actual,selected)
  opened_mutations=[]
  for key,value in (('requested_kfd_index',0),('logical_device_id',1),('filtered_kfd_nodes',['1']),
    ('gpu_id',54982),('pci','0000:03:00.0'),('render_minor',128),('renderer_target','KFD:0+AMD:LLVM:gfx1100'),
    ('is_usb',True),('native_pci',True),('can_recover',True)):
    bad={**actual,key:value};reject(lambda:C.validate_opened_identity(bad,selected));opened_mutations.append(key)
  calls=[];fake=SimpleNamespace(gpu_id=27211,dev_sysfs_path='/sys/class/drm/renderD129/device')
  budget={'requested_bytes':0,'calls':0};wrapped=P.allocation_guard(lambda obj,size:calls.append(size) or 'fake',selected,budget)
  assert wrapped(fake,4096)=='fake'
  reject(lambda:wrapped(fake,C.LIMIT))
  reject(lambda:wrapped(SimpleNamespace(gpu_id=54982,dev_sysfs_path=fake.dev_sysfs_path),4096))
  reject(lambda:wrapped(SimpleNamespace(gpu_id=27211,dev_sysfs_path='/sys/class/drm/renderD128/device'),4096))
  assert calls==[4096]
  guard=P.hardware_open_guard({'/dev/kfd','/dev/dri/renderD129'})
  guard('open',('/dev/kfd',));guard('open',('/dev/dri/renderD129',))
  for path in ('/dev/dri/renderD128','/dev/dri/renderD130','/dev/bus/usb/006/022','/dev/nvidia0'):
    reject(lambda path=path:guard('open',(path,)))
  topology_mutations=[];old_topology=C.passive_topology
  try:
    for key,value in (('pci','0000:03:00.0'),('node','1'),('gpu_id',54982),('render_minor',128),
                      ('driver','vfio-pci'),('render_rw_access',False),('gfx_target_version',120000)):
      bad=copy.deepcopy(topology);bad['nodes'][C.SELECTED_INDEX][key]=value
      C.passive_topology=lambda bad=bad:bad
      reject(lambda:P.verify(args.protocol_sha));topology_mutations.append(key)
    bad=copy.deepcopy(topology);bad['nodes'][0],bad['nodes'][1]=bad['nodes'][1],bad['nodes'][0]
    C.passive_topology=lambda:bad
    reject(lambda:P.verify(args.protocol_sha));topology_mutations.append('ordinal_swap')
  finally:C.passive_topology=old_topology
  snapshot=C.passive_availability()
  positive={**snapshot,'observed_selected_kfd_owners':[],'kfd_pid_coverage_complete':True,'vram_free_bytes':C.LIMIT}
  C.require_no_observed_conflict(positive)
  for key,value in (('observed_selected_kfd_owners',[123]),('kfd_pid_coverage_complete',False),('vram_free_bytes',C.LIMIT-1)):
    reject(lambda key=key,value=value:C.require_no_observed_conflict({**positive,key:value}))
  reject(lambda:P.verify('0'*64))
  check=C.import_numeric('agent-distill-tinygrad-check',C.NUMERIC_CHECK_SHA)
  assert check.C is C.B and Path(check.__file__).resolve()==C.NUMERIC/'agent-distill-tinygrad-check.py'
  assert C.file_sha(Path(check.__file__))==protocol['numerical_check_sha256']==C.NUMERIC_CHECK_SHA
  assert C.file_sha(Path(C.B.__file__))==protocol['numerical_common_sha256']==C.NUMERIC_COMMON_SHA
  # The unchanged numerical execution uses no old target, PCI, output, or cache constants.
  tree=ast.parse(inspect.getsource(check.execute))
  accesses={n.attr for n in ast.walk(tree) if isinstance(n,ast.Attribute) and isinstance(n.value,ast.Name) and n.value.id=='C'}
  assert accesses=={'arrays','Block','recovery_loss','compare'}
  for path in C.EXP.glob('agent-distill-kfd1-*.py'):ast.parse(path.read_text(),filename=str(path))
  assert 'tinygrad.runtime.ops_amd' not in sys.modules and not Device._opened_devices
  C.no_gpu_fds()
  result={'complete':True,'gpu_executed':False,'opened_devices':[],'protocol_sha256':args.protocol_sha,
    'actual_target':repr(target),'actual_visibility_filter':['2'],'index0_negative_rejected':True,
    'post_open_identity_mutations_rejected':opened_mutations,'topology_mutations_rejected':topology_mutations,
    'allocation_limit_and_wrong_gpu_rejected_before_allocator':True,'unselected_open_rejected':True,
    'availability_owner_coverage_capacity_negatives_rejected':True,'bad_protocol_hash_rejected':True,
    'unchanged_numerical_function_sha256':C.B.sha(inspect.getsource(check.execute).encode()),
    'unchanged_numeric_common_accesses':sorted(accesses),'numerical_source_reused_without_edits':True,
    'native_amd_runtime_imported':False,'parent_source_sha256':C.file_sha(Path(P.__file__)),
    'availability_snapshot':snapshot}
  out=C.ROOT/'guardcheck.json';assert not out.exists();C.write(out,result)
  print(json.dumps({k:v for k,v in result.items() if k!='availability_snapshot'},indent=2))

if __name__=='__main__':main()
