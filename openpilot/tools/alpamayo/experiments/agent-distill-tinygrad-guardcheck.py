"""CPU-only selector, allocation-bound and fail-closed protocol controls. No runtime device opened."""
import argparse
import ast
import copy
import importlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

C=importlib.import_module('agent-distill-tinygrad-common')
P=importlib.import_module('agent-distill-tinygrad-parent')

def reject(f):
  try:f()
  except AssertionError:return
  raise AssertionError('Negative control was accepted')

def main():
  parser=argparse.ArgumentParser();parser.add_argument('--protocol-sha',required=True);args=parser.parse_args()
  C.no_gpu_fds();p=P.verify(args.protocol_sha);topology=C.passive_topology();selected=topology['nodes'][0]
  # Actual pinned helper API; importing helpers and the filtering function opens no runtime device.
  from tinygrad.helpers import Target,Context,select_by_name
  from tinygrad.runtime.support.system import filter_visible_devices
  from tinygrad import Device
  target=Target.parse(C.TARGET)
  assert target.device=='AMD' and target.renderer=='LLVM' and target.arch=='gfx1100' and target.interface=='KFD' and target.indices=='0'
  names=['KFD','PCI','USB'];candidates=[SimpleNamespace(name=n) for n in names]
  assert [x.name for x in select_by_name(candidates,lambda x:x.name,target.interface,'failure')]==['KFD']
  with Context(DEV=C.TARGET):assert filter_visible_devices([r['node'] for r in topology['nodes']],'AMD')==[selected['node']]
  # No ctypes/KFD call: test the private guard with a fake allocator and identity.
  calls=[];fake=SimpleNamespace(gpu_id=selected['gpu_id'],dev_sysfs_path=f'/sys/class/drm/renderD{selected["render_minor"]}/device')
  budget={'requested_bytes':0,'calls':0};wrapped=P.allocation_guard(lambda obj,size: calls.append(size) or 'fake',selected,budget)
  assert wrapped(fake,4096)=='fake'
  reject(lambda:wrapped(fake,P.LIMIT));assert calls==[4096]
  reject(lambda:wrapped(SimpleNamespace(gpu_id=-1,dev_sysfs_path=fake.dev_sysfs_path),4096));assert calls==[4096]
  guard=P.hardware_open_guard({'/dev/kfd','/dev/dri/renderD128'})
  guard('open',('/dev/kfd',));guard('open',('/dev/dri/renderD128',))
  for path in ('/dev/dri/renderD129','/dev/dri/renderD130','/dev/bus/usb/006/022','/dev/nvidia0'):
    reject(lambda path=path:guard('open',(path,)))
  old=C.passive_topology
  mutations=[]
  try:
    for key,value in (('pci','0000:0c:00.0'),('driver','vfio-pci'),('render_rw_access',False),('gfx_target_version',120000)):
      changed=copy.deepcopy(topology);changed['nodes'][0][key]=value
      C.passive_topology=lambda changed=changed:changed
      reject(lambda:P.verify(args.protocol_sha));mutations.append(key)
  finally:C.passive_topology=old
  reject(lambda:P.verify('0'*64))
  # Parse all new code without creating bytecode on the nearly full root filesystem.
  for path in C.EXP.glob('agent-distill-tinygrad-*.py'):ast.parse(path.read_text(),filename=str(path))
  assert 'tinygrad.runtime.ops_amd' not in sys.modules and not Device._opened_devices
  C.no_gpu_fds()
  report={'complete':True,'gpu_executed':False,'opened_devices':[],'protocol_sha256':args.protocol_sha,
    'actual_selector_target':repr(target),'actual_visibility_filter':[selected['node']],
    'usb_pci_fallback_excluded':True,'allocation_limit_rejected_before_allocator_call':True,
    'wrong_gpu_id_rejected':True,'unselected_hardware_open_rejected':True,'topology_mutations_rejected':mutations,
    'bad_protocol_hash_rejected':True,'native_amd_runtime_imported':False,
    'parent_source_sha256':C.file_sha(Path(P.__file__)),'source_sha256':C.file_sha(Path(__file__))}
  out=C.ROOT/'guardcheck.json';assert not out.exists();C.write(out,report);print(json.dumps(report,indent=2))

if __name__=='__main__':main()
