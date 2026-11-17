"""Execute real production imports and child pre-open path with actual Device access disabled."""
import argparse
import ast
import copy
import importlib
import inspect
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

C=importlib.import_module('agent-distill-kfd1-import-common')
P=importlib.import_module('agent-distill-kfd1-import-parent')

def reject(f,kind=AssertionError):
  try:f()
  except kind:return
  raise AssertionError('Negative control was accepted')

def unchanged_parent_except_imports():
  old=ast.parse((C.PRIOR/'frozen/agent-distill-kfd1-parent.py').read_text())
  new=ast.parse(Path(P.__file__).read_text())
  new.body[0]=copy.deepcopy(old.body[0])
  for node in ast.walk(new):
    if isinstance(node,ast.Constant) and node.value=='agent-distill-kfd1-import-common':node.value='agent-distill-kfd1-common'
    if isinstance(node,ast.ImportFrom) and node.module=='tinygrad.helpers' and [a.name for a in node.names]==['DEV','PROFILE','MAX_BUFFER_SIZE']:
      node.names=[ast.alias(name=x) for x in ('DEV','PROFILE','PMC','SQTT','MAX_BUFFER_SIZE')]
    if isinstance(node,ast.ImportFrom) and node.module=='tinygrad.runtime.ops_amd':node.names=[ast.alias(name='KFDIface')]
  assert ast.dump(new,include_attributes=False)==ast.dump(old,include_attributes=False),'Unreviewed production behavior change'

def main():
  parser=argparse.ArgumentParser();parser.add_argument('--protocol-sha',required=True);args=parser.parse_args()
  C.no_gpu_fds();unchanged_parent_except_imports();protocol=P.verify(args.protocol_sha)
  # An unconditional hardware-open guard is installed before importing the real AMD module.
  sys.addaudithook(P.hardware_open_guard(set()))
  from tinygrad import Device
  from tinygrad.helpers import ALLOW_DEVICE_USAGE,Context,Target,select_by_name
  assert ALLOW_DEVICE_USAGE.value==0 and not Device._opened_devices
  target=Target.parse(C.TARGET);assert target.interface=='KFD' and target.indices=='1'
  # Traverse every explicit import in the actual production child and numerical call graph.
  numeric=C.import_numeric('agent-distill-tinygrad-check',C.NUMERIC_CHECK_SHA)
  assert numeric.C is C.B
  import_records=[]
  for owner in (P.child,numeric.execute,C.B.Block,C.B.arrays,C.B.compare,C.B.recovery_loss):
    tree=ast.parse(inspect.getsource(owner))
    for node in ast.walk(tree):
      if not isinstance(node,(ast.Import,ast.ImportFrom)):continue
      namespace={};exec(compile(ast.Module(body=[node],type_ignores=[]),inspect.getfile(owner),'exec'),namespace)
      names=[alias.asname or alias.name.split('.')[0] for alias in node.names]
      import_records.append({'owner':owner.__qualname__,'statement':ast.unparse(node),
        'resolved':{name:{'type':type(namespace[name]).__name__,'module':getattr(namespace[name],'__module__',getattr(namespace[name],'__name__',None))}
                    for name in names}})
  from tinygrad.runtime.ops_amd import KFDIface,PMC,SQTT
  from tinygrad.helpers import PROFILE,DEV,MAX_BUFFER_SIZE
  import tinygrad.helpers as helpers
  import tinygrad.runtime.ops_amd as amd
  assert PMC is amd.PMC and SQTT is amd.SQTT and PROFILE is helpers.PROFILE
  assert not hasattr(helpers,'PMC') and not hasattr(helpers,'SQTT')
  reject(lambda:exec('from tinygrad.helpers import PMC',{}),ImportError)
  reject(lambda:exec('from tinygrad.helpers import SQTT',{}),ImportError)
  assert repr(DEV)==C.TARGET and PROFILE.value==PMC.value==SQTT.value==0
  assert MAX_BUFFER_SIZE.value==128*1024**2 and KFDIface.kfd is None
  from tinygrad.runtime.support.system import filter_visible_devices
  topology=C.passive_topology();selected=C.selected_node(topology)
  with Context(DEV=C.TARGET):assert filter_visible_devices([r['node'] for r in topology['nodes']],'AMD')==['2']
  choices=[SimpleNamespace(name=n) for n in ('KFD','PCI','USB')]
  assert [x.name for x in select_by_name(choices,lambda x:x.name,target.interface,'error')]==['KFD']
  # Run the actual production child, with no substitute Device or runtime modules.
  # Only its output directory is redirected into CPU evidence; actual ALLOW_DEVICE_USAGE=0 blocks opening.
  parent_file=str(Path(P.__file__).resolve());device_file=str(C.TINY/'tinygrad/device.py')
  fulltree=ast.parse(Path(P.__file__).read_text())
  child_node=next(n for n in fulltree.body if isinstance(n,ast.FunctionDef) and n.name=='child')
  device_line=next(n.lineno for n in ast.walk(child_node) if isinstance(n,ast.Assign) and isinstance(n.value,ast.Subscript)
    and isinstance(n.value.value,ast.Name) and n.value.value.id=='Device')
  zero_guard_line=next(n.lineno for n in ast.walk(child_node) if isinstance(n,ast.Assert)
    and {'PMC','SQTT'}<={v.id for v in ast.walk(n) if isinstance(v,ast.Name)})
  original_output=P.OUTPUT;original_alloc=KFDIface.alloc;trials=[]
  def attempt(label,context,expect_device):
    calls=[];lines=[];P.OUTPUT=C.ROOT/'cpu-preopen'/label
    assert not P.OUTPUT.exists()
    def trace(frame,event,arg):
      if frame.f_code.co_filename==parent_file and frame.f_code.co_name=='child' and event=='line':lines.append(frame.f_lineno)
      if frame.f_code.co_filename==device_file and frame.f_code.co_name=='__getitem__' and event=='call':calls.append(frame.f_locals['ix'])
      return trace
    try:
      with Context(**context):
        sys.settrace(trace)
        try:P.child(args.protocol_sha)
        except AssertionError as exc:
          if expect_device:assert str(exc)=='usage of device AMD disallowed'
          else:assert lines[-1]==zero_guard_line
        else:raise AssertionError('Production path did not stop at its expected CPU guard')
        finally:sys.settrace(None)
      if expect_device:
        assert calls==['AMD'] and device_line in lines
        saved=C.read(P.OUTPUT/'report.json')
        assert not saved['complete'] and not saved['gpu_executed'] and saved['allocation_budget']['calls']==0
      else:assert calls==[] and device_line not in lines
      assert not Device._opened_devices and KFDIface.kfd is None
      trials.append({'label':label,'device_gate_calls':calls,'last_production_line':lines[-1],
        'expected_device_block':expect_device,'hardware_opened':False})
    finally:KFDIface.alloc=original_alloc;P.OUTPUT=original_output;sys.settrace(None)
  attempt('real-imports-to-device-gate',{},True)
  for key in ('PROFILE','PMC','SQTT'):attempt('reject-'+key.lower(),{key:1},False)
  calls=[];fake=SimpleNamespace(gpu_id=27211,dev_sysfs_path='/sys/class/drm/renderD129/device')
  wrapped=P.allocation_guard(lambda obj,size:calls.append(size) or 'fake',selected,{'requested_bytes':0,'calls':0})
  assert wrapped(fake,4096)=='fake'
  reject(lambda:wrapped(fake,C.LIMIT))
  reject(lambda:wrapped(SimpleNamespace(gpu_id=54982,dev_sysfs_path=fake.dev_sysfs_path),4096));assert calls==[4096]
  guard=P.hardware_open_guard({'/dev/kfd','/dev/dri/renderD129'})
  for path in ('/dev/dri/renderD128','/dev/dri/renderD130','/dev/bus/usb/006/022'):
    reject(lambda path=path:guard('open',(path,)))
  actual={'requested_kfd_index':1,'logical_device_id':0,'filtered_kfd_nodes':['2'],'interface':'KFDIface',
    'gpu_id':27211,'pci':'0000:0c:00.0','render_minor':129,'arch':'gfx1100','renderer_target':C.TARGET,
    'is_usb':False,'native_pci':False,'can_recover':False,'allocator':'AMDAllocator'}
  C.validate_opened_identity(actual,selected)
  for key,value in (('requested_kfd_index',0),('filtered_kfd_nodes',['1']),('gpu_id',54982),('pci','0000:03:00.0'),('render_minor',128)):
    reject(lambda key=key,value=value:C.validate_opened_identity({**actual,key:value},selected))
  drm=Path('/sys/bus/pci/devices')/C.SELECTED_PCI/'drm'
  drm_names=sorted(p.name for p in drm.iterdir())
  assert 'card2' in drm_names and 'renderD129' in drm_names
  C.no_gpu_fds();assert not Device._opened_devices and KFDIface.kfd is None
  result={'complete':True,'gpu_executed':False,'opened_devices':[],'protocol_sha256':args.protocol_sha,
    'parent_change_limited_to_imports_and_module_path':True,'actual_imports':import_records,
    'wrong_helpers_module_negatives_rejected':True,'real_production_child_trials':trials,
    'actual_device_open_block':'Pinned ALLOW_DEVICE_USAGE=0; unconditional hardware-open audit guard',
    'kfd_fd_is_none':True,'allocation_and_wrong_gpu_negatives_passed':True,'identity_and_hardware_open_negatives_passed':True,
    'selected_drm_nodes':drm_names,'source_sha256':{str(p):C.file_sha(p) for p in
      (Path(__file__),Path(P.__file__),Path(C.__file__),Path(helpers.__file__),Path(amd.__file__),Path(numeric.__file__))}}
  output=C.ROOT/'cpu-import-check.json';assert not output.exists();C.write(output,result)
  print(json.dumps({k:v for k,v in result.items() if k not in ('actual_imports','source_sha256')},indent=2))

if __name__=='__main__':main()
