"""Run the frozen production child through runpy with real Device opening prohibited."""
import argparse
import ast
import copy
import importlib
import inspect
import json
import os
from pathlib import Path
import runpy
import sys
from types import SimpleNamespace

C=importlib.import_module('agent-distill-full-common')
P=importlib.import_module('agent-distill-full-parent')

def reject(fn,kind=AssertionError):
  try:fn()
  except kind:return
  raise AssertionError('Negative control accepted')

def main():
  ap=argparse.ArgumentParser();ap.add_argument('--protocol-sha',required=True);args=ap.parse_args()
  assert not (C.ROOT/'cpu-check.json').exists();C.no_gpu_fds();protocol=P.verify(args.protocol_sha)
  sys.addaudithook(P.hardware_open_guard(set()))
  from tinygrad import Device
  from tinygrad.helpers import ALLOW_DEVICE_USAGE,Context,Target,select_by_name
  from tinygrad.runtime.ops_amd import KFDIface,PMC,SQTT
  from tinygrad.helpers import PROFILE,DEV,MAX_BUFFER_SIZE
  import tinygrad.helpers as helpers
  import tinygrad.runtime.ops_amd as amd
  assert ALLOW_DEVICE_USAGE.value==0 and not Device._opened_devices and KFDIface.kfd is None
  assert PMC is amd.PMC and SQTT is amd.SQTT and PROFILE is helpers.PROFILE
  assert PROFILE.value==PMC.value==SQTT.value==0 and MAX_BUFFER_SIZE.value==C.MAX_BUFFER
  reject(lambda:exec('from tinygrad.helpers import PMC',{}),ImportError)
  reject(lambda:exec('from tinygrad.helpers import SQTT',{}),ImportError)
  for key in ('PMC','SQTT'):
    prior=os.environ.get(key)
    try:
      os.environ[key]='0';assert helpers.getenv.__wrapped__(key,False) is True
      os.environ[key]='';assert helpers.getenv.__wrapped__(key,False) is False
    finally:
      if prior is None:os.environ.pop(key,None)
      else:os.environ[key]=prior
  env=P.environment();assert env['PROFILE']==env['VIZ']=='0' and env['PMC']==env['SQTT']==''
  target=Target.parse(C.TARGET);assert target.interface=='KFD' and target.indices=='1' and repr(DEV)==C.TARGET
  from tinygrad.runtime.support.system import filter_visible_devices
  topology=C.K.passive_topology();selected=C.K.selected_node(topology)
  assert filter_visible_devices([r['node'] for r in topology['nodes']],'AMD')==['2']
  choices=[SimpleNamespace(name=n) for n in ('KFD','PCI','USB')]
  assert [x.name for x in select_by_name(choices,lambda x:x.name,target.interface,'error')]==['KFD']
  check=importlib.import_module('agent-distill-full-check');imports=[]
  for owner in (P.child,P.AllocationBudget,check.execute,C.block,C.array_manifest,C.loss,C.B.Block,C.B.compare,C.B.recovery_loss):
    for node in ast.walk(ast.parse(inspect.getsource(owner))):
      if not isinstance(node,(ast.Import,ast.ImportFrom)):continue
      ns={};exec(compile(ast.Module(body=[node],type_ignores=[]),inspect.getfile(owner),'exec'),ns)
      imports.append({'owner':owner.__qualname__,'statement':ast.unparse(node)})
  # Inspect the real production entry point with runpy; no substitute Device or runtime module.
  parent=str(Path(P.__file__).resolve());device_file=str(C.TINY/'tinygrad/device.py');trials=[]
  original_methods=(KFDIface.alloc,KFDIface.free,KFDIface.unmap);original_argv=sys.argv[:]
  def attempt(label,context,expect_device):
    calls=[];lines=[];output=C.ROOT/'cpu-preopen'/label;assert not output.exists()
    def trace(frame,event,arg):
      if frame.f_code.co_filename==parent and frame.f_code.co_name=='child' and event=='line':lines.append(frame.f_lineno)
      if frame.f_code.co_filename==device_file and frame.f_code.co_name=='__getitem__' and event=='call':calls.append(frame.f_locals['ix'])
      return trace
    sys.argv=[parent,'--protocol-sha',args.protocol_sha,'--child','--output',str(output)]
    try:
      with Context(**context):
        sys.settrace(trace)
        try:runpy.run_path(parent,run_name='__main__')
        except AssertionError as exc:
          if expect_device:assert str(exc)=='usage of device AMD disallowed',str(exc)
          else:assert not calls and str(exc)=='',str(exc)
        else:raise AssertionError('Production child did not stop at its expected guard')
        finally:sys.settrace(None)
      assert calls==(['AMD'] if expect_device else [])
      if expect_device:
        saved=C.read(output/'report.json')
        assert not saved['complete'] and not saved['gpu_executed'] and saved['allocation_budget']['calls']==0
      assert not Device._opened_devices and KFDIface.kfd is None
      trials.append({'label':label,'actual_device_gate_calls':calls,'last_child_line':lines[-1],'hardware_opened':False})
    finally:
      KFDIface.alloc,KFDIface.free,KFDIface.unmap=original_methods;sys.argv=original_argv;sys.settrace(None)
  attempt('real-runpy-device-block',{},True)
  for key in ('PROFILE','PMC','SQTT'):attempt('reject-'+key.lower(),{key:1},False)
  for key,value in (('DEV','KFD:0+AMD:LLVM:gfx1100'),('MAX_BUFFER_SIZE',0),('NO_MEMORY_PLANNER',0),('TC',1),('ALLOW_TF32',1)):
    attempt('reject-'+key.lower(),{key:value},False)
  actual={'requested_kfd_index':1,'logical_device_id':0,'filtered_kfd_nodes':['2'],'interface':'KFDIface','arch':'gfx1100',
    'gpu_id':27211,'pci':'0000:0c:00.0','render_minor':129,'renderer_target':C.TARGET,'allocator':'AMDAllocator',
    'can_recover':False,'is_usb':False,'native_pci':False}
  C.K.validate_opened_identity(actual,selected)
  for key,value in (('requested_kfd_index',0),('filtered_kfd_nodes',['1']),('gpu_id',54982),('pci','0000:03:00.0'),
                    ('render_minor',128),('interface','PCI'),('is_usb',True),('can_recover',True)):
    reject(lambda key=key,value=value:C.K.validate_opened_identity({**actual,key:value},selected))
  snapshot=C.K.passive_availability();P.require_available(snapshot,initial=True)
  bad=copy.deepcopy(snapshot);bad['observed_selected_kfd_owners']=[273252]
  reject(lambda:P.require_available(bad,os.getpid()))
  bad=copy.deepcopy(snapshot);bad['vram_free_bytes']=C.FREE_RESERVE-1
  reject(lambda:P.require_available(bad))
  bad=copy.deepcopy(snapshot);bad['kfd_proc_root_read']=False
  reject(lambda:P.require_available(bad))
  budget=P.AllocationBudget(selected,lambda:snapshot);calls=[]
  fake=SimpleNamespace(gpu_id=27211,dev_sysfs_path='/sys/class/drm/renderD129/device')
  def allocate(iface,size,*a,**kw):
    calls.append(size);return SimpleNamespace(meta=SimpleNamespace(handle=len(calls),size=size))
  alloc=budget.wrap_alloc(allocate);free=budget.wrap_free(lambda *a:None);unmap=budget.wrap_unmap(lambda *a:None)
  a=alloc(fake,4096);b=alloc(fake,8192);assert budget.report['live_bytes']==12288
  free(fake,a);unmap(fake,SimpleNamespace(meta=(b.meta.handle,True)));assert budget.report['live_bytes']==0
  unmap(fake,SimpleNamespace(meta=(123456,False)))
  reject(lambda:alloc(fake,C.MAX_BUFFER+1));reject(lambda:alloc(SimpleNamespace(gpu_id=54982,dev_sysfs_path=fake.dev_sysfs_path),4096))
  budget.report['requested_bytes']=C.TOTAL_LIMIT;reject(lambda:alloc(fake,1));budget.report['requested_bytes']=12288
  budget.report['live_bytes']=C.LIVE_LIMIT;reject(lambda:alloc(fake,1));budget.report['live_bytes']=0
  assert calls==[4096,8192] and budget.report['free_calls']==budget.report['owned_unmap_calls']==1
  guard=P.hardware_open_guard({'/dev/kfd','/dev/dri/renderD129'})
  for path in ('/dev/dri/renderD128','/dev/dri/card2','/dev/bus/usb/006/022','/dev/nvidia0'):
    reject(lambda path=path:guard('open',(path,)))
  oracle,arrays=C.array_manifest();assert len([k for k in arrays if k.startswith('grad:')])==9
  import numpy as np
  assert all(np.isfinite(a).all() for a in arrays.values())
  assert not C.B.compare(np.zeros_like(arrays['grad:mlp.fc2.weight']),arrays['grad:mlp.fc2.weight'],'gradient')['passed']
  assert oracle['padding_input_gradient_zero'] and oracle['causal_future_exact'] and oracle['noncausal_negative_max_absolute']>1e-4
  C.no_gpu_fds();assert not Device._opened_devices and KFDIface.kfd is None
  drm=sorted(p.name for p in (Path('/sys/bus/pci/devices')/C.K.SELECTED_PCI/'drm').iterdir())
  assert 'card2' in drm and 'renderD129' in drm
  result={'complete':True,'gpu_executed':False,'opened_devices':[],'protocol_sha256':args.protocol_sha,
    'production_runpy_trials':trials,'actual_imports':imports,'allocation_free_owned_unmap_and_budget_negatives_passed':True,
    'identity_availability_and_hardware_open_negatives_passed':True,'all_nine_oracle_gradients_finite':True,
    'boolean_zero_string_pitfall_verified':True,'wrong_module_imports_rejected':True,'selected_drm_nodes':drm,
    'sources':{str(p):C.file_sha(p) for p in (Path(__file__),Path(P.__file__),Path(C.__file__),Path(check.__file__))}}
  C.write(C.ROOT/'cpu-check.json',result)
  print(json.dumps({k:v for k,v in result.items() if k not in ('actual_imports','sources')},indent=2))

if __name__=='__main__':main()
