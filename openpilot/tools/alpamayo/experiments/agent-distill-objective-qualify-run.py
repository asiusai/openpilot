"""Parent-admitted KFD1 resource qualification: one connected update, exact recovery, no selection."""
import argparse
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('objective_qualify_core_run',HERE/'agent-distill-objective-qualify-core.py')
Q=importlib.util.module_from_spec(spec);sys.modules[spec.name]=Q;spec.loader.exec_module(Q)
LEDGER=Q.WORK/'agent-distill-continue-artifacts/live-v1/frozen/agent-distill-continue-ledger.py'
LEDGER_SHA='f070ca77dedbc6f22027a2dcb063189b6c66647ae1436d022d0cbf7386b1663d'
TARGET='KFD:1+AMD:LLVM:gfx1100'


def admission(args,context):
  if args.admission is None or Q.sha(args.admission)!=args.reviewed_admission_sha:raise ValueError('Parent-reviewed admission required')
  value=Q.read(args.admission)
  assert value['parent_reviewed'] is True and value['no_reset_or_owner_displacement'] is True
  assert value['target']==TARGET and value['selected_pci']=='0000:0c:00.0' and value['selected_gpu_id']==27211
  assert value['aligned_report_sha256']==context['aligned_report_sha256'] and value['no_memory_planner']==1
  checked=value['cpu_check'];assert Q.sha(checked['path'])==checked['sha256']
  report=Q.read(checked['path']);assert report['all_passed'] and report['gpu_executed'] is False
  for name in ('agent-distill-objective-qualify-core.py','agent-distill-objective-qualify-run.py','agent-distill-objective-core.py'):
    assert report['source_sha256'][str(HERE/name)]==Q.sha(HERE/name)
  reviewed=value['availability_review'];assert Q.sha(reviewed['path'])==reviewed['sha256']
  current=Q.read(reviewed['path'])
  assert 0<=time.time_ns()-current['captured_unix_ns']<=60_000_000_000
  assert current['selected_node']['pci']=='0000:0c:00.0' and current['selected_node']['gpu_id']==27211
  assert current['kfd_pid_coverage_complete'] and not current['observed_selected_kfd_owners']
  assert current['vram_free_bytes']>=Q.BOUNDS['live_allocation_limit']+Q.BOUNDS['free_reserve']
  return value


def child(args,row,context,receipt):
  K=Q.module('objective_qualify_identity',HERE/'agent-distill-kfd1-common.py')
  K.no_gpu_fds();before=K.passive_availability();K.require_no_observed_conflict(before)
  assert before['vram_free_bytes']>=Q.BOUNDS['live_allocation_limit']+Q.BOUNDS['free_reserve']
  root=args.output_dir.resolve();Q.S.atomic_json(root/'availability-before-open.json',before)
  def hardware_guard(event,values):
    if event=='open' and isinstance(values[0],(str,bytes)):
      name=os.fsdecode(values[0])
      if name.startswith(('/dev/dri','/dev/kfd','/dev/nvidia','/dev/bus/usb')):assert name in {'/dev/kfd','/dev/dri/renderD129'}
  sys.addaudithook(hardware_guard)
  from tinygrad import Device
  from tinygrad.helpers import DEV,PROFILE,MAX_BUFFER_SIZE,USE_TC,ALLOW_TF32,NO_MEMORY_PLANNER,LRU
  from tinygrad.runtime.ops_amd import KFDIface,PMC,SQTT
  assert not Device._opened_devices and repr(DEV)==TARGET and DEV.target('AMD').indices=='1'
  assert not PROFILE.value and not PMC.value and not SQTT.value
  assert USE_TC.value==ALLOW_TF32.value==LRU.value==0 and NO_MEMORY_PLANNER.value==1
  assert MAX_BUFFER_SIZE.value==Q.BOUNDS['max_single_buffer'] and Q.sha(LEDGER)==LEDGER_SHA
  implementation=Q.module('objective_qualify_ledger',LEDGER)
  ledger=implementation.Ledger(dict(Q.BOUNDS,selected_gpu_id=27211,selected_pci='0000:0c:00.0'),K.passive_availability)
  KFDIface.alloc=ledger.alloc(KFDIface.alloc);KFDIface.free=ledger.free(KFDIface.free);KFDIface.unmap=ledger.unmap(KFDIface.unmap)
  report=dict(complete=False,gpu_executed=False,kind=Q.KIND,quality_qualified=False,training_cohort_sample=False,
    checkpoint_selection_allowed=False,default_model_modified=False,allocation_budget=ledger.report,
    context=context,bounds=Q.BOUNDS,estimate=Q.estimate(),qualified_ledger_sha256=LEDGER_SHA,
    source_sha256={str(p):Q.sha(p) for p in (*HERE.glob('agent-distill-objective-qualify-*.py'),HERE/'agent-distill-objective-core.py')})
  try:
    ledger.phase('device-and-starting-masters');dev=Device['AMD'];selected=K.selected_node(K.passive_topology())
    identity=dict(requested_kfd_index=1,logical_device_id=dev.device_id,filtered_kfd_nodes=list(KFDIface.gpus),
      interface=type(dev.iface).__name__,gpu_id=dev.iface.gpu_id,pci=Path(dev.iface.dev_sysfs_path).resolve().name,
      render_minor=dev.iface.props['drm_render_minor'],arch=dev.arch,renderer_target=repr(dev.renderer.target),
      is_usb=dev.is_usb,native_pci=dev.is_am(),can_recover=dev.can_recover,allocator=type(dev.allocator).__name__)
    K.validate_opened_identity(identity,selected);report.update(gpu_executed=True,actual_device=identity)
    report.update(Q.run_step(Q.seed_arrays(),lambda device:Q.C.load_record(row,device),root,context,'AMD',ledger.phase,resume=args.resume_from))
    for path,digest in Q.SEEDS.values():assert Q.sha(path)==digest
  except BaseException as error:report['failure']=dict(type=type(error).__name__,message=str(error));raise
  finally:
    report.update(allocator_guards_retained_through_finalizers=True,allocation_snapshot_before_native_finalizers=True)
    Q.S.atomic_json(root/'report.json',report)


def main():
  parser=argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--aligned-report',type=Path);parser.add_argument('--aligned-report-sha')
  parser.add_argument('--output-dir',type=Path);parser.add_argument('--resume-from',type=Path)
  parser.add_argument('--admission',type=Path);parser.add_argument('--reviewed-admission-sha')
  parser.add_argument('--execute-qualification',action='store_true');parser.add_argument('--child',action='store_true')
  parser.add_argument('--estimate-only',action='store_true');args=parser.parse_args()
  if args.estimate_only:
    assert not args.execute_qualification and not args.child
    print(json.dumps(dict(gpu_executed=False,estimate=Q.estimate(),bounds=Q.BOUNDS),indent=2));return
  if args.aligned_report is None or args.aligned_report_sha is None:raise ValueError('Explicit completed aligned report and SHA required')
  row,context=Q.control(args.aligned_report,args.aligned_report_sha)
  if args.resume_from and not args.resume_from.resolve().is_relative_to(Q.ROOT):raise ValueError('Resume only an isolated qualification checkpoint')
  if not args.execute_qualification:
    os.environ['DEV']='CPU';sys.path.insert(0,str(Q.BASE/'tinygrad'))
    Q.seed_arrays();Q.C.load_record(row,'CPU')
    K=Q.module('objective_qualify_preflight_fds',HERE/'agent-distill-kfd1-common.py');K.no_gpu_fds()
    print(json.dumps(dict(ready=True,gpu_executed=False,context=context,estimate=Q.estimate()),indent=2));return
  receipt=admission(args,context)
  if args.output_dir is None or not args.output_dir.resolve().is_relative_to(Q.ROOT):raise ValueError('Owned qualification output required')
  if args.child:
    assert os.environ.get('AGENT_DISTILL_OBJECTIVE_QUALIFY_PARENT_PID')==str(os.getppid())
    sys.path.insert(0,str(Q.BASE/'tinygrad'));return child(args,row,context,receipt)
  root=args.output_dir.resolve()
  if root.exists():raise ValueError('Fresh output directory required')
  if shutil.disk_usage(Q.WORK).free<Q.estimate()['workspace_free_required_bytes']:raise ValueError('Keep checkpoint workspace reserve')
  with Path('/tmp/opencode/alpamayo-chestnut-speed.lock').open('a') as lock:
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    root.mkdir(parents=True);(root/'tmp').mkdir()
    old=Q.module('objective_qualify_environment',HERE/'agent-distill-fit-train.py')
    env=old.environment(root,dict(Q.BOUNDS,target=TARGET),receipt)
    env['AGENT_DISTILL_OBJECTIVE_QUALIFY_PARENT_PID']=str(os.getpid())
    command=[str(Q.BASE/'upstream/.venv/bin/python'),str(Path(__file__).resolve()),*sys.argv[1:],'--child']
    Q.S.atomic_json(root/'parent-invocation.json',dict(command=command,context=context,admission_sha256=args.reviewed_admission_sha))
    started=time.monotonic();timed_out=False
    with (root/'run.log').open('w') as stream:
      process=subprocess.Popen(command,env=env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
      try:code=process.wait(timeout=900)
      except subprocess.TimeoutExpired:
        timed_out=True;os.killpg(process.pid,signal.SIGTERM)
        try:code=process.wait(timeout=5)
        except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);code=process.wait()
    Q.S.atomic_json(root/'wall.json',dict(seconds=time.monotonic()-started,returncode=code,timed_out=timed_out))
    assert code==0 and not timed_out and Q.read(root/'report.json')['complete']


if __name__=='__main__':main()
