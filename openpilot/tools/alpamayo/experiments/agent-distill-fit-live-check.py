"""CPU proof of persistent/repeated allocation accounting and the full-qualified, availability-gated runner."""
import ast
import copy
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
ROOT=Path('/mnt/nvme-500/tmp/alpamayo-speed-20261116/agent-distill-fit-artifacts')
LIVE=ROOT/'live-v4'

def module(name,path):
  s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m

def main():
  root=ROOT/'live-cpu-controls-v2';root.mkdir(exist_ok=False)
  def audit(event,args):
    if event=='open' and isinstance(args[0],(str,bytes)):
      value=os.fsdecode(args[0]);assert not value.startswith(('/dev/dri','/dev/kfd','/dev/nvidia','/dev/bus/usb'))
      assert not ('sealed_test' in value and value.endswith('.npz'))
  sys.addaudithook(audit)
  C=module('agent_distill_fit_live_control_common',LIVE/'frozen/agent-distill-fit-live-common.py')
  protocol=LIVE/'agent-distill-fit-live-protocol.json';p=C.protocol(protocol);full=C.verify_full()
  assert full['allocation_budget']['peak_live_bytes']==1802057192 and full['optimizer_updates']==0
  L=module('agent_distill_fit_live_control_ledger',LIVE/'frozen/agent-distill-fit-ledger.py')
  limits=dict(selected_gpu_id=27211,selected_pci='0000:0c:00.0',live_allocation_limit=176,
    cumulative_allocation_limit=400,max_single_buffer=64,free_reserve=20)
  availability=dict(selected_node=dict(gpu_id=27211,pci='0000:0c:00.0'),kfd_proc_root_read=True,
    kfd_processes=[],observed_selected_kfd_owners=[],vram_free_bytes=10000)
  ledger=L.Ledger(limits,lambda:availability,own_pid=123)
  iface=SimpleNamespace(gpu_id=27211,dev_sysfs_path='/sys/bus/pci/devices/0000:0c:00.0')
  next_handle=[0]
  def allocate(iface,size,*args,**kwargs):
    next_handle[0]+=1;return SimpleNamespace(meta=SimpleNamespace(handle=next_handle[0],size=size))
  a=ledger.alloc(allocate);f=ledger.free(lambda *x:None);u=ledger.unmap(lambda *x:None)
  persistent=[a(iface,32) for _ in range(3)]
  for step in range(3):
    ledger.phase('step-'+str(step));temporary=a(iface,64);f(iface,temporary)
    assert ledger.report['live_bytes']==96
  host=a(iface,16,host=True);u(iface,SimpleNamespace(meta=(host.meta.handle,True)))
  assert ledger.report['live_bytes']==96 and ledger.report['requested_bytes']==304
  assert ledger.report['peak_live_bytes']==160 and ledger.report['owned_unmap_calls']==1
  u(iface,SimpleNamespace(meta=(persistent[0].meta.handle,False)));assert ledger.report['live_bytes']==96
  rejected=[]
  def reject(name,fn):
    try:fn()
    except AssertionError:rejected.append(name)
    else:raise AssertionError('Negative control accepted: '+name)
  reject('single buffer cap',lambda:a(iface,65))
  reject('wrong card',lambda:a(SimpleNamespace(gpu_id=54982,dev_sysfs_path='/sys/bus/pci/devices/0000:03:00.0'),16))
  other=copy.deepcopy(availability);other['observed_selected_kfd_owners']=[9]
  conflict=L.Ledger(limits,lambda:other,own_pid=123)
  reject('other workload',lambda:conflict.alloc(allocate)(iface,16))
  # Cumulative usage survives frees and phase changes.
  x=a(iface,64);f(iface,x);assert ledger.report['requested_bytes']==368
  ledger.phase('next-step');reject('cumulative after frees',lambda:a(iface,64))
  x=a(iface,32);reject('live includes persistent states',lambda:a(iface,64));f(iface,x)
  for value in persistent:f(iface,value)
  assert ledger.report['live_bytes']==0
  # The generated numerical loop/optimizer/state machinery stays identical.
  original=ast.parse((HERE/'agent-distill-fit-train.py').read_text())
  current=ast.parse((LIVE/'frozen/agent-distill-fit-live-train.py').read_text())
  original_seed=next(n for n in original.body if isinstance(n,ast.FunctionDef) and n.name=='seed_arrays')
  current_seed=next(n for n in current.body if isinstance(n,ast.FunctionDef) and n.name=='seed_arrays')
  assert ast.dump(original_seed,include_attributes=False)==ast.dump(current_seed,include_attributes=False)
  for name in ('tensors','objective'):
    a=next(n for n in ast.walk(original) if isinstance(n,ast.FunctionDef) and n.name==name)
    b=next(n for n in ast.walk(current) if isinstance(n,ast.FunctionDef) and n.name==name)
    assert ast.dump(a,include_attributes=False)==ast.dump(b,include_attributes=False)
  def optimizer_calls(tree):
    return [ast.dump(n,include_attributes=False) for n in ast.walk(tree) if isinstance(n,ast.Call) and (
      isinstance(n.func,ast.Name) and n.func.id=='AdamW' or isinstance(n.func,ast.Attribute) and n.func.attr in ('gradient','step','zero_grad'))]
  assert optimizer_calls(original)==optimizer_calls(current)
  assert (LIVE/'frozen/agent-distill-fit-state.py').read_bytes()==(HERE/'agent-distill-fit-state.py').read_bytes()
  # Actual entrypoint checks, always stopped before tinygrad imports.
  helper=ROOT/'guard-controls-v1/agent-distill-fit-guarded-cli.py'
  base=[str(C.BASE/'upstream/.venv/bin/python'),str(helper),str(LIVE/'frozen/agent-distill-fit-live-train.py'),'--protocol-sha',C.sha(protocol)]
  control=subprocess.run(base,capture_output=True,text=True,timeout=60)
  assert control.returncode==0 and json.loads(control.stdout)['full_length_qualification_passed']
  blocked=subprocess.run(base+['--execute-training'],capture_output=True,text=True,timeout=60)
  assert blocked.returncode==42 and json.loads(blocked.stdout)['blocked_before_tinygrad_import']
  stale=ROOT/'agent-distill-fit-stale-synthetic-availability-v2.json'
  value=C.read(C.WORK/'agent-distill-full-artifacts/parent-gfx1100-kfd1/availability-before-launch.json')
  value['captured_unix_ns']=0;C.write(stale,value)
  expired=subprocess.run(base+['--execute-training','--availability-snapshot',str(stale),'--reviewed-availability-sha',C.sha(stale)],capture_output=True,text=True,timeout=60)
  assert expired.returncode==42 and json.loads(expired.stdout)['blocked_before_tinygrad_import']
  assert not (ROOT/'pilot-live-v1').exists()
  C.no_devices();result=dict(complete=True,all_passed=True,gpu_executed=False,actual_optimizer_updates=0,
    full_length_report_bound=True,full_length_report_sha256=C.FULL_REPORT_SHA,
    persistent_allocations_included=True,owned_host_unmap_accounted=True,unowned_mapping_does_not_free_storage=True,
    cumulative_never_resets=True,ledger_negative_controls=rejected,
    original_BF16_seed_loader_AST_identical=True,production_optimizer_and_loss='Original native AdamW and qualified graph/loss; unchanged CPU numerical/recovery controls apply.',
    original_target_objective_gradient_optimizer_AST_identical=True,recovery_source_byte_identical=True,
    actual_default_preflight_passed=True,missing_or_stale_availability_blocked_before_tinygrad=True,
    exclusive_output_untouched=True,protocol_sha256=C.sha(protocol))
  C.write(root/'agent-distill-fit-live-check.json',result);print(json.dumps(result,indent=2))

if __name__=='__main__':main()
