"""CPU preflight and negative parent-gate controls. Never invokes a model or device."""
import ast
import importlib.util
import json
import os
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
WORK=Path('/mnt/nvme-500/tmp/alpamayo-speed-20261116')
DATA=WORK/'agent-nearest-live-data'
sys.dont_write_bytecode=True


def module(path,name):
  spec=importlib.util.spec_from_file_location(name,path);value=importlib.util.module_from_spec(spec);sys.modules[name]=value;spec.loader.exec_module(value);return value


def main():
  if os.environ.get('DEV')!='CPU':raise ValueError('Set DEV=CPU')
  parent=module(HERE/'agent-nearest-live-run.py','agent_nearest_live_parent_check_run')
  policy=module(HERE/'agent-nearest-live-policy.py','agent_nearest_live_parent_check_policy')
  plan=parent.read(parent.DEFAULT);root,startup,dynamic=parent.verify(plan,policy)
  failures={}
  for name,report in (('pending_missing_gate',{}),('actual64FLOORgate_is_not_nearest4',parent.read(plan['startup_candidate']))):
    try:parent.verify_parity4(report,plan,policy,startup)
    except (ValueError,KeyError) as error:failures[name]=str(error)
    else:raise AssertionError('Invalid nearest gate accepted')
  source=ast.parse((HERE/'agent-nearest-live-run.py').read_text())
  functions={f.name:f for f in source.body if isinstance(f,ast.FunctionDef)}
  def contexts(node):
    return sum(isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='imports' and
      isinstance(n.func.value,ast.Name) and n.func.value.id=='policy' for n in ast.walk(node))
  assert contexts(functions['parity'])==1 and contexts(functions['main'])==1
  assert all(not Path(job['output']).parent.exists() for job in plan['jobs'])
  from tinygrad import Device
  assert not Device._opened_devices
  result=dict(cpu_only=True,gpu_executed=False,device_opens=[],startup_actual64GPU_gate_verified=True,
    negative_nearest_gate_controls=failures,single_import_hook_scope_per_GPU_path=True,
    protocol_sha256=policy.sha(parent.DEFAULT),sources_sha256={p.name:policy.sha(p) for p in HERE.glob('agent-nearest-live-*.py')},
    four_query_GPU_gate_pending=True,paced_GPU_trial_pending=True,automatic_launch=False)
  with (DATA/'agent-nearest-live-parent-check.json').open('x') as f:f.write(json.dumps(result,indent=2)+'\n')
  print(json.dumps({k:v for k,v in result.items() if k!='sources_sha256'},indent=2))


if __name__=='__main__':main()
