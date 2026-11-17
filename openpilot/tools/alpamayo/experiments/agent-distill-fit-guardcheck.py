"""Actual default/blocked trainer paths and negative protocol controls, without hardware access."""
import copy
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('agent_distill_fit_common_guard',HERE/'agent-distill-fit-common.py')
C=importlib.util.module_from_spec(spec);sys.modules[spec.name]=C;spec.loader.exec_module(C)

def main():
  C.no_devices();root=C.ROOT/'guard-controls-v1';root.mkdir(exist_ok=False)
  path=C.ROOT/'agent-distill-fit-protocol.json';original=C.protocol(path);digest=C.sha(path)
  rejected=[]
  for key,value in [('learning_rate',7e-6),('updates',201),('validate_every',1),('dtype','bfloat16'),
                    ('optimizer','torch.AdamW'),('fused',True),('batch_size',2),('length',4)]:
    altered=copy.deepcopy(original);altered['hyperparameters'][key]=value
    target=root/('agent-distill-fit-bad-'+key+'.json');C.write(target,altered)
    try:C.protocol(target)
    except (AssertionError,KeyError,ValueError):rejected.append(key)
    else:raise AssertionError('Modified hyperparameter accepted: '+key)
  for label,change in [
    ('wrong gpu',lambda p:p.update(selected_gpu_id=54982)),
    ('expanded memory cap',lambda p:p.update(live_allocation_limit=20*1024**3)),
    ('source omitted',lambda p:p['source_sha256'].pop(str(HERE/'agent-distill-fit-train.py'))),
    ('source changed',lambda p:p['source_sha256'].update({str(C.GRAPH):'0'*64})),
    ('wrong BF16 master',lambda p:next(iter(p['master_weights'].values())).update(bf16_sha256='0'*64)),
    ('wrong teacher precision',lambda p:p['teacher_contract'].update(fixed_language=False)),
    ('sealed target enabled',lambda p:p.update(sealed_targets_opened=True))]:
    altered=copy.deepcopy(original);change(altered);target=root/('agent-distill-fit-bad-'+label.replace(' ','-')+'.json');C.write(target,altered)
    try:C.protocol(target)
    except (AssertionError,KeyError,ValueError):rejected.append(label)
    else:raise AssertionError('Modified protocol accepted: '+label)
  trainer=C.module('agent_distill_fit_trainer_guard',HERE/'agent-distill-fit-train.py')
  env=trainer.environment(root,original,dict(no_memory_planner=1))
  assert env['DEV']=='KFD:1+AMD:LLVM:gfx1100' and env['PMC']==env['SQTT']=='' and env['PROFILE']=='0'
  assert env['TC']==env['ALLOW_TF32']==env['LRU']=='0' and env['OPTIM_DTYPE']=='float32'
  assert env['CACHEDB']==str(root/'compiler-cache.db') and env['TMPDIR']==str(root/'tmp')
  assert env['PYTHONPATH']==str(C.BASE/'tinygrad')
  # Execute actual runner entrypoints in a fresh process, behind an independent
  # hardware-open denial and a check that tinygrad never imports at these gates.
  launch=root/'agent-distill-fit-guarded-cli.py'
  launch.write_text('''import json,os,runpy,sys
def guard(event,args):
 if event=='open' and isinstance(args[0],(str,bytes)):
  assert not os.fsdecode(args[0]).startswith(('/dev/dri','/dev/kfd','/dev/nvidia','/dev/bus/usb'))
sys.addaudithook(guard)
program=sys.argv.pop(1)
sys.argv[0]=program
try:runpy.run_path(program,run_name='__main__')
except (AssertionError,KeyError,ValueError,FileNotFoundError) as error:
 assert 'tinygrad' not in sys.modules
 print(json.dumps({'blocked_before_tinygrad_import':True,'error':str(error)}))
 raise SystemExit(42)
assert 'tinygrad' not in sys.modules
''')
  command=[str(C.BASE/'upstream/.venv/bin/python'),str(launch),str(HERE/'agent-distill-fit-train.py'),'--protocol-sha',digest]
  default=subprocess.run(command,capture_output=True,text=True,timeout=60)
  assert default.returncode==0 and json.loads(default.stdout)['data_preflight_passed']
  blocked=subprocess.run(command+['--execute-training'],capture_output=True,text=True,timeout=60)
  assert blocked.returncode==42 and json.loads(blocked.stdout)['blocked_before_tinygrad_import']
  template=C.ROOT/'agent-distill-fit-admission-template.json'
  unreviewed=subprocess.run(command+['--execute-training','--admission',str(template),'--reviewed-admission-sha',C.sha(template)],capture_output=True,text=True,timeout=60)
  assert unreviewed.returncode==42 and json.loads(unreviewed.stdout)['blocked_before_tinygrad_import']
  child=subprocess.run(command+['--child'],capture_output=True,text=True,timeout=60)
  assert child.returncode==42 and json.loads(child.stdout)['blocked_before_tinygrad_import']
  assert not (C.ROOT/'pilot-v1').exists()
  C.no_devices();result=dict(complete=True,all_passed=True,gpu_executed=False,optimizer_updates=0,
    protocol_sha256=digest,negative_protocols_rejected=rejected,actual_default_preflight_passed=True,
    execution_without_qualification_rejected_before_tinygrad=True,unreviewed_template_rejected_before_tinygrad=True,
    direct_child_mode_rejected=True,actual_runner_environment_verified=True,exclusive_output_untouched=True)
  C.write(root/'agent-distill-fit-guardcheck.json',result);print(json.dumps(result,indent=2))

if __name__=='__main__':main()
