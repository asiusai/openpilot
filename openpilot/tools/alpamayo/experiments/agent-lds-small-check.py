"""Offline proofs for the selected binary and reuse of the unchanged CPU JIT/validator checks."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('agent_lds_small_check_kernel',HERE/'agent-lds-small-kernel.py')
S=importlib.util.module_from_spec(spec)
spec.loader.exec_module(S)
C=S.load('agent-lds-cold-run')
D=S.load('agent-lds-cold-check')


def cpu_checks(render,compiled):
  from tinygrad import Device,dtypes
  from tinygrad.uop.ops import UOp
  assert not Device._opened_devices
  assert S.K.CONFIGS=={'amd3':(256,160,128,4,2),'amd4':(192,192,128,2,4)}
  assert set(S.CONFIGS)=={'small128','small96'}
  source_equivalence=[]
  for label,config in S.CONFIGS.items():
    for shape,row_scale,bias in ((C.SHAPE,False,False),((193,208,256),True,True)):
      old=S.K.source(*shape,'amd4',row_scale,bias)
      expected=old.replace('192,192,128,2,4,true',','.join(map(str,config))+',true')
      expected=expected.replace('amdgpu_flat_work_group_size(256,256)','amdgpu_flat_work_group_size(128,128)')
      assert S.source(*shape,label,row_scale,bias)==expected
      source_equivalence.append(dict(label=label,shape_mnk=shape,row_scale=row_scale,bias=bias,
        only_template_tile_and_launch_size_changed=True))
  queues={}
  for label,fn in S.functions(render).items():
    if label not in ('current','small96'): continue
    program,specs=D.numeric_program(fn,render)
    expected_binary=compiled['actual'][label]['binary_sha256']
    assert S.H.E.sha(program.src[3].arg)==expected_binary
    out=UOp.param(7,dtypes.bfloat16,C.SHAPE[0]*C.SHAPE[1],device='AMD')
    a=UOp.param(0,dtypes.fp8e4m3,C.SHAPE[0]*C.SHAPE[2],device='AMD')
    sx,sw=(UOp.param(i,dtypes.float32,1,device='AMD') for i in (1,2))
    weights=[UOp.param(i+3,dtypes.fp8e4m3,C.SHAPE[1]*C.SHAPE[2],device='AMD') for i in range(4)]
    queues[label]={}
    for workload,order in C.ORDERS.items():
      calls=[program.call(out,a,weights[index],sx,sw) for index in order]
      queues[label][workload]=C.offline_queue(calls,order,expected_binary)
    order=C.ORDERS['rotating']
    calls=[program.call(out,a,weights[index],sx,sw) for index in order]
    bad_output=UOp.param(8,dtypes.bfloat16,C.SHAPE[0]*C.SHAPE[1],device='AMD')
    queues[label]['negative_controls']={
      'dropped_call':D.rejected(lambda:C.offline_queue(calls[:-1],order,expected_binary)),
      'all_same_weight':D.rejected(lambda:C.offline_queue([program.call(out,a,weights[0],sx,sw)]*40,order,expected_binary)),
      'reordered_weights':D.rejected(lambda:C.offline_queue(calls[::-1],order,expected_binary)),
      'independent_output':D.rejected(lambda:C.offline_queue(calls[:-1]+[program.call(bad_output,a,weights[3],sx,sw)],order,expected_binary)),
      'wrong_binary':D.rejected(lambda:C.offline_queue(calls,order,'0'*64)),
    }
  artifact=HERE/'agent-lds-small-artifacts'
  artifact.mkdir(exist_ok=True)
  destination=artifact/'host.json'
  child=subprocess.run([sys.executable,str(HERE/'agent-lds-cold-check.py'),'--host','--output',str(destination)],
    env=dict(os.environ,DEV='CPU',ALLOW_DEVICE_USAGE='1'),capture_output=True,text=True,timeout=120)
  (artifact/'host.log').write_text(child.stdout+child.stderr)
  assert child.returncode==0,child.stdout+child.stderr
  host=json.loads(destination.read_text())
  assert host['complete'] and not host['gpu_executed'] and not Device._opened_devices
  return dict(source_equivalence=source_equivalence,offline_actual_binary_queues=queues,unchanged_validator_and_real_cpu_jit=host,
    new_GPU_arithmetic_qualified=False,all_six_small_families_and_six_post_timing_actual_mutations_required=True)
