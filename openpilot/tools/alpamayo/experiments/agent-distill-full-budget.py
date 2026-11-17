"""Actual L2048 forward/backward schedule inspection without device opening or buffer allocation."""
import importlib
import json
from pathlib import Path

C=importlib.import_module('agent-distill-full-common')

def main():
  C.no_gpu_fds()
  from tinygrad import Device
  from tinygrad.helpers import Context,ALLOW_DEVICE_USAGE
  from tinygrad.uop.ops import Ops
  assert ALLOW_DEVICE_USAGE.value==0 and not Device._opened_devices
  inventories={}
  for planner in (0,1):
    with Context(NO_MEMORY_PLANNER=planner,TC=0,ALLOW_TF32=0):
      b,x,c,s,targets,outputs,activation,objective,gradients=C.graph()
      linear=objective.schedule_linear(*outputs,activation,*gradients)
    buffers={u.key.hex():{'bytes':u.nbytes(),'dtype':str(u.dtype),'shape':list(u.shape),'device':u.device}
             for u in linear.toposort() if u.op is Ops.BUFFER}
    inventories[str(planner)]={'calls':len(linear.src),'unique_buffers':len(buffers),
      'total_buffer_bytes':sum(v['bytes'] for v in buffers.values()),'maximum_buffer_bytes':max(v['bytes'] for v in buffers.values()),
      'buffers':buffers,'linear_key':linear.key.hex()}
  fwd_linear=2*C.LENGTH*(2*2048*2048+2*2048*1024+2*2048*9216)
  fwd_attention=4*16*C.LENGTH*C.LENGTH*128
  report={'complete':True,'gpu_executed':False,'opened_devices':sorted(Device._opened_devices),'length':C.LENGTH,
    'parameter_count':50335744,'fp32_parameter_bytes':50335744*4,'fp32_gradient_bytes':50335744*4,
    'q_bytes':C.LENGTH*2048*4,'kv_each_bytes':C.LENGTH*1024*4,'mlp_activation_bytes':C.LENGTH*9216*4,
    'attention_matrix_bytes':16*C.LENGTH*C.LENGTH*4,'layouts':'B1; Q16xLx128; K/V8xLx128; row-major projection masters',
    'forward_matmul_flops':fwd_linear+fwd_attention,'forward_backward_matmul_flops_estimate':3*(fwd_linear+fwd_attention),
    'one_backward_plus_two_causal_control_forwards_flops_estimate':5*(fwd_linear+fwd_attention),
    'teacher_four_forward_flops_estimate':4*(fwd_linear+fwd_attention),'schedules_by_NO_MEMORY_PLANNER':inventories,
    'requested_limits':{'live':C.LIVE_LIMIT,'cumulative':C.TOTAL_LIMIT,'single_buffer':C.MAX_BUFFER,'free_reserve':C.FREE_RESERVE},
    'source_sha256':{str(p):C.file_sha(p) for p in (Path(__file__),C.EXP/'agent-distill-full-common.py',C.OLD/'frozen/agent-distill-tinygrad-common.py')}}
  assert not Device._opened_devices;C.no_gpu_fds()
  path=C.ROOT/'budget.json';assert not path.exists();C.write(path,report)
  print(json.dumps({k:v for k,v in report.items() if k!='schedules_by_NO_MEMORY_PLANNER'},indent=2))
  print(json.dumps({k:{n:v for n,v in x.items() if n!='buffers'} for k,x in inventories.items()},indent=2))

if __name__=='__main__':main()
