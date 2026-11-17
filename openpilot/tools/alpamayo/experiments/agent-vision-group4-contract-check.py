"""CPU exercise of the actual parent control function and strict comparison gate."""
import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np

HERE=Path(__file__).resolve().parent
s=importlib.util.spec_from_file_location('agent_vision_group4_contract_common',HERE/'agent-vision-group4-common.py')
c=importlib.util.module_from_spec(s);s.loader.exec_module(c)


def main():
  c.require(os.environ.get('DEV')=='CPU','Set DEV=CPU')
  import sys
  sys.path[:0]=[str(c.BASE),str(c.BASE/'experiments')]
  from tinygrad import Device,Tensor
  c.require(Device.DEFAULT=='CPU','CPU only')
  runtime=c.module(HERE/'agent-vision-group4-run.py','agent_vision_group4_contract_run')
  cache_module=c.module(c.BASE/'experiments/vision-cache.py','agent_vision_group4_contract_cache')
  values=(np.arange(4*720*8,dtype=np.float32).reshape(2880,8)%101)/101+1
  pixels=Tensor(values,device='CPU').realize();positions=Tensor.zeros(2880,8,device='CPU').realize()
  all_records=[]
  for limit in (2,4):
    cache=SimpleNamespace(image_view=cache_module.VisionCache.image_view,encoders={},cache={},group_limit=limit,
      batch_encoder=lambda *xs:tuple((p+v).contiguous().realize() for p,v in zip(xs[::2],xs[1::2],strict=True)))
    record=[];runtime.controls(cache,pixels,positions,record,lambda _:None)
    all_records.append(record)
  for a,b in zip(*all_records,strict=True):
    for key in ('images','original','mutated','restored','initial_capture_replay_exact','mutation_replay_exact','earlier_images_independent'):
      c.require(a[key]==b[key],'CPU control partition parity failed')
  def coupled(*xs):
    maximum=Tensor.stack(*(p.max() for p in xs[::2])).max()
    return tuple((p+maximum).contiguous().realize() for p in xs[::2])
  bad=SimpleNamespace(image_view=cache_module.VisionCache.image_view,encoders={},cache={},group_limit=4,batch_encoder=coupled)
  try:runtime.controls(bad,pixels,positions,[],lambda _:None)
  except ValueError as error:
    c.require('mutation coupled earlier images' in str(error),'Unexpected negative-control failure')
  else:raise AssertionError('Cross-image scale coupling was not rejected')
  positive=runtime.tensor_bits(Tensor(np.array([0.],np.float32),device='CPU'))
  negative=runtime.tensor_bits(Tensor(np.array([-0.],np.float32),device='CPU'))
  c.require(positive['sha256']!=negative['sha256'],'Bit check hid signed zero')
  output=c.WORK/'agent-vision-group4-contract-check-v2.json'
  c.write(output,dict(cpu_only=True,gpu_executed=False,parent_control_function_actual_cpu_jit_passed=True,
    remainders=[1,2,3,4],group_limits=[2,4],cross_image_coupling_negative_control_rejected=True,signed_zero_distinguished=True,
    limits='Independent arithmetic CPU fixture, not original vision/GPU model parity',
    source_sha256={str(path):c.sha(path) for path in (Path(__file__),HERE/'agent-vision-group4-run.py',HERE/'agent-vision-group4-compare.py')}))
  print(json.dumps(dict(cpu_contract_passed=True,gpu_executed=False),indent=2))


if __name__=='__main__':main()
