"""Compare one actual language layer, including changing inputs and KV writes."""
import argparse
import importlib.util
import json
from pathlib import Path
import time

import numpy as np
from tinygrad import Tensor, Device, TinyJit, dtypes
from tinygrad.device import Compiled


def local(name):
  spec=importlib.util.spec_from_file_location(name.replace('-','_'),Path(__file__).with_name(name+'.py'))
  module=importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module


def main():
  parser=argparse.ArgumentParser()
  parser.add_argument('--checkpoint',required=True)
  parser.add_argument('--output',type=Path,required=True)
  args=parser.parse_args()
  Compiled.wait_timeout_ms=10000
  import benchmark
  local('benchmark-fast')
  local('wgp_compiler').install(Device[Device.DEFAULT])
  model=benchmark.EdgeModel(args.checkpoint,precision='fp8')
  local('fast-quantize').install(model,materialize=True)
  local('prefill-kernels').fix_dispatch_packet()
  local('quantize-reduction').install(model)
  local('online-attention').install(model)
  local('fused-norm').install(model)
  Tensor.manual_seed(81)
  rows=3261
  x=Tensor.randn(1,rows,2048).cast(dtypes.bfloat16).realize()
  cos=Tensor.ones(1,1,rows,128,dtype=dtypes.bfloat16).contiguous().realize()
  sin=Tensor.zeros(1,1,rows,128,dtype=dtypes.bfloat16).contiguous().realize()
  cache=Tensor.zeros(2,1,8,3584,128,dtype=dtypes.bfloat16).contiguous().realize()
  inputs=(x,cos,sin,cache)
  def new_jit():
    return TinyJit(lambda value,c,s,kv:model.text_layer(value,c,s,'vlm.model.language_model.layers.0',kv,cache_pos=0))
  original=new_jit()
  timings={}
  for name in ('separate','packed'):
    if name=='packed':
      count=local('quantize-packed').install(model)
      candidate=new_jit()
    fn=original if name=='separate' else candidate
    elapsed=[]
    for iteration in range(12):
      started=time.perf_counter()
      result=fn(*inputs).realize()
      model.synchronize()
      if iteration>=2:elapsed.append((time.perf_counter()-started)*1000)
    timings[name]=float(np.median(elapsed))
    if name=='separate':
      expected=result.float().numpy().copy()
      expected_cache=cache.float().numpy().copy()
    else:
      np.testing.assert_array_equal(result.float().numpy(),expected)
      np.testing.assert_array_equal(cache.float().numpy(),expected_cache)
  for _ in range(3):
    x.assign((x+.03125).cast(dtypes.bfloat16)).realize()
    expected=original(*inputs).float().numpy().copy()
    expected_cache=cache.float().numpy().copy()
    actual=candidate(*inputs).float().numpy()
    assert np.isfinite(actual).all()
    np.testing.assert_array_equal(actual,expected)
    np.testing.assert_array_equal(cache.float().numpy(),expected_cache)
  report=dict(rows=rows,packed_matrices=count,median_ms=timings,exact_output=True,exact_kv=True,changing_inputs=3)
  args.output.write_text(json.dumps(report,indent=2)+'\n')
  print(json.dumps(report),flush=True)


if __name__=='__main__':main()
