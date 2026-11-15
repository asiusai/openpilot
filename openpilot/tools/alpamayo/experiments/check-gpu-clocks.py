"""Compare stock automatic and high clocks under a real prefill-layer load."""
import argparse
import importlib.util
import json
from pathlib import Path
import time

import numpy as np
from tinygrad import Device, Tensor, TinyJit, dtypes
from tinygrad.device import Compiled


def local(name):
  spec=importlib.util.spec_from_file_location(name.replace('-','_'),Path(__file__).with_name(name+'.py'))
  module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
  return module


def main():
  parser=argparse.ArgumentParser()
  parser.add_argument('--checkpoint',required=True)
  parser.add_argument('--output',type=Path,required=True)
  args=parser.parse_args()
  Compiled.wait_timeout_ms=10000
  import benchmark
  local('benchmark-fast')
  device=Device[Device.DEFAULT]
  local('wgp_compiler').install(device)
  model=benchmark.EdgeModel(args.checkpoint,precision='fp8')
  local('fast-quantize').install(model,materialize=True)
  local('prefill-kernels').fix_dispatch_packet()
  local('quantize-reduction').install(model)
  local('online-attention').install(model)
  local('fused-norm').install(model)
  metrics=local('gpu-metrics')
  smu=device.iface.dev_impl.smu
  Tensor.manual_seed(81)
  x=Tensor.randn(1,3261,2048).cast(dtypes.bfloat16).realize()
  cos=Tensor.ones(1,1,3261,128,dtype=dtypes.bfloat16).realize()
  sin=Tensor.zeros(1,1,3261,128,dtype=dtypes.bfloat16).realize()
  cache=Tensor.zeros(2,1,8,3584,128,dtype=dtypes.bfloat16).realize()
  jit=TinyJit(lambda value,c,s,kv:model.text_layer(value,c,s,'vlm.model.language_model.layers.0',kv,cache_pos=0).realize())
  for _ in range(3):
    jit(x,cos,sin,cache);model.synchronize()
  expected=None
  results=[]
  try:
    for name,level in (('auto',None),('high',-1),('auto-repeat',None)):
      smu.set_clocks(level)
      timings=[]
      for iteration in range(120):
        started=time.perf_counter();output=jit(x,cos,sin,cache);model.synchronize()
        if iteration>=20:timings.append((time.perf_counter()-started)*1000)
      telemetry=metrics.read(device)
      values=output.float().numpy()
      assert np.isfinite(values).all()
      if expected is None:expected=values.copy()
      else:np.testing.assert_array_equal(values,expected)
      row=dict(mode=name,median_ms=float(np.median(timings)),mean_ms=float(np.mean(timings)),metrics=telemetry,exact=True)
      results.append(row);args.output.write_text(json.dumps(results,indent=2)+'\n');print(json.dumps(row),flush=True)
  finally:
    model.synchronize()
    smu.set_clocks(None)


if __name__=='__main__':main()
