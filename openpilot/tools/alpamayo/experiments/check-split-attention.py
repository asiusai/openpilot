"""Compare key partitions with FP32 attention, including causal empty partitions."""
import argparse
import importlib.util
import json
from pathlib import Path
import time

import numpy as np
import torch
from tinygrad import Device, Tensor, TinyJit, dtypes
from tinygrad.device import Compiled


def local(name):
  spec=importlib.util.spec_from_file_location(name.replace('-','_'),Path(__file__).with_name(name+'.py'))
  module=importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module


def main():
  parser=argparse.ArgumentParser()
  parser.add_argument('--output',type=Path,required=True)
  parser.add_argument('--prefill-only',action='store_true')
  args=parser.parse_args()
  Compiled.wait_timeout_ms=10000
  torch.set_num_threads(4)
  local('prefill-kernels').fix_dispatch_packet()
  online=local('online-attention')
  split=local('split-attention')
  Tensor.manual_seed(2031)
  results=[]
  shapes=[
      (2,2,1,17,33,72,True),(2,2,1,17,33,72,False),
      (1,16,8,3261,3261,128,True),(4,16,16,720,720,72,False),(8,16,16,364,364,72,False)]
  if args.prefill_only: shapes=shapes[:3]
  for batch,heads,kv_heads,rows,keys,width,causal in shapes:
    q=Tensor.randn(batch,heads,rows,width).cast(dtypes.bfloat16).realize()
    k=Tensor.randn(batch,kv_heads,keys,width).cast(dtypes.bfloat16).realize()
    v=Tensor.randn(batch,kv_heads,keys,width).cast(dtypes.bfloat16).realize()
    tq,tk,tv=[torch.from_numpy(x.float().numpy()) for x in (q,k,v)]
    expected=torch.nn.functional.scaled_dot_product_attention(tq,tk.repeat_interleave(heads//kv_heads,dim=1),
      tv.repeat_interleave(heads//kv_heads,dim=1),is_causal=causal).numpy()
    baseline_error=None
    baseline=None
    for name, fn in [('online',lambda a,b,c: online.attention(a,b,c,causal))]+[
        (f'split-{s}-waves-{w}',lambda a,b,c,s=s,w=w:split.attention(a,b,c,causal,splits=s,waves=w))
        for s,w in ((1,8),(2,8),(4,8),(8,8),(4,4))]:
      jit=TinyJit(lambda a,b,c:fn(a,b,c).realize())
      times=[]
      for iteration in range(7):
        Device[Device.DEFAULT].synchronize()
        start=time.perf_counter()
        output=jit(q,k,v)
        Device[Device.DEFAULT].synchronize()
        if iteration>=2: times.append((time.perf_counter()-start)*1000)
      actual=output.float().numpy()
      assert np.isfinite(actual).all(),name
      error=float(np.sqrt(np.mean((actual-expected)**2,dtype=np.float64)))
      if baseline_error is None:
        baseline_error=error
        baseline=actual
      assert error<=baseline_error*1.1+1e-7,(name,error,baseline_error)
      record={'shape':[batch,heads,kv_heads,rows,keys,width,causal],'name':name,
              'median_ms':float(np.median(times)),'fp32_rmse':error,
              'baseline_exact':bool(np.array_equal(actual,baseline)),
              'max_baseline_delta':float(np.max(np.abs(actual-baseline)))}
      results.append(record)
      args.output.write_text(json.dumps(results,indent=2)+'\n')
      print(json.dumps(record),flush=True)


if __name__=='__main__':
  main()
