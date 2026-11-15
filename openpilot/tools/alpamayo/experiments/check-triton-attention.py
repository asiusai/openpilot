"""Validate offline Triton kernels through the existing Chestnut USB runtime."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import time

import numpy as np
import torch
from tinygrad import Tensor, TinyJit, Device, dtypes
from tinygrad.device import Compiled
from amd_attention import attention as original


def local(name):
  spec=importlib.util.spec_from_file_location(name.replace('-','_'),Path(__file__).with_name(name+'.py'))
  module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
  return module


def main():
  parser=argparse.ArgumentParser()
  parser.add_argument('--output',type=Path,required=True)
  parser.add_argument('--pipeline',action='store_true')
  args=parser.parse_args()
  Compiled.wait_timeout_ms=10000
  torch.set_num_threads(4)
  Tensor.manual_seed(2031)
  local('prefill-kernels').fix_dispatch_packet()
  candidate=local('triton-attention').attention
  online=local('online-attention').attention
  os.environ.update(ALPAMAYO_ONLINE_BLOCK_KEYS='32',ALPAMAYO_ONLINE_WAVES='8',ALPAMAYO_ONLINE_STAGE_KV='1')
  results=[]
  # Small unequal query/key lengths check dispatch and padding before model-sized work.
  shapes=[(2,2,1,17,33,72,False),(1,16,8,3261,3261,128,True),
          (4,16,16,720,720,72,False),(8,16,16,364,364,72,False)]
  if args.pipeline: shapes=shapes[:2]
  for batch,heads,kv_heads,rows,keys,width,causal in shapes:
    q=Tensor.randn(batch,heads,rows,width).cast(dtypes.bfloat16).realize()
    k,v=[Tensor.randn(batch,kv_heads,keys,width).cast(dtypes.bfloat16).realize() for _ in range(2)]
    qt,kt,vt=[torch.from_numpy(t.float().numpy()) for t in (q,k,v)]
    expected=torch.nn.functional.scaled_dot_product_attention(qt,kt.repeat_interleave(heads//kv_heads,1),
                 vt.repeat_interleave(heads//kv_heads,1),is_causal=causal).numpy()
    variants=[('original',64,64,4,1,1),('online',64,64,4,1,1)]
    if args.pipeline:
      variants += [('triton',64,32,4,stages,waves) for waves in (1,2) for stages in (1,2,3)]
      variants += [('triton',128,32,4,stages,1) for stages in (2,3)]
    else:
      variants += [('triton',bm,bn,warps,1,1) for bm,bn,warps in ((32,32,4),(64,32,4),(64,64,4),(128,32,4),(128,64,8),(64,64,8))]
    baseline_error=None
    triton_baseline=None
    for name,bm,bn,warps,stages,waves in variants:
      os.environ.update(ALPAMAYO_TRITON_BLOCK_M=str(bm),ALPAMAYO_TRITON_BLOCK_N=str(bn),ALPAMAYO_TRITON_WARPS=str(warps),
                        ALPAMAYO_TRITON_STAGES=str(stages),ALPAMAYO_TRITON_WAVES_PER_EU=str(waves))
      fn={'original':original,'online':online,'triton':candidate}[name]
      jit=TinyJit(lambda a,b,c:fn(a,b,c,causal).realize())
      timings=[]
      for iteration in range(7):
        start=time.perf_counter();output=jit(q,k,v);Device[Device.DEFAULT].synchronize()
        if iteration>=2: timings.append((time.perf_counter()-start)*1000)
      values=output.float().numpy()
      assert np.isfinite(values).all()
      rmse=float(np.sqrt(np.mean((values-expected)**2)))
      if baseline_error is None: baseline_error=rmse
      assert rmse<=baseline_error*1.1,(name,bm,bn,warps,rmse,baseline_error)
      record=dict(batch=batch,heads=heads,kv_heads=kv_heads,rows=rows,keys=keys,width=width,causal=causal,
                   name=name,block_m=bm,block_n=bn,warps=warps,median_ms=float(np.median(timings)),
                   stages=stages,waves_per_eu=waves,rmse_fp32=rmse,max_error_fp32=float(np.max(np.abs(values-expected))))
      if name=='triton' and args.pipeline:
        if triton_baseline is None:triton_baseline=values.copy()
        record['exact_triton_baseline']=bool(np.array_equal(values,triton_baseline))
      results.append(record);args.output.write_text(json.dumps(results,indent=2)+'\n');print(json.dumps(record),flush=True)


if __name__=='__main__': main()
