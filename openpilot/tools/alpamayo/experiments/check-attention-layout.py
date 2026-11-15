"""Check exact attention parity while changing only staged-key LDS layout."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import time

import numpy as np
from tinygrad import Tensor, TinyJit, Device, dtypes
from tinygrad.device import Compiled


def local(name):
  spec=importlib.util.spec_from_file_location(name.replace('-','_'),Path(__file__).with_name(name+'.py'))
  module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
  return module


def main():
  parser=argparse.ArgumentParser()
  parser.add_argument('--output',type=Path,required=True)
  parser.add_argument('--padded',action='store_true')
  parser.add_argument('--local-fences',action='store_true')
  parser.add_argument('--dpp',action='store_true')
  parser.add_argument('--kv-load',action='store_true')
  parser.add_argument('--value-layout',action='store_true')
  parser.add_argument('--probability-mma',action='store_true')
  parser.add_argument('--block-keys',type=int,nargs='+',choices=(16,32,64))
  args=parser.parse_args()
  Compiled.wait_timeout_ms=10000
  Tensor.manual_seed(2031)
  local('prefill-kernels').fix_dispatch_packet()
  attention=local('online-attention').attention
  os.environ.update(ALPAMAYO_ONLINE_STAGE_KV='1',ALPAMAYO_ONLINE_WAVE_SYNC='0',ALPAMAYO_HIP_WGP='0',
                    ALPAMAYO_ONLINE_NATIVE_EXP='0',ALPAMAYO_ONLINE_REGISTER_PROB='0')
  results=[]
  for batch,heads,kv_heads,rows,keys,width,causal in ((2,2,1,17,33,72,False),
      (1,16,8,3261,3261,128,True),(4,16,16,720,720,72,False),(8,16,16,364,364,72,False)):
    q=Tensor.randn(batch,heads,rows,width).cast(dtypes.bfloat16).realize()
    k,v=[Tensor.randn(batch,kv_heads,keys,width).cast(dtypes.bfloat16).realize() for _ in range(2)]
    for block_keys in (args.block_keys or ((32,) if args.padded or args.dpp or args.kv_load or args.value_layout else (32,64))):
      expected=None
      variants=((False,8,0,0),(True,8,0,0),(False,8,2,0),(False,8,4,0),(False,8,2,2),(False,8,4,4)) if args.padded else ((False,8,0,0),(True,8,0,0),(True,4,0,0))
      variants=[(*variant,False) for variant in variants]
      if args.local_fences:
        variants=[(False,8,0,0,False),(False,8,0,0,True),(False,4,0,0,True),(False,8,4,0,True)]
      variants=[(*variant,False) for variant in variants]
      if args.dpp:
        variants=[(False,8,4,0,False,False),(False,8,4,0,False,True),(False,16,4,0,False,True)]
      variants=[(*variant,0) for variant in variants]
      if args.kv_load:
        variants=[(False,8,4,0,False,False,0),(False,8,4,0,False,False,1),(False,8,4,0,False,False,2),
                  (False,16,4,0,False,False,1),(False,16,4,0,False,False,2)]
      variants=[(*variant,False) for variant in variants]
      if args.value_layout:
        variants=[(False,8,4,0,False,False,1,False),(False,8,4,0,False,False,1,True),
                  (False,16,4,0,False,False,1,True)]
      variants=[(*variant,False) for variant in variants]
      if args.probability_mma:
        variants=[(False,8,4,0,False,False,1,False,False),(False,8,4,0,False,False,1,False,True),
                  (False,16,4,0,False,False,1,False,True),(False,8,4,0,False,False,1,True,True),
                  (False,16,4,0,False,False,1,True,True)]
      for packed,waves,key_padding,prob_padding,local_fences,dpp,kv_load,value_layout,probability_mma in variants:
        os.environ.update(ALPAMAYO_ONLINE_PACKED_K=str(int(packed)),ALPAMAYO_ONLINE_WAVES=str(waves),
                          ALPAMAYO_ONLINE_BLOCK_KEYS=str(block_keys),ALPAMAYO_ONLINE_KEY_PADDING=str(key_padding),
                          ALPAMAYO_ONLINE_PROB_PADDING=str(prob_padding),ALPAMAYO_ONLINE_LOCAL_FENCES=str(int(local_fences)),
                          ALPAMAYO_ONLINE_DPP=str(int(dpp)),ALPAMAYO_ONLINE_KV_LOAD=str(kv_load),
                          ALPAMAYO_ONLINE_VALUE_LAYOUT=str(int(value_layout)),
                          ALPAMAYO_ONLINE_PROBABILITY_MMA=str(int(probability_mma)))
        jit=TinyJit(lambda a,b,c:attention(a,b,c,causal).realize())
        timings=[]
        for iteration in range(12):
          start=time.perf_counter();output=jit(q,k,v);Device[Device.DEFAULT].synchronize()
          if iteration>=2:timings.append((time.perf_counter()-start)*1000)
        values=output.float().numpy()
        assert np.isfinite(values).all()
        if expected is None:expected=values.copy()
        else:np.testing.assert_array_equal(values,expected)
        record=dict(batch=batch,heads=heads,kv_heads=kv_heads,rows=rows,keys=keys,width=width,causal=causal,
                     block_keys=block_keys,packed_k=packed,waves=waves,key_padding=key_padding,prob_padding=prob_padding,
                      local_fences=local_fences,dpp=dpp,kv_load=kv_load,value_layout=value_layout,probability_mma=probability_mma,
                    median_ms=float(np.median(timings)),exact=True)
        results.append(record);args.output.write_text(json.dumps(results,indent=2)+'\n');print(json.dumps(record),flush=True)


if __name__=='__main__':main()
