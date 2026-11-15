"""Check whether one submission helps independent, unchanged GPU operations."""
import argparse
import importlib.util
import json
from pathlib import Path
import time

import numpy as np
from tinygrad import Device, Tensor, TinyJit, dtypes
from tinygrad.device import Compiled
from amd_gemm import pack, unpack


def local(name):
  spec=importlib.util.spec_from_file_location(name.replace('-','_'),Path(__file__).with_name(name+'.py'))
  module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
  return module


def main():
  parser=argparse.ArgumentParser()
  parser.add_argument('--output',type=Path,required=True)
  args=parser.parse_args()
  Compiled.wait_timeout_ms=10000
  local('prefill-kernels').fix_dispatch_packet()
  local('wgp_compiler').install(Device[Device.DEFAULT])
  attention,gemm=local('online-attention').attention,local('tiled-gemm').gemm
  Tensor.manual_seed(9182)
  q=Tensor.randn(1,16,3261,128).cast(dtypes.bfloat16).realize()
  k,v=[Tensor.randn(1,8,3261,128).cast(dtypes.bfloat16).realize() for _ in range(2)]
  # Layout transformations are complete before either launch. This isolates
  # scheduling of independent kernels, not an end-to-end pipeline speed.
  x=unpack(pack((Tensor.randn(3261,2048)*8).cast(dtypes.fp8e4m3)).realize())
  w=unpack(pack((Tensor.randn(9216,2048)*8).cast(dtypes.fp8e4m3)).realize())
  sx,sw=Tensor([.125]).realize(),Tensor([.125]).realize()
  attn=TinyJit(lambda a,b,c:attention(a,b,c,causal=True).realize())
  matmul=TinyJit(lambda a,b,s,t:gemm(a,b,(s,t)).realize())
  def joined(a,b,c,d,e,s,t):
    values=(attention(a,b,c,causal=True),gemm(d,e,(s,t)))
    Tensor.realize(*values)
    return values
  joint=TinyJit(joined)
  variants={'attention':lambda:(attn(q,k,v),), 'gemm':lambda:(matmul(x,w,sx,sw),),
            'separate':lambda:(attn(q,k,v),matmul(x,w,sx,sw)),
            'joint':lambda:joint(q,k,v,x,w,sx,sw)}
  references={}
  results=[]
  for name in ('attention','gemm','separate','joint','separate','joint'):
    run=variants[name]
    times=[]
    for iteration in range(33):
      start=time.perf_counter();outputs=run();Device[Device.DEFAULT].synchronize()
      if iteration>=3:times.append((time.perf_counter()-start)*1000)
    actual=[value.float().numpy() for value in outputs]
    if name in ('attention','gemm'):references[name]=actual[0]
    else:
      np.testing.assert_array_equal(actual[0],references['attention'])
      np.testing.assert_array_equal(actual[1],references['gemm'])
    assert all(np.isfinite(value).all() for value in actual)
    record={'variant':name,'median_ms':float(np.median(times)),'exact':True}
    results.append(record);args.output.write_text(json.dumps(results,indent=2)+'\n');print(json.dumps(record),flush=True)


if __name__=='__main__':main()
