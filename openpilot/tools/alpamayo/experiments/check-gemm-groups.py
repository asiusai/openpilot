"""Check exact GEMM results while grouping neighboring tiles into workgroups."""
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
  parser.add_argument('--tiles',action='store_true',help='Compare rectangular register tiles and reduction unrolling')
  parser.add_argument('--swizzle',action='store_true',help='Compare tile ordering for cache reuse')
  args=parser.parse_args()
  assert not(args.tiles and args.swizzle)
  Compiled.wait_timeout_ms=10000
  local('wgp_compiler').install(Device[Device.DEFAULT])
  tiles=local('tiled-gemm')
  Tensor.manual_seed(42)
  results=[]
  shapes=[(3261,9216,2048),(3261,2048,9216),(2880,4304,1152),(2912,1152,4304)]
  if args.swizzle:shapes.insert(0,(81,144,64))
  for rows,columns,width in shapes:
    x=(Tensor.randn(rows,width)*8).cast(dtypes.fp8e4m3).realize()
    w=unpack(pack((Tensor.randn(columns,width)*8).cast(dtypes.fp8e4m3)).realize())
    scales=(Tensor([.125]).realize(),Tensor([.125]).realize())
    expected=None
    variants=[(group,swap,(4,4),1) for group,swap in
      (((1,1),False),((2,1),False),((4,1),False),((1,2),False),((1,4),False),((2,2),False),((1,1),True),((4,1),True))]
    if args.tiles:
      variants=[((1,1),False,tile,unroll) for tile,unroll in
        (((4,4),1),((3,4),1),((4,3),1),((3,3),1),((2,4),1),((4,2),1),((8,2),1),((2,8),1),
         ((4,4),2),((4,4),4),((3,4),2),((4,3),2))]
    variants=[(*variant,1) for variant in variants]
    if args.swizzle:variants=[((1,1),False,(4,4),1,group_m) for group_m in (1,2,4,8,16)]
    for group,swap,tile,unroll,group_m in variants:
      jit=TinyJit(lambda a,b,s,t:tiles.gemm(a,b,(s,t),local=group,swap=swap,tile=tile,unroll_k=unroll,group_m=group_m).realize())
      timings=[]
      for iteration in range(12):
        start=time.perf_counter();out=jit(x,w,*scales);Device[Device.DEFAULT].synchronize()
        if iteration>=2:timings.append((time.perf_counter()-start)*1000)
      values=out.float().numpy()
      assert np.isfinite(values).all()
      if expected is None:expected=values.copy()
      else:np.testing.assert_array_equal(values,expected)
      record=dict(shape=[rows,columns,width],local=group,swap=swap,tile=tile,unroll_k=unroll,group_m=group_m,median_ms=float(np.median(timings)),exact=True)
      results.append(record);args.output.write_text(json.dumps(results,indent=2)+'\n');print(json.dumps(record),flush=True)


if __name__=='__main__':main()
