"""Validate shared loads separately from the model, including padded tiles."""
import argparse
import importlib.util
import json
from pathlib import Path
import time

import numpy as np
from tinygrad import Tensor, Device, TinyJit, dtypes
from tinygrad.device import Compiled
from amd_gemm import pack,unpack


def local(name):
  spec=importlib.util.spec_from_file_location(name.replace('-','_'),Path(__file__).with_name(name+'.py'))
  module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
  return module


def main():
  parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
  parser.add_argument('--smaller-tiles',action='store_true')
  args=parser.parse_args();Compiled.wait_timeout_ms=10000
  local('prefill-kernels').fix_dispatch_packet()
  local('wgp_compiler').install(Device[Device.DEFAULT])
  original,candidate=local('tiled-gemm'),local('shared-vector-gemm')
  Tensor.manual_seed(42);results=[]
  for rows,columns,width in ((17,48,32),(145,144,64),(3261,9216,2048),(3261,2048,9216)):
    x=(Tensor.randn(rows,width)*8).cast(dtypes.fp8e4m3).realize()
    w=unpack(pack((Tensor.randn(columns,width)*8).cast(dtypes.fp8e4m3)).realize())
    packed=candidate.layout.pack32(w).realize()
    scales=(Tensor([.125]).realize(),Tensor([.125]).realize());expected=None
    variants=[(group,prefetch,(4,4)) for group,prefetch in ((None,False),((2,2),False),((2,2),True),((4,2),True),((2,4),True))]
    if args.smaller_tiles:
      variants=[(None,False,(4,4)),((2,2),True,(4,4)),((4,2),True,(2,4)),((2,4),True,(4,2)),
                ((4,4),True,(2,2)),((4,4),False,(2,2)),((2,2),True,(2,4)),((2,2),True,(4,2))]
    for group,prefetch,tile in variants:
      if group is None:
        run=TinyJit(lambda a,b,s,t:original.gemm(a,b,(s,t)).realize());weight=w
      else:
        run=TinyJit(lambda a,b,s,t:candidate.gemm(a,b,columns,(s,t),group,prefetch,tile).realize());weight=packed
      timings=[]
      for iteration in range(12):
        start=time.perf_counter();out=run(x,weight,*scales);Device[Device.DEFAULT].synchronize()
        if iteration>=2:timings.append((time.perf_counter()-start)*1000)
      values=out.float().numpy();assert np.isfinite(values).all()
      if expected is None:expected=values.copy()
      else:np.testing.assert_array_equal(values,expected)
      record=dict(shape=[rows,columns,width],group=group,prefetch=prefetch,tile=tile,median_ms=float(np.median(timings)),exact=True)
      results.append(record);args.output.write_text(json.dumps(results,indent=2)+'\n');print(json.dumps(record),flush=True)


if __name__=='__main__':main()
