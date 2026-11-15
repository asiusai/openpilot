"""Compare compiled FP8 matrix kernels with the current packed implementation."""
import argparse
import importlib.util
import json
from pathlib import Path
import time

import numpy as np
from tinygrad import Device,Tensor,TinyJit,dtypes
from tinygrad.device import Compiled
from amd_gemm import pack,unpack


def local(name):
  spec=importlib.util.spec_from_file_location(name.replace('-','_'),Path(__file__).with_name(name+'.py'))
  module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
  return module


def main():
  parser=argparse.ArgumentParser()
  parser.add_argument('--output',type=Path,required=True)
  parser.add_argument('--diagnose-rounding',action='store_true',help='Record all timings and FP64 checks, then fail if exact parity differs')
  parser.add_argument('--row-major',action='store_true',help='Materialize separate row-major weights before timing Triton')
  args=parser.parse_args()
  Compiled.wait_timeout_ms=10000
  local('wgp_compiler').install(Device[Device.DEFAULT])
  original=local('tiled-gemm').gemm
  candidate=local('triton-gemm').gemm
  Tensor.manual_seed(42)
  results=[]
  for rows,columns,width in ((17,48,32),(3261,9216,2048),(3261,2048,9216),(2880,4304,1152),(2912,1152,4304)):
    x=(Tensor.randn(rows,width)*8).cast(dtypes.fp8e4m3).realize()
    w=unpack(pack((Tensor.randn(columns,width)*8).cast(dtypes.fp8e4m3)).realize())
    scales=(Tensor([.125]).realize(),Tensor([.125]).realize())
    row_weight=w.contiguous().realize() if args.row_major else w
    expected=None
    host_x=host_w=None
    variants=[('original',(64,64,64),4,1)]+[
      ('triton',tile,4,stages) for tile,stages in
      (((64,64,64),1),((64,128,64),1),((128,64,64),1),((128,128,64),1),((64,128,128),2),((128,64,128),2))]
    for name,tile,warps,stages in variants:
      fn=(lambda a,b,s,t:original(a,b,(s,t))) if name=='original' else (
        lambda a,b,s,t:candidate(a,b,(s,t),tile=tile,warps=warps,stages=stages,packed=not args.row_major))
      jit=TinyJit(lambda a,b,s,t:fn(a,b,s,t).realize())
      timings=[]
      for iteration in range(12):
        start=time.perf_counter();output=jit(x,w if name=='original' else row_weight,*scales);Device[Device.DEFAULT].synchronize()
        if iteration>=2:timings.append((time.perf_counter()-start)*1000)
      values=output.float().numpy()
      assert np.isfinite(values).all()
      if expected is None:expected=values.copy()
      different=np.flatnonzero(values.reshape(-1)!=expected.reshape(-1))
      record=dict(shape=[rows,columns,width],name=name,tile=tile,warps=warps,stages=stages,row_major=args.row_major and name=='triton',
        median_ms=float(np.median(timings)),exact=not different.size,differing_values=int(different.size),
        max_delta=float(np.max(np.abs(values-expected))))
      if different.size and args.diagnose_rounding:
        if host_x is None:host_x,host_w=x.float().numpy(),w.float().numpy()
        selected=different[np.linspace(0,different.size-1,min(1024,different.size),dtype=np.int64)]
        rr,cc=np.divmod(selected,columns)
        precise=np.sum(host_x[rr].astype(np.float64)*host_w[cc].astype(np.float64),axis=1)*(.125*.125)
        original_error=np.abs(expected[rr,cc].astype(np.float64)-precise)
        candidate_error=np.abs(values[rr,cc].astype(np.float64)-precise)
        record['fp64_checked_changed_values']=len(selected)
        record['fp64_original_mean_error']=float(original_error.mean())
        record['fp64_candidate_mean_error']=float(candidate_error.mean())
        record['fp64_candidate_better_count']=int(np.count_nonzero(candidate_error<original_error))
        record['fp64_candidate_worse_count']=int(np.count_nonzero(candidate_error>original_error))
      results.append(record);args.output.write_text(json.dumps(results,indent=2)+'\n');print(json.dumps(record),flush=True)
      if not args.diagnose_rounding:np.testing.assert_array_equal(values,expected)
  assert all(row['exact'] for row in results),'Some candidates differ; see recorded independent FP64 checks. No implementation selected.'


if __name__=='__main__':main()
