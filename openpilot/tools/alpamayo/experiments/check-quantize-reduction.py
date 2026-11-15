"""Check every activation scale and FP8 byte before selecting the reduction."""
import argparse
import importlib.util
import json
from pathlib import Path
import time

import numpy as np
from tinygrad import Tensor, TinyJit, Device, dtypes
from tinygrad.device import Compiled
from model import EdgeModel


def local(name):
  spec=importlib.util.spec_from_file_location(name.replace('-','_'),Path(__file__).with_name(name+'.py'))
  module=importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module


def main():
  parser=argparse.ArgumentParser()
  parser.add_argument('--output',type=Path,required=True)
  args=parser.parse_args()
  Compiled.wait_timeout_ms=10000
  local('prefill-kernels').fix_dispatch_packet()
  candidate=local('quantize-reduction').quantize
  grouped_candidate=local('quantize-reduction').grouped_quantize
  existing=local('fast-quantize').quantize
  reference=EdgeModel.__new__(EdgeModel)
  reference.cuda_kernels=False
  Tensor.manual_seed(2491)
  cases=[(1,2048,'zero'),(1,2048,'tiny'),(17,1152,'random'),(364,1152,'random'),
         (720,1152,'random'),(364,4304,'random'),(3261,2048,'random'),(3261,9216,'random')]
  results=[]
  for rows,width,kind in cases:
    x=Tensor.randn(rows,width)
    if kind=='zero': x=x*0
    if kind=='tiny': x=x*1e-13
    x=x.cast(dtypes.bfloat16).realize()
    expected, scale = reference.quantize(x)
    expected_bytes=expected.bitcast(dtypes.uint8).numpy()
    expected_scale=scale.numpy()
    timings={}
    for name,fn in [('existing',lambda v: existing(reference,v)),('candidate',candidate)]:
      def operation(v):
        q,s=fn(v)
        return q.realize(),s.realize()
      jit=TinyJit(operation)
      values=[]
      for iteration in range(7):
        start=time.perf_counter()
        quantized, actual_scale=jit(x)
        Device[Device.DEFAULT].synchronize()
        if iteration>=2: values.append((time.perf_counter()-start)*1000)
      np.testing.assert_array_equal(actual_scale.numpy(),expected_scale)
      np.testing.assert_array_equal(quantized.bitcast(dtypes.uint8).numpy(),expected_bytes)
      timings[name]=float(np.median(values))
    result=dict(rows=rows,width=width,kind=kind,exact_scale=True,exact_fp8_bytes=True,median_ms=timings)
    results.append(result)
    args.output.write_text(json.dumps(results,indent=2)+'\n')
    print(json.dumps(result),flush=True)

  for rows,width in ((364,1152),(720,1152),(364,4304),(720,4304)):
    images=4
    # Different image ranges detect accidental sharing of activation scales.
    x=(Tensor.randn(images,rows,width)*Tensor.arange(images).reshape(images,1,1)).cast(dtypes.bfloat16).realize()
    expected_values=[]
    expected_scales=[]
    for i in range(images):
      q,s=reference.quantize(x[i])
      expected_values.append(q.bitcast(dtypes.uint8).numpy())
      expected_scales.extend([s.numpy().item()]*rows)
    expected_bytes=np.concatenate(expected_values)
    expected_scales=np.asarray(expected_scales,dtype=np.float32)
    def grouped_existing(v):
      scale=(v.float().abs().max(axis=(1,2),keepdim=True)/448).clamp(min_=1e-12).realize()
      q=(v.float()/scale).clamp(-448,448).cast(dtypes.fp8e4m3).reshape(-1,width).contiguous()
      return q,scale.reshape(images,1).expand(images,rows).reshape(-1).contiguous()
    timings={}
    for name,fn in [('existing',grouped_existing),('candidate',lambda v:grouped_candidate(v,images))]:
      def operation(v):
        q,s=fn(v)
        return q.realize(),s.realize()
      jit=TinyJit(operation)
      values=[]
      for iteration in range(7):
        start=time.perf_counter()
        q,s=jit(x)
        Device[Device.DEFAULT].synchronize()
        if iteration>=2: values.append((time.perf_counter()-start)*1000)
      np.testing.assert_array_equal(q.bitcast(dtypes.uint8).numpy(),expected_bytes)
      np.testing.assert_array_equal(s.numpy(),expected_scales)
      timings[name]=float(np.median(values))
    result=dict(rows=rows,width=width,images=images,exact_scale=True,exact_fp8_bytes=True,median_ms=timings)
    results.append(result)
    args.output.write_text(json.dumps(results,indent=2)+'\n')
    print(json.dumps(result),flush=True)


if __name__=='__main__':
  main()
