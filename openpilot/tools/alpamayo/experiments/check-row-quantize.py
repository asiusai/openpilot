"""Check device row maxima against the existing per-token FP8 quantizer."""
import argparse
import importlib.util
import json
from pathlib import Path
import statistics
import time

import numpy as np
from tinygrad import Tensor, TinyJit, Device, dtypes
from tinygrad.device import Compiled


def original(x):
  values=x.reshape(-1,x.shape[-1]).float().contiguous().realize()
  scales=(values.abs().max(axis=1,keepdim=True)/448).clamp(min_=1e-12).realize()
  return (values/scales).clamp(-448,448).cast(dtypes.fp8e4m3).contiguous().realize(),scales.reshape(-1).realize()


def main():
  parser=argparse.ArgumentParser()
  parser.add_argument('--output',type=Path,required=True)
  args=parser.parse_args()
  Compiled.wait_timeout_ms=10000
  spec=importlib.util.spec_from_file_location('prefill',Path(__file__).with_name('prefill-kernels.py'))
  prefill=importlib.util.module_from_spec(spec)
  spec.loader.exec_module(prefill)
  prefill.fix_dispatch_packet()
  Tensor.manual_seed(7213)
  spec=importlib.util.spec_from_file_location('row_reduction',Path(__file__).with_name('quantize-reduction.py'))
  candidate=importlib.util.module_from_spec(spec)
  spec.loader.exec_module(candidate)
  results=[]
  for rows,width,kind in ((17,1152,'zero'),(17,2048,'tiny'),(128,2048,'random'),(3261,2048,'random'),(3261,9216,'random')):
    x=Tensor.randn(rows,width).cast(dtypes.bfloat16)
    if kind=='zero': x=x*0
    elif kind=='tiny': x=x*1e-13
    x=x.realize()
    baseline=None
    for name,fn in (('original',original),('device-row-max',candidate.row_quantize)):
      run=TinyJit(lambda value:tuple(item.realize() for item in fn(value)))
      timings=[]
      for iteration in range(12):
        start=time.perf_counter()
        output,scales=run(x)
        Device[Device.DEFAULT].synchronize()
        if iteration>=2: timings.append((time.perf_counter()-start)*1000)
      actual=(output.bitcast(dtypes.uint8).numpy(),scales.numpy())
      assert np.isfinite(actual[1]).all()
      if baseline is None: baseline=actual
      else:
        for value,expected in zip(actual,baseline,strict=True): np.testing.assert_array_equal(value,expected)
      row=dict(shape=[rows,width],kind=kind,name=name,median_ms=statistics.median(timings),exact=True)
      results.append(row)
      print(json.dumps(row),flush=True)
      args.output.write_text(json.dumps(results,indent=2)+'\n')


if __name__=='__main__':
  main()
