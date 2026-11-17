"""Check fused FP8 MLP output bytes and amortized matrix latency on the target GPU."""
import argparse
import importlib.util
import json
from pathlib import Path
from functools import partial
import time
import numpy as np
from tinygrad import Device, Tensor, TinyJit, dtypes
from amd_gemm import pack, _gemm


def module(name):
  spec=importlib.util.spec_from_file_location(name.replace('-','_'),Path(__file__).with_name(name+'.py'))
  value=importlib.util.module_from_spec(spec);spec.loader.exec_module(value);return value


def main():
  parser=argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--output',type=Path,required=True)
  args=parser.parse_args()
  fused, quantize = module('gemm-epilogue').mlp_input,module('quantize-packed').quantize
  rng=np.random.default_rng(810)
  results=[]
  for rows,columns,width in ((17,48,32),(145,2048,2048),(2027,9216,2048)):
    x=pack(Tensor(rng.normal(size=(rows,width)).astype(np.float32)).cast(dtypes.fp8e4m3)).realize()
    w=pack(Tensor(rng.normal(size=(columns,width)).astype(np.float32)).cast(dtypes.fp8e4m3)).realize()
    a,b,c=(Tensor([v],dtype=dtypes.float32).reshape(()).realize() for v in (.3,.2,.35))
    bias=Tensor(rng.normal(size=columns).astype(np.float32)).cast(dtypes.bfloat16).realize()
    def baseline(x,w):
      out=Tensor.empty(rows,columns,dtype=dtypes.bfloat16)
      h=Tensor.custom_kernel(out,x,w,a,b,bias,fxn=partial(_gemm,packed_input=True))[0]
      return quantize(h.relu().square(),c)[0]
    def candidate(x,w):return fused(x,w,a,b,c,rows,bias)
    expected=baseline(x,w).bitcast(dtypes.uint8).numpy()
    actual=candidate(x,w).bitcast(dtypes.uint8).numpy()
    np.testing.assert_array_equal(actual,expected)
    timings={}
    for name,function in (('baseline',baseline),('fused',candidate)):
      jit=TinyJit(function)
      for _ in range(3):jit(x,w)
      Device[Device.DEFAULT].synchronize()
      start=time.perf_counter()
      for _ in range(40):jit(x,w)
      Device[Device.DEFAULT].synchronize()
      timings[name]=(time.perf_counter()-start)*1000/40
    results.append({'shape':[rows,columns,width],'exact':True,'amortized_ms':timings})
    print(json.dumps(results[-1]),flush=True)
  args.output.write_text(json.dumps(results,indent=2)+'\n')


if __name__=='__main__':main()
