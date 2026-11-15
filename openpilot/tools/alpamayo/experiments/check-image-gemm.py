"""Check shared matrix dispatch against separate image calls, including tails."""
import argparse
import importlib.util
import json
from pathlib import Path
import time

import numpy as np
from tinygrad import Tensor, TinyJit, Device, dtypes
from tinygrad.device import Compiled
from model import EdgeModel
from amd_gemm import gemm, pack, unpack
import wgp_compiler


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
  wgp_compiler.install(Device[Device.DEFAULT])
  local('prefill-kernels').fix_dispatch_packet()
  candidate=local('image-gemm').gemm
  quantizer=EdgeModel.__new__(EdgeModel)
  quantizer.cuda_kernels=False
  Tensor.manual_seed(2718)
  results=[]
  for count,rows,columns,width in ((3,81,144,32),(4,364,4304,1152),(8,364,4304,1152),(4,720,4304,1152)):
    source=(Tensor.randn(count,rows,width)*Tensor.arange(1,count+1).reshape(count,1,1)).cast(dtypes.bfloat16).realize()
    parts=[quantizer.quantize(source[i]) for i in range(count)]
    parts=[(q.realize(),s.realize()) for q,s in parts]
    inputs=Tensor.stack(*[q for q,_ in parts]).contiguous().realize()
    scales=Tensor.cat(*[s.reshape(1) for _,s in parts]).contiguous().realize()
    weight,ws=quantizer.quantize(Tensor.randn(columns,width).cast(dtypes.bfloat16))
    weight,ws=unpack(pack(weight).realize()),ws.realize()
    bias=Tensor.randn(columns).cast(dtypes.bfloat16).realize()
    for use_bias in (False,True):
      chosen_bias=bias if use_bias else None
      expected=None
      for mode in ('individual','image-axis'):
        def operation():
          if mode=='individual':
            return Tensor.stack(*[gemm(q,weight,(s,ws),bias=chosen_bias) for q,s in parts]).realize()
          return candidate(inputs,weight,(scales,ws),bias=chosen_bias).realize()
        jit=TinyJit(operation)
        elapsed=[]
        for iteration in range(12):
          start=time.perf_counter()
          result=jit()
          Device[Device.DEFAULT].synchronize()
          if iteration>=2: elapsed.append((time.perf_counter()-start)*1000)
        actual=result.float().numpy()
        assert np.isfinite(actual).all()
        if expected is None: expected=actual.copy()
        record=dict(shape=[count,rows,columns,width],bias=use_bias,mode=mode,median_ms=float(np.median(elapsed)),
                    exact=bool(np.array_equal(actual,expected)),different_values=int(np.count_nonzero(actual!=expected)),
                    max_delta=float(np.max(np.abs(actual-expected))))
        results.append(record)
        args.output.write_text(json.dumps(results,indent=2)+'\n')
        print(json.dumps(record),flush=True)
        np.testing.assert_array_equal(actual,expected)


if __name__=='__main__':main()
