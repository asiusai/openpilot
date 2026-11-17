"""Verify native signed INT4 arithmetic and compare model-sized matrix costs.

Inputs are already quantized integers. Timing excludes calibration, activation
quantization and model quality; this does not qualify a four-bit model.
"""
import argparse
import importlib.util
import json
from pathlib import Path
from functools import partial
import time
import numpy as np
from tinygrad import Device,Tensor,TinyJit,dtypes
from amd_gemm import pack,_gemm


def main():
  p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
  spec=importlib.util.spec_from_file_location('int4_gemm',Path(__file__).with_name('int4-gemm.py'))
  module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
  rng=np.random.default_rng(9060);results=[]
  scale=Tensor([1.],dtype=dtypes.float32).reshape(()).realize()
  for m,n,k in ((17,48,32),(511,9216,2048),(2027,9216,2048),(2027,2048,9216)):
    a,b=(rng.integers(-8,8,(r,k),dtype=np.int32) for r in (m,n))
    def packed(x):
      x=np.pad(x,((0,(-x.shape[0])%16),(0,0)))
      tiles=x.reshape(-1,16,k//16,2,8).transpose(0,2,3,1,4).reshape(-1,k//16,32,8)
      values=np.bitwise_or.reduce((tiles.astype(np.uint32)&15)<<np.arange(0,32,4,dtype=np.uint32),axis=-1)
      return Tensor(values.view(np.int32)).realize()
    ap,bp=packed(a),packed(b)
    af,bf=(pack(Tensor(x.astype(np.float32)).cast(dtypes.fp8e4m3)).realize() for x in (a,b))
    def baseline(x,w):
      out=Tensor.empty(m,n,dtype=dtypes.bfloat16)
      return Tensor.custom_kernel(out,x,w,scale,scale,fxn=partial(_gemm,packed_input=True))[0]
    expected=baseline(af,bf).bitcast(dtypes.uint16).numpy()
    if m==17:
      cpu=Tensor((a@b.T).astype(np.float32),device='CPU').cast(dtypes.bfloat16).bitcast(dtypes.uint16).numpy()
      np.testing.assert_array_equal(expected,cpu)
    record={'shape':[m,n,k],'exact':True,'amortized_ms':{}}
    cases=[('fp8',baseline,af,bf)]+[(f'int4_tile{q}',partial(module.gemm,rows=m,columns=n,parts=q),ap,bp) for q in (1,2,4)]
    for name,fn,x,w in cases:
      np.testing.assert_array_equal(fn(x,w).bitcast(dtypes.uint16).numpy(),expected)
      jit=TinyJit(fn)
      for _ in range(3):jit(x,w)
      Device[Device.DEFAULT].synchronize();start=time.perf_counter()
      for _ in range(40):jit(x,w)
      Device[Device.DEFAULT].synchronize()
      record['amortized_ms'][name]=(time.perf_counter()-start)*1000/40
    results.append(record);print(json.dumps(record),flush=True)
  args.output.write_text(json.dumps(results,indent=2)+'\n')


if __name__=='__main__':main()
