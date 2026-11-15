"""Check padded edge tiles and random FP8 products, then time actual model shapes."""
import importlib.util
import json
from pathlib import Path
import time
import numpy as np
from tinygrad import Device,Tensor,TinyJit,dtypes

def module(name):
  spec=importlib.util.spec_from_file_location(name,Path(__file__).with_name(name+'.py'))
  result=importlib.util.module_from_spec(spec);spec.loader.exec_module(result);return result

module('prefill-kernels').fix_dispatch_packet()
lds=module('lds-gemm')
Tensor.manual_seed(2026)
results=[]
for m,n,k in ((65,97,128),(196,1152,1152),(2027,9216,2048),(2027,2048,9216)):
  x=Tensor.randn(m,k).cast(dtypes.fp8e4m3).realize()
  w=Tensor.randn(n,k).cast(dtypes.fp8e4m3).realize()
  expected=x.float().numpy()@w.float().numpy().T
  for tile in ((128,96,64,2,2),(256,256,64,4,4),(256,160,128,4,2),(192,192,128,2,4)):
    jit=TinyJit(lambda a,b:lds.gemm(a,b,tile).contiguous().realize())
    times=[]
    for iteration in range(7):
      start=time.perf_counter();output=jit(x,w);Device[Device.DEFAULT].synchronize()
      if iteration>=2:times.append((time.perf_counter()-start)*1000)
    actual=output.numpy()
    correct=bool(np.allclose(actual,expected,rtol=1e-5,atol=.0001))
    row={'shape':[m,n,k],'tile':tile,'correct':correct,'max_abs_error':float(np.max(np.abs(actual-expected))),'median_ms':float(np.median(times))}
    results.append(row);print(json.dumps(row),flush=True)
    Path('/tmp/opencode/alpamayo-lds-check.json').write_text(json.dumps(results,indent=2))
