"""Validate the fused LDS epilogue against the existing packed GEMM."""
import importlib.util
import json
from pathlib import Path
import time
import numpy as np
from tinygrad import Tensor,TinyJit,Device,dtypes
from amd_gemm import gemm,pack,unpack

def module(name):
  spec=importlib.util.spec_from_file_location(name,Path(__file__).with_name(name+'.py'))
  value=importlib.util.module_from_spec(spec);spec.loader.exec_module(value);return value

module('prefill-kernels').fix_dispatch_packet()
lds=module('lds-gemm')
Tensor.manual_seed(2026)
results=[]
for m,n,k in ((65,97,128),(784,4304,1152),(2027,9216,2048),(2027,2048,9216)):
  x=Tensor.randn(m,k).cast(dtypes.fp8e4m3).realize()
  w=Tensor.randn(n,k).cast(dtypes.fp8e4m3).realize()
  # The standard packed layout requires a multiple of 16 output channels.
  padded=w.pad(((0,(-n)%16),(0,0))).contiguous().realize()
  wp=unpack(pack(padded).realize())
  xs=Tensor([.125]).realize();ws=Tensor([.0625]).realize()
  bias=Tensor.randn(n).cast(dtypes.bfloat16).realize()
  bp=bias.pad(((0,(-n)%16),)).contiguous().realize()
  for tile in ((128,96,64,2,2),(256,256,64,4,4)):
    functions={
      'original':lambda a,b,s,t,c:gemm(a,b,(s,t),c)[:,:n].contiguous().realize(),
      'lds':lambda a,b,s,t,c:lds.scaled_gemm(a,b,(s,t),c,tile).realize(),
    }
    outputs={};timings={}
    for label,fn in functions.items():
      jit=TinyJit(fn);times=[]
      for i in range(8):
        start=time.perf_counter();out=jit(x,wp if label=='original' else w,xs,ws,bp if label=='original' else bias)
        Device[Device.DEFAULT].synchronize()
        if i>=2:times.append((time.perf_counter()-start)*1000)
      timings[label]=float(np.median(times));outputs[label]=out.float().numpy()
    expected=x.float().numpy()@w.float().numpy().T*(.125*.0625)+bias.float().numpy()
    errors={key:float(np.sqrt(np.mean((value-expected)**2))) for key,value in outputs.items()}
    assert errors['lds']<=errors['original']*1.05+1e-6,errors
    result={'shape':[m,n,k],'tile':tile,'timings_ms':timings,'rmse_fp32':errors,
      'max_delta':float(np.max(np.abs(outputs['original']-outputs['lds'])))}
    results.append(result);print(json.dumps(result),flush=True)
    Path('/tmp/opencode/alpamayo-lds-scaled.json').write_text(json.dumps(results,indent=2))
