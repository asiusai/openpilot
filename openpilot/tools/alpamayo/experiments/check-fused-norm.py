"""Compare fused normalizations with independent FP32 and production operations."""
import importlib.util
import json
from pathlib import Path
import time
import numpy as np
import torch
from tinygrad import Tensor, Device, TinyJit, dtypes

def load(name):
  spec=importlib.util.spec_from_file_location(name,Path(__file__).with_name(name+'.py'))
  module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module

load('prefill-kernels').fix_dispatch_packet()
module=load('fused-norm')
torch.set_num_threads(4)
Tensor.manual_seed(20)
results=[]
for rows,width,rms in ((720,1152,False),(2880,1152,False),(2027,2048,True),(64,512,True)):
  x=(Tensor.randn(rows,width)*3).cast(dtypes.bfloat16).realize()
  w=Tensor.randn(width).cast(dtypes.bfloat16).realize()
  bias=Tensor.randn(width).cast(dtypes.bfloat16).realize()
  eps=1e-5 if rms else 1e-6
  def original(a,b,c):
    normalized=a.float().rmsnorm(eps=eps) if rms else a.float().layernorm(eps=eps)
    return (normalized.cast(a.dtype)*b if rms else (normalized*b.float()+c.float()).cast(a.dtype)).realize()
  def fast(a,b,c): return module.norm(a,b,c,rms,eps).realize()
  a,b,c=(torch.from_numpy(v.float().numpy()) for v in (x,w,bias))
  if rms:
    ref=((a/torch.sqrt((a*a).mean(-1,keepdim=True)+eps)).bfloat16().float()*b).bfloat16().float().numpy()
  else:
    ref=torch.nn.functional.layer_norm(a,(width,),b,c,eps).bfloat16().float().numpy()
  row={'shape':[rows,width],'rms':rms}
  for name,fn in (('original',original),('fused',fast)):
    jit=TinyJit(fn);times=[]
    for i in range(7):
      begin=time.perf_counter();out=jit(x,w,bias);Device[Device.DEFAULT].synchronize();elapsed=(time.perf_counter()-begin)*1000
      if i>=2: times.append(elapsed)
    delta=out.float().numpy()-ref
    row[name]={'median_ms':float(np.median(times)),'rmse':float(np.sqrt(np.mean(delta**2))),'max_abs':float(np.max(np.abs(delta)))}
  assert row['fused']['rmse'] <= max(row['original']['rmse']*2,0.0002),row
  results.append(row);print(json.dumps(row),flush=True)
Path('/tmp/opencode/alpamayo-fused-norm-check.json').write_text(json.dumps(results,indent=2))
