"""Compare online attention with FP32 and the existing BF16 implementation."""
import importlib.util,json,time,os
from pathlib import Path
import numpy as np
import torch
from tinygrad import Tensor,TinyJit,Device,dtypes
from amd_attention import attention as original

def module(name):
  spec=importlib.util.spec_from_file_location(name,Path(__file__).with_name(name+'.py'))
  value=importlib.util.module_from_spec(spec);spec.loader.exec_module(value);return value

module('prefill-kernels').fix_dispatch_packet()
online=module('online-attention').attention
torch.set_num_threads(4)
Tensor.manual_seed(2026)
results=[]
for rows,width,kh,causal in ((17,128,8,True),(128,128,8,True),(2027,128,8,True),(196,72,16,False),(784,72,16,False)):
  q=Tensor.randn(1,16,rows,width).cast(dtypes.bfloat16).realize()
  k=Tensor.randn(1,kh,rows,width).cast(dtypes.bfloat16).realize()
  v=Tensor.randn(1,kh,rows,width).cast(dtypes.bfloat16).realize()
  qt,kt,vt=[torch.from_numpy(t.float().numpy()) for t in (q,k,v)]
  expected=torch.nn.functional.scaled_dot_product_attention(qt,kt.repeat_interleave(16//kh,1),vt.repeat_interleave(16//kh,1),is_causal=causal).numpy()
  errors={};times={}
  def reference(a,b,c,causal):
    if rows>=128:return original(a,b,c,causal)
    k2,v2=(t.repeat_interleave(2,dim=1) for t in (b,c))
    scores=(a.matmul(k2.transpose(-2,-1))*(128**-.5)).float()
    mask=(Tensor.arange(rows).reshape(1,-1)>Tensor.arange(rows).reshape(-1,1)).where(-float('inf'),0)
    return (scores+mask).softmax(-1).cast(a.dtype).matmul(v2)
  for label,fn in (('original',reference),('online',online)):
    jit=TinyJit(lambda a,b,c:fn(a,b,c,causal).realize());durations=[]
    for iteration in range(7):
      begin=time.perf_counter();out=jit(q,k,v);Device[Device.DEFAULT].synchronize()
      if iteration>=2:durations.append((time.perf_counter()-begin)*1000)
    value=out.float().numpy();assert np.isfinite(value).all()
    errors[label]={'rmse':float(np.sqrt(np.mean((value-expected)**2))),'max':float(np.max(np.abs(value-expected)))}
    times[label]=float(np.median(durations))
  assert errors['online']['rmse']<=errors['original']['rmse']*1.1,errors
  result={'rows':rows,'width':width,'causal':causal,'errors_fp32':errors,'timings_ms':times}
  results.append(result);print(json.dumps(result),flush=True)
  Path(os.environ.get('ALPAMAYO_ATTENTION_CHECK_OUTPUT','/tmp/opencode/alpamayo-online-attention-check.json')).write_text(json.dumps(results,indent=2))
