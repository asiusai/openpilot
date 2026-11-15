"""Masked online attention against independent FP32, including an empty prefix."""
import json
from pathlib import Path
import time

import numpy as np
import torch
from tinygrad import Tensor, Device, TinyJit, dtypes
from importlib.util import spec_from_file_location, module_from_spec


def load(name):
  spec=spec_from_file_location(name,Path(__file__).with_name(name+'.py'))
  module=module_from_spec(spec);spec.loader.exec_module(module)
  return module


load('prefill-kernels').fix_dispatch_packet()
online=load('online-attention')
original=load('expert-kernels')
Tensor.manual_seed(2026)
torch.set_num_threads(4)
reports=[]
for keys,length in [(320,237),(2368,2027),(2368,0)]:
  q=Tensor.randn(1,16,64,128).cast(dtypes.bfloat16).realize()
  k,v=[Tensor.randn(1,8,keys,128).cast(dtypes.bfloat16).realize() for _ in range(2)]
  mask=((Tensor.arange(keys)<length)|(Tensor.arange(keys)>=keys-64)).where(0,-float('inf')).clone().realize()
  tq,tk,tv,tm=[torch.from_numpy(x.float().numpy()) for x in (q,k,v,mask)]
  truth=((tq @ tk.repeat_interleave(2,1).transpose(-2,-1)*128**-.5+tm).softmax(-1) @ tv.repeat_interleave(2,1)).numpy()
  case={'keys':keys,'prefix_length':length}
  for name,fn in [('original',lambda a,b,c,m:original.attention(a,b,c,m)),
                  ('online',lambda a,b,c,m:online.attention(a,b,c,False,m))]:
    jit=TinyJit(fn);times=[]
    for i in range(9):
      start=time.perf_counter();out=jit(q,k,v,mask).realize();Device[Device.DEFAULT].synchronize()
      if i>=2:times.append((time.perf_counter()-start)*1000)
    actual=out.float().numpy()
    assert np.isfinite(actual).all()
    case[name]={'median_ms':float(np.median(times)), 'rmse':float(np.sqrt(np.mean((actual-truth)**2))),
                'max_abs_error':float(np.max(np.abs(actual-truth)))}
  assert case['online']['rmse'] <= case['original']['rmse']*1.1
  assert case['online']['max_abs_error'] <= case['original']['max_abs_error']*1.1
  reports.append(case);print(json.dumps(case),flush=True)
Path('/tmp/opencode/alpamayo-online-expert-check.json').write_text(json.dumps(reports,indent=2))
