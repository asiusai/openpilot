"""Separate numerical/performance study; does not select FP8 model attention."""
import argparse
import importlib.util
import json
from pathlib import Path
import time

import numpy as np
import torch
from tinygrad import Tensor, Device, TinyJit, dtypes
from tinygrad.device import Compiled


def module(name):
  spec=importlib.util.spec_from_file_location(name.replace('-','_'),Path(__file__).with_name(name+'.py'))
  result=importlib.util.module_from_spec(spec);spec.loader.exec_module(result)
  return result


def small_oracle(q,k,v,causal,quantized=None):
  """Independent CPU implementation of blockwise FP8 probabilities and online merging."""
  def quantize(value,axes):
    scale=(value.abs().amax(axes,keepdim=True)/448).clamp(min=1e-12)
    return (value/scale).clamp(-448,448).to(torch.float8_e4m3fn).float()*scale
  if quantized is None:
    q=quantize(q,(-1,));k=quantize(k,(-1,));v=quantize(v,(-2,-1))
  else:
    q,k,v=(quantized[i]*quantized[i+3] for i in range(3))
  repeat=q.shape[1]//k.shape[1]
  k=k.repeat_interleave(repeat,1);v=v.repeat_interleave(repeat,1)
  rows,keys=q.shape[2],k.shape[2]
  maximum=torch.full(q.shape[:-1]+(1,),float('-inf'))
  denominator=torch.zeros_like(maximum);acc=torch.zeros_like(q)
  for start in range(0,keys,32):
    scores=q@k[:,:,start:start+32].transpose(-1,-2)*q.shape[-1]**-.5
    if causal:
      scores=scores.masked_fill(torch.arange(start,min(start+32,keys))[None,:]>torch.arange(rows)[:,None],float('-inf'))
    next_max=torch.maximum(maximum,scores.amax(-1,keepdim=True))
    probability=(scores-next_max).exp()
    scale=(maximum-next_max).exp()
    denominator=denominator*scale+probability.sum(-1,keepdim=True)
    rounded=(probability*448).to(torch.float8_e4m3fn).float()/448
    acc=acc*scale+rounded@v[:,:,start:start+32]
    maximum=next_max
  return (acc/denominator).to(torch.bfloat16).float().numpy()


def main():
  parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
  args=parser.parse_args();torch.set_num_threads(4);Compiled.wait_timeout_ms=10000;Tensor.manual_seed(2031)
  module('prefill-kernels').fix_dispatch_packet()
  online=module('online-attention');candidate=module('fp8-attention')
  results=[]
  shapes=[(2,2,1,17,33,72,True),(2,2,1,17,33,72,False),
          (1,16,8,3261,3261,128,True),(4,16,16,720,720,72,False),(8,16,16,364,364,72,False)]
  for batch,heads,kv_heads,rows,keys,width,causal in shapes:
    q=Tensor.randn(batch,heads,rows,width,dtype=dtypes.bfloat16).realize()
    k=Tensor.randn(batch,kv_heads,keys,width,dtype=dtypes.bfloat16).realize()
    v=Tensor.randn(batch,kv_heads,keys,width,dtype=dtypes.bfloat16).realize()
    cpu=[torch.from_numpy(x.float().numpy()) for x in (q,k,v)]
    expected=torch.nn.functional.scaled_dot_product_attention(*cpu,is_causal=causal,enable_gqa=True).numpy()
    oracle=small_oracle(*cpu,causal) if rows==17 else None
    same_input_oracle=None
    if oracle is not None:
      quantized=[torch.from_numpy(x.float().numpy()) for x in candidate.quantized_inputs(q,k,v)]
      same_input_oracle=small_oracle(*cpu,causal,quantized=quantized)
    for name,waves in [('bf16',8),('fp8',8),('fp8',4)]:
      jit=TinyJit(lambda a,b,c: (online.attention(a,b,c,causal) if name=='bf16' else
                               candidate.attention(a,b,c,causal,waves)).realize())
      durations=[]
      for iteration in range(7):
        started=time.perf_counter();output=jit(q,k,v);Device[Device.DEFAULT].synchronize()
        if iteration>=2:durations.append((time.perf_counter()-started)*1000)
      actual=output.float().numpy();assert np.isfinite(actual).all()
      error=actual.astype(np.float64)-expected
      rmse=float(np.sqrt(np.mean(error**2)))
      relative=rmse/float(np.sqrt(np.mean(expected.astype(np.float64)**2)))
      record=dict(shape=[batch,heads,kv_heads,rows,keys,width,causal],name=name,waves=waves,
                  median_ms=float(np.median(durations)),fp32_rmse=rmse,relative_rmse=relative,max_error=float(np.abs(error).max()))
      if name=='fp8' and oracle is not None:
        record['oracle_max_delta']=float(np.abs(actual-oracle).max())
        record['same_quantized_input_oracle_max_delta']=float(np.abs(actual-same_input_oracle).max())
        np.testing.assert_allclose(actual,same_input_oracle,atol=.008,rtol=.02)
      # Numerical smoke bound only. Full-model quality still requires official replay comparisons.
      assert relative<.1,record
      results.append(record);args.output.write_text(json.dumps(results,indent=2));print(json.dumps(record),flush=True)


if __name__=='__main__':main()
