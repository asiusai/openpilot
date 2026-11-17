"""Compare split packed-history attention against the selected expert kernels."""
import argparse
import importlib.util
import json
from pathlib import Path
import time
import numpy as np
from tinygrad import Device,Tensor,TinyJit,dtypes
from amd_attention import _pack


def module(name):
  spec=importlib.util.spec_from_file_location(name,Path(__file__).with_name(name+'.py'))
  value=importlib.util.module_from_spec(spec);spec.loader.exec_module(value);return value


def main():
  parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
  baseline,split=module('expert-kernels'),module('expert-split-cache')
  rng=np.random.default_rng(9060)
  def tensor(shape):return Tensor(rng.standard_normal(shape).astype(np.float32)).cast(dtypes.bfloat16).realize()
  report=[]
  for capacity in (256,2304):
    oldk,oldv=tensor((1,8,capacity,128)),tensor((1,8,capacity,128))
    q,k,v=tensor((3,1,16,64,128)),tensor((3,1,8,64,128)),tensor((3,1,8,64,128))
    mask=Tensor(np.where((np.arange(capacity+64)<capacity-31)|(np.arange(capacity+64)>=capacity),0.,-np.inf).astype(np.float32)).realize()
    def original(q,k,v,oldk,oldv):
      return Tensor.stack(*[baseline.attention(q[i],oldk.cat(k[i],dim=2),oldv.cat(v[i],dim=2),mask).realize() for i in range(3)]).realize()
    def candidate(q,k,v,oldk,oldv):
      history=(_pack(oldk).realize(),_pack(oldv.transpose(-2,-1)).realize())
      return Tensor.stack(*[split.attention(q[i],k[i],v[i],history,mask,capacity).realize() for i in range(3)]).realize()
    results={};outputs=[]
    for name,fn in (('baseline',original),('split',candidate)):
      jit=TinyJit(fn)
      for _ in range(3):output=jit(q,k,v,oldk,oldv)
      Device[Device.DEFAULT].synchronize();start=time.perf_counter()
      for _ in range(40):output=jit(q,k,v,oldk,oldv)
      Device[Device.DEFAULT].synchronize();results[name]=(time.perf_counter()-start)*1000/40
      outputs.append(output.bitcast(dtypes.uint16).numpy())
    np.testing.assert_array_equal(*outputs)
    report.append({'capacity':capacity,'exact':True,'three_attention_calls_ms':results})
  args.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))


if __name__=='__main__':main()
