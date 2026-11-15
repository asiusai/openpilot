"""Compare fused projections with three independent production projections."""
import importlib.util,json,time
from pathlib import Path
import numpy as np
from tinygrad import Tensor,TinyJit,Device,dtypes
from model import EdgeModel
from amd_gemm import pack,unpack

spec=importlib.util.spec_from_file_location('fused',Path(__file__).with_name('fused-qkv.py'))
fused=importlib.util.module_from_spec(spec);spec.loader.exec_module(fused)
Tensor.manual_seed(2026)
results=[]
for rows,width,sizes in ((784,1152,[1152]*3),(2027,2048,[2048,1024,1024]),(64,512,[2048,1024,1024])):
  model=EdgeModel.__new__(EdgeModel);model.fp8_wmma=True;model.cuda_kernels=False
  model.weights={};model.weight_scales={}
  names=['vlm.model.test.self_attn.'+kind+'_proj' for kind in 'qkv']
  for i,(name,size) in enumerate(zip(names,sizes)):
    w=Tensor.randn(size,width).cast(dtypes.fp8e4m3).realize()
    model.weights[name+'.weight']=unpack(pack(w).realize())
    model.weights[name+'.bias']=Tensor.randn(size).cast(dtypes.bfloat16).realize()
    model.weight_scales[name+'.weight']=Tensor([.015625*(i+1)]).realize()
  x=Tensor.randn(1,rows,width).cast(dtypes.bfloat16).realize()
  def call(a):
    quantized=model.quantize(a)
    output=tuple(model.linear(a,name,quantized) for name in names)
    Tensor.realize(*output)
    return output
  outputs={};timings={}
  for label in ('original','fused'):
    if label=='fused':fused.install(model)
    jit=TinyJit(call);times=[]
    for iteration in range(7):
      begin=time.perf_counter();out=jit(x);Device[Device.DEFAULT].synchronize()
      if iteration>=2:times.append((time.perf_counter()-begin)*1000)
    outputs[label]=[o.float().numpy() for o in out];timings[label]=float(np.median(times))
  for a,b in zip(outputs['original'],outputs['fused']):np.testing.assert_array_equal(a,b)
  result={'rows':rows,'width':width,'sizes':sizes,'timings_ms':timings,'exact':True}
  results.append(result);print(json.dumps(result),flush=True)
  Path('/tmp/opencode/alpamayo-qkv-check.json').write_text(json.dumps(results,indent=2))
