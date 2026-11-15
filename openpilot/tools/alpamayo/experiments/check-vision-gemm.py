"""Measure independent and batched partial vision tiles against FP32 arithmetic."""
import argparse
import importlib.util
import json
from pathlib import Path
import time

import numpy as np
import torch
from tinygrad import Tensor, TinyJit, Device, dtypes
from tinygrad.device import Compiled
from model import EdgeModel
from amd_gemm import gemm
import wgp_compiler


def local(name):
  spec = importlib.util.spec_from_file_location(name.replace('-','_'),Path(__file__).with_name(name+'.py'))
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--output',type=Path,required=True)
  args = parser.parse_args()
  Compiled.wait_timeout_ms = 10000
  torch.set_num_threads(4)
  wgp_compiler.install(Device[Device.DEFAULT])
  local('prefill-kernels').fix_dispatch_packet()
  batched = local('tiled-gemm').gemm
  quantizer = EdgeModel.__new__(EdgeModel)
  quantizer.cuda_kernels = False
  Tensor.manual_seed(2718)
  results = []
  for rows in (364,720):
    width, columns, count = 1152,4304,4
    source = (Tensor.randn(count,rows,width)*Tensor.arange(1,count+1).reshape(count,1,1)).cast(dtypes.bfloat16).realize()
    parts = [quantizer.quantize(source[i]) for i in range(count)]
    parts = [(q.realize(),s.realize()) for q,s in parts]
    weight,weight_scale = quantizer.quantize(Tensor.randn(columns,width).cast(dtypes.bfloat16))
    weight,weight_scale = weight.realize(),weight_scale.realize()
    bias = Tensor.randn(columns).cast(dtypes.bfloat16).realize()
    q = Tensor.cat(*[p[0] for p in parts]).realize()
    scales = Tensor.cat(*[p[1].reshape(1).expand(rows) for p in parts]).contiguous().realize()
    padded_rows=(rows+63)//64*64
    padded_q=q.reshape(count,rows,width).pad(((0,0),(0,padded_rows-rows),(0,0))).reshape(-1,width).contiguous().realize()
    padded_scales=Tensor.cat(*[p[1].reshape(1).expand(padded_rows) for p in parts]).contiguous().realize()
    qt,wt = [torch.from_numpy(v.float().numpy()) for v in (q,weight)]
    scale_array = scales.numpy()*weight_scale.numpy().item()
    base_expected = (qt@wt.T).numpy()*scale_array[:,None]
    for with_bias in (False,True):
      b = bias if with_bias else None
      expected = base_expected + (bias.float().numpy() if with_bias else 0)
      reference = None
      for name in ('individual','batched','padded'):
        def operation():
          if name == 'individual':
            return Tensor.cat(*[gemm(p,weight,(s,weight_scale),bias=b) for p,s in parts]).realize()
          if name == 'padded':
            return batched(padded_q,weight,(padded_scales,weight_scale),row_scale=True,bias=b).reshape(count,padded_rows,columns)[:,:rows].reshape(-1,columns).contiguous().realize()
          return batched(q,weight,(scales,weight_scale),row_scale=True,bias=b).realize()
        jit = TinyJit(operation)
        times = []
        for iteration in range(12):
          start = time.perf_counter()
          output = jit()
          Device[Device.DEFAULT].synchronize()
          if iteration >= 2:
            times.append((time.perf_counter()-start)*1000)
        actual = output.float().numpy()
        assert np.isfinite(actual).all()
        if reference is None:
          reference = actual.copy()
        result = dict(rows=rows,images=count,bias=with_bias,name=name,median_ms=float(np.median(times)),
                      exact_vs_individual=bool(np.array_equal(actual,reference)),
                      different_values=int(np.count_nonzero(actual-reference)),
                      max_delta=float(np.max(np.abs(actual-reference))),
                      rmse_fp32=float(np.sqrt(np.mean((actual-expected)**2))))
        results.append(result)
        args.output.write_text(json.dumps(results,indent=2)+'\n')
        print(json.dumps(result),flush=True)
  metrics = local('gpu-metrics').read(Device[Device.DEFAULT])
  args.output.with_suffix('.metrics.json').write_text(json.dumps(metrics,indent=2)+'\n')


if __name__ == '__main__':
  main()
