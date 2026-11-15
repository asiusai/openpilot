"""Check every packed byte, scale, and matrix output before selecting fusion."""
import argparse
from functools import partial
import importlib.util
import json
from pathlib import Path
import time

import numpy as np
from tinygrad import Tensor, TinyJit, Device, dtypes
from tinygrad.device import Compiled
from amd_gemm import pack, unpack, gemm, _gemm


def local(name):
  spec = importlib.util.spec_from_file_location(name.replace('-','_'), Path(__file__).with_name(name+'.py'))
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--output', type=Path, required=True)
  args = parser.parse_args()
  Compiled.wait_timeout_ms = 10000
  local('prefill-kernels').fix_dispatch_packet()
  import wgp_compiler
  wgp_compiler.install(Device[Device.DEFAULT])
  reference = local('quantize-reduction')
  candidate = local('quantize-packed')
  Tensor.manual_seed(6151)
  records = []
  cases = [(17,48,32,'zero'), (17,48,32,'tiny'), (17,48,32,'random'),
           (364,4304,1152,'random'), (720,1152,4304,'random'),
           (3261,9216,2048,'random'), (3261,2048,9216,'random')]
  for rows,columns,width,kind in cases:
    x = Tensor.randn(rows,width)
    if kind == 'zero': x = x*0
    if kind == 'tiny': x = x*1e-13
    x = x.cast(dtypes.bfloat16).contiguous().realize()
    weights = unpack(pack((Tensor.randn(columns,width)*8).cast(dtypes.fp8e4m3)).realize())
    weight_scale = Tensor([.125],dtype=dtypes.float32).contiguous().realize()
    expected = None
    timings = {}
    for name in ('separate','packed'):
      def operation(value):
        if name == 'separate':
          q,scale = reference.quantize(value)
          y = gemm(q,weights,(scale,weight_scale))
          packed = pack(q)
        else:
          packed,scale = candidate.quantize(value)
          output = Tensor.empty(rows,columns,device=value.device,dtype=dtypes.bfloat16)
          y = Tensor.custom_kernel(output,packed,pack(weights),scale,weight_scale,fxn=partial(_gemm,packed_input=True))[0]
        return y.realize(), packed.realize(), scale.realize()
      jit = TinyJit(operation)
      elapsed = []
      for iteration in range(12):
        started = time.perf_counter()
        y,packed,scale = jit(x)
        Device[Device.DEFAULT].synchronize()
        if iteration >= 2: elapsed.append((time.perf_counter()-started)*1000)
      actual = (y.float().numpy(), packed.bitcast(dtypes.uint8).numpy(), scale.numpy())
      assert np.isfinite(actual[0]).all() and np.isfinite(actual[2]).all()
      if expected is None: expected = actual
      else:
        for result, original in zip(actual,expected,strict=True): np.testing.assert_array_equal(result,original)
      timings[name] = float(np.median(elapsed))
    record = dict(shape=[rows,columns,width],kind=kind,median_ms=timings,
                  exact_output=True,exact_packed_bytes=True,exact_scale=True)
    records.append(record)
    args.output.write_text(json.dumps(records,indent=2)+'\n')
    print(json.dumps(record),flush=True)


if __name__ == '__main__':
  main()
