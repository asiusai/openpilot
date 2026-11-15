"""Compare complete image projectors without changing image detail or model weights."""
import argparse
import importlib.util
import json
from pathlib import Path
import time

import numpy as np
from tinygrad import Device, Tensor, TinyJit, dtypes
from tinygrad.device import Compiled
from amd_gemm import pack, unpack


def local(name):
  spec = importlib.util.spec_from_file_location(name.replace('-', '_'), Path(__file__).with_name(name+'.py'))
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--checkpoint', required=True)
  parser.add_argument('--output', type=Path, required=True)
  args = parser.parse_args()
  Compiled.wait_timeout_ms = 10000
  import benchmark
  local('benchmark-fast')
  local('wgp_compiler').install(Device[Device.DEFAULT])
  local('prefill-kernels').fix_dispatch_packet()
  model = benchmark.EdgeModel(args.checkpoint, precision='fp8')
  local('fast-quantize').install(model, materialize=True)
  quant = local('quantize-reduction')
  quant.install(model, grouped=True)
  local('fused-norm').install(model)
  gemm = local('tiled-gemm').gemm
  names = ['vlm.model.projector.linear_fc1', 'vlm.model.projector.linear_fc2']
  weights = {name: unpack(pack(model.weights[name+'.weight']).realize()) for name in names}

  def normalized(value):
    return model.norm(model.norm(value, 'vlm.model.visual.post_layernorm'), 'vlm.model.projector.norm').reshape(-1, 4608)

  def original(value):
    parts = []
    for i in range(value.shape[0]):
      x = normalized(value[i:i+1])
      x = model.linear(x, names[0]).float().gelu(approximate='none').cast(model.dtype)
      parts.append(model.linear(x, names[1]).realize())
    return Tensor.cat(*parts, dim=0).realize()

  def packed(value, batched):
    images = value.shape[0]
    parts = [normalized(value[i:i+1]).realize() for i in range(images)]
    def project(x, count):
      for index, name in enumerate(names):
        q, scale = (model.grouped_quantize(x, count) if count > 1 else model.quantize(x))
        x = gemm(q, weights[name], (scale, model.weight_scales[name+'.weight']),
                 row_scale=count > 1, bias=model.weights.get(name+'.bias'))
        if index == 0:
          x = x.float().gelu(approximate='none').cast(model.dtype)
      return x.realize()
    if batched:
      return project(Tensor.cat(*parts, dim=0), images)
    return Tensor.cat(*(project(x, 1) for x in parts), dim=0).realize()

  Tensor.manual_seed(1103)
  results = []
  for images, rows in [(4, 720), (8, 364)]:
    value = Tensor.randn(images, rows, 1152).cast(dtypes.bfloat16).realize()
    reference = None
    for name, function in [('original', original), ('packed-independent', lambda x: packed(x, False)),
                           ('packed-batched', lambda x: packed(x, True))]:
      fn = TinyJit(function)
      timings = []
      for iteration in range(12):
        model.synchronize()
        started = time.perf_counter()
        output = fn(value)
        model.synchronize()
        if iteration >= 2:
          timings.append((time.perf_counter()-started)*1000)
      actual = output.float().numpy()
      assert np.isfinite(actual).all()
      if reference is None:
        reference = actual
      delta = actual.astype(np.float64)-reference.astype(np.float64)
      record = dict(images=images, rows=rows, variant=name, median_ms=float(np.median(timings)),
                    differing_values=int(np.count_nonzero(delta)), values=actual.size,
                    max_delta=float(np.max(np.abs(delta))), rmse=float(np.sqrt(np.mean(delta**2))))
      results.append(record)
      args.output.write_text(json.dumps(results, indent=2)+'\n')
      print(json.dumps(record), flush=True)


if __name__ == '__main__':
  main()
