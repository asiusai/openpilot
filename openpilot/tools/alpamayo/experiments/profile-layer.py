"""Profile one model layer at real six-camera dimensions; never a throughput test."""
import argparse
import importlib.util
import json
from pathlib import Path
import time

import numpy as np
from tinygrad import Device, Tensor, TinyJit, dtypes
from tinygrad.device import Compiled


def local(name):
  spec = importlib.util.spec_from_file_location(name.replace('-', '_'), Path(__file__).with_name(name+'.py'))
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--checkpoint', required=True)
  parser.add_argument('--output', type=Path, required=True)
  parser.add_argument('--quantize-reduction', action='store_true')
  args = parser.parse_args()
  Compiled.wait_timeout_ms = 10000
  import benchmark
  local('benchmark-fast')
  local('wgp_compiler').install(Device[Device.DEFAULT])
  model = benchmark.EdgeModel(args.checkpoint, precision='fp8')
  local('fast-quantize').install(model, materialize=True)
  local('prefill-kernels').fix_dispatch_packet()
  if args.quantize_reduction:
    local('quantize-reduction').install(model)
  local('online-attention').install(model)
  local('fused-norm').install(model)
  recorder = local('profile-kernels').Recorder(Device[Device.DEFAULT], benchmark)
  Tensor.manual_seed(81)
  results = []
  for name, rows, width in [('prefill', 3261, 2048), ('vision-front', 720, 1152), ('vision-side', 364, 1152)]:
    x = Tensor.randn(1, rows, width).cast(dtypes.bfloat16).realize()
    if name == 'prefill':
      cos = Tensor.ones(1, 1, rows, 128, dtype=dtypes.bfloat16).realize()
      sin = Tensor.zeros(1, 1, rows, 128, dtype=dtypes.bfloat16).realize()
      cache = Tensor.zeros(2, 1, 8, 3584, 128, dtype=dtypes.bfloat16).realize()
      fn = TinyJit(lambda value, c, s, kv: model.text_layer(value, c, s, 'vlm.model.language_model.layers.0', kv, cache_pos=0))
      inputs = (x, cos, sin, cache)
    else:
      fn = TinyJit(lambda value: model.vision_layer(value, 0))
      inputs = (x,)
    for iteration in range(5):
      model.synchronize()
      recorder.reset()
      recorder.stage = name
      started = time.perf_counter()
      output = fn(*inputs).realize()
      model.synchronize()
      elapsed = (time.perf_counter()-started)*1000
      profile = recorder.report()
      assert not profile['invalid_samples'], profile['invalid_samples']
      if iteration >= 2:
        results.append({'stage': name, 'iteration': iteration, 'wall_ms': elapsed, 'profile': profile})
        args.output.write_text(json.dumps({'profiling': True, 'results': results}, indent=2)+'\n')
        print(name, iteration, elapsed, flush=True)
    assert np.isfinite(output.float().numpy()).all()


if __name__ == '__main__':
  main()
