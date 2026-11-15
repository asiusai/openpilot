"""Compare restored BF16 vision-layer layouts. Diagnostic timings only."""
import argparse
import importlib.util
import json
from pathlib import Path
import time
from types import MethodType

import numpy as np
from tinygrad import Device, Tensor, TinyJit, dtypes
from tinygrad.device import Compiled


def local(name):
  spec=importlib.util.spec_from_file_location(name.replace('-', '_'),Path(__file__).with_name(name+'.py'))
  module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
  return module


def main():
  parser=argparse.ArgumentParser()
  parser.add_argument('--checkpoint',required=True)
  parser.add_argument('--output',type=Path,required=True)
  args=parser.parse_args()
  Compiled.wait_timeout_ms=10000
  import benchmark
  local('benchmark-fast')
  local('wgp_compiler').install(Device[Device.DEFAULT])
  model=benchmark.EdgeModel(args.checkpoint,precision='fp8')
  restoration=local('restore-precision').restore(model,args.checkpoint,vision_layers=2)
  local('prefill-kernels').fix_dispatch_packet()
  local('online-attention').install(model)
  local('fused-norm').install(model)
  recorder=local('profile-kernels').Recorder(Device[Device.DEFAULT],benchmark)
  bf16=local('bf16-linear')
  prefix='vlm.model.visual.encoder.layers.0.'
  packed={name:bf16.pack_weight(value) for name,value in model.weights.items()
          if name.startswith(prefix) and value.ndim==2 and value.dtype==dtypes.bfloat16}
  original=model.linear
  Tensor.manual_seed(81)
  results=[]
  for images,rows in ((4,720),(8,364)):
    x=Tensor.randn(images,rows,1152).cast(dtypes.bfloat16).realize()
    expected=None
    for variant in ('original','materialized','packed'):
      def linear(self,value,name,quantized_input=None,heads=None):
        if not name.startswith(prefix) or f'{name}.weight' not in packed:
          return original(value,name,quantized_input,heads)
        if variant=='packed':
          shape=value.shape[:-1]
          weight=self.weights[f'{name}.weight']
          return bf16.linear(value.reshape(-1,value.shape[-1]),packed[f'{name}.weight'],weight.shape[0],
                             self.weights.get(f'{name}.bias')).reshape(*shape,weight.shape[0])
        value=original(value,name,quantized_input,heads)
        return value.contiguous().realize() if variant=='materialized' else value
      model.linear=MethodType(linear,model)
      run=TinyJit(lambda value:model.vision_layer(value,0))
      timings=[]
      profiles=[]
      for iteration in range(5):
        model.synchronize();recorder.reset();recorder.stage=variant
        start=time.perf_counter();output=run(x).realize();model.synchronize()
        elapsed=(time.perf_counter()-start)*1000
        profile=recorder.report()
        assert not profile['invalid_samples'],profile['invalid_samples']
        if iteration>=2:timings.append(elapsed);profiles.append(profile)
      actual=output.float().numpy()
      assert np.isfinite(actual).all()
      if expected is None:expected=actual.copy()
      record={'shape':list(x.shape),'variant':variant,'profiling':True,'median_wall_ms':float(np.median(timings)),
              'exact_original':bool(np.array_equal(actual,expected)),
              'differing_values':int(np.count_nonzero(actual!=expected)),
              'max_original_delta':float(np.max(np.abs(actual-expected))),'profiles':profiles}
      results.append(record)
      args.output.write_text(json.dumps({'restoration':restoration,'results':results},indent=2)+'\n')
      print(json.dumps({key:value for key,value in record.items() if key!='profiles'}),flush=True)


if __name__=='__main__':main()
