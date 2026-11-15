"""Compare six-camera attention schedules against the same independent FP32 result."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import time

import numpy as np
import torch
from tinygrad import Tensor, TinyJit, Device, dtypes
from tinygrad.device import Compiled
from amd_attention import attention as original


def local(name):
  spec = importlib.util.spec_from_file_location(name.replace('-', '_'), Path(__file__).with_name(name+'.py'))
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--output', type=Path, required=True)
  parser.add_argument('--check-waves', action='store_true', help='Require identical outputs between four and eight waves')
  parser.add_argument('--check-wave-sync', action='store_true', help='Require identical outputs with wave-local probability synchronization')
  parser.add_argument('--check-wgp', action='store_true', help='Check WGP compilation, including larger tiles without register spills')
  parser.add_argument('--check-native-exp', action='store_true', help='Measure native exp2 precision and timing against FP32 attention')
  parser.add_argument('--check-register-prob', action='store_true', help='Check exact probability transposition through wave registers')
  parser.add_argument('--large-groups', action='store_true', help='Compare eight and sixteen waves with sixteen or thirty-two keys')
  args = parser.parse_args()
  Compiled.wait_timeout_ms = 10000
  torch.set_num_threads(4)
  Tensor.manual_seed(2031)
  local('prefill-kernels').fix_dispatch_packet()
  online = local('online-attention').attention
  results = []
  for rows, width, kv_heads, causal in ((3261,128,8,True),(720,72,16,False),(364,72,16,False)):
    q = Tensor.randn(1,16,rows,width).cast(dtypes.bfloat16).realize()
    k, v = [Tensor.randn(1,kv_heads,rows,width).cast(dtypes.bfloat16).realize() for _ in range(2)]
    qt, kt, vt = [torch.from_numpy(t.float().numpy()) for t in (q,k,v)]
    expected = torch.nn.functional.scaled_dot_product_attention(qt, kt.repeat_interleave(16//kv_heads,1),
                 vt.repeat_interleave(16//kv_heads,1), is_causal=causal).numpy()
    baseline_error = None
    variants = [('original',None,None,None)] + [('online',keys,waves,stage)
      for keys,waves,stage in ((32,4,1),(32,2,1),(32,1,1),(32,8,1),(64,4,1),(64,2,1),(64,1,1),(64,8,1),(32,4,0),(64,4,0))]
    if args.check_waves or args.check_wave_sync:
      variants = [('original',None,None,None),('online',32,4,1),('online',32,8,1)]
    if args.check_wave_sync:
      variants += [('wave-sync',32,8,1),('wave-sync',32,4,1)]
    if args.check_wgp:
      variants = [('original',None,None,None),('online',32,4,1),('online',32,8,1),
                  ('wgp',32,4,1),('wgp',32,8,1),('online',64,8,1),('wgp',64,8,1)]
    if args.check_native_exp:
      variants = [('original',None,None,None),('online',32,8,1),('native-exp',32,8,1)]
    if args.check_register_prob:
      variants = [('original',None,None,None),('online',32,8,1),('register-prob',32,8,1),
                   ('register-prob',32,4,1),('register-prob',32,8,0)]
    if args.large_groups:
      variants = [('original',None,None,None),('online',32,8,1),('online',16,8,1),
                  ('large-group',32,16,1),('large-group',16,16,1)]
    four_wave_values = None
    key_reference = {}
    for name, keys, waves, stage in variants:
      os.environ['ALPAMAYO_ONLINE_WAVE_SYNC'] = str(int(name == 'wave-sync'))
      os.environ['ALPAMAYO_HIP_WGP'] = str(int(name == 'wgp'))
      os.environ['ALPAMAYO_ONLINE_NATIVE_EXP'] = str(int(name == 'native-exp'))
      os.environ['ALPAMAYO_ONLINE_REGISTER_PROB'] = str(int(name == 'register-prob'))
      if keys is not None:
        os.environ.update(ALPAMAYO_ONLINE_BLOCK_KEYS=str(keys),ALPAMAYO_ONLINE_WAVES=str(waves),ALPAMAYO_ONLINE_STAGE_KV=str(stage))
      fn = original if name == 'original' else online
      jit = TinyJit(lambda a,b,c: fn(a,b,c,causal).realize())
      timings = []
      for iteration in range(7):
        start = time.perf_counter()
        output = jit(q,k,v)
        Device[Device.DEFAULT].synchronize()
        if iteration >= 2:
          timings.append((time.perf_counter()-start)*1000)
      values = output.float().numpy()
      if (args.check_wgp or args.check_native_exp or args.check_register_prob or args.large_groups) and name == 'online':
        key_reference[keys] = values.copy()
      if name == 'large-group':
        np.testing.assert_array_equal(values, key_reference[keys])
      if args.check_wgp and name == 'wgp':
        np.testing.assert_array_equal(values, key_reference[keys])
      if name == 'register-prob':
        np.testing.assert_array_equal(values, key_reference[keys])
      if (args.check_waves or args.check_wave_sync) and name != 'original':
        if waves == 4 and name == 'online':
          four_wave_values = values.copy()
        else:
          np.testing.assert_array_equal(values, four_wave_values)
      assert np.isfinite(values).all()
      rmse = float(np.sqrt(np.mean((values-expected)**2)))
      if baseline_error is None:
        baseline_error = rmse
      assert rmse <= baseline_error*1.1, (rows,keys,waves,stage,rmse,baseline_error)
      result = dict(rows=rows,width=width,causal=causal,name=name,keys=keys,waves=waves,stage_kv=stage,
                     median_ms=float(np.median(timings)),rmse_fp32=rmse,max_error_fp32=float(np.max(np.abs(values-expected))))
      if (args.check_waves or args.check_wave_sync) and (waves == 8 or name == 'wave-sync'):
        result['exact_vs_four_waves'] = True
      if args.check_wgp and name == 'wgp':
        result['exact_vs_same_key_tile'] = True
      if name == 'native-exp':
        result['exact_vs_standard_exp'] = bool(np.array_equal(values,key_reference[keys]))
        result['max_delta_vs_standard_exp'] = float(np.max(np.abs(values-key_reference[keys])))
      if name == 'register-prob':
        result['exact_vs_shared_probability'] = True
      if name == 'large-group':
        result['exact_vs_eight_waves'] = True
      results.append(result)
      args.output.write_text(json.dumps(results,indent=2)+'\n')
      print(json.dumps(result),flush=True)


if __name__ == '__main__':
  main()
