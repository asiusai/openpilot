"""Check one-context queue separation, dependencies, changing inputs and timing."""
import argparse
import importlib.util
import json
from pathlib import Path
import time

import numpy as np
from tinygrad import Device, Tensor, TinyJit, dtypes
from tinygrad.device import Compiled
from amd_gemm import pack,unpack


def local(name):
  spec=importlib.util.spec_from_file_location(name.replace('-','_'),Path(__file__).with_name(name+'.py'))
  module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
  return module


def main():
  parser=argparse.ArgumentParser()
  parser.add_argument('--output',type=Path,required=True)
  parser.add_argument('--large-only',action='store_true',help='Create queue 1 once for the full-size case')
  parser.add_argument('--usb-serial-file',type=Path,required=True)
  parser.add_argument('--usb-serial',required=True)
  args=parser.parse_args()
  assert args.usb_serial_file.read_text().strip()==args.usb_serial
  Compiled.wait_timeout_ms=10000
  local('prefill-kernels').fix_dispatch_packet()
  device=Device[Device.DEFAULT]
  local('wgp_compiler').install(device)
  attention,gemm=local('online-attention').attention,local('tiled-gemm').gemm
  Tensor.manual_seed(9182)
  records=[]
  shapes=[(33,72,2,1,37,48,32),(3261,128,16,8,3261,9216,2048)]
  if args.large_only:shapes=shapes[-1:]
  for rows,width,heads,kv_heads,gm,gn,gk in shapes:
    q=Tensor.randn(1,heads,rows,width).cast(dtypes.bfloat16).realize()
    k,v=[Tensor.randn(1,kv_heads,rows,width).cast(dtypes.bfloat16).realize() for _ in range(2)]
    x=unpack(pack((Tensor.randn(gm,gk)*8).cast(dtypes.fp8e4m3)).realize())
    w=unpack(pack((Tensor.randn(gn,gk)*8).cast(dtypes.fp8e4m3)).realize())
    sx,sw=Tensor([.125]).realize(),Tensor([.125]).realize()
    def joined(a,b,c,d,e,s,t):
      # Queue 0 produces the attention input. Queue 0 also consumes its output.
      att=attention((a*.75).cast(dtypes.bfloat16),b,c,causal=True)
      mat=gemm(d,e,(s,t))
      checksum=att.float().sum()+mat.float().sum()
      Tensor.realize(att,mat,checksum)
      return att,mat,checksum
    arguments=(q,k,v,x,w,sx,sw)
    serial=TinyJit(joined)
    for _ in range(3):serial(*arguments);device.synchronize()
    print(json.dumps({'stage':'serial_ready','rows':rows}),flush=True)
    scope=local('parallel-queues').SecondComputeQueue(device)
    with scope:
      parallel=TinyJit(joined)
      for iteration in range(3):
        print(json.dumps({'stage':'parallel_warmup_start','rows':rows,'iteration':iteration}),flush=True)
        parallel(*arguments);device.synchronize()
        print(json.dumps({'stage':'parallel_warmup_done','rows':rows,'iteration':iteration}),flush=True)
      assert scope.selected>0 and scope.scratch_checks>0
      for change in range(3):
        q.assign((q+.03125).cast(dtypes.bfloat16)).realize()
        expected=[value.float().numpy() for value in serial(*arguments)]
        actual=[value.float().numpy() for value in parallel(*arguments)]
        for a,b in zip(actual,expected,strict=True):np.testing.assert_array_equal(a,b)
        assert all(np.isfinite(value).all() for value in actual)
      for name,run in (('single',serial),('dual',parallel),('single-repeat',serial),('dual-repeat',parallel)):
        times=[]
        for iteration in range(33):
          start=time.perf_counter();run(*arguments);device.synchronize()
          if iteration>=3:times.append((time.perf_counter()-start)*1000)
        record=dict(rows=rows,width=width,variant=name,median_ms=float(np.median(times)),exact=True,
                    routed_kernels=scope.selected,scratch_checks=scope.scratch_checks)
        records.append(record);args.output.write_text(json.dumps(records,indent=2)+'\n');print(json.dumps(record),flush=True)
    assert scope.drained
    # The original queue and runtime remain usable after removing queue 1.
    probe=Tensor.arange(257,dtype=dtypes.int32)
    np.testing.assert_array_equal((probe*3+7).numpy(),np.arange(257,dtype=np.int32)*3+7)
    print(json.dumps({'stage':'secondary_drained_original_queue_passed','rows':rows}),flush=True)


if __name__=='__main__':main()
