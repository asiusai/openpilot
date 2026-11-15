"""Time real matrix shapes; report all rounding differences before rejecting parity."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
from tinygrad import Tensor, Device, TinyJit, dtypes
from tinygrad.device import Compiled
from amd_gemm import pack, unpack
from importlib.util import spec_from_file_location, module_from_spec


def module(name):
  spec = spec_from_file_location(name,Path(__file__).with_name(name+'.py'))
  value = module_from_spec(spec)
  spec.loader.exec_module(value)
  return value


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--output',type=Path,required=True)
  parser.add_argument('--solutions',type=int,nargs='+',default=[27955,27854])
  parser.add_argument('--prefill-only',action='store_true')
  parser.add_argument('--usb-serial-file',type=Path,required=True)
  parser.add_argument('--usb-serial',required=True)
  args = parser.parse_args()
  assert args.usb_serial_file.read_text().strip() == args.usb_serial
  Compiled.wait_timeout_ms = 10000
  module('prefill-kernels').fix_dispatch_packet()
  device = Device[Device.DEFAULT]
  module('wgp_compiler').install(device)
  original = module('tiled-gemm').gemm
  vendor = module('vendor-gemm')
  Tensor.manual_seed(42)
  results = []
  with vendor.argument_adapter(device):
    shapes = [(3261,9216,2048),(3261,2048,9216)]
    if not args.prefill_only: shapes += [(2880,4304,1152),(2912,1152,4304),(64,512,512)]
    for rows,columns,width in shapes:
      x = (Tensor.randn(rows,width)*8).cast(dtypes.fp8e4m3).contiguous().realize()
      weight = (Tensor.randn(columns,width)*8).cast(dtypes.fp8e4m3).contiguous().realize()
      packed_weight = unpack(pack(weight).realize())
      scales = tuple(Tensor([.125],dtype=dtypes.float32).contiguous().realize() for _ in range(2))
      ones = Tensor.ones(columns,dtype=dtypes.float32).contiguous().realize()
      bias = Tensor.zeros(columns,dtype=dtypes.float32).contiguous().realize()
      zero = Tensor.zeros(rows,columns,dtype=dtypes.bfloat16).contiguous().realize()
      expected = host_x = host_w = None
      for index in (None,*args.solutions):
        fn = (lambda a,b,s,t:original(a,b,(s,t))) if index is None else (
          lambda a,b,s,t:vendor.gemm(a,b,(s,t),ones,bias,zero,index=index))
        run = TinyJit(lambda a,b,s,t:fn(a,b,s,t).realize())
        timings = []
        print('shape',rows,columns,width,'solution',index,flush=True)
        for iteration in range(12):
          begin = time.perf_counter()
          output = run(x,packed_weight if index is None else weight,*scales)
          device.synchronize()
          if iteration >= 2: timings.append((time.perf_counter()-begin)*1000)
        values = output.float().numpy()
        assert np.isfinite(values).all()
        if expected is None: expected = values.copy()
        changed = np.flatnonzero(values.ravel()!=expected.ravel())
        record = dict(shape=[rows,columns,width],solution=index,median_ms=float(np.median(timings)),
          exact=not bool(changed.size),differing_values=int(changed.size),max_delta=float(np.max(np.abs(values-expected))),
          timing_boundary='Row-major FP8 activations to BF16 matrix output; weights prepared before timing')
        if changed.size:
          if host_x is None: host_x,host_w=x.float().numpy(),weight.float().numpy()
          selected = changed[np.linspace(0,len(changed)-1,min(1024,len(changed)),dtype=np.int64)]
          rr,cc = np.divmod(selected,columns)
          precise = np.sum(host_x[rr].astype(np.float64)*host_w[cc].astype(np.float64),axis=1)*.015625
          before,after = np.abs(expected[rr,cc].astype(np.float64)-precise),np.abs(values[rr,cc].astype(np.float64)-precise)
          record.update(fp64_checked_changed_values=len(selected),fp64_original_mean_error=float(before.mean()),
            fp64_candidate_mean_error=float(after.mean()),fp64_candidate_better_count=int(np.count_nonzero(after<before)),
            fp64_candidate_worse_count=int(np.count_nonzero(after>before)))
        results.append(record)
        args.output.write_text(json.dumps(results,indent=2)+'\n')
        print(json.dumps(record),flush=True)
  assert all(r['exact'] for r in results), 'Rounding differs; diagnostic records retained, no model implementation selected'


if __name__ == '__main__': main()
