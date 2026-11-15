"""Small independent CPU-oracle checks before any vendor model experiment."""
import argparse
import importlib.util
import json
from pathlib import Path
import numpy as np
from tinygrad import Tensor, Device, dtypes, TinyJit
from tinygrad.device import Compiled


def module(name):
  spec = importlib.util.spec_from_file_location(name,Path(__file__).with_name(name+'.py'))
  value = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(value)
  return value


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--output',type=Path,required=True)
  parser.add_argument('--solutions',type=int,nargs='+',default=[27955,27854])
  parser.add_argument('--usb-serial-file',type=Path,required=True)
  parser.add_argument('--usb-serial',required=True)
  args = parser.parse_args()
  assert args.usb_serial_file.read_text().strip() == args.usb_serial
  Compiled.wait_timeout_ms = 10000
  module('prefill-kernels').fix_dispatch_packet()
  vendor = module('vendor-gemm')
  device = Device[Device.DEFAULT]
  rng = np.random.default_rng(518)
  records = []
  with vendor.argument_adapter(device):
    for rows,columns,width in ((128,128,64),(145,144,64),(145,144,80),(256,256,256)):
      for index in args.solutions:
        hostx = rng.integers(-2,3,size=(rows,width)).astype(np.float32)
        hostw = rng.integers(-2,3,size=(columns,width)).astype(np.float32)
        x = Tensor(hostx,device=device.device).cast(dtypes.fp8e4m3).contiguous().realize()
        w = Tensor(hostw,device=device.device).cast(dtypes.fp8e4m3).contiguous().realize()
        xs = Tensor([.125],device=device.device,dtype=dtypes.float32).contiguous().realize()
        ws = Tensor([.25],device=device.device,dtype=dtypes.float32).contiguous().realize()
        ones = Tensor.ones(columns,device=device.device,dtype=dtypes.float32).contiguous().realize()
        bias = Tensor.zeros(columns,device=device.device,dtype=dtypes.float32).contiguous().realize()
        zero = Tensor.zeros(rows,columns,device=device.device,dtype=dtypes.bfloat16).contiguous().realize()
        run = TinyJit(lambda a,b:vendor.gemm(a,b,(xs,ws),ones,bias,zero,index=index).realize())
        for iteration in range(3):
          if iteration:
            hostx = rng.integers(-2,3,size=(rows,width)).astype(np.float32)
            x.assign(Tensor(hostx,device=device.device).cast(dtypes.fp8e4m3)).realize()
          print('launch',index,(rows,columns,width),iteration,flush=True)
          result = run(x,w)
          device.synchronize()
          expected = ((hostx.astype(np.float64)@hostw.astype(np.float64).T)*.03125).astype(np.float32)
          bits = expected.view(np.uint32)
          expected = ((bits+np.uint32(0x7fff)+((bits>>16)&1))&np.uint32(0xffff0000)).view(np.float32)
          actual = result.float().numpy()
          assert np.isfinite(actual).all()
          np.testing.assert_array_equal(actual,expected)
        record = dict(solution=index,shape=[rows,columns,width],changing_inputs=3,exact_cpu=True)
        records.append(record)
        args.output.write_text(json.dumps(records,indent=2)+'\n')
        print(record,flush=True)
  print('vendor ABI adapter restored',flush=True)


if __name__ == '__main__': main()
