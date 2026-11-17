"""Independent FP8 controls, exact reference comparisons, and amortized timing.

Run GPU work only through the parent's serialized --program wrapper.
"""
import argparse
from functools import partial
import importlib.util
import json
from pathlib import Path
import time
import numpy as np


def module(name):
  spec = importlib.util.spec_from_file_location(name.replace('-', '_'), Path(__file__).with_name(name+'.py'))
  value = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(value)
  return value


def decode_fp8(bits):
  exponent = ((bits >> 3) & 15).astype(np.int32)
  mantissa = (bits & 7).astype(np.float32)
  magnitude = np.where(exponent == 0, mantissa*2.0**-9, (1+mantissa/8)*np.exp2(exponent-7))
  return np.where(bits & 128, -magnitude, magnitude).astype(np.float32)


def pack_cpu(bits):
  rows, width = bits.shape
  return np.pad(bits, ((0, (-rows)%16), (0,0))).reshape(-1,16,width//16,4,4).transpose(0,2,3,1,4).copy().reshape(-1,width)


def bf16_bits(value):
  bits = np.asarray(value, dtype=np.float32).view(np.uint32)
  regular = bits+((bits >> 16) & 1)+np.uint32(0x7fff)
  special = bits | ((bits & 0xffff) != 0).astype(np.uint32)*np.uint32(0x10000)
  return (np.where(((-bits) & 0x7f800000) != 0, regular, special) >> 16).astype(np.uint16)


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--output', type=Path, required=True)
  parser.add_argument('--suite', choices=('quick','actual','all'), default='quick')
  parser.add_argument('--modes', default='transpose,pipeline')
  parser.add_argument('--runs', type=int, default=40)
  parser.add_argument('--rounds', type=int, default=3)
  parser.add_argument('--no-timing', action='store_true')
  parser.add_argument('--cpu-only', action='store_true')
  args = parser.parse_args()
  rng = np.random.default_rng(906016)
  modes = args.modes.split(',')
  candidate = module('agent-gemm-kernel')
  assert all(mode in candidate.MODES for mode in modes)
  if args.cpu_only:
    # The packing proof uses unique labels and the reference address equation.
    for rows, columns, width in ((17,48,32),(65,80,48),(180,11520,4608)):
      labels = np.arange(rows*width,dtype=np.uint32).reshape(rows,width)
      packed = pack_cpu(labels).reshape(-1)
      for _ in range(2000):
        row = int(rng.integers(rows)); k = int(rng.integers(width))
        index = (row//16*(width//16)+k//16)*256+(k%16//4)*64+(row%16)*4+k%4
        assert packed[index] == labels[row,k]
      for tile in range((rows+63)//64):
        for i in range(4):
          actual = min(tile*4+i, (rows+15)//16-1)
          assert actual*16 < (rows+15)//16*16
    values = rng.integers(0,2**32,100000,dtype=np.uint32)
    mask = np.uint32(0)-(((-values) & 0x7f800000) != 0).astype(np.uint32)
    regular = values+((values >> 16)&1)+np.uint32(0x7fff)
    special = values | (((values & 0xffff) != 0).astype(np.uint32)<<16)
    np.testing.assert_array_equal(((regular&mask)|(special&~mask))>>16,bf16_bits(values.view(np.float32)))
    result = {'cpu_only': True, 'packing_index_checks': 6000, 'bf16_algebra_checks':100000,
              'gpu_correctness': 'not run', 'gpu_timing': 'not run'}
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)
    return

  from tinygrad import Device, Tensor, TinyJit, dtypes
  from tinygrad.device import Compiled
  from amd_gemm import _gemm
  Compiled.wait_timeout_ms = 10000
  module('prefill-kernels').fix_dispatch_packet()
  module('wgp_compiler').install(Device[Device.DEFAULT])
  tiled = module('tiled-gemm')
  image_reference = module('image-gemm')

  # Columns and K retain real model alignment. M exercises every tail condition.
  quick = [(1,17,48,32,False,False), (1,65,80,48,True,True), (1,81,144,64,False,True),
           (2,17,48,32,False,True), (2,33,80,48,True,False)]
  actual = [(1,2027,9216,2048,False,False), (1,2112,2048,9216,False,False),
            (1,1440,4304,1152,True,True), (1,1440,1152,4304,True,True),
            (2,180,11520,4608,False,True), (2,180,2048,11520,False,True),
            (1,64,3840,512,False,False)]
  cases = quick if args.suite == 'quick' else actual if args.suite == 'actual' else quick+actual
  results = []
  for case_number, (images, rows, columns, width, row_scale, with_bias) in enumerate(cases):
    record = dict(images=images, shape=[rows,columns,width], row_scale=row_scale, bias=with_bias, checks=[], timing=[])
    # Two independent random inputs per shape. Integer controls have exact CPU sums.
    for trial in range(2):
      integer_control = case_number == 0 and trial == 0
      def bits(shape):
        if integer_control:
          values = rng.integers(-7,8,shape,dtype=np.int32)
          table = decode_fp8(np.arange(256,dtype=np.uint8))
          lookup = {int(v):i for i,v in enumerate(table) if np.isfinite(v) and -7<=v<=7 and v == int(v)}
          return np.vectorize(lookup.__getitem__,otypes=[np.uint8])(values)
        # Include signed zeros, subnormals, and the complete finite FP8 range.
        magnitude = rng.integers(0,127,shape,dtype=np.uint8)
        return magnitude | (rng.integers(0,2,shape,dtype=np.uint8)<<7)
      abits, wbits = bits((images,rows,width)), bits((columns,width))
      packed_a = np.concatenate([pack_cpu(x) for x in abits],axis=0)
      a = Tensor(packed_a).bitcast(dtypes.fp8e4m3).realize()
      w = Tensor(pack_cpu(wbits)).bitcast(dtypes.fp8e4m3).realize()
      xs_values = (np.arange(images*rows if row_scale else images,dtype=np.float32)%7+1)/64
      xs, ws = Tensor(xs_values).realize(), Tensor(np.array([.03125],dtype=np.float32)).realize()
      bias_values = rng.normal(0,.25,columns).astype(np.float32)
      bias = Tensor(bf16_bits(bias_values)).bitcast(dtypes.bfloat16).realize() if with_bias else None
      def reference(a,w,xs,ws):
        extra = () if bias is None else (bias,)
        if images == 1:
          out = Tensor.empty(rows,columns,dtype=dtypes.bfloat16)
          fn = partial(tiled.kernel,tile_m=4,tile_n=4,row_scale=True) if row_scale else _gemm
          return Tensor.custom_kernel(out,a,w,xs,ws,*extra,fxn=fn)[0]
        # The production image kernel needs a retained global row-tile axis.
        # Tiny controls below one 64-row tile use independent original GEMMs.
        if row_scale or rows <= 64:
          parts = []
          padded_rows = (rows+15)//16*16
          for image in range(images):
            out = Tensor.empty(rows,columns,dtype=dtypes.bfloat16)
            scales = xs[image*rows:(image+1)*rows] if row_scale else xs[image:image+1]
            fn = partial(tiled.kernel,tile_m=4,tile_n=4,row_scale=True) if row_scale else _gemm
            parts.append(Tensor.custom_kernel(out,a[image*padded_rows:(image+1)*padded_rows],w,
              scales,ws,*extra,fxn=fn)[0])
          return Tensor.stack(*parts)
        out = Tensor.empty(images,rows,columns,dtype=dtypes.bfloat16)
        return Tensor.custom_kernel(out,a,w,xs,ws,*extra,fxn=partial(image_reference.kernel,images=images,rows=rows))[0]
      expected = reference(a,w,xs,ws).bitcast(dtypes.uint16).numpy()
      if integer_control:
        cpu = decode_fp8(abits[0]).astype(np.int64) @ decode_fp8(wbits).astype(np.int64).T
        cpu = cpu.astype(np.float32)*(xs_values[0]*np.float32(.03125))
        np.testing.assert_array_equal(expected,bf16_bits(cpu))
      functions = [('reference',reference)]
      for mode in modes:
        def run(a,w,xs,ws,mode=mode):
          return candidate.gemm_packed(a,w,(xs,ws),rows,columns,bias=bias,mode=mode,row_scale=row_scale,images=images)
        actual_bits = run(a,w,xs,ws).bitcast(dtypes.uint16).numpy()
        np.testing.assert_array_equal(actual_bits,expected,err_msg=f'{case_number=} {trial=} {mode=}')
        functions.append((mode,run))
      record['checks'].append(dict(trial=trial,exact=True,independent_integer_cpu=integer_control))
    # Mutate the existing GPU input during graph replay, then check every mode.
    for name, fn in functions:
      jit = TinyJit(fn)
      for _ in range(3): jit(a,w,xs,ws)
      changed = packed_a.copy(); changed.flat[0] ^= np.uint8(128)
      a.assign(Tensor(changed).bitcast(dtypes.fp8e4m3)).realize()
      changed_expected = reference(a,w,xs,ws).bitcast(dtypes.uint16).numpy()
      np.testing.assert_array_equal(jit(a,w,xs,ws).bitcast(dtypes.uint16).numpy(),changed_expected)
      a.assign(Tensor(packed_a).bitcast(dtypes.fp8e4m3)).realize()
      if args.no_timing: continue
      elapsed = []
      for _ in range(args.rounds):
        Device[Device.DEFAULT].synchronize()
        started = time.perf_counter()
        for _ in range(args.runs): jit(a,w,xs,ws)
        Device[Device.DEFAULT].synchronize()
        elapsed.append((time.perf_counter()-started)*1000/args.runs)
      record['timing'].append(dict(mode=name,amortized_ms=elapsed,median_ms=float(np.median(elapsed)),
                                   calls_per_fence=args.runs,host_inclusive=True))
    record['jit_input_mutation_exact'] = True
    results.append(record)
    args.output.write_text(json.dumps(results,indent=2)+'\n')
    print(json.dumps(record),flush=True)


if __name__ == '__main__': main()
