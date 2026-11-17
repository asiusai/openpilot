"""CPU layout proof and parent-only GPU checks for native signed INT4 K32.

Default --suite cpu never initializes a GPU. --suite small checks raw INT32.
--suite actual compares both required shapes with old K16 and current FP8.
Integer inputs are already quantized. Model quality, quantization, and scales
are outside this arithmetic experiment. Timings exclude packing and uploads.
"""
import argparse
from functools import partial
import hashlib
import importlib.util
import json
from pathlib import Path
import time
import numpy as np


def local(name):
  spec = importlib.util.spec_from_file_location(name.replace('-', '_'), Path(__file__).with_name(name+'.py'))
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module


def cases():
  rng = np.random.default_rng(32049060)
  for m, n, k in ((1, 1, 32), (7, 19, 32), (17, 33, 64), (65, 71, 96), (31, 47, 256)):
    a, b = (rng.integers(-8, 8, (r, k), dtype=np.int32) for r in (m, n))
    yield f'random-{m}-{n}-{k}', a, b
    for operand in (0, 1):
      aa, bb = a.copy(), b.copy()
      x = (aa, bb)[operand]
      x[-1, k//2+7] = 7 if x[-1, k//2+7] != 7 else -8
      yield f'mutate-{operand}-{m}-{n}-{k}', aa, bb
  # Every K coordinate, including both nibble and register boundaries.
  for kpos in range(32):
    a, b = np.zeros((17, 32), np.int32), np.zeros((19, 32), np.int32)
    a[:, kpos] = np.arange(17) % 16-8
    b[:, kpos] = (np.arange(19)*3+5) % 16-8
    # A different B coordinate must not contribute.
    b[:, (kpos+9) % 32] = (np.arange(19)*5+2) % 16-8
    yield f'k-basis-{kpos}', a, b
  for av, bv in ((-8, -8), (-8, 7), (7, -8), (7, 7), (0, -8)):
    yield f'extreme-{av}-{bv}', np.full((19, 64), av, np.int32), np.full((23, 64), bv, np.int32)


def unpack_by_isa(packed, rows):
  """Coordinate lookup from the ISA table, separate from reshape-based packing."""
  width = packed.shape[1]*32
  result = np.empty((packed.shape[0]*16, width), np.int32)
  for m in range(result.shape[0]):
    for k in range(width):
      lane = ((k & 16) << 0) | (m & 15)
      word = np.uint32(packed[m//16, k//32, lane, (k >> 3) & 1])
      nibble = int((word >> np.uint32(4*(k & 7))) & np.uint32(15))
      result[m, k] = nibble-16 if nibble & 8 else nibble
  np.testing.assert_array_equal(result[rows:], 0)
  return result[:rows]


def emulated_wmma(packed_a, packed_b, m, n):
  """Execute fragments and map accumulator lanes back through the ISA table."""
  out = np.zeros((m, n), np.int64)
  for mt in range(packed_a.shape[0]):
    for nt in range(packed_b.shape[0]):
      regs = np.zeros((32, 8), np.int64)
      for kt in range(packed_a.shape[1]):
        for k in range(32):
          lane_a = (k//16)*16+np.arange(16)
          wa = packed_a[mt, kt, lane_a, k//8 % 2].astype(np.uint32)
          wb = packed_b[nt, kt, lane_a, k//8 % 2].astype(np.uint32)
          av = ((wa >> np.uint32(4*(k % 8))) & 15).astype(np.int64)
          bv = ((wb >> np.uint32(4*(k % 8))) & 15).astype(np.int64)
          av[av >= 8] -= 16; bv[bv >= 8] -= 16
          for lane in range(32):
            for reg in range(8):
              regs[lane, reg] += av[(lane//16)*8+reg]*bv[lane % 16]
      for lane in range(32):
        for reg in range(8):
          row, col = mt*16+(lane//16)*8+reg, nt*16+lane % 16
          if row < m and col < n: out[row, col] = regs[lane, reg]
  return out


def cpu_checks(kernel):
  records = []
  for name, a, b in cases():
    pa, pb = kernel.pack_cpu(a), kernel.pack_cpu(b)
    np.testing.assert_array_equal(unpack_by_isa(pa, len(a)), a)
    np.testing.assert_array_equal(unpack_by_isa(pb, len(b)), b)
    expected = a.astype(np.int64) @ b.astype(np.int64).T
    np.testing.assert_array_equal(emulated_wmma(pa, pb, len(a), len(b)), expected)
    records.append(dict(name=name, shape=[len(a), len(b), a.shape[1]],
                        digest=hashlib.sha256(expected.tobytes()).hexdigest()))
  for invalid in (np.array([[8]*32]), np.array([[-9]*32]), np.zeros((1, 16), np.int32), np.zeros((1, 32))):
    try: kernel.pack_cpu(invalid)
    except ValueError: pass
    else: raise AssertionError('Invalid input accepted')
  return dict(cpu_only=True, exact_layout_and_emulation_cases=len(records), rejected_invalid_inputs=4, cases=records)


def bf16_cpu(x):
  bits = x.astype(np.float32).view(np.uint32)
  return ((bits+0x7fff+((bits >> 16) & 1)) >> 16).astype(np.uint16)


def pack_k16(x):
  k = x.shape[1]
  x = np.pad(x, ((0, (-x.shape[0]) % 16), (0, 0)))
  tiles = x.reshape(-1, 16, k//16, 2, 8).transpose(0, 2, 3, 1, 4).reshape(-1, k//16, 32, 8)
  return np.bitwise_or.reduce((tiles.astype(np.uint32) & 15) << np.arange(0, 32, 4, dtype=np.uint32), axis=-1).view(np.int32)


def gpu_small(kernel):
  from tinygrad import Tensor, dtypes
  records = []
  for name, a, b in cases():
    pa, pb = (Tensor(kernel.pack_cpu(x)).realize() for x in (a, b))
    expected = a.astype(np.int64) @ b.astype(np.int64).T
    raw = kernel.gemm(pa, pb, len(a), len(b), output='i32').numpy()
    np.testing.assert_array_equal(raw, expected)
    rounded = kernel.gemm(pa, pb, len(a), len(b)).bitcast(dtypes.uint16).numpy()
    np.testing.assert_array_equal(rounded, bf16_cpu(expected))
    records.append(dict(name=name, shape=[len(a), len(b), a.shape[1]], i32_exact=True, bf16_exact=True))
    print(json.dumps(records[-1]), flush=True)
  return dict(cpu_only=False, all_pass=True, cases=records)


def gpu_actual(kernel, args):
  from tinygrad import Tensor, Device, TinyJit, dtypes
  from amd_gemm import pack, _gemm
  k16 = local('int4-gemm')
  # Match the current pipeline's FP8 renderer, while retaining each HIP kernel's CU mode.
  local('wgp_compiler').install(Device[Device.DEFAULT])
  scale = Tensor([1.], dtype=dtypes.float32).reshape(()).realize()
  rng = np.random.default_rng(32049060)
  records = []
  for m, n, k in ((2112, 9216, 2048), (2112, 2048, 9216)):
    assert 64*k < 2**24  # Bound every signed partial sum for the independent FP8 control.
    a, b = (rng.integers(-8, 8, (r, k), dtype=np.int32) for r in (m, n))
    def fp8(x, w, dtype=dtypes.bfloat16):
      out = Tensor.empty(m, n, dtype=dtype)
      return Tensor.custom_kernel(out, x, w, scale, scale, fxn=partial(_gemm, packed_input=True))[0]
    variants = [
      ('fp8', fp8, lambda x: pack(Tensor(x.astype(np.float32)).cast(dtypes.fp8e4m3)).realize()),
      ('int4-k16-tile2', partial(k16.gemm, rows=m, columns=n, parts=2), lambda x: Tensor(pack_k16(x)).realize()),
      ('int4-k16-tile4', partial(k16.gemm, rows=m, columns=n, parts=4), lambda x: Tensor(pack_k16(x)).realize()),
      ('int4-k32', partial(kernel.gemm, rows=m, columns=n), lambda x: Tensor(kernel.pack_cpu(x)).realize())]
    inputs = {name: (packer(a), packer(b)) for name, _, packer in variants}
    record = dict(shape=[m, n, k], timings_ms={}, checks=[])
    # Check every output bit against a separate FP8 implementation. These integer
    # dots remain below 2**24, so the FP32 reference has exact integer accumulation.
    for mutation in ('original', 'a-k-boundaries', 'b-last-row'):
      if mutation == 'a-k-boundaries':
        a = a.copy()
        for pos in (0, 7, 8, 15, 16, 23, 24, 31, 32, k-1):
          a[0, pos] = 7 if a[0, pos] != 7 else -8
      elif mutation == 'b-last-row':
        b = b.copy(); b[-1, -1] = -8 if b[-1, -1] != -8 else 7
      if mutation != 'original':
        inputs = {name: (packer(a), packer(b)) for name, _, packer in variants}
      # BF16 equality can hide integer errors. First check every unrounded K32
      # integer against a separate FP8 kernel with FP32 output.
      raw = kernel.gemm(*inputs['int4-k32'], rows=m, columns=n, output='i32').numpy()
      fp32_reference = fp8(*inputs['fp8'], dtype=dtypes.float32).numpy()
      np.testing.assert_array_equal(raw, fp32_reference)
      outputs = {name: fn(*inputs[name]).bitcast(dtypes.uint16).numpy() for name, fn, _ in variants}
      for value in outputs.values(): np.testing.assert_array_equal(value, bf16_cpu(raw))
      # Independent int64 host oracle on asymmetric, boundary-heavy row/column selections.
      ri = np.unique(np.array([0, 1, 7, 8, 15, 16, 31, 32, m//2, m-2, m-1]))
      ci = np.unique(np.array([0, 3, 7, 15, 16, 31, 32, n//2, n-2, n-1]))
      expected = a[ri].astype(np.int64) @ b[ci].astype(np.int64).T
      np.testing.assert_array_equal(raw[np.ix_(ri, ci)], expected)
      for value in outputs.values(): np.testing.assert_array_equal(value[np.ix_(ri, ci)], bf16_cpu(expected))
      record['checks'].append(dict(mutation=mutation, raw_i32_all_elements_exact=True, bf16_all_elements_exact=True,
                                   independent_cpu_elements=int(expected.size)))
    # Matched mutated inputs are now fixed. No packing, conversion, or upload in timing.
    jits = {}
    for name, fn, _ in variants:
      jit = jits[name] = TinyJit(fn)
      for _ in range(3): jit(*inputs[name]).realize()
    Device[Device.DEFAULT].synchronize()
    for name, _, _ in variants: record['timings_ms'][name] = []
    for round_id in range(args.rounds):
      order = list(jits)[round_id % len(jits):]+list(jits)[:round_id % len(jits)]
      for name in order:
        Device[Device.DEFAULT].synchronize()
        started = time.perf_counter()
        for _ in range(args.runs): jits[name](*inputs[name]).realize()
        Device[Device.DEFAULT].synchronize()
        record['timings_ms'][name].append((time.perf_counter()-started)*1000/args.runs)
    record['median_ms'] = {name: float(np.median(values)) for name, values in record['timings_ms'].items()}
    record['k32_over_k16_same_tile'] = record['median_ms']['int4-k32']/record['median_ms']['int4-k16-tile2']
    record['k32_over_k16_best_control'] = record['median_ms']['int4-k32']/min(record['median_ms']['int4-k16-tile2'],
                                                                          record['median_ms']['int4-k16-tile4'])
    record['k32_over_fp8'] = record['median_ms']['int4-k32']/record['median_ms']['fp8']
    records.append(record)
    print(json.dumps(record), flush=True)
    args.output.write_text(json.dumps(dict(cpu_only=False, complete=False, records=records), indent=2)+'\n')
  return dict(cpu_only=False, complete=True, all_pass=True, rounds=args.rounds, runs=args.runs,
              timing='amortized host-inclusive, one completion fence per run group; packing/uploads excluded', records=records)


def main():
  p = argparse.ArgumentParser(description=__doc__)
  p.add_argument('--suite', choices=('cpu', 'small', 'actual'), default='cpu')
  p.add_argument('--runs', type=int, default=40)
  p.add_argument('--rounds', type=int, default=4)
  p.add_argument('--output', type=Path, required=True)
  args = p.parse_args()
  if not 1 <= args.runs <= 200 or not 1 <= args.rounds <= 6: p.error('Unbounded timing request')
  kernel = local('agent-int4k32-kernel')
  if args.suite == 'cpu': report = cpu_checks(kernel)
  else:
    local('prefill-kernels').fix_dispatch_packet()
    small = gpu_small(kernel)
    report = small if args.suite == 'small' else dict(gpu_actual(kernel, args), small_controls=small)
  sources = [Path(__file__), Path(__file__).with_name('agent-int4k32-kernel.py'),
             Path(__file__).with_name('int4-gemm.py'), Path(__file__).with_name('wgp_compiler.py'),
             Path(__file__).parent.parent/'amd_gemm.py']
  report['source_hashes'] = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in sources}
  args.output.write_text(json.dumps(report, indent=2)+'\n')
  print(json.dumps({k: v for k, v in report.items() if k not in ('cases', 'records', 'small_controls', 'source_hashes')}), flush=True)


if __name__ == '__main__': main()
