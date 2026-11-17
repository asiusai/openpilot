"""Parent-only GPU format/fallback microharness; never a route speed claim.

Fixed shape M32,N32,K64. Amortize40 independent calls and synchronize once.
Both layouts, overflow fallback, fresh/mutated inputs, and single FP8 scales
must pass before any timing. No model weights or route inputs are loaded.
"""
import argparse
from functools import partial
import hashlib
import importlib.util
import json
from pathlib import Path
import time

import numpy as np


def load(name):
  spec = importlib.util.spec_from_file_location(name.replace('-', '_'), Path(__file__).with_name(name+'.py'))
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module


def forty(operation, *args):
  inputs, shared = args[:40], args[40:]
  for x in inputs: result = operation(x, *shared).realize()
  return result


def error(candidate, reference):
  a, b = np.asarray(candidate, np.float64), np.asarray(reference, np.float64)
  delta = np.abs(a-b)
  return dict(differing=int(np.count_nonzero(a != b)), outputs=int(a.size),
              mean_absolute=float(delta.mean()), max_absolute=float(delta.max()),
              relative_p99=float(np.quantile(delta/np.maximum(np.abs(b), 1e-12), .99)))


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--parent-gpu', action='store_true', help='Only the parent may authorize serial GPU execution')
  parser.add_argument('--output', type=Path, required=True)
  args = parser.parse_args()
  if not args.parent_gpu: parser.error('This harness requires explicit parent GPU execution')
  args.output.parent.mkdir(parents=True, exist_ok=True)
  if args.output.exists(): raise FileExistsError(args.output)
  from tinygrad import Tensor, Device, TinyJit, dtypes, GlobalCounters
  from amd_gemm import _gemm
  device = Device[Device.DEFAULT]
  if not Device.DEFAULT.startswith('AMD') or device.arch != 'gfx1200': raise RuntimeError('Requires actual gfx1200 AMD runtime')
  if not getattr(device, 'is_usb', False): raise RuntimeError('Use the parent USB runtime; no ABI monkeypatch is installed here')
  fmt, kernel = load('agent-sparse-format'), load('agent-sparse-kernel')
  rng = np.random.default_rng(116240)
  shape = rows, columns, width = 32, 32, 64
  sx, sw = (Tensor([value], device=Device.DEFAULT, dtype=dtypes.float32).realize() for value in (.125, .25))
  # Exactly representable signed small-integer E4M3 codes.
  integer_codes = np.array([0, 56, 184], dtype=np.uint8)
  w_codes = rng.choice(integer_codes, size=(columns, width))
  def byte_tensor(packed):
    # Bitcast retains the original bytes; never apply an activation scale again.
    return Tensor(np.ascontiguousarray(packed), device=Device.DEFAULT).bitcast(dtypes.fp8e4m3).realize()
  weight = byte_tensor(fmt.pack(w_codes))
  def original(x, w, xs, ws):
    out = Tensor.empty(rows, columns, device=x.device, dtype=dtypes.bfloat16)
    return Tensor.custom_kernel(out, x, w, xs, ws, fxn=partial(_gemm, packed_input=True))[0]
  def complete(x, w, xs, ws):
    compressed = kernel.compress(x)
    return kernel.multiply(x, w, (xs, ws), compressed).cast(dtypes.bfloat16)
  def dense(x, w, xs, ws):
    return kernel.multiply(x, w, (xs, ws), dummy, mode='dense').cast(dtypes.bfloat16)
  dummy = kernel.compress(byte_tensor(np.zeros((rows, width), dtype=np.uint8)))
  Tensor.realize(*dummy)
  validations = []
  host_cases = []
  for index in range(40):
    q = np.zeros((rows, width), dtype=np.uint8)
    for r in range(rows):
      for group in range(width//4):
        selected = rng.choice(4, size=int(rng.integers(3)), replace=False)
        for p in selected: q[r, group*4+p] = rng.choice(integer_codes[1:])
    # Force genuine dense fallback in alternating K tiles and row halves.
    if index % 2: q[index % rows, (index % 2)*32:(index % 2)*32+4] = [56, 64, 68, 72]
    host_cases.append(q)
  for index in (0, 1, 2, 39):
    q = host_cases[index]
    x = byte_tensor(fmt.pack(q))
    np.testing.assert_array_equal(x.bitcast(dtypes.uint8).numpy(), fmt.pack(q))
    compressed = kernel.compress(x)
    actual_values, actual_meta, actual_flags = (value.numpy() for value in compressed)
    expected_values, expected_meta, expected_flags = fmt.compress(q)
    np.testing.assert_array_equal(actual_values.view(np.uint8).reshape(expected_values.shape), expected_values)
    np.testing.assert_array_equal(actual_meta.reshape(expected_meta.shape), expected_meta)
    np.testing.assert_array_equal(actual_flags.reshape(expected_flags.shape), expected_flags)
    reference = original(x, weight, sx, sw).realize().float().numpy()
    sparse = complete(x, weight, sx, sw).realize().float().numpy()
    fallback = dense(x, weight, sx, sw).realize().float().numpy()
    fp64 = fmt.decode(q) @ fmt.decode(w_codes).T * .03125
    # Small integers and power-of-two scales have an exact FP32/BF16 control.
    # Do not relax this to make a layout, metadata, or scale failure disappear.
    np.testing.assert_array_equal(sparse, reference)
    np.testing.assert_array_equal(fallback, reference)
    np.testing.assert_array_equal(sparse, fp64)
    validations.append(dict(case=index, against_original=error(sparse, reference), against_fp64=error(sparse, fp64),
                            eligibility=fmt.audit(q)))
  # Independent noninteger E4M3 cases expose reduction-order deviations honestly.
  # No new arithmetic tolerance, automatic route pass, or equality label is used.
  general_weight = rng.integers(0, 255, size=(columns, width), dtype=np.uint8)
  general_weight[(general_weight & 127) == 127] = 126
  general_w = byte_tensor(fmt.pack(general_weight))
  for index in range(2):
    q = host_cases[index].copy()
    q[q != 0] = rng.integers(1, 127, size=np.count_nonzero(q), dtype=np.uint8)
    x = byte_tensor(fmt.pack(q))
    reference = original(x, general_w, sx, sw).realize().float().numpy()
    compressed = kernel.compress(x)
    hybrid_fp32 = kernel.multiply(x, general_w, (sx, sw), compressed).realize()
    candidate = hybrid_fp32.cast(dtypes.bfloat16).realize().float().numpy()
    fp64 = fmt.decode(q) @ fmt.decode(general_weight).T * .03125
    if not np.isfinite(candidate).all(): raise RuntimeError('Unexpected nonfinite synthetic output')
    validations.append(dict(case='general-fp8-'+str(index), against_original=error(candidate, reference),
                            hybrid_fp32_against_fp64=error(hybrid_fp32.numpy(), fp64),
                            hybrid_bf16_against_fp64=error(candidate, fp64),
                            original_bf16_against_fp64=error(reference, fp64), eligibility=fmt.audit(q),
                            arithmetic_status='Report deviations; no equality or quality acceptance is inferred'))
  # Compile/capture before measurement.40 distinct input buffers prevent CSE.
  inputs = [byte_tensor(fmt.pack(q)) for q in host_cases]
  shared = (weight, sx, sw)
  modes = {}
  for name, operation, kernels_per_call in (('original', original, 1), ('diagnostic_dense', dense, 2),
                                           ('compression_plus_hybrid', complete, 3)):
    graph = TinyJit(partial(forty, operation))
    for _ in range(2): graph(*inputs, *shared)
    device.synchronize()
    before = GlobalCounters.kernel_count
    output = graph(*inputs, *shared)
    device.synchronize()
    observed = GlobalCounters.kernel_count-before
    # BF16 output conversion is a separate kernel in the diagnostic helpers.
    if observed != 40*kernels_per_call: raise RuntimeError(f'{name}: expected{40*kernels_per_call} kernels, observed{observed}')
    samples = []
    for _ in range(3):
      start = time.perf_counter_ns()
      output = graph(*inputs, *shared)
      device.synchronize()
      samples.append((time.perf_counter_ns()-start)/1e6/40)
    expected = original(inputs[-1], *shared).realize().float().numpy()
    np.testing.assert_array_equal(output.float().numpy(), expected)
    # Replay must read mutation in the last real input, including changed flags.
    poison = host_cases[-1].copy(); poison[:] = 56
    inputs[-1].bitcast(dtypes.uint8).assign(Tensor(fmt.pack(poison), device=Device.DEFAULT)).realize()
    mutated = graph(*inputs, *shared).realize().float().numpy()
    np.testing.assert_array_equal(mutated, original(inputs[-1], *shared).realize().float().numpy())
    inputs[-1].bitcast(dtypes.uint8).assign(Tensor(fmt.pack(host_cases[-1]), device=Device.DEFAULT)).realize()
    modes[name] = dict(amortized_host_ms=samples, median_ms=float(np.median(samples)),
                       observed_kernels_per_40_calls=observed, replay_mutation_passed=True)
  report = dict(shape=list(shape), input='Synthetic finite FP8 small-integer activations only',
                measurement='40 independent calls; 3 replay samples; one completion fence/sample',
                included='Dynamic compression, metadata, eligibility, dense fallback, scale product, BF16 cast for hybrid',
                excluded='FP8 quantization/packing common to both paths, upload, compilation, initial allocation, model/route stages',
                warning='16x16 probe geometry cannot establish production GEMM speed or route improvement',
                gpu_validated=True, validations=validations, modes=modes,
                actual_route_sparsity=None, quality_validated=False)
  report['source_sha256'] = {name:hashlib.sha256(Path(__file__).with_name(name+'.py').read_bytes()).hexdigest()
                             for name in ('agent-sparse-bench', 'agent-sparse-kernel', 'agent-sparse-format')}
  args.output.write_text(json.dumps(report, indent=2)+'\n')
  print(json.dumps(report, indent=2))


if __name__ == '__main__': main()
