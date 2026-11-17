"""CPU alias/mapping checks and parent-serialized GPU expert comparisons."""
import argparse
import importlib.util
import json
from pathlib import Path
import time

import numpy as np


def module(name):
  spec = importlib.util.spec_from_file_location(name.replace('-', '_'), Path(__file__).with_name(name+'.py'))
  result = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(result)
  return result


def pack_numpy(x):
  """Independent scalar-index definition of the selected packed layout."""
  batch, heads, rows, width = x.shape
  assert rows % 16 == 0 and width % 16 == 0
  h, r, d = np.indices((batch*heads, rows, width), dtype=np.int64)
  index = ((h*(rows//16)+r//16)*(width//16)+d//16)*256+(d%16//4)*64+(r%16)*4+d%4
  out = np.empty(x.size, dtype=x.dtype)
  out[index.ravel()] = x.ravel()
  return out.reshape(batch, heads, rows//16, width//16, 4, 16, 4)


def expected_buffers(old, current):
  return (pack_numpy(np.concatenate((old[0], current[0]), axis=2)),
    pack_numpy(np.concatenate((old[1], current[1]), axis=2).swapaxes(-2, -1)))


def cpu_checks(candidate, capacities):
  from tinygrad import Device, Tensor, TinyJit, dtypes
  from tinygrad.uop.ops import Ops
  assert Device.DEFAULT.startswith('CPU'), 'Set DEV=CPU for --cpu-only; do not use the USB wrapper.'
  rng = np.random.default_rng(906064)
  rows = []
  def tensor(x):
    return Tensor(x, device='CPU').bitcast(dtypes.bfloat16).realize()

  for capacity in capacities:
    shapes = ((2, 2, capacity, 128), (1, 8, capacity, 128))
    for shape in shapes:
      batch, heads, _, width = shape
      traced = []
      def run(oldk, oldv, newk, newv):
        history = candidate.PackedHistory(oldk, oldv)
        identity = (history.k.uop.storage_base, history.v.uop.storage_base)
        snapshots = []
        for step in range(3):
          pk, pv = history.update(newk[step], newv[step])
          assert identity == (history.k.uop.storage_base, history.v.uop.storage_base)
          # Copies are explicit consumers before the next update overwrites data.
          snapshots.extend((pk.bitcast(dtypes.uint16).clone().realize(), pv.bitcast(dtypes.uint16).clone().realize()))
        traced.append(identity)
        return tuple(snapshots)
      jit = TinyJit(run)
      inputs = None
      for iteration in range(7):
        old = [rng.integers(0, 65536, shape, dtype=np.uint16) for _ in range(2)]
        new = [rng.integers(0, 65536, (3, batch, heads, 64, width), dtype=np.uint16) for _ in range(2)]
        arrays = old+new
        # Check new allocations and mutation of the same captured input buffers.
        if iteration in (3, 5, 6):
          for target, array in zip(inputs, arrays, strict=True):
            target.assign(tensor(array)).realize()
        else:
          inputs = list(map(tensor, arrays))
        outputs = jit(*inputs)
        for step in range(3):
          expected = expected_buffers(old, [x[step] for x in new])
          for kind in (0, 1):
            np.testing.assert_array_equal(outputs[2*step+kind].numpy(), expected[kind],
              err_msg=f'capacity={capacity}, iteration={iteration}, step={step}, kind={kind}')
        row = {'capacity': capacity, 'batch': batch, 'kv_heads': heads, 'invocation': iteration,
          'same_input_allocation_mutation': iteration in (3, 5, 6), 'all_three_steps_exact': True,
          'new_history_every_invocation': True, 'stable_two_buffer_identity': True}
        rows.append(row)
        print(json.dumps(row), flush=True)
      assert len(traced) == 2, 'Only warmup and capture may run Python; replay must execute captured history writes.'
      # Keep a reviewable CPU capture summary without assuming GPU scheduling.
      linear = jit.captured._linear
      programs = [u for u in linear.toposort(enter_calls=True) if u.op is Ops.PROGRAM]
      assert len(linear.src) == 14, 'Expected eight pack/update kernels and six snapshot copies.'
      rows.append({'capacity': capacity, 'batch': batch, 'kv_heads': heads, 'python_traces': len(traced),
        'captured_calls': len(linear.src), 'cpu_program_names': [p.src[0].arg.name for p in programs]})
  # Model history is captured through self.cache, not explicit JIT arguments.
  # Exercise this exact closure case, with external in-place history updates.
  capacity = max(capacities)
  shape = (1, 8, capacity, 128)
  old = [rng.integers(0, 65536, shape, dtype=np.uint16) for _ in range(2)]
  old_tensors = list(map(tensor, old))
  closure = TinyJit(lambda nk, nv: run(*old_tensors, nk, nv))
  for iteration in range(6):
    old = [rng.integers(0, 65536, shape, dtype=np.uint16) for _ in range(2)]
    for target, array in zip(old_tensors, old, strict=True):
      target.assign(tensor(array)).realize()
    new = [rng.integers(0, 65536, (3, 1, 8, 64, 128), dtype=np.uint16) for _ in range(2)]
    outputs = closure(*map(tensor, new))
    for step in range(3):
      for kind, expected in enumerate(expected_buffers(old, [x[step] for x in new])):
        np.testing.assert_array_equal(outputs[2*step+kind].numpy(), expected)
  return {'cpu_only': True, 'gpu_executed': False, 'cases': rows, 'closure_history_mutation_replays_exact': 6,
    'bits': 'Arbitrary uint16 patterns, including BF16 NaN payloads, infinities, subnormals, and signed zeros.'}


def compile_checks(candidate, capacities):
  from tinygrad import Tensor, dtypes
  from tinygrad.codegen import to_program
  from tinygrad.helpers import Target
  from tinygrad.renderer.llvmir import AMDLLVMRenderer
  import amd_attention
  assert candidate._matmul is amd_attention._matmul and candidate._softmax is amd_attention._softmax
  renderer = AMDLLVMRenderer(Target('AMD', 'LLVM', 'gfx1200'))
  result = []
  for capacity in capacities:
    for transposed in (False, True):
      for new in (False, True):
        rows, width = (128, capacity+64) if transposed else (capacity+64, 128)
        output = Tensor.empty(1, 8, rows//16, width//16, 4, 16, 4, device='CPU', dtype=dtypes.uint16).realize()
        source = Tensor.empty(1, 8, 64 if new else capacity, 128, device='CPU', dtype=dtypes.uint16).realize()
        if transposed:
          view = output[:, :, :, capacity//16:] if new else output[:, :, :, :capacity//16]
        else:
          view = output[:, :, capacity//16:] if new else output[:, :, :capacity//16]
        view.assign(candidate.pack_view(source.transpose(-2, -1) if transposed else source))
        linear = output.schedule_linear()
        assert len(linear.src) == 1
        program = to_program(linear.src[0].src[0], renderer)
        result.append({'capacity': capacity, 'operand': 'V' if transposed else 'K', 'part': 'action' if new else 'history',
          'kernels': 1, 'hsaco_bytes': len(program.src[3].arg), 'original_matrix_functions_unchanged': True})
  return result


def gpu_checks(args, candidate):
  import torch
  from tinygrad import Device, Tensor, TinyJit, dtypes
  from tinygrad.device import Compiled
  from amd_attention import _pack
  module('prefill-kernels').fix_dispatch_packet()
  Compiled.wait_timeout_ms = 10000
  torch.set_num_threads(4)
  baseline = module('expert-kernels').attention
  rng = np.random.default_rng(906064)
  report = {'gpu_executed': True, 'correctness': [], 'timing': []}
  def save():
    args.output.write_text(json.dumps(report, indent=2)+'\n')
  def tensor(array):
    return Tensor(array.astype(np.float32)).cast(dtypes.bfloat16).realize()
  def original(q, k, v, oldk, oldv, mask):
    return Tensor.stack(*[baseline(q[i], oldk.cat(k[i], dim=2), oldv.cat(v[i], dim=2), mask).realize()
      for i in range(3)]).realize()
  def optimized(q, k, v, oldk, oldv, mask):
    history = candidate.PackedHistory(oldk, oldv)
    return Tensor.stack(*[candidate.attention(q[i], k[i], v[i], history, mask) for i in range(3)]).realize()

  for capacity in args.capacities:
    keys = capacity+64
    shapes = [(3, 1, 16, 64, 128), (3, 1, 8, 64, 128), (3, 1, 8, 64, 128),
      (1, 8, capacity, 128), (1, 8, capacity, 128)]
    jits = [('baseline', TinyJit(original)), ('cachedpack', TinyJit(optimized))]
    inputs = None
    for iteration, length in enumerate((capacity-31, 0, capacity, 17, capacity//2, capacity-1, 0)):
      arrays = [rng.standard_normal(shape).astype(np.float32) for shape in shapes]
      mask_array = np.where((np.arange(keys) < length) | (np.arange(keys) >= capacity), 0., -np.inf).astype(np.float32)
      if iteration in (3, 5, 6):
        for target, array in zip(inputs[:5], arrays, strict=True):
          target.assign(tensor(array)).realize()
        inputs[5].assign(Tensor(mask_array)).realize()
      else:
        inputs = [tensor(x) for x in arrays]+[Tensor(mask_array).realize()]
      outputs = []
      for _, jit in jits:
        output = jit(*inputs)
        outputs.append(output.bitcast(dtypes.uint16).numpy())
      row = {'capacity': capacity, 'invocation': iteration, 'prefix_length': length,
        'same_input_allocation_mutation': iteration in (3, 5, 6),
        'three_steps_exact': bool(np.array_equal(*outputs)),
        'mismatched_bf16': int(np.count_nonzero(outputs[0] != outputs[1]))}
      report['correctness'].append(row)
      save()
      print(json.dumps(row), flush=True)
      np.testing.assert_array_equal(*outputs, err_msg=str(row))
      # Check selected attention against independent FP32 math, including masks.
      if iteration in (0, 3):
        q, k, v, oldk, oldv, mask = [torch.from_numpy(x.float().numpy()) for x in inputs]
        truths = []
        for step in range(3):
          fullk = torch.cat((oldk, k[step]), dim=2).repeat_interleave(2, dim=1)
          fullv = torch.cat((oldv, v[step]), dim=2).repeat_interleave(2, dim=1)
        truths.append(torch.nn.functional.scaled_dot_product_attention(q[step], fullk, fullv, attn_mask=mask.reshape(1, -1)))
        truth = torch.stack(truths).numpy()
        actual = (outputs[1].astype(np.uint32) << 16).view(np.float32)
        assert np.isfinite(actual).all()
        row['rmse_vs_fp32'] = float(np.sqrt(np.mean((actual-truth)**2)))
        row['max_abs_vs_fp32'] = float(np.max(np.abs(actual-truth)))
        save()

    # Direct packed-buffer checks on the GPU identify mapping errors before
    # model integration. Each snapshot is copied before the next action write.
    q, k, v, oldk, oldv, mask = inputs
    history = candidate.PackedHistory(oldk, oldv)
    for step in range(3):
      pk, pv = history.update(k[step], v[step])
      np.testing.assert_array_equal(pk.bitcast(dtypes.uint16).numpy(), _pack(oldk.cat(k[step], dim=2)).bitcast(dtypes.uint16).numpy())
      np.testing.assert_array_equal(pv.bitcast(dtypes.uint16).numpy(), _pack(oldv.cat(v[step], dim=2).transpose(-2, -1)).bitcast(dtypes.uint16).numpy())
    report['correctness'].append({'capacity': capacity, 'packed_buffer_bits_exact': True})
    save()

    # Restore a representative nonempty prefix before timing both graphs.
    timed_length = capacity-31
    inputs[5].assign(Tensor(np.where((np.arange(keys) < timed_length) | (np.arange(keys) >= capacity),
      0., -np.inf).astype(np.float32))).realize()
    for _, jit in jits:
      jit(*inputs)
    timings = {name: [] for name, _ in jits}
    for block in range(args.blocks):
      for name, jit in (jits if block % 2 == 0 else jits[::-1]):
        Device[Device.DEFAULT].synchronize()
        start = time.perf_counter()
        for _ in range(args.replays):
          jit(*inputs)
        Device[Device.DEFAULT].synchronize()
        timings[name].append((time.perf_counter()-start)*1000/args.replays)
    record = {'capacity': capacity, 'prefix_length': timed_length, 'three_attention_calls_ms': {
      name: {'median': float(np.median(values)), 'samples': values} for name, values in timings.items()},
      'scope': 'Three expert calls, fresh history packing once for candidate, action packing each step, all original matrix kernels.'}
    report['timing'].append(record)
    save()
    print(json.dumps(record), flush=True)
  return report


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--output', type=Path, required=True)
  parser.add_argument('--cpu-only', action='store_true')
  parser.add_argument('--capacities', type=lambda x: tuple(map(int, x.split(','))), default=(256, 2304))
  parser.add_argument('--replays', type=int, default=40)
  parser.add_argument('--blocks', type=int, default=5)
  args = parser.parse_args()
  assert all(x > 0 and x % 16 == 0 for x in args.capacities)
  assert args.replays > 0 and args.blocks > 0
  candidate = module('agent-attention-cachedpack')
  report = cpu_checks(candidate, args.capacities) if args.cpu_only else gpu_checks(args, candidate)
  if args.cpu_only:
    report['gfx1200_compilation'] = compile_checks(candidate, args.capacities)
  args.output.write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
  main()
