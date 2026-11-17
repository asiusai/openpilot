"""CPU compilation/semantics checks and parent-serialized gfx1200 GPU checks.

Use --compile-only without the USB wrapper. GPU execution requires the parent
wrapper --program. No module here changes normal model dispatch.
"""
import argparse
from collections import Counter
import ctypes
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import time

import numpy as np


def module(name):
  spec = importlib.util.spec_from_file_location(name.replace('-', '_'), Path(__file__).with_name(name+'.py'))
  result = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(result)
  return result


def configure_baseline():
  os.environ.update(ALPAMAYO_ONLINE_STAGE_KV='1', ALPAMAYO_ONLINE_WAVE_SYNC='0', ALPAMAYO_HIP_WGP='0',
    ALPAMAYO_ONLINE_NATIVE_EXP='0', ALPAMAYO_ONLINE_REGISTER_PROB='0', ALPAMAYO_ONLINE_PACKED_K='0',
    ALPAMAYO_ONLINE_WAVES='8', ALPAMAYO_ONLINE_BLOCK_KEYS='32', ALPAMAYO_ONLINE_KEY_PADDING='4',
    ALPAMAYO_ONLINE_PROB_PADDING='0', ALPAMAYO_ONLINE_LOCAL_FENCES='0', ALPAMAYO_ONLINE_DPP='0',
    ALPAMAYO_ONLINE_KV_LOAD='1', ALPAMAYO_ONLINE_VALUE_LAYOUT='1', ALPAMAYO_ONLINE_VALUE_MIN_ROWS='1024',
    ALPAMAYO_ONLINE_PROBABILITY_MMA='1')


def transpose_cpu(x):
  """Emulate the kernel's wave shuffle operations, independent of WMMA."""
  lane = np.arange(32)
  x = x.copy()
  for bit in (1, 2, 4):
    for j in range(8):
      if j & bit:
        continue
      a, b = x[:, j].copy(), x[:, j | bit].copy()
      exchanged = np.where(lane & bit, a, b)[lane ^ bit]
      x[:, j] = np.where(lane & bit, exchanged, a)
      x[:, j | bit] = np.where(lane & bit, b, exchanged)
  return x[(lane & 7) | ((lane & 8) << 1) | ((lane & 16) >> 1)]


def cpu_semantics():
  lane = np.arange(32)
  # Labels prove the complete 16x16 transpose, including both wave halves.
  labels = ((lane[:, None] // 16)*8+np.arange(8))*16+lane[:, None] % 16
  expected = (lane[:, None] % 16)*16+(lane[:, None] // 16)*8+np.arange(8)
  np.testing.assert_array_equal(transpose_cpu(labels), expected)
  for width in (72, 128):
    padded = (width+15)//16*16
    staged = np.empty(32*padded, dtype=np.int32)
    for row in range(32):
      for dim in range(0, padded, 8):
        for i in range(8):
          target = ((dim//16)*2+row//16)*256+(row%16//8)*128+((dim+i)%16)*8+row%8
          staged[target] = row*width+dim+i if dim+i < width else -1
    for dim_tile in range(padded//16):
      for sub in range(2):
        for lane_id in range(32):
          for i in range(8):
            key, dim = sub*16+(lane_id//16)*8+i, dim_tile*16+lane_id%16
            expected_value = key*width+dim if dim < width else -1
            assert staged[(dim_tile*2+sub)*256+lane_id*8+i] == expected_value
  rng = np.random.default_rng(906016)
  for _ in range(1000):
    # Scores are exponentials: nonnegative values across a wide dynamic range.
    probs = np.exp(rng.uniform(-100, 0, size=(2, 32, 8)).astype(np.float32)).astype(np.float32)
    probs[rng.random(probs.shape) < .1] = 0
    original = probs[0]+probs[1]
    for bit in (8, 4, 2, 1):
      original = original+original[lane ^ bit]
    rows = transpose_cpu(probs[0])+transpose_cpu(probs[1])
    rows = rows+rows[lane ^ 16]
    rows[:, :4] += rows[:, 4:]
    rows[:, :2] += rows[:, 2:4]
    reduced = rows[:, 0]+rows[:, 1]
    broadcast = reduced[(lane[:, None] // 16)*8+np.arange(8)]
    np.testing.assert_array_equal(original.view(np.uint32), broadcast.view(np.uint32))
  return {'transpose_labels_exact': True, 'staged_values_labels_exact': [72, 128],
    'sum_tree_random_cases_exact': 1000, 'gpu_executed': False}


def disassemble(binary):
  # COMGR supports gfx1200; the host llvm-objdump 18 does not decode this ISA.
  from tinygrad.runtime.support.compiler_amd import comgr, check, _get_comgr_data
  check(comgr.amd_comgr_create_action_info(ctypes.byref(info := comgr.amd_comgr_action_info_t())))
  check(comgr.amd_comgr_action_info_set_isa_name(info, b'amdgcn-amd-amdhsa--gfx1200'))
  check(comgr.amd_comgr_create_data_set(ctypes.byref(source := comgr.amd_comgr_data_set_t())))
  check(comgr.amd_comgr_create_data_set(ctypes.byref(dest := comgr.amd_comgr_data_set_t())))
  check(comgr.amd_comgr_create_data(comgr.AMD_COMGR_DATA_KIND_EXECUTABLE, ctypes.byref(data := comgr.amd_comgr_data_t())))
  try:
    check(comgr.amd_comgr_set_data(data, len(binary), binary))
    check(comgr.amd_comgr_set_data_name(data, b'agent-attention.hsaco'))
    check(comgr.amd_comgr_data_set_add(source, data))
    check(comgr.amd_comgr_do_action(comgr.AMD_COMGR_ACTION_DISASSEMBLE_EXECUTABLE_TO_SOURCE, info, source, dest))
    return _get_comgr_data(dest, comgr.AMD_COMGR_DATA_KIND_SOURCE).decode()
  finally:
    check(comgr.amd_comgr_release_data(data))
    check(comgr.amd_comgr_destroy_data_set(source))
    check(comgr.amd_comgr_destroy_data_set(dest))
    check(comgr.amd_comgr_destroy_action_info(info))


def compile_checks(args, candidate):
  results = []
  for width, rows in ((72, 720), (128, 2027), (128, 2112)):
    for variant, waves in [('baseline', 8)]+[(v, w) for v in args.variants.split(',') for w in args.waves]:
      if variant == 'baseline':
        text, binary = candidate.baseline_module().compiled(16, 16 if width == 72 else 8, rows, rows, width == 128,
          waves=8, width=width, block_keys=32, stage_kv=True, key_padding=4, kv_load=1,
          value_layout=width == 128, probability_mma=True)
      else:
        text, binary = candidate.compiled(16, 16 if width == 72 else 8, rows, rows, width == 128,
          width, waves, width == 128 and variant not in ('stage', 'shuffle-stage'), variant)
      metadata = subprocess.check_output(['/usr/lib/llvm-18/bin/llvm-readelf', '--notes', '-'], input=binary).decode()
      resources = {key: int(value) for key, value in re.findall(
        r'\.(group_segment_fixed_size|private_segment_fixed_size|sgpr_count|sgpr_spill_count|vgpr_count|vgpr_spill_count):\s*(\d+)', metadata)}
      asm = disassemble(binary)
      counts = Counter(re.findall(r'^\s*((?:[sv]_|ds_|global_|buffer_)[a-z0-9_]+)', asm, re.M))
      row = dict(rows=rows, width=width, variant=variant, waves=waves, binary_bytes=len(binary), resources=resources,
        static_instruction_count=sum(counts.values()), instructions=dict(counts))
      if args.artifacts:
        args.artifacts.mkdir(exist_ok=True, parents=True)
        prefix = args.artifacts/f'agent-attention-{width}-{rows}-{variant}-{waves}'
        prefix.with_suffix('.cpp').write_text(text)
        prefix.with_suffix('.hsaco').write_bytes(binary)
        prefix.with_suffix('.asm').write_text(asm)
        prefix.with_suffix('.metadata').write_text(metadata)
      results.append(row)
      print(json.dumps({k: v for k, v in row.items() if k != 'instructions'}), flush=True)
  return results


def gpu_checks(args, candidate):
  import torch
  from tinygrad import Device, Tensor, TinyJit, dtypes
  from tinygrad.device import Compiled
  torch.set_num_threads(4)
  Compiled.wait_timeout_ms = 10000
  module('prefill-kernels').fix_dispatch_packet()
  baseline = candidate.baseline_module().attention
  rng = np.random.default_rng(906016)
  variants = [(v, w) for v in args.variants.split(',') for w in args.waves]
  # Every control is separate from timing. Cases cover tails, GQA, batches,
  # causal prefixes, large logits, low logits, and uniform probabilities.
  cases = [
    (2, 4, 2, 17, 33, 72, False, 1.), (2, 4, 2, 33, 33, 128, True, 1.),
    (1, 4, 4, 129, 129, 72, False, 4.), (1, 4, 2, 65, 97, 128, True, .1),
    (1, 2, 1, 17, 33, 72, False, 0.), (1, 2, 1, 17, 33, 128, True, 8.),
  ]
  targets = []
  if args.target in ('vision', 'all'):
    targets += [(2, 16, 16, 720, 720, 72, False, 1.)]
  if args.target in ('prefill', 'all'):
    targets += [(1, 16, 8, 2027, 2027, 128, True, 1.), (1, 16, 8, 2112, 2112, 128, True, 1.)]
  correctness, timings = [], []
  def tensor(shape, scale=1.):
    return Tensor((rng.standard_normal(shape)*scale).astype(np.float32)).cast(dtypes.bfloat16).realize()
  for batch, heads, kv_heads, rows, keys, width, causal, scale in cases+targets:
    q = tensor((batch, heads, rows, width), scale)
    k, v = [tensor((batch, kv_heads, keys, width), scale if i == 0 else 1.) for i in range(2)]
    qt, kt, vt = [torch.from_numpy(t.float().numpy()) for t in (q, k, v)]
    truth = torch.nn.functional.scaled_dot_product_attention(qt, kt.repeat_interleave(heads//kv_heads, 1),
      vt.repeat_interleave(heads//kv_heads, 1), is_causal=causal).numpy()
    expected = baseline(q, k, v, causal).realize()
    expected_bits = expected.bitcast(dtypes.uint16).numpy()
    expected_float = expected.float().numpy()
    baseline_rmse = float(np.sqrt(np.mean((expected_float-truth)**2)))
    for variant, waves in variants:
      jit = TinyJit(lambda a, b, c: candidate.attention(a, b, c, causal, waves=waves, transpose=variant).realize())
      for _ in range(3):
        output = jit(q, k, v)
      actual = output.float().numpy()
      bits = output.bitcast(dtypes.uint16).numpy()
      row = dict(shape=[batch, heads, kv_heads, rows, keys, width], causal=causal, input_scale=scale,
        variant=variant, waves=waves, exact=bool(np.array_equal(bits, expected_bits)),
        max_delta=float(np.max(np.abs(actual-expected_float))), baseline_rmse=baseline_rmse,
        rmse=float(np.sqrt(np.mean((actual-truth)**2))), mismatched_bf16=int(np.count_nonzero(bits != expected_bits)))
      correctness.append(row)
      args.output.write_text(json.dumps({'correctness': correctness, 'timings': timings}, indent=2)+'\n')
      print(json.dumps(row), flush=True)
      assert np.isfinite(actual).all()
      np.testing.assert_array_equal(bits, expected_bits, err_msg=str(row))
      # Replay fresh contents through the captured graph. For causal attention,
      # mutate only future K/V and check that earlier outputs cannot change.
      if rows < 200:
        before = actual.copy()
        split = min(rows, keys)//2
        changed_k, changed_v = [t.float().numpy() for t in (k, v)]
        changed_k[:, :, split:] *= -3
        changed_v[:, :, split:] += 17
        k.assign(Tensor(changed_k).cast(dtypes.bfloat16)).realize()
        v.assign(Tensor(changed_v).cast(dtypes.bfloat16)).realize()
        updated = jit(q, k, v).bitcast(dtypes.uint16).numpy()
        updated_reference = baseline(q, k, v, causal).bitcast(dtypes.uint16).numpy()
        np.testing.assert_array_equal(updated, updated_reference)
        if causal:
          np.testing.assert_array_equal(jit(q, k, v).float().numpy()[:, :, :split], before[:, :, :split])
        row['mutation_replay_exact'] = True
        row['causal_future_invariance'] = causal
        # Restore inputs before testing another implementation.
        k.assign(Tensor(kt.numpy()).cast(dtypes.bfloat16)).realize()
        v.assign(Tensor(vt.numpy()).cast(dtypes.bfloat16)).realize()
    if (batch, heads, kv_heads, rows, keys, width, causal, scale) not in targets:
      continue
    funcs = [('baseline', lambda a, b, c: baseline(a, b, c, causal))]
    funcs += [(f'{variant}-{waves}', lambda a, b, c, variant=variant, waves=waves:
      candidate.attention(a, b, c, causal, waves=waves, transpose=variant)) for variant, waves in variants]
    jits = []
    for name, fn in funcs:
      def chain(a, b, c, fn=fn):
        for _ in range(args.chain):
          a = fn(a, b, c).realize()
        return a
      jit = TinyJit(chain)
      for _ in range(4):
        jit(q, k, v)
      jits.append((name, jit))
    collected = {name: [] for name, _ in jits}
    for block in range(args.blocks):
      for name, jit in (jits if block % 2 == 0 else jits[::-1]):
        Device[Device.DEFAULT].synchronize()
        start = time.perf_counter()
        for _ in range(args.replays):
          jit(q, k, v)
        Device[Device.DEFAULT].synchronize()
        collected[name].append((time.perf_counter()-start)*1000/(args.replays*args.chain))
    timing = dict(shape=[batch, heads, kv_heads, rows, keys, width], causal=causal,
      chain=args.chain, replays=args.replays, blocks=args.blocks,
      ms_per_attention={name: {'median': float(np.median(values)), 'samples': values} for name, values in collected.items()})
    timings.append(timing)
    print(json.dumps(timing), flush=True)
    args.output.write_text(json.dumps({'correctness': correctness, 'timings': timings}, indent=2)+'\n')
  return {'correctness': correctness, 'timings': timings, 'gpu_executed': True,
    'timing_scope': 'Amortized host graph submission and GPU completion. Includes contiguous copies and value packing.'}


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--output', required=True, type=Path)
  parser.add_argument('--compile-only', action='store_true')
  parser.add_argument('--artifacts', type=Path)
  parser.add_argument('--variants', default='shuffle')
  parser.add_argument('--waves', nargs='+', type=int, choices=(4, 8), default=[8])
  parser.add_argument('--target', choices=('vision', 'prefill', 'all'), default='all')
  parser.add_argument('--chain', type=int, default=4)
  parser.add_argument('--replays', type=int, default=30)
  parser.add_argument('--blocks', type=int, default=5)
  args = parser.parse_args()
  assert set(args.variants.split(',')) <= {'shuffle', 'lds', 'stage', 'shuffle-stage'}
  assert args.chain > 0 and args.replays > 0 and args.blocks > 0
  configure_baseline()
  candidate = module('agent-attention-rowsoftmax')
  report = {'cpu_semantics': cpu_semantics()}
  report.update({'compilation': compile_checks(args, candidate)} if args.compile_only else gpu_checks(args, candidate))
  args.output.write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
  main()
