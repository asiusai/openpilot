"""Bounded checks for shuffle-fma; GPU execution belongs to the parent wrapper.

CPU mode checks the arithmetic counterexample, complete online recurrences,
and actual gfx1200 assembly. GPU mode reuses the original strict BF16 gate.
"""
import argparse
from collections import Counter
import ctypes
import importlib.util
import json
from pathlib import Path
import re
import subprocess

import numpy as np


def load(name):
  spec = importlib.util.spec_from_file_location(name.replace('-', '_'), Path(__file__).with_name(name+'.py'))
  result = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(result)
  return result


helpers = load('agent-attention-check')
candidate = load('agent-attention-rowsoftmax')
fmaf = ctypes.CDLL('libm.so.6').fmaf
fmaf.argtypes, fmaf.restype = [ctypes.c_float]*3, ctypes.c_float


def fma_array(a, b, c):
  a, b, c = np.broadcast_arrays(a, b, c)
  return np.fromiter((fmaf(float(x), float(y), float(z)) for x, y, z in zip(a.flat, b.flat, c.flat)),
    dtype=np.float32).reshape(a.shape)


def bf16_bits(x):
  bits = np.asarray(x, dtype=np.float32).view(np.uint32)
  return ((bits+0x7fff+((bits>>16)&1)) >> 16).astype(np.uint16)


def counterexample():
  raw, scale, maximum = np.float32(-15.8331298828125), np.float32(72**-.5), np.float32(4.402106761932373)
  fused = np.float32(fmaf(float(raw), float(scale), -float(maximum)))
  separated = np.float32(np.float32(raw*scale)-maximum)
  assert fused == np.float32(-6.268059253692627) and separated == np.float32(-6.268058776855469)
  probabilities = np.exp2(np.array([fused, separated], dtype=np.float32)*np.float32(1.4426950408889634))
  bits = bf16_bits(probabilities)
  np.testing.assert_array_equal(bits, np.array([0x3af8, 0x3af9], dtype=np.uint16))
  return dict(raw=float(raw), scale=float(scale), maximum=float(maximum), fused_difference=float(fused),
    separated_difference=float(separated), fused_probability_bf16=hex(int(bits[0])),
    separated_probability_bf16=hex(int(bits[1])), cpu_exp2_only=True)


def recurrence_checks():
  """Compare baseline layout and row layout, including masks and FP32 state.

  CPU exp2 is shared by both emulations. It does not emulate GPU exp2 error.
  Check every intermediate probability, BF16 probability, maximum, rescale,
  and denominator bit. The independent original sum-tree check also runs.
  """
  rng = np.random.default_rng(906072)
  lanes = np.arange(32)
  register = np.arange(8)
  log2e = np.float32(1.4426950408889634)
  transpose = helpers.transpose_cpu
  checked_tiles = 0
  for width, keys, causal in ((72, 33, False), (72, 47, False), (72, 48, False), (72, 49, False),
      (72, 63, False), (72, 64, False), (72, 65, False), (72, 720, False), (128, 2027, True), (128, 2112, True)):
    scale = np.float32(width**-.5)
    fused_fragments = candidate.score_fma_fragments(keys, causal)
    for query_start in (0, (keys//32)*32):
      oldmax = np.full((32, 8), -np.inf, dtype=np.float32)
      olddenom = np.zeros((32, 8), dtype=np.float32)
      newmax = np.full(32, -np.inf, dtype=np.float32)
      newdenom = np.zeros(32, dtype=np.float32)
      row = query_start+(lanes[:, None]//16)*8+register
      # Mimic the causal group's bounded key loop. Padded queries still execute.
      end = min(keys, ((query_start//128)+1)*128) if causal else keys
      for start in range(0, end, 32):
        raw = (rng.standard_normal((2, 32, 8))*8).astype(np.float32)
        scaled = np.empty_like(raw)
        for sub in range(2):
          col = start+sub*16+lanes[:, None]%16
          valid = (col < keys) & ((col <= row) if causal else True)
          scaled[sub] = np.where(valid, raw[sub]*scale, -np.inf)
        oldlocal = np.maximum(scaled[0], scaled[1])
        for bit in (8, 4, 2, 1):
          oldlocal = np.maximum(oldlocal, oldlocal[lanes ^ bit])
        oldnext = np.maximum(oldmax, oldlocal)
        oldrescale = np.exp2((oldmax-oldnext)*log2e)
        oldprob = np.empty_like(raw)
        for sub in range(2):
          delta = fma_array(raw[sub], scale, -oldnext) if fused_fragments[sub] else scaled[sub]-oldnext
          oldprob[sub] = np.exp2(delta*log2e)
        oldsum = oldprob[0]+oldprob[1]
        for bit in (8, 4, 2, 1):
          oldsum = oldsum+oldsum[lanes ^ bit]
        olddenom = fma_array(olddenom, oldrescale, oldsum)
        oldmax = oldnext

        moved, retained = [], []
        for sub in range(2):
          value = transpose(raw[sub] if fused_fragments[sub] else scaled[sub])
          retained.append(value)
          moved.append(value*scale if fused_fragments[sub] else value)
        newlocal = np.maximum(moved[0], moved[1]).max(axis=1)
        newnext = np.maximum(newmax, np.maximum(newlocal, newlocal[lanes ^ 16]))
        newrescale = np.exp2((newmax-newnext)*log2e)
        newprob = np.empty_like(raw)
        for sub in range(2):
          delta = (fma_array(retained[sub], scale, -newnext[:, None]) if fused_fragments[sub]
                   else moved[sub]-newnext[:, None])
          newprob[sub] = np.exp2(delta*log2e)
          np.testing.assert_array_equal(newprob[sub].view(np.uint32), transpose(oldprob[sub]).view(np.uint32))
          np.testing.assert_array_equal(bf16_bits(newprob[sub]), transpose(bf16_bits(oldprob[sub])))
        newsum = newprob[0]+newprob[1]
        newsum = newsum+newsum[lanes ^ 16]
        newsum[:, :4] += newsum[:, 4:]
        newsum[:, :2] += newsum[:, 2:4]
        newdenom = fma_array(newdenom, newrescale, newsum[:, 0]+newsum[:, 1])
        newmax = newnext
        for old, new in ((oldmax, newmax), (olddenom, newdenom), (oldrescale, newrescale)):
          broadcast = new[(lanes[:, None]//16)*8+register]
          np.testing.assert_array_equal(old.view(np.uint32), broadcast.view(np.uint32))
        checked_tiles += 1
  return dict(online_tiles_exact=checked_tiles, checked_state=['FP32 probabilities', 'BF16 probabilities',
    'maximum', 'accumulator rescale', 'FMA denominator'], gpu_executed=False)


def compile_checks(args):
  from tinygrad import Device
  assert Device.DEFAULT.startswith('CPU'), 'Compile-only checks require DEV=CPU'
  # Real shapes, original small controls, and both sides of fragment boundaries.
  shapes = [(72, 720, 720, False), (128, 2027, 2027, True), (128, 2112, 2112, True),
    (72, 17, 33, False), (128, 33, 33, True), (72, 129, 129, False), (128, 65, 97, True),
    (128, 17, 33, True)] + [(72, keys, keys, False) for keys in (47, 48, 49, 63, 64, 65)]
  result = []
  for width, rows, keys, causal in shapes:
    fused = candidate.score_fma_fragments(keys, causal)
    scale_hex = hex(int(np.array(width**-.5, dtype=np.float32).view(np.uint32)))
    for variant in ('baseline', 'shuffle-fma'):
      value_layout = rows >= 1024
      if variant == 'baseline':
        text, binary = candidate.baseline_module().compiled(16, 8 if causal else 16, rows, keys, causal,
          waves=8, width=width, block_keys=32, stage_kv=True, key_padding=4, kv_load=1,
          value_layout=value_layout, probability_mma=True)
      else:
        text, binary = candidate.compiled(16, 8 if causal else 16, rows, keys, causal, width, 8, value_layout, variant)
      asm = helpers.disassemble(binary)
      scale_fmas = [line.strip() for line in asm.splitlines() if 'v_fma_f32' in line and scale_hex in line]
      assert len(scale_fmas) == 8*sum(fused), (variant, width, rows, keys, causal, scale_fmas)
      metadata = subprocess.check_output(['/usr/lib/llvm-18/bin/llvm-readelf', '--notes', '-'], input=binary).decode()
      resources = {key: int(value) for key, value in re.findall(
        r'\.(group_segment_fixed_size|private_segment_fixed_size|sgpr_count|sgpr_spill_count|vgpr_count|vgpr_spill_count):\s*(\d+)', metadata)}
      counts = Counter(re.findall(r'^\s*((?:[sv]_|ds_|global_|buffer_)[a-z0-9_]+)', asm, re.M))
      row = dict(width=width, rows=rows, keys=keys, causal=causal, variant=variant,
        fused_fragments=fused, score_scale_minus_maximum_fma_count=len(scale_fmas),
        score_scale_minus_maximum_fma_instructions=scale_fmas, resources=resources,
        static_instruction_count=sum(counts.values()))
      if args.artifacts:
        args.artifacts.mkdir(parents=True, exist_ok=True)
        prefix = args.artifacts/f'agent-attention-rowsoftmax-{width}-{rows}-{keys}-{variant}'
        prefix.with_suffix('.cpp').write_text(text)
        prefix.with_suffix('.asm').write_text(asm)
        prefix.with_suffix('.metadata').write_text(metadata)
      result.append(row)
      print(json.dumps({k: v for k, v in row.items() if k != 'score_scale_minus_maximum_fma_instructions'}), flush=True)
  return result


def fusion_replay_checks(args):
  """Extra exact replay gates exercise one and two contracted fragments."""
  from tinygrad import Tensor, TinyJit, dtypes
  from tinygrad.device import Compiled
  Compiled.wait_timeout_ms = 10000
  helpers.module('prefill-kernels').fix_dispatch_packet()
  rng = np.random.default_rng(906073)
  results = []
  for keys in (48, 64, 720):
    def tensor(heads, scale):
      return Tensor((rng.standard_normal((1, heads, keys, 72))*scale).astype(np.float32)).cast(dtypes.bfloat16).realize()
    q, k, v = tensor(2, 4.), tensor(1, 4.), tensor(1, 1.)
    jit = TinyJit(lambda a, b, c: candidate.attention(a, b, c, False, transpose='shuffle-fma').realize())
    for _ in range(3):
      jit(q, k, v)
    for step in range(3):
      if step:
        # Change every Q/K/V element in the existing captured buffers.
        for target, heads, scale in ((q, 2, 1.), (k, 1, 4.), (v, 1, 1.)):
          target.assign(tensor(heads, scale)).realize()
      expected = candidate.baseline_module().attention(q, k, v, False).bitcast(dtypes.uint16).numpy()
      actual = jit(q, k, v).bitcast(dtypes.uint16).numpy()
      row = dict(keys=keys, step=step, fused_fragments=candidate.score_fma_fragments(keys, False),
        exact=bool(np.array_equal(actual, expected)), mismatched_bf16=int(np.count_nonzero(actual != expected)))
      results.append(row)
      print(json.dumps({'fusion_replay': row}), flush=True)
      args.output.write_text(json.dumps({'fusion_replay': results}, indent=2)+'\n')
      np.testing.assert_array_equal(actual, expected, err_msg=str(row))
  return results


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--output', required=True, type=Path)
  parser.add_argument('--compile-only', action='store_true')
  parser.add_argument('--artifacts', type=Path)
  parser.add_argument('--target', choices=('vision', 'prefill', 'all'), default='all')
  parser.add_argument('--chain', type=int, default=4)
  parser.add_argument('--replays', type=int, default=30)
  parser.add_argument('--blocks', type=int, default=5)
  args = parser.parse_args()
  assert 1 <= args.chain <= 4 and 1 <= args.replays <= 40 and 1 <= args.blocks <= 5
  args.variants, args.waves = 'shuffle-fma', [8]
  helpers.configure_baseline()
  report = dict(variant='shuffle-fma', counterexample=counterexample(),
    cpu_semantics=helpers.cpu_semantics(), cpu_recurrence=recurrence_checks())
  if args.compile_only:
    report.update(compilation=compile_checks(args), gpu_executed=False)
  else:
    report['fusion_replay'] = fusion_replay_checks(args)
    # Original six controls and all requested real shapes, with unchanged seed,
    # BF16 bit equality, mutation/causal gates, independent RMSE, and timing.
    report.update(helpers.gpu_checks(args, candidate))
  args.output.write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
  main()
