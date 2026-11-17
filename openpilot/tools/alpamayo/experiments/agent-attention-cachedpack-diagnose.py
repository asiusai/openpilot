"""CPU-only evidence for the separate shuffle-attention numerical failure.

This does not change or execute any GPU prototype. The cachedpack prefix
follows this agent's current file ownership boundary.
"""
import argparse
import ctypes
import importlib.util
import json
from pathlib import Path
import re

import numpy as np


def load(name):
  spec = importlib.util.spec_from_file_location(name.replace('-', '_'), Path(__file__).with_name(name+'.py'))
  result = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(result)
  return result


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--output', type=Path, required=True)
  args = parser.parse_args()
  from tinygrad import Device
  assert Device.DEFAULT.startswith('CPU')
  online = load('online-attention')
  candidate = load('agent-attention-rowsoftmax')
  disassemble = load('agent-attention-check').disassemble
  compiled = []
  for rows, keys in ((17, 33), (129, 129), (720, 720)):
    for variant in ('baseline', 'stage', 'shuffle-stage'):
      if variant == 'baseline':
        _, binary = online.compiled(16, 16, rows, keys, False, waves=8, width=72, block_keys=32,
          stage_kv=True, key_padding=4, kv_load=1, value_layout=False, probability_mma=True)
      else:
        _, binary = candidate.compiled(16, 16, rows, keys, False, 72, 8, False, variant)
      asm = disassemble(binary)
      instructions = [line.strip() for line in asm.splitlines() if re.search(r'v_fma_f32 .*0x3df15bef', line)]
      compiled.append({'rows': rows, 'keys': keys, 'variant': variant,
        'score_scale_minus_maximum_fma_count': len(instructions), 'instructions': instructions})

  # These are finite FP32 operands. The maximum is a rounded scaled score.
  qk = np.float32(-15.8331298828125)
  scale = np.float32(72**-.5)
  maximum = np.float32(4.402106761932373)
  log2e = np.float32(1.4426950408889634)
  fmaf = ctypes.CDLL('libm.so.6').fmaf
  fmaf.argtypes, fmaf.restype = [ctypes.c_float]*3, ctypes.c_float
  fused = np.float32(fmaf(float(qk), float(scale), -float(maximum)))
  separated = np.float32(np.float32(qk*scale)-maximum)
  # CPU exp2 demonstrates that the difference can cross a BF16 threshold.
  # It is independent of the GPU exp2 approximation, not a device replay.
  probabilities = np.exp2(np.array([fused, separated], dtype=np.float32)*log2e).astype(np.float32)
  bits = probabilities.view(np.uint32)
  bf16 = ((bits+0x7fff+((bits>>16)&1)) >> 16).astype(np.uint16)
  assert fused != separated and bf16[0] != bf16[1]
  example = {'qk': float(qk), 'scale': float(scale), 'maximum': float(maximum),
    'fused_difference': float(fused), 'separated_difference': float(separated),
    'fused_probability': float(probabilities[0]), 'separated_probability': float(probabilities[1]),
    'fused_probability_bf16': hex(int(bf16[0])), 'separated_probability_bf16': hex(int(bf16[1]))}
  result = {'gpu_executed': False, 'compilation': compiled, 'cpu_counterexample': example,
    'finding': 'Moving scaled scores through wave permutations prevents baseline score-scale/subtraction FMA contraction.',
    'limit': 'Concrete arithmetic difference established. GPU attribution of all 178 output mismatches remains uninstrumented.'}
  args.output.write_text(json.dumps(result, indent=2)+'\n')
  print(json.dumps(result, indent=2))


if __name__ == '__main__':
  main()
