"""Independent CPU FP32 oracle and immutable BF16 fixtures for one actual shape.

No tinygrad, torch, HIP, GPU, model, or device import occurs here.
"""
import hashlib
import json
import os
from pathlib import Path
import time

os.environ.update(OPENBLAS_NUM_THREADS='4', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4')
import numpy as np

ROOT = Path(__file__).with_name('agent-ck-attention-v2')
SHAPE = {'q': (1, 16, 2048, 128), 'k': (1, 8, 2048, 128), 'v': (1, 8, 2048, 128)}

# Exploratory kernel gates, declared before any candidate GPU result.
# These do not replace route accuracy gates or constitute a safety claim.
LIMITS = {
  'output_rmse': {'baseline_multiplier': 1.10, 'absolute_floor': 2e-5},
  'output_max_abs': {'baseline_multiplier': 1.25, 'absolute_floor': 0.002},
  'per_head_rmse': {'baseline_multiplier': 1.20, 'absolute_floor': 3e-5},
  'lse_rmse_abs': 0.001,
  'lse_max_abs': 0.005,
  'mutation_min_output_max_abs_change': 1e-4,
  'finite_fraction': 1.0,
  'unchanged_causal_prefix': 'bitwise',
  'ordered_replay_same_input': 'bitwise against first candidate result for the same fixture',
  'redzones': 'bitwise unchanged',
  'output_poison': 'all elements overwritten with finite values',
  'speed_screen': {'both_boundary_median_ratio_max': 0.95, 'both_boundary_p95_ratio_max': 1.05},
}


def bf16_bits(x):
  bits = np.asarray(x, dtype=np.float32).view(np.uint32)
  return ((bits + np.uint32(0x7fff) + ((bits >> 16) & 1)) >> 16).astype(np.uint16)


def fp32(bits):
  return (bits.astype(np.uint32) << 16).view(np.float32)


def reference(inputs):
  q, k, v = (fp32(inputs[n]) for n in ('q', 'k', 'v'))
  out = np.empty(SHAPE['q'], dtype=np.float32)
  lse = np.empty((1, 16, 2048), dtype=np.float32)
  keys = np.arange(2048)[None, :]
  scale = np.float32(128**-0.5)
  for head in range(16):
    kh = head//2
    for begin in range(0, 2048, 128):
      end = begin+128
      scores = (q[0, head, begin:end] @ k[0, kh].T) * scale
      scores[keys > np.arange(begin, end)[:, None]] = -np.inf
      maximum = np.max(scores, axis=-1, keepdims=True)
      probability = np.exp(scores-maximum)
      total = np.sum(probability, axis=-1, keepdims=True, dtype=np.float32)
      probability /= total
      out[0, head, begin:end] = probability @ v[0, kh]
      lse[0, head, begin:end] = (maximum+np.log(total))[:, 0]
  assert np.isfinite(out).all() and np.isfinite(lse).all()
  return out, lse


def fixtures():
  rng = np.random.default_rng(202611170128)
  a = {n: bf16_bits(rng.standard_normal(shape, dtype=np.float32)) for n, shape in SHAPE.items()}
  b = {n: bf16_bits(rng.standard_normal(shape, dtype=np.float32)) for n, shape in SHAPE.items()}
  yield 'a', a
  yield 'b', b
  for name in ('q', 'k', 'v'):
    changed = {n: x.copy() for n, x in a.items()}
    x = fp32(changed[name])
    if name == 'q': x[0, 3, 768:800, 32:64] += np.float32(0.5)
    elif name == 'k': x[0, 5, 512:544, 8:40] -= np.float32(0.25)
    else: x[0, 2, 128:160, :] += np.float32(0.5)
    changed[name] = bf16_bits(x)
    yield f'mutate_{name}', changed
  changed = {n: x.copy() for n, x in a.items()}
  for n in ('k', 'v'):
    changed[n][:, :, 1024:] = bf16_bits(rng.standard_normal((1, 8, 1024, 128), dtype=np.float32))
  yield 'causal_future', changed
  gqa = {n: x.copy() for n, x in a.items()}
  for head in range(0, 16, 2): gqa['q'][:, head+1] = gqa['q'][:, head]
  gqa['k'].fill(0)
  for head in range(8):
    constant = np.broadcast_to(np.float32((head+1)/8), (2048, 128))
    gqa['v'][0, head] = bf16_bits(constant)
  yield 'gqa', gqa


def main():
  target = ROOT/'oracle-r2'
  target.mkdir(exist_ok=False)
  manifest = {'shape': SHAPE, 'dtype': 'BF16 inputs encoded as uint16; FP32 independent references',
              'numpy_version': np.__version__, 'gpu_executed': False,
              'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'limits': LIMITS, 'fixtures': [],
              'replay_order': ['a', 'b', 'mutate_q', 'a', 'mutate_k', 'b', 'mutate_v', 'causal_future']*5}
  baseline_out = None
  started = time.monotonic()
  for name, inputs in fixtures():
    then = time.monotonic()
    folder = target/name
    folder.mkdir()
    out, lse = reference(inputs)
    if name == 'a': baseline_out = out.copy()
    checks = {'finite': True}
    if name.startswith('mutate_'):
      checks['max_output_change'] = float(np.max(np.abs(out-baseline_out)))
      assert checks['max_output_change'] > LIMITS['mutation_min_output_max_abs_change']
    if name == 'causal_future':
      checks['prefix_bitwise_equal'] = np.array_equal(out[:, :, :1024], baseline_out[:, :, :1024])
      assert checks['prefix_bitwise_equal']
      checks['suffix_changed'] = not np.array_equal(out[:, :, 1024:], baseline_out[:, :, 1024:])
      assert checks['suffix_changed']
    if name == 'gqa':
      checks['paired_outputs_equal'] = all(np.array_equal(out[:, h], out[:, h+1]) for h in range(0, 16, 2))
      checks['max_constant_error'] = float(max(np.max(np.abs(out[0, h]-np.float32((h//2+1)/8))) for h in range(16)))
      exact = np.broadcast_to(((np.arange(16)//2+1)/8).astype(np.float32)[None, :, None, None], SHAPE['q']).copy()
      checks['bf16_matches_analytic_constant'] = np.array_equal(bf16_bits(out), bf16_bits(exact))
      assert checks['paired_outputs_equal'] and checks['bf16_matches_analytic_constant']
      checks['reference_method'] = 'Analytic constant PV and log(causal key count); independently checked by FP32 matmul.'
      out = exact
      lse = np.broadcast_to(np.log(np.arange(1, 2049, dtype=np.float32))[None, None, :], (1, 16, 2048)).copy()
    files = {}
    for key, array in {**inputs, 'output_fp32': out, 'lse_fp32': lse}.items():
      path = folder/f'{key}.npy'
      np.save(path, array, allow_pickle=False)
      files[key] = {'path': str(path.relative_to(target)), 'shape': array.shape, 'dtype': str(array.dtype),
                    'sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'bytes': path.stat().st_size}
    record = {'name': name, 'files': files, 'checks': checks, 'cpu_seconds': time.monotonic()-then}
    manifest['fixtures'].append(record)
    print(json.dumps({'name': name, 'checks': checks, 'cpu_seconds': record['cpu_seconds']}), flush=True)
  manifest['cpu_seconds'] = time.monotonic()-started
  (target/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')


if __name__ == '__main__':
  main()
