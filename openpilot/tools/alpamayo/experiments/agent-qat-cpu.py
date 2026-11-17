#!/usr/bin/env python3
"""CPU-only K32 quantization math and PTQ screening. This file never imports a GPU runtime.

Commands: self-test; inventory CHECKPOINT; checkpoint-probe CHECKPOINT;
          screen CAPTURE.npz [--nonnegative]
Capture keys: W [N,K], X_cal [M,K], X_eval [T,K]. Optional string provenance.
Calibration fits transforms and clips. Evaluation rows never select parameters.
This is a linear surrogate, not model, trajectory, QAT, or GPU validation.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import struct

import numpy as np


def safetensor_headers(checkpoint):
  result = {}
  for shard in sorted(Path(checkpoint).glob('*.safetensors')):
    with shard.open('rb') as stream:
      size = struct.unpack('<Q', stream.read(8))[0]
      header = json.loads(stream.read(size))
    for name, info in header.items():
      if name != '__metadata__':
        if name in result:
          raise ValueError(f'Duplicate checkpoint tensor: {name}')
        result[name] = dict(info, shard=str(shard), payload_offset=8 + size)
  if not result:
    raise ValueError('No local safetensors found')
  return result


def read_weight_rows(info, rows):
  """Read only a contiguous output-row subset, retaining every input channel."""
  shape = info['shape']
  if len(shape) != 2:
    raise ValueError('Expected a matrix')
  count = min(rows, shape[0]) * shape[1]
  offset = info['payload_offset'] + info['data_offsets'][0]
  if info['dtype'] == 'BF16':
    words = np.memmap(info['shard'], mode='r', dtype='<u2', offset=offset, shape=(count,))
    values = (np.asarray(words).astype(np.uint32) << 16).view(np.float32)
  elif info['dtype'] in ('F32', 'F16'):
    values = np.asarray(np.memmap(info['shard'], mode='r',
                                 dtype='<f4' if info['dtype'] == 'F32' else '<f2',
                                 offset=offset, shape=(count,))).astype(np.float32)
  else:
    raise ValueError(f"Unsupported dtype: {info['dtype']}")
  return values.reshape(-1, shape[1]).astype(np.float64)


def inventory(checkpoint):
  groups = {}
  for name, info in safetensor_headers(checkpoint).items():
    group = ('expert' if name.startswith('expert.') else 'vision' if '.visual.' in name else
             'projector' if '.projector.' in name else 'embedding' if 'embed_tokens' in name else
             'lm_head' if 'lm_head' in name else 'language')
    entry = groups.setdefault(group, dict(parameters=0, tensors=0, mlp_parameters=0, matrices=0))
    count = math.prod(info['shape'])
    entry['parameters'] += count
    entry['tensors'] += 1
    if len(info['shape']) == 2:
      entry['matrices'] += 1
      if '.mlp.' in name:
        entry['mlp_parameters'] += count
  config = Path(checkpoint) / 'config.json'
  return dict(groups=groups, total_parameters=sum(g['parameters'] for g in groups.values()),
              config_sha256=hashlib.sha256(config.read_bytes()).hexdigest(),
              source='local safetensors headers; no tensor loading')


def hadamard32(values, signs):
  """Right multiply each padded K32 group by diag(signs) H32 / sqrt(32)."""
  if values.shape[-1] % 32 or signs.shape != (values.shape[-1],):
    raise ValueError('Pad K to 32 before rotation')
  out = (values * signs).reshape(*values.shape[:-1], -1, 32).copy()
  stride = 1
  while stride < 32:
    view = out.reshape(*out.shape[:-1], -1, 2, stride)
    left, right = view[..., 0, :].copy(), view[..., 1, :].copy()
    view[..., 0, :], view[..., 1, :] = left + right, left - right
    stride *= 2
  return out.reshape(values.shape) / math.sqrt(32)


def smooth_scale(x_cal, weight, alpha):
  activation = np.maximum(np.quantile(np.abs(x_cal), 0.999, axis=0), 1e-12)
  columns = np.maximum(np.max(np.abs(weight), axis=0), 1e-12)
  scale = activation ** alpha / columns ** (1 - alpha)
  scale /= np.exp(np.mean(np.log(scale)))
  return np.clip(scale, 1 / 32, 32)


def integer_quant(values, maximum, nonnegative=False):
  high, low = (15, 0) if nonnegative else (7, -7)
  step = np.maximum(np.asarray(maximum) / high, 1e-12)
  codes = np.clip(np.rint(values / step), low, high).astype(np.int8)
  return codes, step


def fp8(values):
  """Simulate tensor-scaled E4M3FN, nearest-even; accumulation remains float64."""
  scale = max(float(np.max(np.abs(values))) / 448, 1e-12)
  levels = np.array([m / 512 for m in range(8)] +
                    [(1 + m / 8) * 2.0 ** (e - 7) for e in range(1, 16)
                     for m in range(8 if e < 15 else 7)])
  target = np.minimum(np.abs(values) / scale, 448)
  upper = np.clip(np.searchsorted(levels, target), 1, len(levels) - 1)
  lower = upper - 1
  dl, du = target - levels[lower], levels[upper] - target
  choose_lower = (dl < du) | ((dl == du) & ((lower % 2) == 0))
  return np.copysign(levels[np.where(choose_lower, lower, upper)], values) * scale


def pack_nibbles(codes, signed=True):
  if codes.shape[-1] % 2:
    raise ValueError('Packing requires an even K')
  if np.any(codes < (-8 if signed else 0)) or np.any(codes > (7 if signed else 15)):
    raise ValueError('Code outside nibble range')
  bits = codes.astype(np.int16) & 15
  return (bits[..., 0::2] | (bits[..., 1::2] << 4)).astype(np.uint8)


def unpack_nibbles(packed, signed=True):
  bits = np.stack((packed & 15, packed >> 4), axis=-1).reshape(*packed.shape[:-1], -1)
  values = bits.astype(np.int16)
  return np.where(values >= 8, values - 16, values).astype(np.int8) if signed else values.astype(np.int8)


def k32_reference(a_codes, w_codes):
  """Factorable-scale contract: one INT32 dot over all K, then row/output scales."""
  if a_codes.shape[1] != w_codes.shape[1] or a_codes.shape[1] % 32:
    raise ValueError('K32 alignment or shape mismatch')
  out = np.zeros((a_codes.shape[0], w_codes.shape[0]), np.int64)
  for start in range(0, a_codes.shape[1], 32):
    out += a_codes[:, start:start + 32].astype(np.int64) @ w_codes[:, start:start + 32].astype(np.int64).T
  if np.max(np.abs(out), initial=0) > np.iinfo(np.int32).max:
    raise OverflowError('INT32 accumulator overflow')
  return out.astype(np.int32)


def error(prediction, reference):
  delta = prediction - reference
  return dict(relative_rmse=float(np.linalg.norm(delta) / max(np.linalg.norm(reference), 1e-12)),
              max_absolute=float(np.max(np.abs(delta))),
              cosine=float(np.vdot(prediction, reference) /
                           max(np.linalg.norm(prediction) * np.linalg.norm(reference), 1e-12)))


def fit(weight, x_cal, *, alpha=0.0, clip=1.0, fraction=0.0, rotation=False,
        nonnegative=False, static=False, seed=17):
  if nonnegative and np.any(x_cal < 0):
    raise ValueError('Unsigned A4 requires proven nonnegative inputs')
  scale = np.ones(weight.shape[1]) if alpha == 0 else smooth_scale(x_cal, weight, alpha)
  x, w = x_cal / scale, weight * scale
  padding = (-w.shape[1]) % 32
  x, w = np.pad(x, ((0, 0), (0, padding))), np.pad(w, ((0, 0), (0, padding)))
  signs = np.random.default_rng(seed).choice([-1., 1.], w.shape[1]) if rotation else None
  if signs is not None:
    x, w = hadamard32(x, signs), hadamard32(w, signs)
    nonnegative = False
  rank = math.ceil(weight.shape[1] * fraction)
  score = np.sqrt(np.mean(x * x, axis=0)) * np.linalg.norm(w, axis=0)
  selected = np.argsort(-score, kind='stable')[:rank]
  w_side = w[:, selected].copy()
  w[:, selected] = 0
  x[:, selected] = 0
  w_max = np.max(np.abs(w), axis=1, keepdims=True)
  w_codes, w_step = integer_quant(w, w_max)
  # Clip is a deterministic multiplier of calibrated absolute maximum.
  a_max = float(np.max(np.abs(x))) * clip
  return dict(scale=scale, padding=padding, signs=signs, selected=selected, w_side=w_side,
              w_codes=w_codes, w_step=w_step, a_max=a_max, clip=clip,
              static=static, nonnegative=nonnegative, alpha=alpha, fraction=fraction)


def predict(fitted, values, correction='fp8'):
  x = np.pad(values / fitted['scale'], ((0, 0), (0, fitted['padding'])))
  if fitted['signs'] is not None:
    x = hadamard32(x, fitted['signs'])
  if fitted['nonnegative'] and np.any(x < 0):
    raise ValueError('Unsigned A4 input domain changed')
  side = x[:, fitted['selected']].copy()
  x[:, fitted['selected']] = 0
  a_max = fitted['a_max'] if fitted['static'] else np.max(np.abs(x), axis=1, keepdims=True) * fitted['clip']
  a_codes, a_step = integer_quant(x, a_max, fitted['nonnegative'])
  dot = k32_reference(a_codes, fitted['w_codes']).astype(np.float64)
  output = dot * a_step * fitted['w_step'].T
  if side.size:
    if correction == 'fp8':
      output += fp8(side) @ fp8(fitted['w_side']).T
    elif correction == 'exact':
      output += side @ fitted['w_side'].T
    else:
      raise ValueError('Unknown correction mode')
  saturation = float(np.mean(np.abs(x) > a_max))
  return output, saturation


def screen(weight, x_cal, x_eval, nonnegative=False):
  if weight.ndim != 2 or x_cal.ndim != 2 or x_eval.ndim != 2:
    raise ValueError('Expected rank-two arrays')
  if not (weight.shape[1] == x_cal.shape[1] == x_eval.shape[1]) or min(weight.shape + x_cal.shape + x_eval.shape) <= 0:
    raise ValueError('Invalid input shapes')
  if not all(np.isfinite(a).all() for a in (weight, x_cal, x_eval)):
    raise ValueError('Nonfinite capture')
  reference_cal, reference_eval = x_cal @ weight.T, x_eval @ weight.T
  records = []
  # This fixed menu exposes both static and per-token activation scaling costs.
  variants = [('signed_row', False, False, False, 0),
              ('native_domain_row', nonnegative, False, False, 0),
              ('smooth_row', nonnegative, False, False, 0),
              ('smooth_side1_row', nonnegative, False, False, .01),
              ('smooth_side1_static', nonnegative, True, False, .01),
              ('smooth_rot32_row', False, False, True, 0)]
  for name, unsigned, static, rotation, fraction in variants:
    best = None
    for alpha in ([0.] if name in ('signed_row', 'native_domain_row') else [.25, .5, .75]):
      for clip in ([1.] if name in ('signed_row', 'native_domain_row') else [.8, .9, 1.]):
        fitted = fit(weight, x_cal, alpha=alpha, clip=clip, fraction=fraction,
                     rotation=rotation, nonnegative=unsigned, static=static)
        prediction, _ = predict(fitted, x_cal)
        loss = error(prediction, reference_cal)['relative_rmse']
        if best is None or loss < best[0]:
          best = loss, fitted
    loss, fitted = best
    prediction, saturation = predict(fitted, x_eval)
    records.append(dict(name=name, calibration_relative_rmse=loss,
                        evaluation=error(prediction, reference_eval), evaluation_saturation=saturation,
                        alpha=fitted['alpha'], clip=fitted['clip'],
                        correction_columns=len(fitted['selected']),
                        correction_padded_K=math.ceil(len(fitted['selected']) / 32) * 32,
                        activation_unsigned=fitted['nonnegative'], activation_static=static,
                        rotation_block=32 if rotation else None))
  return dict(shapes=dict(W=list(weight.shape), X_cal=list(x_cal.shape), X_eval=list(x_eval.shape)),
              reference='float64 CPU matmul; not BF16 reference inference',
              fp8_tensor_scaled=error(fp8(x_eval) @ fp8(weight).T, reference_eval), variants=records,
              warning='Calibration-only parameter selection; synthetic/linear errors cannot qualify driving quality.')


def self_test():
  rng = np.random.default_rng(20261116)
  checks = []
  for unsigned in (False, True):
    values = np.arange(16, dtype=np.int8) if unsigned else np.arange(-8, 8, dtype=np.int8)
    np.testing.assert_array_equal(unpack_nibbles(pack_nibbles(values, not unsigned), not unsigned), values)
    for k in (32, 64, 96):
      a = rng.integers(0 if unsigned else -7, 16 if unsigned else 8, (3, k), dtype=np.int8)
      w = rng.integers(-7, 8, (5, k), dtype=np.int8)
      expected = np.array([[sum(int(a[m, q]) * int(w[n, q]) for q in range(k))
                            for n in range(5)] for m in range(3)], dtype=np.int32)
      np.testing.assert_array_equal(k32_reference(a, w), expected)
    checks.append('unsigned' if unsigned else 'signed')
  for k in (32, 64, 96):
    x, w = rng.normal(size=(11, k)), rng.normal(size=(7, k))
    s = np.exp(rng.normal(size=k))
    np.testing.assert_allclose((x / s) @ (w * s).T, x @ w.T, rtol=1e-12, atol=1e-12)
    signs = rng.choice([-1., 1.], k)
    xr, wr = hadamard32(x, signs), hadamard32(w, signs)
    np.testing.assert_allclose(xr @ wr.T, x @ w.T, rtol=1e-12, atol=1e-12)
    # Independent explicit Sylvester matrix checks the butterfly and right-multiply convention.
    h = np.array([[1.]])
    for _ in range(5):
      h = np.block([[h, h], [h, -h]])
    explicit = np.concatenate([(x[:, q:q + 32] * signs[q:q + 32]) @ h / math.sqrt(32)
                               for q in range(0, k, 32)], axis=1)
    np.testing.assert_allclose(xr, explicit, rtol=1e-12, atol=1e-12)
  checks.extend(['positive_diagonal_exact', 'K32_rotation_exact'])
  x, w1, w2 = rng.normal(size=(13, 32)), rng.normal(size=(64, 32)), rng.normal(size=(9, 64))
  b1, b2 = rng.normal(size=64), rng.normal(size=9)
  d = np.exp(rng.normal(size=64))
  original = np.maximum(x @ w1.T + b1, 0) ** 2 @ w2.T + b2
  folded = np.maximum(x @ (w1 / np.sqrt(d)[:, None]).T + b1 / np.sqrt(d), 0) ** 2 @ (w2 * d).T + b2
  np.testing.assert_allclose(folded, original, rtol=1e-12, atol=1e-11)
  checks.append('relu2_sqrt_fold_exact_with_bias')
  gamma, beta, d = rng.normal(size=32), rng.normal(size=32), np.exp(rng.normal(size=32))
  normed = (x - x.mean(axis=1, keepdims=True)) / np.sqrt(x.var(axis=1, keepdims=True) + 1e-6)
  np.testing.assert_allclose((normed * gamma + beta) @ w1.T,
                             (normed * (gamma / d) + beta / d) @ (w1 * d).T,
                             rtol=1e-12, atol=1e-12)
  checks.append('layernorm_affine_fold_exact')
  # Moving a rotation through ReLU-squared is invalid.
  signs = rng.choice([-1., 1.], 32)
  assert np.max(np.abs(hadamard32(np.maximum(x, 0) ** 2, signs) -
                       np.maximum(hadamard32(x, signs), 0) ** 2)) > 1
  checks.append('rotation_does_not_commute_with_relu2')
  fitted = fit(w1, x, alpha=.5, fraction=.1)
  xm, wm = x / fitted['scale'], w1 * fitted['scale']
  selected = fitted['selected']
  main = xm.copy()
  main[:, selected] = 0
  np.testing.assert_allclose(main @ wm.T + xm[:, selected] @ wm[:, selected].T,
                             x @ w1.T, rtol=1e-12, atol=1e-12)
  checks.append('outlier_split_exact_without_quantization')
  zero = fit(np.zeros((4, 35)), np.zeros((6, 35)), nonnegative=True, static=True)
  np.testing.assert_array_equal(predict(zero, np.zeros((2, 35)))[0], np.zeros((2, 4)))
  checks.append('zero_scale_and_K_padding')
  fixed = fit(np.ones((2, 32)), np.ones((3, 32)), static=True)
  _, saturation = predict(fixed, np.full((2, 32), 100.))
  assert saturation == 1.0
  checks.append('static_scale_shift_detects_saturation')
  np.testing.assert_array_equal(fp8(np.array([0., 1 / 512, .5, 1., 240., 256., 448.])),
                                np.array([0., 1 / 512, .5, 1., 240., 256., 448.]))
  np.testing.assert_array_equal(fp8(np.array([0., -448., 1.0625, 1.1875])),
                                np.array([0., -448., 1., 1.25]))
  checks.append('fp8_representables_and_even_ties')
  # Independent seeds create calibration/evaluation tensors with the same channel structure.
  channel = np.ones(256)
  channel[[3, 91]] = [32., 64.]
  x_cal = np.maximum(rng.normal(size=(96, 256)) * channel, 0) ** 2
  x_eval = np.maximum(rng.normal(size=(80, 256)) * channel, 0) ** 2
  weight = rng.normal(size=(48, 256)) / math.sqrt(256)
  return dict(passed=True, checks=checks,
              synthetic_relu2=screen(weight, x_cal, x_eval, nonnegative=True))


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  sub = parser.add_subparsers(dest='command', required=True)
  sub.add_parser('self-test')
  sub.add_parser('inventory').add_argument('checkpoint')
  probe = sub.add_parser('checkpoint-probe')
  probe.add_argument('checkpoint')
  probe.add_argument('--rows', type=int, default=64)
  capture = sub.add_parser('screen')
  capture.add_argument('capture')
  capture.add_argument('--nonnegative', action='store_true')
  args = parser.parse_args()
  if args.command == 'self-test':
    result = self_test()
  elif args.command == 'inventory':
    result = inventory(args.checkpoint)
  elif args.command == 'screen':
    with np.load(args.capture, allow_pickle=False) as data:
      result = screen(*(np.asarray(data[k], dtype=np.float64) for k in ('W', 'X_cal', 'X_eval')),
                      nonnegative=args.nonnegative)
    result['capture_sha256'] = hashlib.sha256(Path(args.capture).read_bytes()).hexdigest()
  else:
    if not 1 <= args.rows <= 256:
      raise ValueError('Probe reads at most 256 output rows per matrix')
    names = ['vlm.model.language_model.layers.0.mlp.fc2.weight',
             'vlm.model.language_model.layers.14.mlp.fc2.weight',
             'vlm.model.visual.encoder.layers.14.mlp.fc2.weight',
             'expert.expert.layers.14.mlp.fc2.weight']
    headers, result = safetensor_headers(args.checkpoint), {}
    rng = np.random.default_rng(43)
    for name in names:
      w = read_weight_rows(headers[name], args.rows)
      relu2 = '.visual.' not in name
      x_cal, x_eval = rng.normal(size=(24, w.shape[1])), rng.normal(size=(20, w.shape[1]))
      if relu2:
        x_cal, x_eval = np.maximum(x_cal, 0) ** 2, np.maximum(x_eval, 0) ** 2
      result[name] = screen(w, x_cal, x_eval, relu2)
    result['warning'] = 'Actual checkpoint weight rows, SYNTHETIC inputs; no route or model quality claim.'
  print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == '__main__':
  main()
