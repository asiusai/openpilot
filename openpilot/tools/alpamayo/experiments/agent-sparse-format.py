"""Exact FP8-byte compression and conservative tile eligibility on the CPU.

Input codes must come AFTER the original BF16 ReLU-square and FP8 quantizer.
This code does not prune, quantize, train, load weights, or predict sparsity.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def matrix(value):
  q = np.asarray(value)
  if q.dtype != np.uint8 or q.ndim != 2 or q.shape[0] % 16 or q.shape[1] % 32:
    raise ValueError('Expected uint8 [M,K], M multiple of 16, K multiple of 32')
  return q


def pack(q):
  q = matrix(q)
  m, k = q.shape
  return q.reshape(m//16, 16, k//16, 4, 4).transpose(0, 2, 3, 1, 4).copy().reshape(m, k)


def unpack(q):
  q = matrix(q)
  m, k = q.shape
  return q.reshape(m//16, k//16, 4, 16, 4).transpose(0, 3, 1, 2, 4).copy().reshape(m, k)


def decode(q):
  """OCP E4M3 finite values, both zero signs, and both NaN encodings."""
  q = np.asarray(q, dtype=np.uint8)
  magnitude = q & 127
  exponent, mantissa = (magnitude >> 3).astype(np.int16), (magnitude & 7).astype(np.float64)
  values = np.where(exponent == 0, mantissa*2.0**-9, (1+mantissa/8)*np.exp2(exponent-7))
  values = np.where(magnitude == 127, np.nan, values)
  return np.where(q & 128, -values, values)


def compress(q):
  """Return [M16,K32,32 lanes,8 bytes], metadata, and ALL-row eligibility.

  Invalid tiles contain placeholders. Consumers MUST use the dense original.
  Metadata bits[15:0] map to this lane's four K4 groups; bits[31:16] are zero.
  The FP8 mapping differs from the FP16 fragment mapping in AMD's handbook.
  """
  q = matrix(q)
  m, k = q.shape
  groups = q.reshape(m, k//4, 4)
  nz = (groups & 127) != 0
  group_ok = (nz.sum(-1) <= 2) & ~((groups & 127) == 127).any(-1)
  # Take nonzeros first, then the lowest zero fillers, then sort two positions.
  priority = np.where(nz, np.arange(4), np.arange(4)+4)
  positions = np.sort(np.argsort(priority, axis=-1, kind='stable')[..., :2], axis=-1)
  values = np.take_along_axis(groups, positions, axis=-1)
  nibbles = (positions[..., 0] | (positions[..., 1] << 2)).astype(np.uint32)
  # Compressed K0..7 -> lanes0..15; K8..15 -> lanes16..31.
  fragments = values.reshape(m//16, 16, k//32, 2, 4, 2).transpose(0, 2, 3, 1, 4, 5).copy().reshape(m//16, k//32, 32, 8)
  nibble_lanes = nibbles.reshape(m//16, 16, k//32, 2, 4).transpose(0, 2, 3, 1, 4).reshape(m//16, k//32, 32, 4)
  metadata = (nibble_lanes << (4*np.arange(4, dtype=np.uint32))).sum(-1, dtype=np.uint32)
  eligible = group_ok.reshape(m//16, 16, k//32, 8).all(axis=(1, 3))
  return fragments, metadata, eligible


def expand(fragments, metadata, eligible, original):
  """Sparse expansion with genuine original-byte fallback for every invalid tile."""
  q = matrix(original).copy()
  mt, kt = q.shape[0]//16, q.shape[1]//32
  if fragments.shape != (mt, kt, 32, 8) or metadata.shape != (mt, kt, 32) or eligible.shape != (mt, kt):
    raise ValueError('Fragment, metadata, eligibility, or original shape mismatch')
  for i in range(mt):
    for t in range(kt):
      if not eligible[i, t]: continue
      tile = np.zeros((16, 32), dtype=np.uint8)
      for lane in range(32):
        for e in range(8):
          local_group, within = e//2, e % 2
          pos = (int(metadata[i, t, lane]) >> (local_group*4+within*2)) & 3
          other = (int(metadata[i, t, lane]) >> (local_group*4+2)) & 3
          first = (int(metadata[i, t, lane]) >> (local_group*4)) & 3
          if first >= other: raise ValueError('SWMMAC requires index0 < index1')
          tile[lane % 16, (lane//16)*16+local_group*4+pos] = fragments[i, t, lane, e]
      q[i*16:(i+1)*16, t*32:(t+1)*32] = tile
  return q


def audit(q):
  q = matrix(q)
  nonzeros = ((q & 127) != 0).reshape(q.shape[0], -1, 4).sum(-1)
  finite = ~((q & 127) == 127).reshape(q.shape[0], -1, 4).any(-1)
  group_ok = (nonzeros <= 2) & finite
  m, k = q.shape
  tile64_ok = group_ok.reshape(m//16, 16, k//32, 8)
  ok = tile64_ok.all(axis=(1, 3))
  output = dict(shape=[m, k], logical_zero_fraction=float(((q & 127) == 0).mean()),
                k4_nonzero_histogram=[int((nonzeros == i).sum()) for i in range(5)],
                nonfinite_codes=int(((q & 127) == 127).sum()),
                eligible_k4_fraction=float(group_ok.mean()),
                eligible_m16_k32_tiles=int(ok.sum()), total_m16_k32_tiles=int(ok.size),
                eligible_m16_k32_fraction=float(ok.mean()),
                eligible_full_k_m16_rows=int(ok.all(-1).sum()),
                conservative_m64_k32_fraction=None,
                extra_bytes=dict(compressed_values=m*k//2, metadata_padded_uint32=m*k//8,
                                 eligibility_uint32=(m//16)*(k//32)*4))
  if m % 64 == 0:
    output['conservative_m64_k32_fraction'] = float(tile64_ok.reshape(m//64, 4, 16, k//32, 8).all(axis=(1, 2, 4)).mean())
  return output


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--capture', type=Path, required=True, help='NPZ containing codes:uint8 and manifest:JSON scalar')
  parser.add_argument('--output', type=Path, required=True)
  args = parser.parse_args()
  with np.load(args.capture, allow_pickle=False) as capture:
    manifest = json.loads(str(capture['manifest']))
    if manifest['stage'] != 'original-post-relu2-post-fp8-language-fc2':
      raise ValueError('Capture must use the original fc2 activation and quantization')
    for key in ('activation_scale', 'weight_scale'):
      if not np.isfinite(manifest[key]) or manifest[key] <= 0: raise ValueError(key)
    for key in ('layer', 'checkpoint_sha256', 'scales_sha256', 'actual_prompt_rows', 'padded_rows', 'layout'):
      if key not in manifest: raise ValueError('Missing provenance: '+key)
    q = matrix(capture['codes'])
    if manifest['layout'] == 'amd_gemm.pack': q = unpack(q)
    elif manifest['layout'] != 'row-major': raise ValueError('Unknown capture layout')
    if q.shape[0] != manifest['padded_rows'] or not 0 < manifest['actual_prompt_rows'] <= q.shape[0]:
      raise ValueError('Actual/padded prompt row mismatch')
    stats = audit(q)
  result = dict(measurement='CPU eligibility audit; no GPU timing or speed/quality claim', manifest=manifest,
                capture_sha256=hashlib.sha256(args.capture.read_bytes()).hexdigest(), statistics=stats)
  args.output.write_text(json.dumps(result, indent=2)+'\n')
  print(json.dumps(result, indent=2))


if __name__ == '__main__': main()
