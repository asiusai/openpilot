"""Independent CPU boundary checks and one gfx1200 fusion compilation.

CPU checks do not emulate AMD exp2/reciprocal. No GPU admission is produced.
"""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import re

import numpy as np

HERE = Path(__file__).resolve().parent


def load(name):
  spec = importlib.util.spec_from_file_location('vision_gelu_check_'+name.replace('-', '_'), HERE/(name+'.py'))
  module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
  return module


def bf16(value):
  value = np.asarray(value, np.float32)
  assert np.isfinite(value).all()
  bits = value.view(np.uint32)
  return ((bits+np.uint32(0x7fff)+((bits >> 16) & 1)) >> 16).astype(np.uint16)


def f32(bits): return (np.asarray(bits, np.uint16).astype(np.uint32) << 16).view(np.float32)


def fp8(value):
  """Independent finite E4M3 nearest-even table; retain signed zero."""
  value = np.asarray(value, np.float32)
  assert np.isfinite(value).all() and np.max(np.abs(value)) <= 448
  codes = np.arange(127, dtype=np.uint8)
  exponent, fraction = (codes >> 3).astype(np.int32), (codes & 7).astype(np.float32)
  table = np.where(exponent == 0, fraction*2.**-9, (1+fraction/8)*np.exp2(exponent-7)).astype(np.float32)
  magnitude = np.abs(value)
  high = np.minimum(np.searchsorted(table, magnitude), 126)
  low = np.maximum(high-1, 0)
  dl, dh = magnitude-table[low], table[high]-magnitude
  choose_high = (dh < dl) | ((dh == dl) & ((high & 1) == 0))
  return np.where(choose_high, high, low).astype(np.uint8) | (np.signbit(value).astype(np.uint8) << 7)


def packed_formula(codes):
  rows, width = codes.shape
  output = np.zeros((((rows+15)//16)*16, width), np.uint8)
  k = np.arange(width)[None, :]
  for start in range(0, rows, 16):
    r = np.arange(start, min(start+16, rows))[:, None]
    address = (r//16*(width//16)+k//16)*256+(k%16//4)*64+(r%16)*4+k%4
    output.reshape(-1)[address] = codes[start:start+16]
  return output


def packed_view(codes):
  rows, width = codes.shape
  padded = np.pad(codes, ((0, (-rows) % 16), (0, 0)))
  return padded.reshape(-1, 16, width//16, 4, 4).transpose(0, 2, 3, 1, 4).copy().reshape(-1, width)


def reference(bits):
  values = f32(bits)
  maximum = np.abs(values).max(axis=(1, 2)).astype(np.float32)
  scale = np.maximum(maximum/np.float32(448), np.float32(1e-12))
  codes = fp8(np.clip(values/scale[:, None, None], -448, 448))
  return maximum, scale, packed_formula(codes.reshape(-1, bits.shape[-1]))


def chunked(bits):
  images, rows, width = bits.shape
  count = rows*width
  chunks = (count+4095)//4096
  # Implement the candidate's integer-sign-clear abs and masked chunk traversal.
  maxima = np.empty((images, chunks), np.float32)
  for image in range(images):
    for block in range(chunks):
      valid = bits[image].reshape(-1)[block*4096:min((block+1)*4096, count)]
      padded = np.zeros(4096, np.uint16)
      padded[:len(valid)] = valid
      maxima[image, block] = f32(padded & np.uint16(0x7fff)).max()
  maximum = maxima.max(axis=1)
  scale = np.maximum(maximum/np.float32(448), np.float32(1e-12))
  codes = fp8(np.clip(f32(bits)/scale[:, None, None], -448, 448))
  return maximum, scale, packed_view(codes.reshape(-1, width)), maxima


def independent_checks(inspect):
  import torch
  # Cast controls compare independent integer/table conversion with CPU PyTorch.
  boundaries = np.array([0., -0., 1.00390625, 1.01171875, -1.00390625, 255.5, -255.5, 2.**-126, 2.**-133], np.float32)
  np.testing.assert_array_equal(bf16(boundaries), torch.from_numpy(boundaries).to(torch.bfloat16).view(torch.uint16).numpy())
  positive = f32(np.arange(0x0000, 0x43e1, dtype=np.uint16))
  numbers = np.concatenate((positive, -positive))
  numbers = numbers[np.abs(numbers) <= 448]
  np.testing.assert_array_equal(fp8(numbers), torch.from_numpy(numbers).to(torch.float8_e4m3fn).view(torch.uint8).numpy())
  # Ensure all FP8 midpoint ties and adjacent float32 values are checked.
  table = []
  for code in range(127):
    exp, fraction = code >> 3, code & 7
    table.append(fraction*2.**-9 if exp == 0 else (1+fraction/8)*2.**(exp-7))
  mids = (np.asarray(table[:-1], np.float32)+np.asarray(table[1:], np.float32))/2
  mids = np.concatenate((mids, -mids))
  probes = np.concatenate((mids, np.nextafter(mids, np.float32(-np.inf)), np.nextafter(mids, np.float32(np.inf))))
  np.testing.assert_array_equal(fp8(probes), torch.from_numpy(probes).to(torch.float8_e4m3fn).view(torch.uint8).numpy())
  records = []
  for shape in ((1, 720, 4304), (2, 720, 4304), (2, 17, 48)):
    images, rows, width = shape
    pattern = np.array([0., -0., 8., 8.0625, 16., 64., 160., 2.**-133, -.125], np.float32)
    original = bf16(np.resize(pattern, shape))
    if images == 2: original[1] = bf16(f32(original[1])*2)
    previous = None
    for state in range(6):
      bits = original.copy()
      if state == 1: bits[0, -1, -1] = bf16([704.])[0]
      if state == 2 and images == 2: bits[1, -1, -1] = bf16([1408.])[0]
      if state == 3: bits[-1].fill(0x8000)
      if state == 4: bits[0, rows//2, width//2] = bf16([448.])[0]
      expected, actual = reference(bits), chunked(bits)
      for x, y in zip(expected, actual[:3], strict=True):
        np.testing.assert_array_equal(x.view(np.uint8), y.view(np.uint8))
      if state == 3:
        assert not np.signbit(actual[0][-1]) and not np.signbit(actual[3][-1]).any()
        assert actual[1][-1] == np.float32(1e-12)
      if state == 2 and images == 2:
        np.testing.assert_array_equal(expected[1][:1], reference(original)[1][:1])
      digest = inspect.sha(b''.join(value.tobytes() for value in expected))
      if state == 0: previous = digest
      if state == 5: assert digest == previous
      records.append(dict(shape=list(shape), state=state, boundary_sha256=digest, maxima=actual[0].tolist(),
                          exact_max_scale_pack=True, gelu_native_not_executed=True))
  # Early/late BF16 rounding changes are observable and must never be waived.
  pre = np.array([8.031249, 8.031251, 16.062498, 16.062502], np.float32)
  rounded = f32(bf16(pre))
  assert np.any(pre != rounded)
  maximum_pre, maximum_bf16 = pre.max(), rounded.max()
  assert maximum_pre != maximum_bf16
  # A tanh/erf host library is not an exact oracle for the pinned native SFU path.
  values = f32(np.arange(0xbe00, 0xc100, dtype=np.uint16))
  with np.errstate(over='ignore', under='ignore'):
    a = values*(1+np.tanh(np.float32(np.sqrt(2/np.pi))*(values+np.float32(.044715)*values**3)))*np.float32(.5)
    b = values/(1+np.exp2(np.float32(-np.sqrt(8/np.pi)/np.log(2))*(values+np.float32(.044715)*values**3)))
  mismatches = np.flatnonzero(bf16(a) != bf16(b))
  assert len(mismatches)
  witness = int(mismatches[0])
  return dict(records=records, fp8_midpoint_probes=len(probes), fp8_bf16_domain_probes=len(numbers),
    bf16_rounding_controls=True, host_tanh_not_exact=dict(mismatched_bf16=len(mismatches), input=float(values[witness]),
      tanh_bits=int(bf16(a)[witness]), sigmoid_exp2_bits=int(bf16(b)[witness])),
    limitation='Independent max/scale/cast/pack checks start from BF16 activations. They do not prove native AMD GELU equality.')


def scalar_expression(source):
  """Canonicalize the LLVM dataflow through the first GELU BF16 rounding."""
  definitions = {}
  for line in source.splitlines():
    if match := re.match(r'\s*(%\w+) = (.*)', line): definitions[match[1]] = match[2]
  roots = [name for name, value in definitions.items() if re.match(r'trunc i32 %\w+ to i16$', value)]
  assert roots
  memo = {}
  def expand(name):
    if name in memo: return memo[name]
    value = definitions[name]
    if value.startswith(('load bfloat,', 'phi bfloat ')): result = 'INPUT_BF16'
    else: result = re.sub(r'%\w+', lambda m:'('+expand(m.group())+')', value)
    memo[name] = result
    return result
  result = expand(roots[0])
  assert 'llvm.exp2.float' in result and 'fdiv contract afn float' in result and '32767' in result
  return result


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--output', type=Path, required=True)
  parser.add_argument('--artifacts', type=Path, default=HERE/'agent-vision-gelu-artifacts')
  args = parser.parse_args()
  os.environ.update(DEV='CPU', ALLOW_DEVICE_USAGE='0', TC_OPT='2', BEAM='0', JITBEAM='0', PARALLEL='0', PROFILE='0', DEBUG='0')
  from tinygrad import Tensor, dtypes, Device
  from tinygrad.helpers import Target
  from tinygrad.renderer.llvmir import AMDLLVMRenderer
  from amd_gemm import pack
  inspect, kernel, reduction = (load(name) for name in ('agent-vision-gelu-inspect', 'agent-vision-gelu-kernel', 'quantize-reduction'))
  production_sources = inspect.production_sources()
  renderer = AMDLLVMRenderer(Target('AMD', 'LLVM', 'gfx1200'))
  renderer.compiler = inspect.load('wgp_compiler').WGPCompiler('gfx1200')
  args.artifacts.mkdir(parents=True, exist_ok=True)
  baseline_calls, _ = inspect.original_chain(renderer=renderer)
  original = inspect.compile_calls(baseline_calls, renderer, args.artifacts, prefix='original')
  x = inspect.fc1_output(renderer)
  with inspect.schedules() as calls:
    quantized, scale = kernel.quantize(x, 2, reduction)
    pack(quantized).realize(); scale.realize()
  candidate = inspect.compile_calls(calls, renderer, args.artifacts, prefix='candidate')
  assert len(original) == 8 and len(candidate) == 7
  assert original[0]['source_sha256'] == candidate[0]['source_sha256']
  assert original[0]['binary_sha256'] == candidate[0]['binary_sha256'], 'Original fc1 changed'
  for old, new in ((3, 2), (4, 3), (6, 5), (7, 6)):
    assert original[old]['source_sha256'] == candidate[new]['source_sha256']
    assert original[old]['binary_sha256'] == candidate[new]['binary_sha256'], 'Unchanged downstream boundary changed'
  def source(label, index): return (args.artifacts/f'agent-vision-gelu-{label}-{index}.ll').read_text()
  expression = scalar_expression(source('original', 1))
  assert expression == scalar_expression(source('original', 5)) == scalar_expression(source('candidate', 1))
  fused = candidate[1]
  assert all(fused['resources'][key] == 0 for key in ('vgpr_spill_count', 'sgpr_spill_count', 'private_segment_fixed_size'))
  assert not Device._opened_devices
  result = dict(gpu_executed=False, opened_devices=[], production_sources=production_sources,
    source_sha256={str(HERE/(name+'.py')):inspect.sha((HERE/(name+'.py')).read_bytes())
    for name in ('agent-vision-gelu-inspect', 'agent-vision-gelu-kernel', 'agent-vision-gelu-check')},
    original=original, candidate=candidate, independent_cpu=independent_checks(inspect),
    unchanged_fc1_binary=True, unchanged_final_max_scale_pack_binaries=True,
    identical_llvm_gelu_bf16_expression_sha256=inspect.sha(expression.encode()),
    native_gelu_equality_proven=False, gpu_admission=False, model_hook_installed=False,
    blocker='LLVM scalar GELU dataflow matches, but native exp2/reciprocal and FMA behavior lacks an exact GPU boundary comparison.',
    runtime_attribution_limit='The existing profile aggregates names and does not identify source/binary hashes.')
  args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
  print(json.dumps(dict(gpu_executed=False, original_kernels=len(original), candidate_kernels=len(candidate),
    exact_llvm_scalar_expression=True, independent_controls=len(result['independent_cpu']['records']),
    candidate_resources=fused['resources'], gpu_admission=False), indent=2))


if __name__ == '__main__': main()
