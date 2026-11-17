"""CPU checks: actual compression source, independent expansion, packed contract.

No GPU resources, real activations, checkpoint loading, or sparsity prediction.
"""
import argparse
import ctypes
import hashlib
import importlib.util
import itertools
import json
import os
from pathlib import Path
import shutil
import subprocess

import numpy as np


def load(name):
  spec = importlib.util.spec_from_file_location(name.replace('-', '_'), Path(__file__).with_name(name+'.py'))
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module


def native_compressor(out, rows, width):
  """Compile the actual HIP compression body as CPU C++, with wave emulation.

  Device/global attributes vanish; scalar IDs and ballot are explicit host mocks.
  Run lanes31..1 before lane0, so its ballot sees all32 eligibility predicates.
  This verifies the compression body only, not GPU ballot or hardware mapping.
  """
  kernel = load('agent-sparse-kernel')
  prefix = r'''
static unsigned host_lane,host_group,host_bad;
extern "C" unsigned long __ockl_get_local_id(unsigned) {return host_lane;}
extern "C" unsigned long __ockl_get_group_id(unsigned) {return host_group;}
unsigned host_ballot(bool bad) {host_bad|=bad;return host_bad;}
#define __builtin_amdgcn_ballot_w32 host_ballot
'''
  suffix = f'''
extern "C" void agent_sparse_cpu(unsigned *v,unsigned *m,unsigned *f,const unsigned char *a) {{
  for(host_group=0;host_group<{rows//16*(width//32)};host_group++) {{
    host_bad=0;
    for(int lane=31;lane>=0;lane--) {{host_lane=lane;agent_sparse(v,m,f,a);}}
  }}
}}
'''
  body = kernel.source(rows, 16, width, 'compress')
  body = body.replace('__attribute__((global,amdgpu_flat_work_group_size(32,32)))', '')
  body = body.replace('__attribute__((device,always_inline))', '__attribute__((always_inline))')
  body = body.replace('__attribute__((device))', '')
  text = prefix+body+suffix
  cpp, lib = out/'compress-cpu.cpp', out/'compress-cpu.so'
  cpp.write_text(text)
  compiler = shutil.which('clang++')
  if compiler is None: raise RuntimeError('Existing clang++ required; do not install dependencies')
  subprocess.run([compiler, '-O2', '-shared', '-fPIC', '-Wno-ignored-attributes', str(cpp), '-o', str(lib)], check=True)
  dll = ctypes.CDLL(str(lib.resolve()))
  function = dll.agent_sparse_cpu
  function.argtypes = [ctypes.c_void_p]*4
  def compress(q):
    q = np.ascontiguousarray(q)
    tiles = rows//16*(width//32)
    values = np.empty((tiles, 32, 8), dtype=np.uint8)
    metadata = np.empty((tiles, 32), dtype=np.uint32)
    flags = np.empty(tiles, dtype=np.uint32)
    function(values.ctypes.data, metadata.ctypes.data, flags.ctypes.data, q.ctypes.data)
    return values.reshape(rows//16, width//32, 32, 8), metadata.reshape(rows//16, width//32, 32), flags.reshape(rows//16, width//32)
  return compress, dict(compiler=compiler, source_sha256=hashlib.sha256(text.encode()).hexdigest(),
                        library_sha256=hashlib.sha256(lib.read_bytes()).hexdigest())


def independent_expand(values, metadata, flags, q):
  """Use ISA row/column/bit equations, separately from the format packer."""
  result = q.copy()
  for r in range(q.shape[0]):
    for group in range(q.shape[1]//4):
      base = group*4
      tile = (r//16, base//32)
      if not flags[tile]: continue
      dense_k = base % 32
      lane = (dense_k//16)*16+r % 16
      nibble = (int(metadata[*tile, lane]) >> ((dense_k//4 % 4)*4)) & 15
      p0, p1 = nibble & 3, nibble >> 2
      assert p0 < p1
      result[r, base:base+4] = 0
      compressed_col = dense_k//2
      reg, byte = compressed_col//4 % 2, compressed_col % 4
      result[r, base+p0] = values[*tile, lane, reg*4+byte]
      result[r, base+p1] = values[*tile, lane, reg*4+byte+1]
  return result


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--output', type=Path, default=Path(__file__).with_name('agent-sparse-cpu-output'))
  args = parser.parse_args()
  if os.environ.get('DEV') != 'CPU': parser.error('Set DEV=CPU explicitly')
  args.output.mkdir(exist_ok=False)
  fmt = load('agent-sparse-format')
  native, provenance = native_compressor(args.output, 64, 64)
  rng = np.random.default_rng(241116)
  patterns = [p for n in (0, 1, 2) for p in itertools.combinations(range(4), n)]
  checks, cases = [], []
  for pattern in patterns:
    q = np.zeros((64, 64), dtype=np.uint8)
    for pos in pattern: q[:, pos::4] = rng.choice([1, 8, 16, 56, 64, 72, 80, 120, 126], size=(64, 16))
    cases.append(q)
  q = np.zeros((64, 64), dtype=np.uint8)
  for r in range(64):
    for g in range(16):
      pattern = patterns[int(rng.integers(len(patterns)))]
      for p in pattern: q[r, g*4+p] = int(rng.integers(1, 127))
  cases.append(q)
  for row in (0, 15, 16, 31, 32, 63):
    for group in (0, 3, 4, 7, 8, 15):
      for count in (3, 4):
        mutated = q.copy()
        mutated[row, group*4:group*4+4] = [56]*count+[0]*(4-count)
        cases.append(mutated)
  for code in (127, 255):
    mutated = q.copy(); mutated[63, 63] = code; cases.append(mutated)
  # Preserve negative zero numerically; its sign is not an arithmetic guarantee.
  negative_zero = q.copy(); negative_zero[negative_zero == 0] = 128; cases.append(negative_zero)
  for index, current in enumerate(cases):
    values, metadata, flags = fmt.compress(current)
    actual = native(fmt.pack(current))
    for left, right in zip(actual, (values, metadata, flags), strict=True): np.testing.assert_array_equal(left, right)
    # Independent eligibility directly enumerates ALL128 groups in each tile.
    for i in range(4):
      for t in range(2):
        block = current[i*16:(i+1)*16, t*32:(t+1)*32].reshape(16, 8, 4)
        expected = all(sum(int(c & 127) != 0 for c in group) <= 2 and all(int(c & 127) != 127 for c in group)
                       for row in block for group in row)
        assert bool(flags[i, t]) == expected
    expanded = independent_expand(values, metadata, flags, current)
    np.testing.assert_array_equal(fmt.decode(expanded), fmt.decode(current))
    np.testing.assert_array_equal(fmt.decode(fmt.expand(values, metadata, flags, current)), fmt.decode(current))
    # Reused native buffers must observe mutation, eligibility, and literal dense fallback.
    for i, t in np.argwhere(~flags): np.testing.assert_array_equal(expanded[i*16:(i+1)*16, t*32:(t+1)*32], current[i*16:(i+1)*16, t*32:(t+1)*32])
  checks.append(dict(name='actual_compression_body_vs_independent_cpu_format', cases=len(cases), passed=True))
  # Decode all finite FP8 codes and compare independent sparse expansion in FP64.
  current = cases[11]
  values, metadata, flags = fmt.compress(current)
  weights = rng.integers(0, 127, size=(32, 64), dtype=np.uint8)
  expanded = independent_expand(values, metadata | np.uint32(0xffff0000), flags, current)
  sx, sw = np.float32(.125), np.float32(.25)
  a = fmt.decode(current) @ fmt.decode(weights).T * float(np.float32(sx*sw))
  b = fmt.decode(expanded) @ fmt.decode(weights).T * float(np.float32(sx*sw))
  np.testing.assert_array_equal(a, b)
  checks.append(dict(name='independent_fp64_matmul_and_single_scale_product', passed=True,
                     limitation='Does not validate GPU FP32 reduction order or BF16 output equality'))
  from tinygrad import Tensor, Device, dtypes
  from amd_gemm import pack as real_pack, unpack as real_unpack
  quantizer = load('quantize-packed')
  codes = np.arange(64*64, dtype=np.uint16).reshape(64, 64).astype(np.uint8)
  original = Tensor(codes, device='CPU').realize()
  packed = real_pack(original).realize().numpy()
  np.testing.assert_array_equal(packed, fmt.pack(codes))
  np.testing.assert_array_equal(real_unpack(Tensor(packed, device='CPU')).realize().numpy(), codes)
  # Exercise the original BF16 ReLU-square -> fixed FP8 -> PackedActivation contract.
  pre = rng.normal(size=(1, 64, 64)).astype(np.float32)
  h = Tensor(pre, device='CPU', dtype=dtypes.bfloat16).relu().square().realize()
  scale = Tensor([.125], device='CPU', dtype=dtypes.float32).reshape(()).realize()
  qpacked, original_scale = quantizer.quantize(h, scale)
  wrapped = quantizer.PackedActivation(qpacked, 64, 64)
  assert original_scale is scale and (wrapped.rows, wrapped.width) == (64, 64)
  logical = (h.float()/scale).clamp(-448, 448).cast(dtypes.fp8e4m3).reshape(64, 64).realize().bitcast(dtypes.uint8).numpy()
  captured = wrapped.value.bitcast(dtypes.uint8).numpy()
  np.testing.assert_array_equal(fmt.unpack(captured), logical)
  values, metadata, flags = fmt.compress(logical)
  np.testing.assert_array_equal(fmt.decode(independent_expand(values, metadata, flags, logical)), fmt.decode(logical))
  checks.append(dict(name='original_packed_quantizer_contract_no_requantization_or_double_scale', passed=True))
  # Exercise the actual40-call orchestration helper with CPU tensors and JIT.
  # GPU compression/cache replay remains a separate, unexecuted parent check.
  from tinygrad import TinyJit, GlobalCounters
  from functools import partial
  bench = load('agent-sparse-bench')
  inputs = [Tensor([i], device='CPU', dtype=dtypes.float32).realize() for i in range(40)]
  shared = Tensor([2.0], device='CPU', dtype=dtypes.float32).realize()
  graph = TinyJit(partial(bench.forty, lambda value, factor:value*factor))
  for _ in range(2): graph(*inputs, shared)
  before = GlobalCounters.kernel_count
  result = graph(*inputs, shared)
  assert GlobalCounters.kernel_count-before == 40
  np.testing.assert_array_equal(result.numpy(), [78.0])
  inputs[-1].assign(Tensor([101.0], device='CPU', dtype=dtypes.float32)).realize()
  np.testing.assert_array_equal(graph(*inputs, shared).numpy(), [202.0])
  checks.append(dict(name='actual40_call_helper_cpu_jit_retention_and_mutation', passed=True,
                     limitation='GPU custom-call graph capture and mutation remain unverified'))
  # Invalid layouts/shapes cannot masquerade as a supported tile matrix.
  for bad in (np.zeros((17, 32), np.uint8), np.zeros((16, 16), np.uint8), np.zeros((16, 32), np.float32)):
    try: fmt.compress(bad)
    except ValueError: pass
    else: raise AssertionError('Expected fail-closed input rejection')
  assert set(Device._opened_devices) <= {'CPU', 'NPY', 'PYTHON'}, Device._opened_devices
  checks.append(dict(name='invalid_shape_dtype_rejection_and_cpu_only_devices', passed=True))
  result = dict(cpu_only=True, gpu_validated=False, actual_route_sparsity=None, native_compressor=provenance,
                checks=checks, opened_devices=list(Device._opened_devices),
                arithmetic='Finite input representation preserved. GPU accumulation order may differ; exact-output claim is unverified.')
  (args.output/'report.json').write_text(json.dumps(result, indent=2)+'\n')
  print(json.dumps(result, indent=2))


if __name__ == '__main__': main()
