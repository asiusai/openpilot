"""CPU source-contract checks; no claim of compiled CK ABI or GPU correctness."""
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import struct
import subprocess

ROOT = Path(__file__).with_name('agent-ck-attention-work')


def f32(x):
  return struct.unpack('<f', struct.pack('<f', x))[0]


def main():
  selected = json.loads((ROOT / 'selection.json').read_text())
  compiler = json.loads((ROOT / 'compiler-probe.json').read_text())
  ck_probe = next(p for p in compiler['probes'] if p['name'] == 'toolchain-selected')
  assert ck_probe['status'] == 'failed' and 'hip/hip_version.h' in ck_probe['error']
  assert not (ROOT / 'toolchain-selected.hsaco').exists()
  source = (ROOT / 'selected' / selected['filename']).read_bytes()
  assert hashlib.sha256(source).hexdigest() == selected['source_sha256']
  assert selected['selection']['F_tile']['F_bm0'] == 64
  assert selected['selection']['F_tile']['F_bn0'] == 64
  # CK GetTileIndex, BHSD path: flatten the grid, regroup by head, reverse causal tiles.
  coverage = Counter()
  kv_heads = Counter()
  for gy in range(32):
    for gx in range(16):
      linear = gx + 16 * gy
      head, block = divmod(linear, 32)
      tile = 31 - block
      coverage[(head, tile)] += 1
      kv_heads[head // 2] += 1
  assert len(coverage) == 512 and set(coverage.values()) == {1}
  assert kv_heads == Counter({h: 64 for h in range(8)})
  strides = {
    'q': {'shape': [1, 16, 2048, 128], 'elements': [4194304, 262144, 128, 1]},
    'k': {'shape': [1, 8, 2048, 128], 'elements': [2097152, 262144, 128, 1]},
    'v': {'shape': [1, 8, 2048, 128], 'elements': [2097152, 262144, 128, 1]},
    'o': {'shape': [1, 16, 2048, 128], 'elements': [4194304, 262144, 128, 1]},
    'lse': {'shape': [1, 16, 2048], 'elements': [32768, 2048, 1]},
  }
  for value in strides.values():
    assert sum((n-1)*s for n, s in zip(value['shape'], value['elements'])) + 1 == math.prod(value['shape'])
  # CK source arithmetic: BF16 pack=8, 32 LDS banks, Q remains in registers.
  k_pack, banks = 8, 32
  k_space = (32//k_pack-1)*(64+1)*k_pack + (64-1)*k_pack + k_pack
  pixels_per_row = banks*4//2
  v_space = (32//k_pack)*(128//(pixels_per_row//k_pack))*(pixels_per_row+k_pack)
  lds = 2*max(k_space, v_space)
  assert (k_space, v_space, lds) == (2072, 4608, 9216)
  trace = (ROOT / 'compiler-access.strace').read_text()
  prohibited = ['/dev/kfd', '/dev/dri', '/dev/bus/usb', 'libamdhip64', 'libhsa-runtime64', 'libusb']
  assert not any(s in trace for s in prohibited)
  metadata = subprocess.run(['/usr/lib/llvm-18/bin/llvm-readelf', '--notes', str(ROOT/'toolchain-basic.hsaco')],
                            check=True, capture_output=True, text=True).stdout
  (ROOT / 'toolchain-basic-metadata.txt').write_text(metadata)
  assert '.workgroup_processor_mode: 0' in metadata and '.wavefront_size: 32' in metadata
  assert 'amdgcn-amd-amdhsa--gfx1200' in metadata
  report = {
    'status': 'source_contract_passed_compile_blocked',
    'gpu_executed': False, 'no_gpu_or_usb_access_in_compiler_trace': True,
    'ck_binary_exists': False, 'ck_argument_offsets_proven': False,
    'basic_probe_binary_mode': 'CU, ELF metadata workgroup_processor_mode=0',
    'source_derived_grid': [16, 32, 1], 'source_derived_threads': [128, 1, 1],
    'query_tile_coverage_exactly_once': len(coverage), 'gqa_blocks_per_kv_head': dict(kv_heads),
    'head_mapping': 'i_nhead_k = i_nhead / 2', 'layout_contract': strides,
    'scale_s_f32': f32(128**-0.5),
    'fast_exp2_karg_scale_f32': f32(f32(128**-0.5)*math.log2(math.e)),
    'mask': {'type': 'MASK_FROM_TOP_LEFT=1', 'left': -1, 'right': 0, 'sink_size': 0},
    'lds_source_derivation': {'q_bytes': 0, 'k_elements': k_space, 'v_elements': v_space,
                              'epilogue_bytes': 0, 'total_bytes': lds,
                              'compiled_descriptor_verified': False},
    'occupancy': {'launch_bound_min_blocks_per_cu': 2, 'actual': None,
                  'vgpr_count': None, 'scratch': None},
    'abi_next_step': 'Compile one native CK kentry and inspect aggregate layout, hidden arguments, descriptor and relocations.',
  }
  (ROOT / 'cpu-audit.json').write_text(json.dumps(report, indent=2) + '\n')
  print(json.dumps(report, indent=2))


if __name__ == '__main__':
  main()
