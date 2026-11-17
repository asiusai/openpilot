"""CPU-only selection of one unmodified CK forward specialization.

The selected source is a compile candidate, not a validated GPU kernel.
No setup.py, HIP runtime, device query, or downloaded build script executes.
"""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).with_name('agent-ck-attention-work')
CK = ROOT / 'ck'
FMHA = CK / 'example/ck_tile/01_fmha'
NAME = ('fmha_fwd_d128_bf16_batch_b64x64x32x128x32x128_r4x1x1_r4x1x1'
        '_w16x16x16_w16x16x16_qr_vr_pssk_nlogits_nbias_mask_lse_ndropout_nskip_nqscale_ntrload_nsink')


def main():
  sys.path.insert(0, str(FMHA))
  # Reviewed imports: stdlib, arch.py, cmake_config.py, cpp_symbol_map.py, utils.py.
  from codegen.ops.fmha_fwd import get_fwd_blobs
  pool, kernels = get_fwd_blobs(['gfx1200'], NAME, 2, [128], 'simplified')
  assert len(kernels) == 1 and kernels[0].name == NAME
  selected = kernels[0]
  assert selected.F_arch.name == 'gfx12'
  assert selected.F_tile.F_constraint.bool_expr == 'a.max_seqlen_q <= 4096'
  assert selected.F_tile.F_occupancy == -1
  generated = ROOT / 'selected'
  generated.mkdir(exist_ok=True)
  source = selected.render()
  (generated / selected.filename).write_text(source)
  # Record dispatcher predicates without compiling/querying the HIP host runtime.
  (generated / 'dispatch-reference.txt').write_text(pool.render('selected_only'))
  imports = [p for p in sys.modules.values() if getattr(p, '__file__', None)
             and Path(p.__file__).is_relative_to(FMHA)]
  record = {
    'status': 'source_selected_not_compiled', 'gpu_executed': False,
    'ck_commit': 'c56c6750d0fc54ed771d532cc92c316423449614', 'ck_license': 'MIT',
    'flash_attention_commit': '10b74d98fb26113b014b33d947c8bbe4e83fbf7a',
    'flash_attention_license': 'BSD-3-Clause',
    'community_recipe_commit': '525f8096169d416dc15ea15846c179a6e0137840',
    'selection': asdict(selected), 'filename': selected.filename,
    'source_sha256': hashlib.sha256(source.encode()).hexdigest(),
    'generator_sha256': {str(Path(m.__file__).relative_to(CK)): hashlib.sha256(Path(m.__file__).read_bytes()).hexdigest()
                         for m in imports},
    'required_policy': {'arch': 'gfx1200', 'mode': 'CU, explicit compile and descriptor check required',
                        'CK_TILE_FMHA_FWD_FAST_EXP2': 1, 'CK_TILE_FLOAT_TO_BFLOAT16_DEFAULT': 5,
                        'flush_denormals_to_zero': True, 'language': 'c++20'},
    'shape': {'B': 1, 'QH': 16, 'KVH': 8, 'R': 2048, 'K': 2048, 'D': 128, 'causal': True},
    'source_derived_launch': {'grid': [16, 32, 1], 'threads': [128, 1, 1], 'wave_size': 32,
                             'blocks': 512, 'min_blocks_per_cu_launch_bound': 2,
                             'actual_occupancy': None, 'vgprs': None, 'scratch': None},
    'numerical_status': 'variant; no exactness or quality claim',
  }
  (ROOT / 'selection.json').write_text(json.dumps(record, indent=2) + '\n')
  print(json.dumps({'kernel': NAME, 'selected_count': len(kernels), 'status': record['status']}, indent=2))


if __name__ == '__main__':
  main()
