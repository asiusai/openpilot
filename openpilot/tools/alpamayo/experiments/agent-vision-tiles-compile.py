"""CPU-only gfx1200 image-tile compilation and 40-call dependency proofs."""
import argparse
from functools import lru_cache, partial
import importlib.util
import json
import os
from pathlib import Path
import re
import sys

HERE = Path(__file__).resolve().parent


@lru_cache(None)
def load(name):
  spec = importlib.util.spec_from_file_location('agent_vision_tiles_compile_'+name.replace('-', '_'), HERE/(name+'.py'))
  module = importlib.util.module_from_spec(spec)
  sys.modules[spec.name] = module
  spec.loader.exec_module(module)
  return module


def functions(renderer, images, rows, tile):
  E, V, T = (load(name) for name in ('agent-expert-gemm', 'agent-vision-tiles', 'tiled-gemm'))
  return {'current': partial(E.bind_renderer(T.kernel, renderer), tile_m=4, tile_n=4, row_scale=True),
    'candidate': partial(E.bind_renderer(V.kernel, renderer), images=images, rows=rows, tile_m=tile[0], tile_n=tile[1])}


def compile_one(renderer, images, shape, tile, variant, artifact=None):
  from tinygrad import dtypes
  from tinygrad.codegen import to_program
  from tinygrad.helpers import round_up
  from tinygrad.uop.ops import AxisType, Ops, UOp
  E, C = load('agent-expert-gemm'), load('agent-gemm-compile')
  rows, columns, width = shape
  out_shape = (images, rows, columns) if variant == 'candidate' else (images*rows, columns)
  x_shape = (images*round_up(rows, 16), width) if variant == 'candidate' else (round_up(images*rows, 16), width)
  specs = [(out_shape, dtypes.bfloat16), (x_shape, dtypes.fp8e4m3), ((columns, width), dtypes.fp8e4m3),
    ((images*rows,), dtypes.float32), ((1,), dtypes.float32), ((columns,), dtypes.bfloat16)]
  args = [UOp.placeholder(s, dtype, i) for i, (s, dtype) in enumerate(specs)]
  ast = functions(renderer, images, rows, tile)[variant](*args)
  reductions = [(u.axis_type.name, int(u.vmax+1)) for u in ast.toposort() if u.op is Ops.RANGE and u.axis_type in (AxisType.LOOP, AxisType.WEAK)]
  assert reductions == [('WEAK', width//16)], reductions
  program = to_program(ast, renderer)
  source, binary = program.src[2].arg, program.src[3].arg
  wmma = [line for line in source.splitlines() if 'call ' in line and '@llvm.amdgcn.wmma.f32.16x16x16.fp8.fp8' in line]
  expected_tile = tile if variant == 'candidate' else (4, 4)
  assert len(wmma) == expected_tile[0]*expected_tile[1]
  metadata = C.metadata(binary)
  assert metadata['private_segment_fixed_size'] == metadata['group_segment_fixed_size'] == 0
  assert metadata['vgpr_spill_count'] == metadata['sgpr_spill_count'] == 0
  params = [UOp.param(slot, arg.dtype, arg.numel(), device='AMD') for slot, arg in zip((5, 0, 1, 2, 3, 4), args, strict=True)]
  proof = E.prove_cpu_queue([program.call(*params)]*40, device='AMD')
  if artifact:
    artifact.with_suffix('.ll').write_text(source)
    artifact.with_suffix('.hsaco').write_bytes(binary)
    artifact.with_suffix('.s').write_text(C.disassemble(binary))
  return {'images': images, 'shape_mnk': shape, 'variant': variant, 'tile': expected_tile, 'resources': metadata,
    'global_size': program.arg.global_size, 'local_size': program.arg.local_size, 'k16_reduction': reductions,
    'wmma_per_iteration': len(wmma), 'source_sha256': E.sha(source.encode()), 'binary_sha256': E.sha(binary),
    'serial_40_call_queue_proof': proof}


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--output', type=Path, required=True)
  args = parser.parse_args()
  os.environ.update(DEV='CPU', ALLOW_DEVICE_USAGE='0', BEAM='0', PROFILE='0')
  from tinygrad import Device
  from tinygrad.helpers import Target
  from tinygrad.renderer.llvmir import AMDLLVMRenderer
  from tinygrad.runtime.support.compiler_llvm import LLVMCompiler
  E, V = load('agent-expert-gemm'), load('agent-vision-tiles')
  renderer = AMDLLVMRenderer(Target('AMD', 'LLVM', 'gfx1200'))
  renderer.compiler = LLVMCompiler('AMDGPU', 'gfx1200', '-cumode')
  report = {'complete': False, 'gpu_executed': False, 'records': [], 'source_sha256': {
    str(p): E.sha(p.read_bytes()) for p in (Path(__file__), HERE/'agent-vision-tiles.py', HERE/'image-gemm.py', HERE/'tiled-gemm.py')}}
  args.output.parent.mkdir(parents=True, exist_ok=True)
  for images in (1, 2):
    for fc in ('fc1', 'fc2'):
      for variant in ('current', 'candidate'):
        prefix = args.output.parent/f'agent-vision-tiles-b{images}-{fc}-{variant}'
        row = compile_one(renderer, images, V.SHAPES[fc], V.TILES[fc], variant, prefix)
        report['records'].append(row)
        print(json.dumps(row), flush=True)
        E.write_json(args.output, report)
    # Per-image row padding differs from concatenation in this control.
    for tile in V.TILES.values():
      prefix = args.output.parent/f'agent-vision-tiles-b{images}-tail-{tile[0]}x{tile[1]}'
      report['records'].append(compile_one(renderer, images, (65, 80, 64), tile, 'candidate', prefix))
  # Tie the B=1 arithmetic to the completed GPU experiment, not just a fresh
  # self-reference. The only allowed LLVM difference is the kernel symbol.
  work = Path('/mnt/nvme-500/tmp/alpamayo-speed-20261116')
  measured_path = work/'agent-matrix-tiles-kernels/report.json'
  measured = json.loads(measured_path.read_text())
  assert measured['complete'] and measured['gpu_executed']
  report['measured_single_image_evidence'] = {'report_sha256': E.sha(measured_path.read_bytes()), 'cases': []}
  normalize = lambda text: re.sub(r'@(?:alpamayo_fp8_gemm_[0-9_]+|agent_vision_tiles_[0-9x]+_b1)\(', '@test(', text)
  for fc, label in (('fc1', 'vision-up'), ('fc2', 'vision-down')):
    tile = 'x'.join(map(str, V.TILES[fc]))
    old = (work/f'agent-matrix-tiles-{label}-{tile}.ll').read_text()
    new = (args.output.parent/f'agent-vision-tiles-b1-{fc}-candidate.ll').read_text()
    result = next(c for c in measured['cases'] if c['case'] == label)['variants'][tile]
    assert E.sha(old.encode()) == result['llvm_sha256'], 'Saved LLVM does not match the measured GPU result'
    assert normalize(old) == normalize(new), 'B=1 arithmetic changed from the measured candidate'
    report['measured_single_image_evidence']['cases'].append({'case': label, 'tile': tile,
      'llvm_identical_except_symbol': True, 'measured_llvm_sha256': result['llvm_sha256']})
  assert not Device._opened_devices
  report['complete'] = True
  E.write_json(args.output, report)


if __name__ == '__main__': main()
