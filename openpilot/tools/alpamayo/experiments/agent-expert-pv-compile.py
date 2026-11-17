"""CPU-only gfx1200 PV compilation, original binary controls, and WAW proofs."""
import argparse
from functools import lru_cache, partial
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys

HERE = Path(__file__).resolve().parent


@lru_cache(None)
def load(name):
  spec = importlib.util.spec_from_file_location('agent_expert_pv_'+name.replace('-', '_'), HERE/(name+'.py'))
  module = importlib.util.module_from_spec(spec)
  sys.modules[spec.name] = module
  spec.loader.exec_module(module)
  return module


def provenance():
  import tinygrad
  E, K = load('agent-expert-gemm'), load('agent-expert-pv-kernel')
  root = Path(tinygrad.__file__).resolve().parent.parent
  revision = subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip()
  assert revision == K.TINYGRAD_REVISION
  assert E.sha((HERE.parent/'amd_attention.py').read_bytes()) == K.ORIGINAL_SHA256
  paths = [HERE.parent/'amd_attention.py', *[HERE/(n+'.py') for n in (
    'agent-expert-pv-kernel', 'agent-expert-pv-compile', 'agent-expert-pv-check', 'agent-expert-gemm',
    'agent-gemm-compile', 'agent-vision-tiles-transfer-check', 'prefill-kernels', 'wgp_compiler')]]
  # Hash the pinned compiler, scheduler, math, JIT, queue, and runtime sources too.
  paths += sorted((root/'tinygrad').rglob('*.py'))
  return {'tinygrad_revision': revision, 'source_sha256': {str(p): E.sha(p.read_bytes()) for p in paths}}


def functions(renderer, shape):
  from amd_attention import _matmul
  E, K = load('agent-expert-gemm'), load('agent-expert-pv-kernel')
  result = {'original': partial(E.bind_renderer(_matmul, renderer), shape=shape)}
  for tm, tn in ((4, 4), *K.TILES):
    result[f'{tm}x{tn}'] = partial(E.bind_renderer(K.kernel, renderer), shape=shape, tile_m=tm, tile_n=tn)
  return result


def specs(shape):
  batch, heads, kv_heads, rows, columns, width = shape
  r, n, k = ((x+15)//16*16 for x in (rows, columns, width))
  return (batch, heads, rows, columns), (batch, heads, r, k), (batch, kv_heads, n//16, k//16, 4, 16, 4)


def compile_one(renderer, shape, label, artifact=None):
  from tinygrad import dtypes
  from tinygrad.codegen import to_program
  from tinygrad.uop.ops import AxisType, Ops, UOp
  E, C = load('agent-expert-gemm'), load('agent-gemm-compile')
  args = [UOp.placeholder(s, dtypes.bfloat16, i) for i, s in enumerate(specs(shape))]
  ast = functions(renderer, shape)[label](*args)
  reductions = [(u.axis_type.name, int(u.vmax+1)) for u in ast.toposort()
                if u.op is Ops.RANGE and u.axis_type in (AxisType.LOOP, AxisType.WEAK)]
  assert reductions == [('WEAK', (shape[-1]+15)//16)], reductions
  program = to_program(ast, renderer)
  source, binary = program.src[2].arg, program.src[3].arg
  wmma = [line for line in source.splitlines() if 'call ' in line and '@llvm.amdgcn.wmma.f32.16x16x16.bf16' in line]
  tile = (4, 4) if label == 'original' else tuple(map(int, label.split('x')))
  # Whole masked tiles can disappear from the one-step control. With a loop,
  # the rendered IR can retain masked accumulator phis until LLVM optimization.
  assert 1 <= len(wmma) <= math.prod(tile), wmma
  if shape[3:5] == (64, 128): assert len(wmma) == math.prod(tile), wmma
  metadata = C.metadata(binary)
  assert metadata['group_segment_fixed_size'] == 0
  # The untouched actual-shape original has 75 VGPR spills and 292 private
  # bytes. Keep that baseline intact; candidates must eliminate the spills.
  if label not in ('original', '4x4'):
    assert metadata['private_segment_fixed_size'] == 0
    assert metadata['vgpr_spill_count'] == metadata['sgpr_spill_count'] == 0
  assembly = C.disassemble(binary)
  asm_wmma = re.findall(r'v_wmma_f32_16x16x16_bf16 (v\[\d+:\d+\]), (v\[\d+:\d+\]), (v\[\d+:\d+\]), (v\[\d+:\d+\]|0)', assembly)
  assert asm_wmma
  arithmetic = {'assembly_wmma_instructions': len(asm_wmma),
    'in_place_accumulator_recurrence': all(dest == accum for dest, _, _, accum in asm_wmma),
    'scratch_load_instructions': len(re.findall(r'\bscratch_load_', assembly)),
    'scratch_store_instructions': len(re.findall(r'\bscratch_store_', assembly))}
  if shape[3:5] == (64, 128):
    assert arithmetic['in_place_accumulator_recurrence']
    # The rendered loop remains K16. LLVM independently unrolls each narrow
    # machine loop twice; each WMMA still updates the same FP32 accumulator.
    induction = re.findall(r'(%v\d+) = phi i32 \[ 0, %loop_entry_3 \]', source)
    assert len(induction) == 1
    iv = induction[0]
    assert f'{iv}phi = add i32 {iv}, 1' in source and f'icmp ult i32 {iv}, 148' in source
    assert re.search(r'= shl i32 '+re.escape(iv)+r', 8\b', source)
    assert not re.search(r'\b(fadd|fsub|fmul|fdiv|fptrunc)\b', source)
    assert len(re.findall(r'add i32 %v\d+, 32767', source)) == math.prod(tile)*8
    assert len({dest for dest, _, _, _ in asm_wmma}) == math.prod(tile)
    arithmetic.update(ascending_148_k16_steps=True, packed_k16_stride_elements=256,
      extra_fp_arithmetic=False, original_integer_bf16_rne_conversions=math.prod(tile)*8,
      llvm_machine_loop_unroll=len(asm_wmma)//math.prod(tile))
  params = [UOp.param(slot, arg.dtype, arg.numel(), device='AMD') for slot, arg in zip((2, 0, 1), args, strict=True)]
  proof = E.prove_cpu_queue([program.call(*params)]*40, device='AMD')
  assert proof['output_input_slot'] == 2
  if artifact:
    artifact.with_suffix('.ll').write_text(source)
    artifact.with_suffix('.hsaco').write_bytes(binary)
    artifact.with_suffix('.s').write_text(assembly)
  return {'variant': label, 'shape_b_h_kvh_m_n_k': shape, 'tile': tile, 'resources': metadata,
          'global_size': program.arg.global_size, 'local_size': program.arg.local_size,
          'workgroups': math.prod(program.arg.global_size), 'k16_reduction': reductions, 'wmma_per_iteration': len(wmma),
          'llvm_sha256': E.sha(source.encode()), 'binary_sha256': E.sha(binary),
          'arithmetic_and_assembly_checks': arithmetic, 'serial_40_call_queue_proof': proof}


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--output', type=Path, required=True)
  args = parser.parse_args()
  os.environ.update(DEV='CPU', ALLOW_DEVICE_USAGE='0', BEAM='0', PROFILE='0', DEBUG='0')
  from tinygrad import Device
  from tinygrad.helpers import Target
  from tinygrad.renderer.llvmir import AMDLLVMRenderer
  from tinygrad.runtime.support.compiler_llvm import LLVMCompiler
  E, K = load('agent-expert-gemm'), load('agent-expert-pv-kernel')
  renderer = AMDLLVMRenderer(Target('AMD', 'LLVM', 'gfx1200'))
  renderer.compiler = LLVMCompiler('AMDGPU', 'gfx1200', '-cumode')
  args.output.parent.mkdir(parents=True, exist_ok=True)
  report = dict(provenance(), complete=False, gpu_executed=False, wgp=True, cases=[])
  E.write_json(args.output, report)
  for name, shape in K.CASES.items():
    row = {'case': name, 'variants': {}}
    for label in functions(renderer, shape):
      artifact = args.output.parent/f'agent-expert-pv-{name}-{label}'
      row['variants'][label] = compile_one(renderer, shape, label, artifact)
      print(json.dumps(row['variants'][label]), flush=True)
    for key in ('llvm_sha256', 'binary_sha256'):
      assert row['variants']['original'][key] == row['variants']['4x4'][key], f'Original control changed: {key}'
    row['original_4x4_llvm_and_binary_identical'] = True
    report['cases'].append(row)
    E.write_json(args.output, report)
  assert not Device._opened_devices
  assert provenance()['source_sha256'] == report['source_sha256'], 'Sources changed during CPU compilation'
  report['complete'] = True
  E.write_json(args.output, report)


if __name__ == '__main__': main()
