"""Compile six shared-weight GEMVs without opening any GPU device."""
import argparse
from collections import Counter
from functools import partial
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import sys

os.environ.update(DEV='CPU', ALLOW_DEVICE_USAGE='0', BEAM='0', DEBUG='0')
HERE = Path(__file__).resolve().parent


def load(name):
  spec = importlib.util.spec_from_file_location('agent_spec_head_compile_'+name.replace('-', '_'), HERE/(name+'.py'))
  module = importlib.util.module_from_spec(spec)
  sys.modules[spec.name] = module
  spec.loader.exec_module(module)
  return module


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--output', type=Path, required=True)
  args = parser.parse_args()
  from tinygrad import Device, Tensor, dtypes
  from tinygrad.codegen import to_program
  from tinygrad.helpers import Target
  from tinygrad.renderer.llvmir import AMDLLVMRenderer
  from tinygrad.runtime.support.compiler_llvm import LLVMCompiler
  from tinygrad.uop.ops import Ops, UOp
  from amd_decode import _gemv, gemv
  from model import EdgeModel
  C, H = load('agent-gemm-compile'), load('agent-spec-head')
  renderer = AMDLLVMRenderer(Target('AMD', 'LLVM', 'gfx1200'))
  renderer.compiler = LLVMCompiler('AMDGPU', 'gfx1200', '-cumode')
  report = {'gpu_executed': False, 'complete': False, 'records': [], 'source_sha256': {
    str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in
    (Path(__file__), HERE/'agent-spec-head.py', HERE.parent/'amd_decode.py', HERE.parent/'amd_attention.py')}}
  args.output.parent.mkdir(parents=True, exist_ok=True)
  # A unit-length K range trips this pinned tinygrad's original GEMV lowering.
  for vocab, width in ((64, 256), (2112, 2048)):
    for variant in ('original-single', 'shared-six'):
      if variant == 'original-single':
        specs = [((vocab,), dtypes.bfloat16), ((1, width), dtypes.fp8e4m3), ((vocab, width), dtypes.fp8e4m3),
                 ((1,), dtypes.float32), ((1,), dtypes.float32)]
        fn = partial(_gemv, packed=False)
      else:
        specs = [((6, vocab), dtypes.bfloat16), ((vocab, width), dtypes.fp8e4m3), ((1,), dtypes.float32)]
        specs += [spec for _ in range(6) for spec in (((1, width), dtypes.fp8e4m3), ((1,), dtypes.float32))]
        fn = H.kernel
      ast = fn(*(UOp.placeholder(shape, dtype, i) for i, (shape, dtype) in enumerate(specs)))
      program = to_program(ast, renderer)
      text = next(src.arg for src in program.src if src.op == Ops.SOURCE)
      binary = next(src.arg for src in program.src if src.op == Ops.BINARY)
      asm, resources = C.disassemble(binary), C.metadata(binary)
      # Ignore the unused standalone FP8 conversion helper emitted in the ELF.
      kernel_asm = asm[asm.index('\n'+ast.arg.name+':'):]
      ops = Counter(part.strip().split()[0] for line in kernel_asm.splitlines() for part in line.split('//')[0].split('::')
        if re.match(r'^\s*[vs]_[a-z0-9_]+\s|^\s*(global_|ds_)[a-z0-9_]+\s', part))
      instructions = {op: count for op, count in ops.items() if any(term in op for term in
        ('load', 'store', 'cvt', 'wmma', 'fma', 'mul_f32', 'add_f32', 'swizzle', 's_cbranch'))}
      assert resources['private_segment_fixed_size'] == resources['group_segment_fixed_size'] == 0, resources
      assert resources['vgpr_spill_count'] == resources['sgpr_spill_count'] == 0
      assert not any('wmma' in op for op in ops)
      prefix = args.output.parent/f'agent-spec-head-{vocab}-{width}-{variant}'
      prefix.with_suffix('.ll').write_text(text)
      prefix.with_suffix('.s').write_text(asm)
      prefix.with_suffix('.hsaco').write_bytes(binary)
      record = {'shape': [6 if variant == 'shared-six' else 1, vocab, width], 'variant': variant, 'resources': resources,
        'instructions': instructions, 'source': str(prefix.with_suffix('.ll')), 'assembly': str(prefix.with_suffix('.s')),
        'source_sha256': hashlib.sha256(text.encode()).hexdigest(), 'binary_sha256': hashlib.sha256(binary).hexdigest()}
      report['records'].append(record)
      print(json.dumps(record), flush=True)
      args.output.write_text(json.dumps(report, indent=2)+'\n')
  # Check the materialization change using the original model expressions.
  preparation = {}
  for variant in ('original-lazy-prep', 'candidate-prep-realized'):
    model = EdgeModel.__new__(EdgeModel)
    model.cuda_kernels = False
    model.weights = {'norm.weight': Tensor.empty(2048, device='CPU', dtype=dtypes.bfloat16)}
    hidden = Tensor.empty(1, 2112, 2048, device='CPU', dtype=dtypes.bfloat16)
    length = UOp.variable('valid_prefill_draft_length', 1, 2106).bind(2027)
    q, scale = model.quantize(model.norm(hidden[:, length-1:length], 'norm', rms=True).reshape(1, 2048))
    if variant == 'original-lazy-prep':
      weight = Tensor.empty(2112, 2048, device='CPU', dtype=dtypes.fp8e4m3)
      ws = Tensor.empty(1, device='CPU', dtype=dtypes.float32)
      calls = gemv(q, weight, (scale, ws), packed=False).linear_with_vars()[0].src[:-1]
    else: calls = q.linear_with_vars(scale)[0].src
    preparation[variant] = []
    for i, call in enumerate(calls):
      prg = to_program(call.body, renderer)
      source = next(src.arg for src in prg.src if src.op == Ops.SOURCE)
      prefix = args.output.parent/f'agent-spec-head-{variant}-{i}'
      prefix.with_suffix('.ll').write_text(source)
      preparation[variant].append({'source_sha256': hashlib.sha256(source.encode()).hexdigest(), 'source': str(prefix.with_suffix('.ll'))})
  assert [p['source_sha256'] for p in preparation['original-lazy-prep']] == [p['source_sha256'] for p in preparation['candidate-prep-realized']], preparation
  report['single_row_preparation'] = preparation
  report['single_row_preparation_llvm_identical'] = True
  report['complete'] = True
  assert not any(device.startswith('AMD') for device in Device._opened_devices)
  args.output.write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__': main()
