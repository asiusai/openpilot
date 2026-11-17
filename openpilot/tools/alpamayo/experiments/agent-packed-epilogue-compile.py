"""CPU-only resource and assembly comparison for packed FC1 fusion."""
import argparse
from collections import Counter
from functools import partial
import importlib.util
import json
import os
from pathlib import Path
import re
import sys

os.environ.update(DEV='CPU', ALLOW_DEVICE_USAGE='0', BEAM='0', DEBUG='0')
HERE = Path(__file__).resolve().parent


def load(name):
  spec = importlib.util.spec_from_file_location('agent_packed_compile_'+name.replace('-', '_'), HERE/(name+'.py'))
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
  from tinygrad.uop.ops import UOp
  from amd_gemm import _gemm
  E, C, G, F, P = (load(n) for n in ('agent-expert-gemm', 'agent-gemm-compile', 'agent-gemm-kernel', 'gemm-epilogue', 'agent-packed-epilogue'))
  renderer = AMDLLVMRenderer(Target('AMD', 'LLVM', 'gfx1200'))
  renderer.compiler = LLVMCompiler('AMDGPU', 'gfx1200', '-cumode')
  report = {'gpu_executed': False, 'complete': False, 'records': [], 'source_sha256': {str(path): E.sha(path.read_bytes()) for path in
    (Path(__file__), HERE/'agent-packed-epilogue.py', HERE/'gemm-epilogue.py', HERE/'agent-gemm-kernel.py', HERE/'quantize-packed.py', HERE.parent/'amd_gemm.py')}}
  args.output.parent.mkdir(parents=True, exist_ok=True)
  for rows in (2112, 2027):
    columns, width = 9216, 2048
    for label in ('original-gemm', 'baseline-post', 'baseline-reciprocal', 'old-fused', 'transpose-gemm', *P.TILES):
      print(json.dumps({'rows': rows, 'variant': label}), flush=True)
      if label == 'baseline-reciprocal':
        schedule = Tensor.empty(1, device='CPU', dtype=dtypes.float32).reciprocal().schedule_linear()
        assert len(schedule.src) == 1
        program = to_program(schedule.src[0].body, renderer)
        text, binary, extension = program.src[2].arg, program.src[3].arg, '.ll'
      elif label == 'baseline-post':
        from amd_gemm import pack
        h = Tensor.empty(rows, columns, device='CPU', dtype=dtypes.bfloat16)
        output_scale = Tensor.empty(1, device='CPU', dtype=dtypes.float32)
        expression = pack((h.relu().square().float()/output_scale).clamp(-448, 448).cast(dtypes.fp8e4m3))
        schedule = expression.schedule_linear()
        assert len(schedule.src) == 1
        program = to_program(schedule.src[0].body, renderer)
        text, binary, extension = program.src[2].arg, program.src[3].arg, '.ll'
      elif label in ('original-gemm', 'old-fused'):
        old = label == 'old-fused'
        specs = [(((rows+15)//16*16 if old else rows, columns), dtypes.fp8e4m3 if old else dtypes.bfloat16),
                 (((rows+15)//16*16, width), dtypes.fp8e4m3), ((columns, width), dtypes.fp8e4m3),
                 ((1,), dtypes.float32), ((1,), dtypes.float32)]
        if old: specs.append(((1,), dtypes.float32))
        inputs = [UOp.placeholder(shape, dtype, i) for i, (shape, dtype) in enumerate(specs)]
        fn = partial(E.bind_renderer(F.program, renderer), rows=rows) if old else E.bind_renderer(_gemm, renderer)
        program = to_program(fn(*inputs), renderer)
        text, binary, extension = program.src[2].arg, program.src[3].arg, '.ll'
      elif label == 'transpose-gemm': text, binary = G.compiled(rows, columns, width); extension = '.cpp'
      else: text, binary = P.compiled(rows, columns, width, label); extension = '.cpp'
      asm, metadata = C.disassemble(binary), C.metadata(binary)
      ops = Counter(re.findall(r'^\s*(\w+)\s', asm, re.M))
      instructions = {op: count for op, count in ops.items() if any(term in op for term in ('store', 'wmma', 'cvt_pk_fp8', 'div_', 'rcp'))}
      if label in ('baseline-reciprocal', 'baseline-post', 'old-fused'):
        assert instructions.get('v_s_rcp_f32') == 1 and not any('div_' in op for op in instructions), instructions
      assert metadata['private_segment_fixed_size'] == metadata['group_segment_fixed_size'] == 0, metadata
      assert metadata['vgpr_spill_count'] == metadata['sgpr_spill_count'] == 0
      if label in P.TILES:
        assert not any('store' in op and ('b8' in op or 'byte' in op or 'short' in op or 'b16' in op) for op in instructions), instructions
        assert not any('div_' in op or 'f64' in op for op in instructions), instructions
        assert instructions.get('v_s_rcp_f32') == 1, instructions
        tm, tn = P.TILES[label]
        assert sum(count for op, count in instructions.items() if 'wmma' in op) == tm*tn
        assert instructions.get('global_store_b32') == 2*tm*tn and instructions.get('v_cvt_pk_fp8_f32') == 4*tm*tn
        for line in asm.splitlines():
          if 'v_wmma_f32_16x16x16_fp8_fp8' in line:
            registers = re.findall(r'v\[\d+:\d+\]', line)
            assert len(registers) == 4 and registers[0] == registers[-1], line
      prefix = args.output.parent/f'agent-packed-epilogue-{rows}-{label}'
      prefix.with_suffix(extension).write_text(text)
      prefix.with_suffix('.s').write_text(asm)
      prefix.with_suffix('.hsaco').write_bytes(binary)
      report['records'].append({'shape_mnk': [rows, columns, width], 'variant': label, 'resources': metadata,
        'instructions': instructions, 'source_sha256': E.sha(text.encode()), 'binary_sha256': E.sha(binary),
        'source': str(prefix.with_suffix(extension)), 'assembly': str(prefix.with_suffix('.s'))})
      E.write_json(args.output, report)
  report['complete'] = True
  assert not any(device.startswith('AMD') for device in Device._opened_devices)
  E.write_json(args.output, report)


if __name__ == '__main__': main()
