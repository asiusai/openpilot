"""CPU-only HIP compilation, native opcode assertion, and resource inspection."""
import argparse
from collections import Counter
import ctypes
import importlib.util
import json
from pathlib import Path
import re
from types import SimpleNamespace


def local(name):
  spec = importlib.util.spec_from_file_location(name.replace('-', '_'), Path(__file__).with_name(name+'.py'))
  module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
  return module


def disassemble(binary):
  from tinygrad.runtime.support.compiler_amd import comgr, check, _get_comgr_data
  check(comgr.amd_comgr_create_action_info(ctypes.byref(action := comgr.amd_comgr_action_info_t())))
  check(comgr.amd_comgr_action_info_set_isa_name(action, b'amdgcn-amd-amdhsa--gfx1200'))
  check(comgr.amd_comgr_create_data_set(ctypes.byref(data := comgr.amd_comgr_data_set_t())))
  check(comgr.amd_comgr_create_data_set(ctypes.byref(result := comgr.amd_comgr_data_set_t())))
  check(comgr.amd_comgr_create_data(comgr.AMD_COMGR_DATA_KIND_EXECUTABLE, ctypes.byref(item := comgr.amd_comgr_data_t())))
  try:
    check(comgr.amd_comgr_set_data(item, len(binary), binary))
    check(comgr.amd_comgr_set_data_name(item, b'agent-int4k32.hsaco'))
    check(comgr.amd_comgr_data_set_add(data, item))
    check(comgr.amd_comgr_do_action(comgr.AMD_COMGR_ACTION_DISASSEMBLE_EXECUTABLE_TO_SOURCE, action, data, result))
    return _get_comgr_data(result, comgr.AMD_COMGR_DATA_KIND_SOURCE).decode()
  finally:
    check(comgr.amd_comgr_release_data(item))
    check(comgr.amd_comgr_destroy_data_set(data)); check(comgr.amd_comgr_destroy_data_set(result))
    check(comgr.amd_comgr_destroy_action_info(action))


def metadata(binary):
  from tinygrad.runtime.support.elf import elf_loader
  _, sections, _ = elf_loader(binary)
  note = next(section.content for section in sections if section.name == '.note')
  result = {}
  for key in ('vgpr_count', 'sgpr_count', 'vgpr_spill_count', 'sgpr_spill_count', 'private_segment_fixed_size',
              'group_segment_fixed_size', 'wavefront_size', 'workgroup_processor_mode'):
    field = ('.'+key).encode(); pos = note.index(field)+len(field); tag = note[pos]
    if tag < 128: result[key] = tag
    elif tag in (204, 205, 206, 207):
      size = 1 << (tag-204); result[key] = int.from_bytes(note[pos+1:pos+1+size], 'big')
    else: raise ValueError((key, tag))
  return result


def fp8_binary(rows, columns, width, dtype):
  """Compile the independent FP8 controls without opening an AMD device."""
  import amd_gemm
  from tinygrad import dtypes
  from tinygrad.codegen import to_program
  from tinygrad.helpers import Target
  from tinygrad.renderer.llvmir import AMDLLVMRenderer
  from tinygrad.runtime.support.compiler_llvm import LLVMCompiler
  from tinygrad.uop.ops import Ops, UOp
  renderer = AMDLLVMRenderer(Target('AMD', 'LLVM', 'gfx1200'))
  renderer.compiler = LLVMCompiler('AMDGPU', 'gfx1200', '-cumode')
  class Devices:
    DEFAULT = 'offline'
    def __getitem__(self, key): return SimpleNamespace(renderer=renderer)
  original = amd_gemm.Device
  try:
    amd_gemm.Device = Devices()
    out = UOp.placeholder((rows, columns), dtype, 0)
    a = UOp.placeholder((((rows+15)//16)*16, width), dtypes.fp8e4m3, 1)
    b = UOp.placeholder((columns, width), dtypes.fp8e4m3, 2)
    xs, ws = (UOp.placeholder((1,), dtypes.float32, i) for i in (3, 4))
    program = to_program(amd_gemm._gemm(out, a, b, xs, ws), renderer)
    return next(item.arg for item in program.src if item.op == Ops.BINARY)
  finally: amd_gemm.Device = original


def main():
  p = argparse.ArgumentParser(description=__doc__); p.add_argument('--output', type=Path, required=True); args = p.parse_args()
  args.output.mkdir(parents=True, exist_ok=False)
  k32, k16 = local('agent-int4k32-kernel'), local('int4-gemm')
  from tinygrad import dtypes
  records = []
  for m, n, k in ((17, 33, 64), (2112, 9216, 2048), (2112, 2048, 9216)):
    for mode in ('k16-tile2-bf16', 'k16-tile4-bf16', 'k32-bf16', 'k32-i32', 'fp8-f32', 'fp8-bf16'):
      if mode.startswith('fp8'):
        text, binary = '', fp8_binary(m, n, k, dtypes.float32 if mode == 'fp8-f32' else dtypes.bfloat16)
      elif mode.startswith('k16'): text, binary = k16.compiled(m, n, k, 4 if 'tile4' in mode else 2)
      else: text, binary = k32.compiled(m, n, k, mode[4:])
      asm = disassemble(binary)
      instructions = Counter(re.findall(r'^\s*([a-z][a-z0-9_]+)\s', asm, re.M))
      if mode.startswith('k32'):
        assert instructions['v_wmma_i32_16x16x32_iu4'] > 0, 'Native K32 instruction missing'
        assert instructions['v_wmma_i32_16x16x16_iu4'] == 0, 'Unexpected K16 expansion'
      stem = f'{m}-{n}-{k}-{mode}'
      (args.output/(stem+'.cpp')).write_text(text)
      (args.output/(stem+'.s')).write_text(asm)
      (args.output/(stem+'.hsaco')).write_bytes(binary)
      record = dict(shape=[m, n, k], mode=mode, resources=metadata(binary), binary_bytes=len(binary),
                    static_instructions={key: value for key, value in instructions.items() if any(token in key for token in
                      ('wmma', 'global_load', 'global_store', 's_wait_load', 's_cbranch', 's_branch'))})
      records.append(record); print(json.dumps(record), flush=True)
  (args.output/'report.json').write_text(json.dumps(dict(cpu_only=True, records=records), indent=2)+'\n')


if __name__ == '__main__': main()
