"""Compile and inspect GEMMs on the CPU. Never initialize a GPU device."""
import argparse
from collections import Counter
import ctypes
import importlib.util
import json
from pathlib import Path
import re
from types import SimpleNamespace


def local(name):
  spec = importlib.util.spec_from_file_location(name.replace('-','_'),Path(__file__).with_name(name+'.py'))
  value = importlib.util.module_from_spec(spec); spec.loader.exec_module(value)
  return value


def disassemble(binary):
  from tinygrad.runtime.support.compiler_amd import comgr, check, _get_comgr_data
  check(comgr.amd_comgr_create_action_info(ctypes.byref(action:=comgr.amd_comgr_action_info_t())))
  check(comgr.amd_comgr_action_info_set_isa_name(action,b'amdgcn-amd-amdhsa--gfx1200'))
  check(comgr.amd_comgr_create_data_set(ctypes.byref(data:=comgr.amd_comgr_data_set_t())))
  check(comgr.amd_comgr_create_data_set(ctypes.byref(result:=comgr.amd_comgr_data_set_t())))
  check(comgr.amd_comgr_create_data(comgr.AMD_COMGR_DATA_KIND_EXECUTABLE,ctypes.byref(item:=comgr.amd_comgr_data_t())))
  try:
    check(comgr.amd_comgr_set_data(item,len(binary),binary))
    check(comgr.amd_comgr_set_data_name(item,b'agent.hsaco'))
    check(comgr.amd_comgr_data_set_add(data,item))
    check(comgr.amd_comgr_do_action(comgr.AMD_COMGR_ACTION_DISASSEMBLE_EXECUTABLE_TO_SOURCE,action,data,result))
    return _get_comgr_data(result,comgr.AMD_COMGR_DATA_KIND_SOURCE).decode()
  finally:
    check(comgr.amd_comgr_release_data(item))
    check(comgr.amd_comgr_destroy_data_set(data)); check(comgr.amd_comgr_destroy_data_set(result))
    check(comgr.amd_comgr_destroy_action_info(action))


def metadata(binary):
  # Extract only integer fields. This avoids adding a MessagePack dependency.
  from tinygrad.runtime.support.elf import elf_loader
  _, sections, _ = elf_loader(binary)
  note = next(section.content for section in sections if section.name == '.note')
  result = {}
  for key in ('vgpr_count','sgpr_count','vgpr_spill_count','sgpr_spill_count','private_segment_fixed_size',
              'group_segment_fixed_size','wavefront_size','workgroup_processor_mode'):
    field = ('.'+key).encode()
    pos = note.index(field)+len(field)
    tag = note[pos]
    if tag < 128: result[key] = tag
    elif tag in (204,205,206,207):
      size = 1 << (tag-204)
      result[key] = int.from_bytes(note[pos+1:pos+1+size],'big')
    else: raise ValueError((key,tag))
  return result


def reference(rows, columns, width):
  import amd_gemm
  from tinygrad import dtypes
  from tinygrad.codegen import to_program
  from tinygrad.helpers import Target
  from tinygrad.renderer.llvmir import AMDLLVMRenderer
  from tinygrad.runtime.support.compiler_llvm import LLVMCompiler
  from tinygrad.uop.ops import Ops, UOp
  renderer = AMDLLVMRenderer(Target('AMD','LLVM','gfx1200'))
  renderer.compiler = LLVMCompiler('AMDGPU','gfx1200','-cumode')
  class Devices:
    DEFAULT = 'offline'
    def __getitem__(self,key): return SimpleNamespace(renderer=renderer)
  old = amd_gemm.Device
  try:
    amd_gemm.Device = Devices()
    out = UOp.placeholder((rows,columns),dtypes.bfloat16,0)
    a = UOp.placeholder((((rows+15)//16)*16,width),dtypes.fp8e4m3,1)
    w = UOp.placeholder((columns,width),dtypes.fp8e4m3,2)
    xs,ws = (UOp.placeholder((1,),dtypes.float32,i) for i in (3,4))
    program = to_program(amd_gemm._gemm(out,a,w,xs,ws),renderer)
    return next(item.arg for item in program.src if item.op == Ops.BINARY)
  finally: amd_gemm.Device = old


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--output',type=Path,required=True)
  args = parser.parse_args()
  args.output.mkdir(parents=True,exist_ok=False)
  kernel = local('agent-gemm-kernel')
  records = []
  cases = [(2027,9216,2048,False,False,1), (2112,2048,9216,False,False,1),
           (1440,4304,1152,True,True,1), (180,11520,4608,False,True,2),
           (180,2048,11520,False,True,2)]
  for rows, columns, width, row_scale, bias, images in cases:
    for mode in ('reference',*kernel.MODES):
      if mode == 'reference':
        if row_scale or bias or images != 1: continue
        text, binary = '', reference(rows,columns,width)
      else: text,binary = kernel.compiled(rows,columns,width,mode,row_scale,bias,images)
      stem = f'{rows}-{columns}-{width}-{images}-{mode}'
      asm = disassemble(binary)
      (args.output/(stem+'.s')).write_text(asm)
      (args.output/(stem+'.hsaco')).write_bytes(binary)
      if text: (args.output/(stem+'.cpp')).write_text(text)
      instructions = Counter(re.findall(r'^\s*([a-z][a-z0-9_]+)\s',asm,re.M))
      selected = {key:value for key,value in instructions.items() if any(token in key for token in
                  ('global_load','global_store','wmma','s_wait_load','s_cbranch','s_branch'))}
      record = dict(shape=[rows,columns,width],images=images,mode=mode,row_scale=row_scale,bias=bias,
                    binary_bytes=len(binary),resources=metadata(binary),static_instructions=selected)
      records.append(record)
      print(json.dumps(record),flush=True)
      (args.output/'report.json').write_text(json.dumps({'cpu_only':True,'records':records},indent=2)+'\n')


if __name__ == '__main__': main()
