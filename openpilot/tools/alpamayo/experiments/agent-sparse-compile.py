"""CPU-only gfx1200 sparse FP8 compiler probe. Never opens a GPU."""
import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import re

os.environ.setdefault('DEV', 'CPU')
os.environ.setdefault('ALLOW_DEVICE_USAGE', '0')


def disassemble(binary):
  from tinygrad.runtime.support.compiler_amd import comgr, check, _get_comgr_data
  check(comgr.amd_comgr_create_action_info(ctypes.byref(action := comgr.amd_comgr_action_info_t())))
  check(comgr.amd_comgr_action_info_set_isa_name(action, b'amdgcn-amd-amdhsa--gfx1200'))
  check(comgr.amd_comgr_create_data_set(ctypes.byref(data := comgr.amd_comgr_data_set_t())))
  check(comgr.amd_comgr_create_data_set(ctypes.byref(result := comgr.amd_comgr_data_set_t())))
  check(comgr.amd_comgr_create_data(comgr.AMD_COMGR_DATA_KIND_EXECUTABLE, ctypes.byref(item := comgr.amd_comgr_data_t())))
  try:
    check(comgr.amd_comgr_set_data(item, len(binary), binary))
    check(comgr.amd_comgr_set_data_name(item, b'agent-sparse.hsaco'))
    check(comgr.amd_comgr_data_set_add(data, item))
    check(comgr.amd_comgr_do_action(comgr.AMD_COMGR_ACTION_DISASSEMBLE_EXECUTABLE_TO_SOURCE, action, data, result))
    return _get_comgr_data(result, comgr.AMD_COMGR_DATA_KIND_SOURCE).decode()
  finally:
    check(comgr.amd_comgr_release_data(item))
    check(comgr.amd_comgr_destroy_data_set(data))
    check(comgr.amd_comgr_destroy_data_set(result))
    check(comgr.amd_comgr_destroy_action_info(action))


def metadata(binary):
  from tinygrad.runtime.support.elf import elf_loader
  _, sections, _ = elf_loader(binary)
  note = next(section.content for section in sections if section.name == '.note')
  fields = {}
  for key in ('vgpr_count', 'sgpr_count', 'vgpr_spill_count', 'sgpr_spill_count',
              'private_segment_fixed_size', 'group_segment_fixed_size', 'wavefront_size', 'workgroup_processor_mode'):
    field = ('.'+key).encode()
    pos = note.index(field)+len(field)
    tag = note[pos]
    if tag < 128: fields[key] = tag
    elif tag in (204, 205, 206, 207):
      size = 1 << (tag-204)
      fields[key] = int.from_bytes(note[pos+1:pos+1+size], 'big')
    else: raise ValueError((key, tag))
  return fields


def probe_source(sparse=True):
  return r'''
typedef int i2 __attribute__((ext_vector_type(2)));
typedef int i4 __attribute__((ext_vector_type(4)));
typedef float f8 __attribute__((ext_vector_type(8)));
extern "C" __attribute__((device)) unsigned long __ockl_get_local_id(unsigned);
extern "C" __attribute__((global,amdgpu_flat_work_group_size(32,32)))
void agent_sparse_probe(float *out,const i2 *a,const i4 *b,const int *meta) {
  unsigned lane=__ockl_get_local_id(0);
  f8 c=(f8){0};
''' + (r'''
  c=__builtin_amdgcn_swmmac_f32_16x16x32_fp8_fp8_w32(a[lane],b[lane],c,meta[lane]);
''' if sparse else r'''
  i4 bv=b[lane];
  c=__builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(a[lane],(i2){bv[0],bv[1]},c);
''') + r'''
  for(unsigned i=0;i<8;i++) out[lane*8+i]=c[i];
}
'''


def save_compile(out, name, text):
  from tinygrad.runtime.support.compiler_amd import HIPCompiler
  binary = HIPCompiler('gfx1200').compile(text)
  asm = disassemble(binary)
  (out/(name+'.cpp')).write_text(text)
  (out/(name+'.hsaco')).write_bytes(binary)
  (out/(name+'.s')).write_text(asm)
  return dict(name=name, source_sha256=hashlib.sha256(text.encode()).hexdigest(),
              binary_sha256=hashlib.sha256(binary).hexdigest(), binary_bytes=len(binary), resources=metadata(binary),
              compiler_identification=[value.decode() for value in re.findall(rb'[\x20-\x7e]{12,}', binary)
                                       if b'AMD clang version' in value or b'Linker: AMD LLD' in value],
              matrix_instructions=re.findall(r'^\s*(v_(?:s?wmma\w+))\s', asm, re.M))


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--output', type=Path, default=Path(__file__).with_name('agent-sparse-compiler-output'))
  args = parser.parse_args()
  args.output.mkdir(exist_ok=False)
  records = [save_compile(args.output, 'sparse-probe', probe_source()),
             save_compile(args.output, 'dense-probe', probe_source(False))]
  kernel_path = Path(__file__).with_name('agent-sparse-kernel.py')
  if kernel_path.exists():
    import importlib.util
    spec = importlib.util.spec_from_file_location('agent_sparse_kernel', kernel_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for mode in ('compress', 'dense', 'hybrid'):
      records.append(save_compile(args.output, mode, module.source(32, 32, 64, mode)))
      records.append(save_compile(args.output, mode+'-fc2-shape', module.source(2112, 2048, 9216, mode)))
    # Build actual custom-call UOps on the CPU, including all scratch outputs.
    from tinygrad import dtypes
    from tinygrad.uop.ops import UOp, Ops
    contracts = []
    for mode in ('compress', 'dense', 'hybrid'):
      shapes = ((4,32,2), (4,32), (4,), (32,64)) if mode == 'compress' else (
        (32,32), (32,64), (32,64), (4,32,2), (4,32), (4,), (1,), (1,))
      types = (dtypes.uint32,)*3+(dtypes.fp8e4m3,) if mode == 'compress' else (
        dtypes.float32, dtypes.fp8e4m3, dtypes.fp8e4m3, dtypes.uint32, dtypes.uint32, dtypes.uint32, dtypes.float32, dtypes.float32)
      inputs = [UOp.placeholder(shape, dtype, i) for i,(shape,dtype) in enumerate(zip(shapes,types,strict=True))]
      program = module.program(*inputs, rows=32, columns=32, width=64, mode=mode)
      sink = program.src[0]
      slots = sorted(value.arg.slot for value in sink.toposort() if value.op == Ops.PARAM)
      stores = sum(value.op == Ops.STORE for value in sink.toposort())
      assert slots == list(range(len(inputs))) and stores == (3 if mode == 'compress' else 1)
      contracts.append(dict(mode=mode, dynamic_parameter_slots=slots, written_outputs=stores))
  from tinygrad import Device
  assert not Device._opened_devices, Device._opened_devices
  report = dict(cpu_only=True, gpu_opened=False, target='gfx1200', records=records,
                custom_call_contracts=contracts if kernel_path.exists() else [])
  (args.output/'report.json').write_text(json.dumps(report, indent=2)+'\n')
  print(json.dumps(report, indent=2))


if __name__ == '__main__': main()
