"""CPU-only final ABI, graph, provenance, and numerical-gate audit."""
import difflib
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import sys
from types import SimpleNamespace

EXP = Path(__file__).parent
OLD = EXP/'agent-ck-attention-v2'
ROOT = Path('/mnt/nvme-500/tmp/alpamayo-speed-20261116/agent-ck-attention-v3')
os.environ.update(DEV='CPU', ALLOW_DEVICE_USAGE='0', TMPDIR=str(ROOT), CACHEDB=str(ROOT/'compiler-cache.db'),
                  OPENBLAS_NUM_THREADS='4', OMP_NUM_THREADS='4')
sys.path.insert(0, '/tmp/opencode/alpamayo-asius-20261114/tinygrad')


def load(name):
  spec = importlib.util.spec_from_file_location(name.replace('-', '_'), EXP/(name+'.py'))
  result = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(result)
  return result


def sha(data): return hashlib.sha256(data).hexdigest()


def main():
  from tinygrad import dtypes
  from tinygrad.uop.ops import UOp
  from tinygrad.renderer.llvmir import AMDLLVMRenderer
  from tinygrad.helpers import Target
  from tinygrad.codegen import to_program
  from tinygrad.runtime import ops_amd
  from tinygrad.runtime.support.hcq2 import layout_args
  from tinygrad.engine.realize import get_call_kernel_args
  adapter = load('agent-ck-attention-v2-adapter')
  runner = load('agent-ck-attention-v3-run')
  plan, oracle = runner.verify()
  shapes = [adapter.QSHAPE, adapter.LSHAPE, adapter.QSHAPE, adapter.KVSHAPE, adapter.KVSHAPE]
  params = [UOp.param(i, dtypes.float32 if i == 1 else dtypes.bfloat16, shape, 'NULL') for i, shape in enumerate(shapes)]
  raw = adapter.program(*params)
  program = to_program(raw, AMDLLVMRenderer(Target('AMD', 'LLVM', 'gfx1200', 'USB')))
  assert program.src[3].arg == adapter.artifacts()[1]
  assert program.arg.outs == (0, 1) and program.arg.ins == (2, 3, 4)
  dev = SimpleNamespace(arch='gfx1200', target=(12, 0, 0), iface=SimpleNamespace(props={'lds_size_in_kb':64}))
  data, image = ops_amd._amd_program_image(dev, adapter.artifacts()[1])
  # Runtime CALL buffers are flat storage views, after the scheduler removes logical reshapes.
  storage = [UOp.alloc((math.prod(shape)+512,), dtypes.float32 if i == 1 else dtypes.bfloat16, device='NULL')
             for i, shape in enumerate(shapes)]
  views = [s[256:256+math.prod(shape)] for s, shape in zip(storage, shapes)]
  call = program.call(*views)
  args = get_call_kernel_args(call, program, ('NULL',))
  assert [o for o, _ in layout_args(args)] == [0, 8, 16, 24, 32]
  with adapter.abi_scope():
    packed = ops_amd.AMDComputeQueue.kernargs(SimpleNamespace(dev=dev, devs=('NULL',)), call, program, data)
  assert packed.op.name == 'GETADDR'
  assert not adapter.ACTIVE
  import numpy as np
  reference = np.zeros(adapter.QSHAPE, np.float32)
  good = runner.metrics(reference, reference)
  runner.error_gate(good, good, plan['limits'])
  bad = runner.metrics(np.ones(adapter.QSHAPE, np.float32), reference)
  try: runner.error_gate(bad, good, plan['limits'])
  except AssertionError: pass
  else: raise AssertionError('Numerical gate did not reject a wrong output')
  original = (EXP/'agent-ck-attention-work/ck/include/ck_tile/host/device_prop.hpp').read_text()
  overlay = (OLD/'overlay/ck_tile/host/device_prop.hpp').read_text()
  expected = original.replace('inline size_t get_num_xccs()',
    '// Local device-only compilation overlay: this unused host query needs HIP newer than 6.4.\n#if !defined(__HIP_DEVICE_COMPILE__)\ninline size_t get_num_xccs()')
  expected = expected.replace('    return 1;\n}\n\n} // namespace ck_tile', '    return 1;\n}\n#endif\n\n} // namespace ck_tile')
  assert overlay == expected
  output = ROOT/'final-cpu-audit-r2'
  output.mkdir(exist_ok=False)
  (output/'device_prop.diff').write_text(''.join(difflib.unified_diff(original.splitlines(True), overlay.splitlines(True),
    fromfile='CK c56c6750/device_prop.hpp', tofile='device-only overlay/device_prop.hpp')))
  licenses = []
  hip = OLD/'sysroot/include/hip/hip_runtime.h'
  license_text = hip.read_text().split('*/', 1)[0]+'*/\n'
  assert 'Permission is hereby granted, free of charge' in license_text
  (output/'HIP-MIT.txt').write_text(license_text)
  llvm = OLD/'llvm-licenses/opt/rocm-6.4.0/share/doc/rocm-llvm/LICENSE.TXT'
  for source in (hip, llvm, EXP/'agent-ck-attention-work/ck/LICENSE'):
    licenses.append({'source_path':str(source), 'sha256':sha(source.read_bytes())})
  headers = {}
  for name, field in [('header-manifest.json','headers'), ('resource-header-manifest.json','files')]:
    record = json.loads((OLD/name).read_text())
    for file in record[field]: assert sha((OLD/file['extracted_path']).read_bytes()) == file['sha256']
    headers[name] = {'manifest_sha256':sha((OLD/name).read_bytes()), 'verified_files':len(record[field])}
  trace_text = '\n'.join(p.read_text() for p in OLD.glob('*.trace'))
  forbidden = re.findall(r'.*(?:/dev/dri/|/dev/kfd|/dev/bus/usb/|libamdhip64|libhsa-runtime64).*',trace_text)
  assert not forbidden
  disassembly = (OLD/'inspection/disassembly.txt').read_text()
  metadata = (OLD/'inspection/readelf.txt').read_text()
  assert 'There are no relocations in this file.' in metadata
  assert '.workgroup_processor_mode: 0' in metadata
  assert 's_load_b64 s[2:3], s[0:1], 0x28' in disassembly
  result = {'gpu_executed':False, 'full_lowering_keeps_exact_binary':True,
    'outputs':program.arg.outs, 'inputs':program.arg.ins, 'parameter_slots':[p.arg.slot for p in program.kernel_params],
    'flat_guarded_buffer_pointer_offsets':[o for o,_ in layout_args(args)], 'scoped_kernargs_constructed':True,
    'scope_restored':True, 'mode':'CU', 'mode_descriptor_bit':(data.rsrc1>>29)&1,
    'dispatch_pointer':data.enable_dispatch_ptr, 'image_bytes':len(image), 'image_sha256':sha(image),
    'scratch_metadata_bytes_per_thread':data.private_segment_size,
    'scratch_instruction_lines':len(re.findall(r'^\s*scratch_',disassembly,re.M)),
    'wmma_instruction_lines':len(re.findall(r'^\s*v_wmma_',disassembly,re.M)),
    'real_kernel_reads_block_count_x_y_at_offset40':True, 'relocations':0,
    'numerical_gate_accepts_exact_and_rejects_wrong':True, 'oracle_fixtures':len(oracle['fixtures']),
    'overlay_only_guards_unused_host_query':True, 'headers':headers, 'licenses':licenses,
    'compiler_trace_forbidden_access_matches':forbidden,
    'inspection_note':'COMGR disassembly of final ELF succeeded. A separate BC-to-assembly inspection failed; no alternate binary was substituted.'}
  (output/'audit.json').write_text(json.dumps(result,indent=2)+'\n')
  print(json.dumps(result,indent=2))


if __name__ == '__main__': main()
