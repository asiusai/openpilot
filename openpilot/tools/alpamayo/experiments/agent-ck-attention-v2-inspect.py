"""Inspect the compiled CK code object using CPU-only COMGR and tinygrad parsing."""
import ctypes
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

ROOT = Path(__file__).with_name('agent-ck-attention-v2')
sys.path.insert(0, '/tmp/opencode/alpamayo-asius-20261114/tinygrad')
os.environ.update(CPU='1', AMD='0', TMPDIR=str(ROOT), CACHEDB=str(ROOT/'compiler-cache.db'))
from tinygrad.runtime.support.compiler_amd import comgr, check, set_options, _get_comgr_data
from tinygrad.runtime.ops_amd import _amd_program_image


def comgr_action(blob, kind, action, output_kind, flags):
  ai, src = comgr.amd_comgr_action_info_t(), comgr.amd_comgr_data_t()
  inp, out = comgr.amd_comgr_data_set_t(), comgr.amd_comgr_data_set_t()
  check(comgr.amd_comgr_create_action_info(ctypes.byref(ai)))
  check(comgr.amd_comgr_create_data(kind, ctypes.byref(src)))
  check(comgr.amd_comgr_create_data_set(ctypes.byref(inp)))
  check(comgr.amd_comgr_create_data_set(ctypes.byref(out)))
  try:
    check(comgr.amd_comgr_action_info_set_language(ai, comgr.AMD_COMGR_LANGUAGE_HIP))
    check(comgr.amd_comgr_action_info_set_isa_name(ai, b'amdgcn-amd-amdhsa--gfx1200'))
    check(set_options(ai, ' '.join(flags).encode()))
    check(comgr.amd_comgr_set_data(src, len(blob), blob))
    check(comgr.amd_comgr_set_data_name(src, b'kernel'))
    check(comgr.amd_comgr_data_set_add(inp, src))
    check(comgr.amd_comgr_do_action(action, ai, inp, out))
    return _get_comgr_data(out, output_kind)
  finally:
    check(comgr.amd_comgr_release_data(src))
    for ds in (inp, out): check(comgr.amd_comgr_destroy_data_set(ds))
    check(comgr.amd_comgr_destroy_action_info(ai))


if __name__ == '__main__':
  folder = ROOT/'inspection'
  folder.mkdir(exist_ok=False)
  build = ROOT/'native-overlay'
  record = json.loads((build/'record.json').read_text())
  blob = (build/'kernel.hsaco').read_bytes()
  assert hashlib.sha256(blob).hexdigest() == record['binary_sha256']
  text = subprocess.check_output(['llvm-readelf-18', '--all', str(build/'kernel.hsaco')], text=True)
  (folder/'readelf.txt').write_text(text)
  dev = SimpleNamespace(target=(12, 0, 0), iface=SimpleNamespace(props={'lds_size_in_kb': 64}))
  data, image = _amd_program_image(dev, blob)
  (folder/'loader.json').write_text(json.dumps({'data': asdict(data), 'image_bytes': len(image),
    'image_sha256': hashlib.sha256(image).hexdigest(), 'gpu_executed': False}, indent=2)+'\n')
  (folder/'disassembly.txt').write_bytes(comgr_action(blob, comgr.AMD_COMGR_DATA_KIND_EXECUTABLE,
    comgr.AMD_COMGR_ACTION_DISASSEMBLE_EXECUTABLE_TO_SOURCE, comgr.AMD_COMGR_DATA_KIND_SOURCE, []))
  (folder/'assembly.s').write_bytes(comgr_action((build/'kernel.bc').read_bytes(), comgr.AMD_COMGR_DATA_KIND_BC,
    comgr.AMD_COMGR_ACTION_CODEGEN_BC_TO_ASSEMBLY, comgr.AMD_COMGR_DATA_KIND_SOURCE, record['codegen_options']))
  print((folder/'loader.json').read_text())
