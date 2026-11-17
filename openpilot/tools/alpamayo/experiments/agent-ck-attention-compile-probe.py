"""CPU-only COMGR probe. Explicit CU mode; no renderer or GPU runtime edits.

Run under strace to audit device access. All files belong to this experiment.
"""
import ctypes
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).with_name('agent-ck-attention-work')
TINYGRAD = Path('/tmp/opencode/alpamayo-asius-20261114/tinygrad')
sys.path.insert(0, str(TINYGRAD))
os.environ['TMPDIR'] = str(ROOT)
os.environ['CACHEDB'] = str(ROOT / 'probe-cache.db')
os.environ['CPU'] = '1'
os.environ['AMD'] = '0'

from tinygrad.runtime.support.compiler_amd import comgr, check, set_options, _get_comgr_data


def compile_source(name, source, include_options=()):
  record = {'name': name, 'source_sha256': hashlib.sha256(source.encode()).hexdigest(),
            'arch': 'gfx1200', 'mode': 'CU', 'gpu_executed': False}
  (ROOT / f'{name}.cpp').write_text(source)
  options = ['-O3', '-mcumode', '--hip-version=6.4.60400', '-DHIP_VERSION_MAJOR=6',
             '-DHIP_VERSION_MINOR=4', '-DHIP_VERSION_PATCH=60400', '-D__HIPCC_RTC__',
             '-std=c++20', '-nogpuinc', '--offload-arch=gfx1200', '-Wno-missing-prototypes',
             '-I/opt/rocm/include', *include_options,
             '-Xclang', '-disable-llvm-passes', '-Xclang', '-aux-triple', '-Xclang', 'x86_64-unknown-linux-gnu']
  record['frontend_options'] = options
  record['codegen_options'] = ['-O3', '-mcumode', '-mllvm', '-amdgpu-internalize-symbols']
  datasets, ai, src = [], None, None
  started = time.monotonic()
  try:
    ai = comgr.amd_comgr_action_info_t()
    check(comgr.amd_comgr_create_action_info(ctypes.byref(ai)))
    check(comgr.amd_comgr_action_info_set_language(ai, comgr.AMD_COMGR_LANGUAGE_HIP))
    check(comgr.amd_comgr_action_info_set_isa_name(ai, b'amdgcn-amd-amdhsa--gfx1200'))
    check(comgr.amd_comgr_action_info_set_logging(ai, True))
    for _ in range(4):
      ds = comgr.amd_comgr_data_set_t()
      check(comgr.amd_comgr_create_data_set(ctypes.byref(ds)))
      datasets.append(ds)
    src = comgr.amd_comgr_data_t()
    check(comgr.amd_comgr_create_data(comgr.AMD_COMGR_DATA_KIND_SOURCE, ctypes.byref(src)))
    encoded = source.encode()
    check(comgr.amd_comgr_set_data(src, len(encoded), encoded))
    check(comgr.amd_comgr_set_data_name(src, f'{name}.cpp'.encode()))
    check(comgr.amd_comgr_data_set_add(datasets[0], src))
    stages = [
      ('frontend', comgr.AMD_COMGR_ACTION_COMPILE_SOURCE_WITH_DEVICE_LIBS_TO_BC, options),
      ('codegen', comgr.AMD_COMGR_ACTION_CODEGEN_BC_TO_RELOCATABLE, record['codegen_options']),
      ('link', comgr.AMD_COMGR_ACTION_LINK_RELOCATABLE_TO_EXECUTABLE, []),
    ]
    for i, (stage, action, flags) in enumerate(stages):
      check(set_options(ai, ' '.join(flags).encode()))
      status = comgr.amd_comgr_do_action(action, ai, datasets[i], datasets[i+1])
      if status:
        log = _get_comgr_data(datasets[i+1], comgr.AMD_COMGR_DATA_KIND_LOG).decode(errors='replace')
        (ROOT / f'{name}-{stage}.log').write_text(log)
        record.update(status='failed', failed_stage=stage, error=log)
        break
    else:
      blob = _get_comgr_data(datasets[-1], comgr.AMD_COMGR_DATA_KIND_EXECUTABLE)
      (ROOT / f'{name}.hsaco').write_bytes(blob)
      record.update(status='compiled_not_executed', binary_bytes=len(blob),
                    binary_sha256=hashlib.sha256(blob).hexdigest())
  finally:
    if src is not None: check(comgr.amd_comgr_release_data(src))
    for ds in datasets: check(comgr.amd_comgr_destroy_data_set(ds))
    if ai is not None: check(comgr.amd_comgr_destroy_action_info(ai))
  record['seconds'] = time.monotonic() - started
  return record


def main():
  basic = 'extern "C" __attribute__((global)) void ck_toolchain_probe(float *p) { p[0] = 1.0f; }\n'
  ck_include = '#include "ck_tile/core/config.hpp"\n' + basic
  selection = json.loads((ROOT / 'selection.json').read_text())
  selected_source = (ROOT / 'selected' / selection['filename']).read_text()
  records = [compile_source('toolchain-basic', basic),
             compile_source('toolchain-ck-header', ck_include, [f'-I{ROOT}/ck/include']),
             compile_source('toolchain-standard-header', '#include <type_traits>\n' + basic),
             compile_source('toolchain-selected', selected_source, [f'-I{ROOT}/ck/include',
               f'-I{ROOT}/ck/example/ck_tile/01_fmha', '-DCK_TILE_FMHA_FWD_FAST_EXP2=1',
               '-DCK_TILE_FLOAT_TO_BFLOAT16_DEFAULT=5', '-fgpu-flush-denormals-to-zero', '-fbracket-depth=1024'])]
  libraries = sorted({line.split()[-1] for line in Path('/proc/self/maps').read_text().splitlines()
                      if 'libamd_comgr' in line and line.split()[-1].startswith('/')})
  report = {'gpu_executed': False, 'probes': records, 'compiler_libraries': [
    {'path': p, 'sha256': hashlib.sha256(Path(p).read_bytes()).hexdigest()} for p in libraries]}
  (ROOT / 'compiler-probe.json').write_text(json.dumps(report, indent=2) + '\n')
  print(json.dumps(report, indent=2))


if __name__ == '__main__':
  main()
