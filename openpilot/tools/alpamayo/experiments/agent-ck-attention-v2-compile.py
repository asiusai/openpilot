"""CPU-only COMGR compilation using isolated official HIP 6.4.0 headers."""
import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import sys
import time

BASE = Path(__file__).parent
ROOT = BASE / 'agent-ck-attention-v2'
CK = BASE / 'agent-ck-attention-work/ck'
TINYGRAD = Path('/tmp/opencode/alpamayo-asius-20261114/tinygrad')
sys.path.insert(0, str(TINYGRAD))
os.environ.update(TMPDIR=str(ROOT), CACHEDB=str(ROOT/'compiler-cache.db'), CPU='1', AMD='0')
from tinygrad.runtime.support.compiler_amd import comgr, check, set_options, _get_comgr_data


def compile_source(source_path, name, rtc=False, extra=()):
  assert name.replace('-', '').replace('_', '').isalnum()
  target = ROOT / name
  target.mkdir(exist_ok=False)
  source = source_path.read_bytes()
  (target/'source.cpp').write_bytes(source)
  options = ['-O3', '-mcumode', '--hip-version=6.4.43482', '-D__HIP_PLATFORM_AMD__=1',
             '-std=c++20', '-nogpuinc', '--offload-arch=gfx1200', '-Wno-missing-prototypes',
             f'-resource-dir={ROOT}/sysroot/clang-resource',
             f'-I{ROOT}/sysroot/include', f'-I{ROOT}/overlay', f'-I{CK}/include', f'-I{CK}/example/ck_tile/01_fmha',
             '-include', '__clang_hip_runtime_wrapper.h',
             '-include', 'hip/hip_runtime.h',
             '-DCK_TILE_FMHA_FWD_FAST_EXP2=1', '-DCK_TILE_FLOAT_TO_BFLOAT16_DEFAULT=5',
             '-fgpu-flush-denormals-to-zero', '-fbracket-depth=1024', *extra,
             '-Xclang', '-disable-llvm-passes', '-Xclang', '-aux-triple', '-Xclang', 'x86_64-unknown-linux-gnu']
  if rtc: options += ['-D__HIPCC_RTC__']
  codegen = ['-O3', '-mcumode', '-mllvm', '-amdgpu-internalize-symbols',
             '-mllvm', '-enable-post-misched=0', '-mllvm', '-amdgpu-early-inline-all=true',
             '-mllvm', '-amdgpu-function-calls=false']
  record = {'source': str(source_path), 'source_sha256': hashlib.sha256(source).hexdigest(),
            'compile_script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'arch': 'gfx1200', 'mode': 'CU', 'gpu_executed': False,
            'frontend_options': options, 'codegen_options': codegen, 'rtc': rtc,
            'compiler_path': '/opt/rocm-6.4.0/lib/libamd_comgr.so.3.0.60400',
            'compiler_sha256': 'e3102d0fb8ec1dfb9e219309f08f82100694e33398be14c3b2028b61b8b0595e',
            'hip_package_sha256': '82da61c3cfbb122df6fd2b14e57d7091a0a16cfe79cdbb2d4f7a56394e52314e',
            'overlay_sha256': hashlib.sha256((ROOT/'overlay/ck_tile/host/device_prop.hpp').read_bytes()).hexdigest(),
            'ck_commit': 'c56c6750d0fc54ed771d532cc92c316423449614'}
  assert hashlib.sha256(Path(record['compiler_path']).read_bytes()).hexdigest() == record['compiler_sha256']
  manifest = json.loads((ROOT/'header-manifest.json').read_text())
  for h in manifest['headers']:
    assert hashlib.sha256((ROOT/h['extracted_path']).read_bytes()).hexdigest() == h['sha256']
  resource_manifest = json.loads((ROOT/'resource-header-manifest.json').read_text())
  for h in resource_manifest['files']:
    assert hashlib.sha256((ROOT/h['extracted_path']).read_bytes()).hexdigest() == h['sha256']
  (target/'record.json').write_text(json.dumps(record,indent=2)+'\n')
  started = time.monotonic()
  sets, ai, src = [], None, None
  try:
    ai = comgr.amd_comgr_action_info_t()
    check(comgr.amd_comgr_create_action_info(ctypes.byref(ai)))
    check(comgr.amd_comgr_action_info_set_language(ai, comgr.AMD_COMGR_LANGUAGE_HIP))
    check(comgr.amd_comgr_action_info_set_isa_name(ai, b'amdgcn-amd-amdhsa--gfx1200'))
    check(comgr.amd_comgr_action_info_set_logging(ai, True))
    for _ in range(4):
      ds = comgr.amd_comgr_data_set_t()
      check(comgr.amd_comgr_create_data_set(ctypes.byref(ds)))
      sets.append(ds)
    src = comgr.amd_comgr_data_t()
    check(comgr.amd_comgr_create_data(comgr.AMD_COMGR_DATA_KIND_SOURCE, ctypes.byref(src)))
    check(comgr.amd_comgr_set_data(src, len(source), source))
    check(comgr.amd_comgr_set_data_name(src, b'agent_ck_attention.cpp'))
    check(comgr.amd_comgr_data_set_add(sets[0], src))
    stages = [('frontend', comgr.AMD_COMGR_ACTION_COMPILE_SOURCE_WITH_DEVICE_LIBS_TO_BC, options),
              ('codegen', comgr.AMD_COMGR_ACTION_CODEGEN_BC_TO_RELOCATABLE, codegen),
              ('link', comgr.AMD_COMGR_ACTION_LINK_RELOCATABLE_TO_EXECUTABLE, [])]
    for i, (stage, action, flags) in enumerate(stages):
      check(set_options(ai, ' '.join(flags).encode()))
      status = comgr.amd_comgr_do_action(action, ai, sets[i], sets[i+1])
      if status:
        log = _get_comgr_data(sets[i+1], comgr.AMD_COMGR_DATA_KIND_LOG).decode(errors='replace')
        (target/f'{stage}.log').write_text(log)
        record.update(status='failed', failed_stage=stage, error=log)
        break
      if stage == 'frontend':
        (target/'kernel.bc').write_bytes(_get_comgr_data(sets[1], comgr.AMD_COMGR_DATA_KIND_BC))
    else:
      blob = _get_comgr_data(sets[-1], comgr.AMD_COMGR_DATA_KIND_EXECUTABLE)
      (target/'kernel.hsaco').write_bytes(blob)
      record.update(status='compiled_not_executed', binary_bytes=len(blob), binary_sha256=hashlib.sha256(blob).hexdigest())
  finally:
    if src is not None: check(comgr.amd_comgr_release_data(src))
    for ds in sets: check(comgr.amd_comgr_destroy_data_set(ds))
    if ai is not None: check(comgr.amd_comgr_destroy_action_info(ai))
    record['seconds'] = time.monotonic()-started
    (target/'record.json').write_text(json.dumps(record,indent=2)+'\n')
  print(json.dumps(record,indent=2))
  return record


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--source', required=True, type=Path)
  parser.add_argument('--name', required=True)
  parser.add_argument('--rtc', action='store_true')
  args = parser.parse_args()
  compile_source(args.source, args.name, args.rtc)
