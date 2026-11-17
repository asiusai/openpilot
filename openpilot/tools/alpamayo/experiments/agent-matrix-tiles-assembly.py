"""Emit assembly and LLVM occupancy estimates from saved IR, without GPU use."""
import argparse
import ctypes
import json
import os
from pathlib import Path
import re

os.environ.update(DEV='CPU', ALLOW_DEVICE_USAGE='0')
from tinygrad.runtime.autogen import llvm
from tinygrad.runtime.support.compiler_llvm import LLVMCompiler, expect, cerr
from tinygrad import Device


def assembly(compiler, source):
  encoded = source.encode()
  buffer = llvm.LLVMCreateMemoryBufferWithMemoryRangeCopy(ctypes.create_string_buffer(encoded), len(encoded), b'agent-matrix-tiles')
  module = expect(llvm.LLVMParseIRInContext(compiler.context, buffer, ctypes.pointer(value:=llvm.LLVMModuleRef()), error:=cerr()), error, value)
  layout = llvm.LLVMCreateTargetDataLayout(compiler.target_machine)
  llvm.LLVMSetModuleDataLayout(module, layout)
  llvm.LLVMDisposeTargetData(layout)
  expect(llvm.LLVMVerifyModule(module, llvm.LLVMReturnStatusAction, error:=cerr()), error)
  expect(llvm.LLVMRunPasses(module, compiler.passes, compiler.target_machine, compiler.pbo), 'LLVM pass failure')
  result = expect(llvm.LLVMTargetMachineEmitToMemoryBuffer(compiler.target_machine, module, llvm.LLVMAssemblyFile,
    error:=cerr(), ctypes.pointer(value:=llvm.LLVMMemoryBufferRef())), error, value)
  text = ctypes.string_at(llvm.LLVMGetBufferStart(result), llvm.LLVMGetBufferSize(result)).decode()
  llvm.LLVMDisposeMemoryBuffer(result)
  llvm.LLVMDisposeModule(module)
  return text


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--compiled', type=Path, required=True)
  parser.add_argument('--output', type=Path, required=True)
  args = parser.parse_args()
  compiler = LLVMCompiler('AMDGPU', 'gfx1200', '-cumode')
  llvm.LLVMSetTargetMachineAsmVerbosity(compiler.target_machine, True)
  report = {'gpu_executed': False, 'scope': 'LLVM compiler estimates, not measured occupancy', 'records': []}
  for case in json.loads(args.compiled.read_text())['cases']:
    # Controls have constant dimensions where LLVM can remove discarded fragments.
    # Resource checks cover controls; this assembly count checks the real shapes.
    if case['case'].startswith('control'): continue
    for label, info in case['variants'].items():
      path = args.compiled.parent/f"agent-matrix-tiles-{case['case']}-{label}.ll"
      text = assembly(compiler, path.read_text())
      path.with_suffix('.s').write_text(text)
      estimates = {key.strip(): int(value) for key, value in re.findall(r';\s*(Occupancy|NumVgprs|NumSgprs|ScratchSize|LDSByteSize):\s*(\d+)', text)}
      wmma = [line.strip() for line in text.splitlines() if line.strip().startswith('v_wmma_f32_16x16x16_fp8_fp8')]
      assert len(wmma) == info['wmma_instructions_per_iteration'], (case['case'], label, len(wmma))
      assert estimates['NumVgprs'] == info['resources']['vgpr_count'], estimates
      report['records'].append({'case': case['case'], 'variant': label, 'compiler_estimates': estimates,
                                'wmma_per_iteration': len(wmma), 'wmma_instructions': wmma})
  assert not Device._opened_devices
  args.output.write_text(json.dumps(report, indent=2)+'\n')
  print(json.dumps({'records': len(report['records']), 'gpu_executed': False}))
