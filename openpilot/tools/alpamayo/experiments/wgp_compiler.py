"""Experimental WGP scheduling for the existing LLVM kernels; no arithmetic changes."""
from tinygrad.runtime.support.compiler_llvm import AMDLLVMCompiler, LLVMCompiler


class WGPCompiler(AMDLLVMCompiler):
  def __init__(self, arch):
    self.arch = arch
    LLVMCompiler.__init__(self, 'AMDGPU', arch, '-cumode')

  def __reduce__(self):
    return WGPCompiler, (self.arch,)


def install(device):
  assert device.arch in ('gfx1200', 'gfx1201')
  assert isinstance(device.compiler, AMDLLVMCompiler)
  device.renderer.compiler = WGPCompiler(device.arch)
