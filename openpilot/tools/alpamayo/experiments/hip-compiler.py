"""Private compiler experiment; select WGP mode without changing tinygrad."""
from types import FunctionType
from tinygrad.device import Compiler
from tinygrad.runtime.support import compiler_amd


class WGPCompiler(Compiler):
  def __init__(self, arch='gfx1200'):
    self.arch = arch
    original = compiler_amd.compile_hip
    def options(action, value):
      return compiler_amd.set_options(action, value.replace(b'-mcumode', b'-mno-cumode'))
    self.compile_source = FunctionType(original.__code__, dict(original.__globals__, set_options=options),
                                       argdefs=original.__defaults__)
    super().__init__(f'alpamayo_hip_wgp_{arch}')

  def compile(self, source):
    return self.compile_source(source, self.arch)


def compiler(arch='gfx1200'):
  import os
  return WGPCompiler(arch) if os.environ.get('ALPAMAYO_HIP_WGP') == '1' else compiler_amd.HIPCompiler(arch)
