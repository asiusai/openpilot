"""Process-local LLVM backend configuration for the selected recipe."""
import ctypes
import hashlib
from tinygrad import Device, dtypes
from tinygrad.codegen import do_to_program
from tinygrad.device import Compiled
from tinygrad.runtime.support.compiler_llvm import AMDLLVMCompiler, LLVMCompiler
from tinygrad.uop.ops import UOp
from .frozen import ASSETS, checked, manifest, verify_assets
from .kernels.expert_pv import kernel


class WGPCompiler(AMDLLVMCompiler):
  def __init__(self, arch):
    self.arch = arch
    LLVMCompiler.__init__(self, 'AMDGPU', arch, '-cumode')

  def __reduce__(self):
    return WGPCompiler, (self.arch,)


class Backend:
  active = False

  def __init__(self, timeout_ms):
    if Backend.active:
      raise RuntimeError('Use one selected runtime per dedicated process')
    if timeout_ms <= 0:
      raise ValueError('Device timeout must be positive')
    verify_assets()
    from tinygrad.runtime.autogen import llvm
    major, minor, patch = (ctypes.c_uint32() for _ in range(3))
    llvm.LLVMGetVersion(ctypes.byref(major), ctypes.byref(minor), ctypes.byref(patch))
    if major.value != 22:
      raise RuntimeError(f'Selected arithmetic requires LLVM22; found {major.value}')
    from tinygrad.runtime import ops_amd
    self.device = Device[Device.DEFAULT]
    if self.device.arch != 'gfx1200' or str(self.device.renderer.target) != 'USB+AMD:LLVM:gfx1200':
      raise ValueError('Selected inference requires USB+AMD:LLVM:gfx1200')
    if not isinstance(self.device.renderer.compiler, AMDLLVMCompiler):
      raise ValueError('Selected inference requires AMD LLVM')
    self.compiler = self.device.renderer.compiler
    self.old_packet = ops_amd.dispatch_packet
    self.old_timeout = Compiled.wait_timeout_ms
    self.closed = False
    self.device.renderer.compiler = WGPCompiler(self.device.arch)
    original = self.old_packet
    def typed_packet(*args, **kwargs):
      return [UOp.const(word, dtypes.uint32) if isinstance(word, int) else word for word in original(*args, **kwargs)]
    ops_amd.dispatch_packet = self.packet = typed_packet
    Compiled.wait_timeout_ms = timeout_ms
    Backend.active = True

  def pv_program(self):
    expected = manifest()['expert_pv']
    shapes = ((1,16,64,128), (1,16,64,2368), (1,8,8,148,4,16,4))
    args = [UOp.placeholder(shape, dtypes.bfloat16, i) for i, shape in enumerate(shapes)]
    ast = kernel(*args, shape=(1,16,8,64,128,2368), tile_m=2, tile_n=2)
    binary = checked(ASSETS/'gfx1200/expert-pv.elf', expected['binary_sha256'])
    def frozen_compile(source):
      if hashlib.sha256(source.encode()).hexdigest() != expected['llvm_sha256']:
        raise ValueError('Expert PV LLVM source changed')
      return binary
    compiler = self.device.renderer.compiler
    previous = compiler.__dict__.get('compile_cached')
    had_override = 'compile_cached' in compiler.__dict__
    compiler.compile_cached = frozen_compile
    try:
      value = do_to_program(ast, self.device.renderer)
    finally:
      if had_override:
        compiler.compile_cached = previous
      else:
        del compiler.compile_cached
    actual = {'program_key':value.key.hex(), 'input_ast_key':ast.key.hex(),
              'binary_sha256':hashlib.sha256(value.src[3].arg).hexdigest(),
              'llvm_sha256':hashlib.sha256(value.src[2].arg.encode()).hexdigest()}
    if actual != expected:
      raise ValueError('Expert PV PROGRAM identity changed')
    return value

  def close(self):
    if self.closed:
      return
    from tinygrad.runtime import ops_amd
    try:
      self.device.synchronize()
    finally:
      self.device.renderer.compiler = self.compiler
      ops_amd.dispatch_packet = self.old_packet
      Compiled.wait_timeout_ms = self.old_timeout
      self.closed = True
      Backend.active = False
