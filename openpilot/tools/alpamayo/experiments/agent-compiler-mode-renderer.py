"""Optional gfx1200 LLVM WGP renderer. Import this module by its canonical name.

Private experiment; no tinygrad source edits. The distinct class is the process
cache namespace. Its constructor and pickle protocol always restore -cumode.
The unchanged LLVM renderer name preserves AMD:LLVM device selection.
"""
from tinygrad.renderer.llvmir import AMDLLVMRenderer as BaseAMDLLVMRenderer
from tinygrad.runtime.support.compiler_llvm import AMDLLVMCompiler, LLVMCompiler


class WGPCompiler(AMDLLVMCompiler):
  def __init__(self, arch):
    if arch != 'gfx1200': raise ValueError('This experiment admits only gfx1200')
    self.arch = arch
    LLVMCompiler.__init__(self, 'AMDGPU', arch, '-cumode')

  def __reduce__(self): return WGPCompiler, (self.arch,)


class AMDLLVMRenderer(BaseAMDLLVMRenderer):
  def __init__(self, target):
    if target.device != 'AMD' or target.renderer != 'LLVM' or target.arch != 'gfx1200':
      raise ValueError('This experiment admits only AMD:LLVM:gfx1200')
    super().__init__(target)
    self.compiler = WGPCompiler(target.arch)

  def __reduce__(self):
    if type(self.compiler) is not WGPCompiler: raise ValueError('WGP compiler was replaced')
    return AMDLLVMRenderer, (self.target,)


def install(device):
  """Replace one device's renderer factory before model compilation.

  Existing base-class cache entries remain intact but cannot match this class.
  Other device instances, renderer factories, and architectures are untouched.
  """
  from tinygrad.codegen import to_program_cache
  from tinygrad.engine import worker
  if device.device.split(':')[0] != 'AMD' or device.arch != 'gfx1200':
    raise ValueError('Only the selected gfx1200 AMD device may change')
  old = device.renderer
  if type(old) is not BaseAMDLLVMRenderer: raise ValueError('Expected the original LLVM renderer')
  if worker.worker_pool is not None: raise ValueError('Install before creating compilation workers')
  if any(k[2].device == 'AMD' for k in to_program_cache):
    raise ValueError('Install before compiling model programs')
  if device.renderers.count(BaseAMDLLVMRenderer) != 1: raise ValueError('Ambiguous renderer factory')
  factories = device.renderers
  device.renderers = [AMDLLVMRenderer if r is BaseAMDLLVMRenderer else r for r in factories]
  try:
    new = device.renderer
    assert type(new) is AMDLLVMRenderer and new.target == old.target
    assert type(new.compiler) is WGPCompiler
  except BaseException:
    device.renderers = factories
    raise
  return {'old_class': type(old).__module__+'.'+type(old).__name__,
          'new_class': type(new).__module__+'.'+type(new).__name__,
          'target': repr(new.target), 'compiler_cachekey': new.compiler.cachekey,
          'prior_model_cache_entries': 0, 'prior_worker_pool': False}
