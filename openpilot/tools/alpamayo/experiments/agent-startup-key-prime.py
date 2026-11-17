"""Compile five scalar token concatenations with the installed parent compiler.

No tensor is realized, no device is opened, and no global runtime function changes.
The normal tinygrad compilation cache retains the actual resulting programs.
"""
import hashlib
from pathlib import Path
import time

ORDINARY = {
  2:'69410a361264641938c4d8708c64ea19973cb060e795fb769825af56231b05b5',
  3:'d7fb99841b398e9357c13db4d3bbe6590cbee81e96919c5740b3c38d2c5c3e9a',
  4:'f3e8def008619c2909ce49fa3e2ea3939c7d7fc7f17650448c1a12960dd6fe3e',
  5:'d3962a29309637e68fb54d6260b883b9fb305bcae978657f7d0bb8eca2ab1c03',
  6:'9bd135113eca86b8be2219fd5ceb302d0cd919910aa32632517115097c30dbbd',
}
CANDIDATE = {
  2:'83e3fc0d31bc58184dbe2be4f3c2f2e4aa9077aaf9bc0a99f7828a549e864691',
  3:'67faf68f64854721311e02903fd9c41594dee87863186d32ece58a952d06016e',
  4:'f827eed2fb8ce2c28d4a3a69f45c2171c17f0e9fc09cf649b39ce7cd29b8a207',
  5:'b459369cdd6000482b2b9173f5ea47dd62f7de62c02dfd73a98e05487634d881',
  6:'8dfbf5763c894c929658aa94e98385c154f3a647fcbf7a992193fe343d19d9c2',
}


def concat_ast(count):
  from tinygrad import Tensor, dtypes
  from tinygrad.uop.ops import UOp
  if count not in ORDINARY: raise ValueError('Only the five reviewed token concatenations are admitted')
  # Unallocated shape placeholders, not camera data, predictions, or KV storage.
  inputs=[Tensor(UOp.new_buffer('AMD',1,dtypes.float32)).reshape(1) for _ in range(count)]
  linear=Tensor.cat(*inputs).schedule_linear()
  if len(linear.src)!=1: raise AssertionError('Token concatenation no longer has exactly one kernel')
  return linear.src[0].body


def prime(renderer):
  from tinygrad import Device
  from tinygrad.codegen import to_program, to_program_key, to_program_cache
  from tinygrad.engine.realize import capturing
  from tinygrad.helpers import Target
  from tinygrad.renderer.llvmir import AMDLLVMRenderer
  if capturing: raise ValueError('Prime before startup capture begins')
  if type(renderer) is not AMDLLVMRenderer or renderer.target!=Target('AMD','LLVM','gfx1200','USB'):
    raise ValueError('Only the measured USB gfx1200 LLVM renderer is admitted')
  if type(renderer.compiler).__name__!='WGPCompiler' or renderer.compiler.arch!='gfx1200':
    raise ValueError('The original parent WGP compiler must already be installed')
  opened=set(Device._opened_devices)
  start=time.perf_counter()
  rows=[]
  for count,expected in ORDINARY.items():
    ast=concat_ast(count)
    cache_key=to_program_key(ast,renderer)
    prior=to_program_cache.get(cache_key)
    if prior is not None and prior.key.hex()!=expected:
      raise AssertionError('A different compiler already populated the concat cache; restart the process')
    program=to_program(ast,renderer)
    if program.key.hex()!=expected:
      raise AssertionError('Parent compiler did not produce the exact ordinary program')
    rows.append(dict(count=count,program_key=program.key.hex(),ast_key=ast.key.hex(),cache_hit=prior is not None,
      binary_sha256=hashlib.sha256(program.src[-1].arg).hexdigest()))
  if set(Device._opened_devices)!=opened: raise AssertionError('Compilation unexpectedly opened a device')
  return dict(rows=rows,host_ms=(time.perf_counter()-start)*1000,kernels_executed=0,
              tensor_contents_read=0,runtime_functions_changed=0,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
