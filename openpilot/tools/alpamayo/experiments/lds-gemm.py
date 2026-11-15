"""Run the pinned AMD LDS reference through tinygrad's existing GPU transport."""
from functools import partial, lru_cache
import importlib.util
from pathlib import Path
from tinygrad import Tensor
from tinygrad.uop.ops import UOp, Ops, KernelInfo
from tinygrad.runtime.support.compiler_amd import HIPCompiler


@lru_cache(None)
def compiled(shape,tile,fused=False):
  spec=importlib.util.spec_from_file_location('lds_source',Path(__file__).with_name('compile-lds-reference.py'))
  module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
  text=module.source(Path('/tmp/opencode/rdna-fp8-reference'),tile,shape,fused)
  compiler_spec=importlib.util.spec_from_file_location('lds_compiler',Path(__file__).with_name('hip-compiler.py'))
  compiler_module=importlib.util.module_from_spec(compiler_spec);compiler_spec.loader.exec_module(compiler_module)
  return text,compiler_module.compiler().compile_cached(text)


def program(out,x,w,*extra,shape,tile):
  rows,cols,_=shape
  m,n,_,wm,wn=tile
  blocks=((rows+m-1)//m)*((cols+n-1)//n)
  value=x.flatten()[0].load().float()+w.flatten()[0].load().float()
  for item in extra: value=value+item.flatten()[0].load().float()
  sink=UOp.sink(out.flatten()[0].store(value.cast(out.dtype)),
                UOp.special(blocks,'gidx0'),UOp.special(wm*wn*32,'lidx0'),arg=KernelInfo(name='alpamayo_lds'))
  text,binary=compiled(shape,tile,bool(extra))
  return UOp(Ops.PROGRAM,src=(sink,UOp(Ops.LINEAR,src=tuple(sink.toposort())),UOp(Ops.SOURCE,arg=text),UOp(Ops.BINARY,arg=binary)))


def gemm(x,w,tile=(128,96,64,2,2)):
  from tinygrad import dtypes
  assert x.ndim==w.ndim==2 and x.shape[1]==w.shape[1]
  assert x.dtype==w.dtype==dtypes.fp8e4m3
  rows,width=x.shape;cols=w.shape[0]
  out=Tensor.empty(cols,rows,device=x.device,dtype=dtypes.float32)
  return Tensor.custom_kernel(out,x.contiguous(),w.contiguous(),fxn=partial(program,shape=(rows,cols,width),tile=tile))[0].T


def scaled_gemm(x,w,scales,bias,tile=(128,96,64,2,2)):
  from tinygrad import dtypes
  assert x.dtype==w.dtype==dtypes.fp8e4m3 and bias.dtype==dtypes.bfloat16
  rows,width=x.shape;cols=w.shape[0]
  out=Tensor.empty(rows,cols,device=x.device,dtype=dtypes.bfloat16)
  return Tensor.custom_kernel(out,x.contiguous(),w.contiguous(),*scales,bias,
    fxn=partial(program,shape=(rows,cols,width),tile=tile))[0]
