"""Narrow lane-order correction. Sparse K32 remains a numerical proposal."""
from functools import lru_cache,partial
import hashlib,importlib.util,json
from pathlib import Path
from types import FunctionType

HERE=Path(__file__).resolve().parent
OLD=Path('/tmp/opencode/openpilot-alpamayo/openpilot/tools/alpamayo/experiments/agent-sparse-fc2-source')
EXPECTED='57b2d434c9d20508548c773b161e7b0b24e2022fbd89c61fa681dd135bc37585'


@lru_cache(None)
def original():
  path=OLD/'agent-sparse-fc2-kernel.py'
  if hashlib.sha256(path.read_bytes()).hexdigest()!=EXPECTED:raise ValueError('Frozen numeric parent kernel changed')
  spec=importlib.util.spec_from_file_location('sparse_numeric_original',path)
  module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def source(m,n,k,mode='hybrid',raw=False,initial=False):
  text=original().source(m,n,k,mode,raw,initial)
  if mode=='compress':return text
  # The pinned TensorCore.frag_a/frag_b put k2 in lane bit 4 and k3 in
  # element bit 2. The previous mirror put k3 in the lane and k2 in the word.
  changes={
    'b0[j]=lane<16?(i2){b32[j][0],b32[j][1]}:(i2){p2,p3};':
      'b0[j]=lane<16?(i2){b32[j][0],b32[j][2]}:(i2){p1,p3};',
    'b1[j]=lane<16?(i2){p0,p1}:(i2){b32[j][2],b32[j][3]};':
      'b1[j]=lane<16?(i2){p0,p2}:(i2){b32[j][1],b32[j][3]};',
    'unsigned row=mr+i*16+lane%16,base=kt*32+(lane/16)*8;':
      'unsigned row=mr+i*16+lane%16,base=kt*32+(lane/16)*4;',
    'i2 a0=(i2){word(x,row,base),word(x,row,base+4)};':
      'i2 a0=(i2){word(x,row,base),word(x,row,base+8)};',
    'i2 a1=(i2){word(x,row,base+16),word(x,row,base+20)};':
      'i2 a1=(i2){word(x,row,base+16),word(x,row,base+24)};',
    'void agent_sparse_fc2_gemm(':'void agent_sparse_numeric_gemm(',
  }
  for before,after in changes.items():
    if text.count(before)!=1:raise ValueError('Frozen mirror source pattern changed: '+before)
    text=text.replace(before,after)
  return text


@lru_cache(None)
def compiled(m,n,k,mode='hybrid',raw=False,initial=False):
  if mode=='compress':return original().compiled(m,n,k,mode,raw,initial)
  text=source(m,n,k,mode,raw,initial)
  report=HERE/'agent-sparse-numeric-cpu.json';stem=f'{m}-{n}-{k}-{mode}-raw{int(raw)}-C{int(initial)}'
  if report.exists():
    data=json.loads(report.read_text());record=next((r for r in data.get('compiler',[])if r['name']==stem),None)
    if data.get('all_pass') and data.get('source_hashes',{}).get(Path(__file__).name)==hashlib.sha256(Path(__file__).read_bytes()).hexdigest() and record:
      root=HERE/'agent-sparse-numeric-cpu-compiled';binary=(root/(stem+'.hsaco')).read_bytes()
      if (root/(stem+'.cpp')).read_text()!=text or hashlib.sha256(binary).hexdigest()!=record['binary_sha256']:
        raise ValueError('Qualified numeric diagnostic binary mismatch')
      return text,binary
  from tinygrad.device import Compiler
  from tinygrad.runtime.support import compiler_amd
  base=compiler_amd.compile_hip
  def options(action,value):return compiler_amd.set_options(action,value.replace(b'-mcumode',b'-mno-cumode'))
  function=FunctionType(base.__code__,dict(base.__globals__,set_options=options),argdefs=base.__defaults__)
  class WGP(Compiler):
    def compile(self,text):return function(text,'gfx1200')
  return text,WGP('agent_sparse_numeric_wgp_gfx1200_v1').compile_cached(text)


def program(*params,m,n,k,mode,raw=False,initial=False):
  from tinygrad import dtypes
  from tinygrad.helpers import Target
  from tinygrad.uop.ops import UOp,Ops,KernelInfo,ProgramInfo
  if mode=='compress':return original().program(*params,m=m,n=n,k=k,mode=mode)
  reads=tuple(range(1,len(params)))
  values=[params[i].flatten().index(0).load().cast(dtypes.uint32)for i in reads]
  dummy=sum(values[1:],values[0]);blocks=m//64*(n//64)
  sink=UOp.sink(params[0].flatten().index(0).store(dummy.cast(params[0].dtype)),
    UOp.special(blocks,'gidx0'),UOp.special(32,'lidx0'),arg=KernelInfo(name='agent_sparse_numeric_gemm'))
  info=ProgramInfo(global_size=(blocks,1,1),local_size=(32,1,1),params=tuple(p.buf_uop for p in params),
    globals=tuple(range(len(params))),outs=(0,),ins=reads,target=Target('AMD','LLVM','gfx1200'))
  text,binary=compiled(m,n,k,mode,raw,initial)
  return UOp(Ops.PROGRAM,src=(sink,UOp(Ops.LINEAR,src=tuple(sink.toposort())),UOp(Ops.SOURCE,arg=text),UOp(Ops.BINARY,arg=binary)),arg=info)


def gemm(x,w,xs,ws,compressed,out,seed,mode='hybrid',initial=False):
  from tinygrad import Tensor,dtypes
  if x.dtype!=dtypes.uint8 or w.dtype!=dtypes.uint8:raise ValueError('Require immutable original packed FP8 bytes')
  m,k=x.shape;n=w.shape[0]
  return Tensor.custom_kernel(out,x,w,xs,ws,*compressed,seed,
    fxn=partial(program,m=m,n=n,k=k,mode=mode,raw=out.dtype==dtypes.float32,initial=initial))[0]


allocate=lambda *a:original().allocate(*a)
compress=lambda *a:original().compress(*a)
reference=lambda *a:original().original(*a)
