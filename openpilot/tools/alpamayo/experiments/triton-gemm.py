"""Experimental FP8 GEMM over Harald's packed inputs; existing USB GPU context."""
from functools import lru_cache, partial
import os
import struct

import triton
import triton.language as tl
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource
from tinygrad import Tensor, dtypes
from tinygrad.uop.ops import UOp, Ops, KernelInfo
from tinygrad.runtime.support.elf import elf_loader
from amd_gemm import pack


@triton.jit
def alpamayo_triton_gemm(Out,X,W,XS,WS,M:tl.constexpr,N:tl.constexpr,K:tl.constexpr,
                        BM:tl.constexpr,BN:tl.constexpr,BK:tl.constexpr,PACKED:tl.constexpr):
  group=tl.program_id(0)
  rows=(group//tl.cdiv(N,BN))*BM+tl.arange(0,BM)
  columns=(group%tl.cdiv(N,BN))*BN+tl.arange(0,BN)
  inner=tl.arange(0,BK)
  accumulator=tl.full((BM,BN),0,tl.float32)
  for start in range(tl.cdiv(K,BK)):
    k=start*BK+inner
    if PACKED:
      xi=(rows[:,None]//16*(K//16)+k[None,:]//16)*256
      xi+=(k[None,:]%16//4)*64+(rows[:,None]%16)*4+k[None,:]%4
      wi=(columns[None,:]//16*(K//16)+k[:,None]//16)*256
      wi+=(k[:,None]%16//4)*64+(columns[None,:]%16)*4+k[:,None]%4
      x=tl.load(X+xi,(rows[:,None]<M)&(k[None,:]<K),0.0)
      w=tl.load(W+wi,(columns[None,:]<N)&(k[:,None]<K),0.0)
    else:
      x=tl.load(X+rows[:,None]*K+k[None,:],(rows[:,None]<M)&(k[None,:]<K),0.0)
      w=tl.trans(tl.load(W+columns[:,None]*K+k[None,:],(columns[:,None]<N)&(k[None,:]<K),0.0))
    accumulator=tl.dot(x,w,accumulator)
  scale=tl.load(XS)*tl.load(WS)
  result=(accumulator*scale).to(tl.bfloat16)
  tl.store(Out+rows[:,None]*N+columns[None,:],result,(rows[:,None]<M)&(columns[None,:]<N))


@lru_cache(None)
def compiled(rows,columns,width,bm,bn,bk,warps,stages,packed=True):
  os.environ.setdefault('TRITON_CACHE_DIR','/tmp/opencode/alpamayo-triton-cache')
  source=ASTSource(alpamayo_triton_gemm,
    signature={'Out':'*bf16','X':'*fp8e4nv','W':'*fp8e4nv','XS':'*fp32','WS':'*fp32'},
    constexprs=dict(M=rows,N=columns,K=width,BM=bm,BN=bn,BK=bk,PACKED=packed))
  result=triton.compile(source,target=GPUTarget('hip','gfx1200',32),
    options=dict(num_warps=warps,num_stages=stages,matrix_instr_nonkdim=16))
  binary=bytearray(result.asm['hsaco'])
  _,sections,_=elf_loader(bytes(binary))
  descriptor=next(section for section in sections if section.name=='.rodata').header.sh_offset
  assert struct.unpack_from('<I',binary,descriptor)[0]==0
  struct.pack_into('<I',binary,descriptor,result.metadata.shared)
  return result.asm['ttir'],bytes(binary),result.metadata._asdict()


def program(out,x,w,xs,ws,*,shape,tile,warps,stages,packed):
  rows,columns,width=shape
  bm,bn,bk=tile
  text,binary,metadata=compiled(rows,columns,width,bm,bn,bk,warps,stages,packed)
  sink=UOp.sink(out.flatten()[0].store(((x.flatten()[0].load().float()*w.flatten()[0].load().float())*
    xs.flatten()[0].load()*ws.flatten()[0].load()).cast(dtypes.bfloat16)),
    UOp.special(((rows+bm-1)//bm)*((columns+bn-1)//bn),'gidx0'),UOp.special(warps*32,'lidx0'),
    arg=KernelInfo(name=metadata['name']))
  return UOp(Ops.PROGRAM,src=(sink,UOp(Ops.LINEAR,src=tuple(sink.toposort())),UOp(Ops.SOURCE,arg=text),UOp(Ops.BINARY,arg=binary)))


def gemm(x,w,scales,tile=(64,64,64),warps=4,stages=1,packed=True):
  assert x.dtype==w.dtype==dtypes.fp8e4m3 and x.shape[1]==w.shape[1] and x.shape[1]%16==0
  assert all(scale.numel()==1 and scale.dtype==dtypes.float32 for scale in scales)
  output=Tensor.empty(x.shape[0],w.shape[0],device=x.device,dtype=dtypes.bfloat16)
  inputs=(pack(x),pack(w)) if packed else (x.contiguous(),w.contiguous())
  return Tensor.custom_kernel(output,*inputs,*scales,
    fxn=partial(program,shape=(x.shape[0],w.shape[0],x.shape[1]),tile=tile,warps=warps,stages=stages,packed=packed))[0]
