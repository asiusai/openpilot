"""Offline Triton attention compilation for tinygrad's existing Chestnut context."""
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


@triton.jit
def alpamayo_triton_attention(Out,Q,K,V,HEADS:tl.constexpr,KVHEADS:tl.constexpr,
                             ROWS:tl.constexpr,KEYS:tl.constexpr,WIDTH:tl.constexpr,
                             CAUSAL:tl.constexpr,BM:tl.constexpr,BN:tl.constexpr,BD:tl.constexpr):
  program = tl.program_id(0)
  groups = tl.cdiv(ROWS,BM)
  head = program // groups
  tile = program % groups
  kvhead = (head//HEADS)*KVHEADS + (head%HEADS)//(HEADS//KVHEADS)
  rows = tile*BM + tl.arange(0,BM)
  dims = tl.arange(0,BD)
  columns = tl.arange(0,BN)
  q = tl.load(Q+head*ROWS*WIDTH+rows[:,None]*WIDTH+dims[None,:],
              (rows[:,None]<ROWS)&(dims[None,:]<WIDTH),0)
  maxima = tl.full((BM,),float('-inf'),tl.float32)
  denominator = tl.full((BM,),0,tl.float32)
  accumulator = tl.full((BM,BD),0,tl.float32)
  end = KEYS
  if CAUSAL:
    end = tl.minimum((tile+1)*BM,KEYS)
  for start in range(0,end,BN):
    keys = start+columns
    k = tl.load(K+kvhead*KEYS*WIDTH+keys[None,:]*WIDTH+dims[:,None],
                (keys[None,:]<KEYS)&(dims[:,None]<WIDTH),0)
    scores = tl.dot(q,k)*(WIDTH**-0.5*1.4426950408889634)
    valid = keys[None,:]<KEYS
    if CAUSAL:
      valid = valid & (keys[None,:]<=rows[:,None])
    scores = tl.where(valid,scores,float('-inf'))
    next_max = tl.maximum(maxima,tl.max(scores,1))
    probability = tl.exp2(scores-next_max[:,None])
    scale = tl.exp2(maxima-next_max)
    denominator = denominator*scale + tl.sum(probability,1)
    accumulator = accumulator*scale[:,None]
    v = tl.load(V+kvhead*KEYS*WIDTH+keys[:,None]*WIDTH+dims[None,:],
                (keys[:,None]<KEYS)&(dims[None,:]<WIDTH),0)
    accumulator = tl.dot(probability.to(tl.bfloat16),v,accumulator)
    maxima = next_max
  result = (accumulator/denominator[:,None]).to(tl.bfloat16)
  tl.store(Out+head*ROWS*WIDTH+rows[:,None]*WIDTH+dims[None,:],result,
            (rows[:,None]<ROWS)&(dims[None,:]<WIDTH))


@lru_cache(None)
def compiled(heads,kv_heads,rows,keys,width,causal,bm=64,bn=64,warps=4,stages=1,waves_per_eu=1):
  os.environ.setdefault('TRITON_CACHE_DIR','/tmp/opencode/alpamayo-triton-cache')
  constants=dict(HEADS=heads,KVHEADS=kv_heads,ROWS=rows,KEYS=keys,WIDTH=width,
                 CAUSAL=causal,BM=bm,BN=bn,BD=triton.next_power_of_2(width))
  source=ASTSource(alpamayo_triton_attention,
                   signature={name:'*bf16' for name in ('Out','Q','K','V')},constexprs=constants)
  result=triton.compile(source,target=GPUTarget('hip','gfx1200',32),
                         options=dict(num_warps=warps,num_stages=stages,waves_per_eu=waves_per_eu,matrix_instr_nonkdim=16))
  binary=bytearray(result.asm['hsaco'])
  _,sections,_=elf_loader(bytes(binary))
  descriptor=next(section for section in sections if section.name=='.rodata').header.sh_offset
  # HIP normally passes Triton's dynamic LDS size at launch. tinygrad reads this descriptor.
  assert struct.unpack_from('<I',binary,descriptor)[0]==0
  struct.pack_into('<I',binary,descriptor,result.metadata.shared)
  return result.asm['ttir'],bytes(binary),result.metadata._asdict()


def program(out,q,k,v,*,shape,causal):
  batch,heads,rows,width=shape
  bm=int(os.environ.get('ALPAMAYO_TRITON_BLOCK_M','64'))
  bn=int(os.environ.get('ALPAMAYO_TRITON_BLOCK_N','64'))
  warps=int(os.environ.get('ALPAMAYO_TRITON_WARPS','4'))
  stages=int(os.environ.get('ALPAMAYO_TRITON_STAGES','1'))
  waves_per_eu=int(os.environ.get('ALPAMAYO_TRITON_WAVES_PER_EU','1'))
  assert bm in (16,32,64,128) and bn in (16,32,64,128) and warps in (4,8)
  assert stages in (1,2,3) and waves_per_eu in (1,2,3,4)
  text,binary,metadata=compiled(heads,k.shape[1],rows,k.shape[2],width,causal,bm,bn,warps,stages,waves_per_eu)
  sink=UOp.sink(out.flatten()[0].store(q.flatten()[0].load()+k.flatten()[0].load()+v.flatten()[0].load()),
                 UOp.special(batch*heads*((rows+bm-1)//bm),'gidx0'),UOp.special(warps*32,'lidx0'),
                 arg=KernelInfo(name=metadata['name']))
  return UOp(Ops.PROGRAM,src=(sink,UOp(Ops.LINEAR,src=tuple(sink.toposort())),
                              UOp(Ops.SOURCE,arg=text),UOp(Ops.BINARY,arg=binary)))


def attention(q,k,v,causal=True):
  assert q.dtype==k.dtype==v.dtype==dtypes.bfloat16
  assert q.shape[-1]==k.shape[-1]==v.shape[-1] and q.shape[-1] in (72,128)
  out=Tensor.empty(q.shape,device=q.device,dtype=q.dtype)
  return Tensor.custom_kernel(out,q.contiguous(),k.contiguous(),v.contiguous(),
                               fxn=partial(program,shape=q.shape,causal=causal))[0]
