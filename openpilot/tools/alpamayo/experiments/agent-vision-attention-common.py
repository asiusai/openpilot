"""Fixed vision shape, reviewed ABI, and shared parent/offline graph construction."""
from contextlib import contextmanager
from functools import lru_cache, partial
import hashlib, importlib.util, math, struct
from pathlib import Path

EXP=Path(__file__).parent
ROOT=Path('/mnt/nvme-500/tmp/alpamayo-speed-20261116/agent-vision-attention-artifacts')
CURRENT=Path('/mnt/nvme-500/tmp/alpamayo-speed-20261116/agent-paced-dynamic-source-cpu02/alpamayo')
NAME='alpamayo_ck_vision72'
ELF_SHA='e67fceade4fca00465613cf1e1ccf21036d1ab4fc7fba80a2e4e85160a9955dd'
SOURCE_SHA='70b099072b7aaf4486d3cecd581e55ae1bd2ffa560cc9b1c2978f07b872990f1'
GUARD=256
ACTIVE=False

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
@lru_cache(None)
def load(path):
  path=Path(path);spec=importlib.util.spec_from_file_location(path.stem.replace('-','_'),path)
  m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def shape(batch):
  assert batch in (1,2);return (batch,16,720,72)

def implicit(batch):
  data=bytearray(256)
  struct.pack_into('<III',data,0,16,12,batch)
  struct.pack_into('<HHH',data,12,128,1,1)
  struct.pack_into('<H',data,64,3)
  return bytes(data)

@lru_cache(None)
def artifacts(arm):
  assert arm in ('baseline','candidate')
  folder=ROOT/arm
  text=(folder/'source.cpp').read_text();blob=(folder/'kernel.hsaco').read_bytes()
  if arm=='candidate':
    assert hashlib.sha256(text.encode()).hexdigest()==SOURCE_SHA
    assert hashlib.sha256(blob).hexdigest()==ELF_SHA
  return text,blob

def program(*args,arm,batch):
  from tinygrad import dtypes
  from tinygrad.uop.ops import UOp,Ops,KernelInfo
  if arm=='candidate': out,lse,q,k,v=args
  else: out,q,k,v=args
  assert all(t.shape==shape(batch) and t.dtype==dtypes.bfloat16 for t in (out,q,k,v))
  val=q.flatten().index(0).load()+k.flatten().index(0).load()+v.flatten().index(0).load()
  stores=[out.flatten().index(0).store(val)]
  if arm=='candidate':
    assert lse.shape==(batch,16,720) and lse.dtype==dtypes.float32
    stores+=[lse.flatten().index(0).store(val.cast(dtypes.float32))]
    launch=[UOp.special(16,'gidx0'),UOp.special(12,'gidx1'),UOp.special(128,'lidx0')]
    if batch==2: launch+=[UOp.special(batch,'gidx2')]
    name=NAME
  else:
    launch=[UOp.special(batch*16*6,'gidx0'),UOp.special(256,'lidx0')];name='alpamayo_online'
  sink=UOp.sink(*stores,*launch,arg=KernelInfo(name=name));source,blob=artifacts(arm)
  return UOp(Ops.PROGRAM,src=(sink,UOp(Ops.LINEAR,src=tuple(sink.toposort())),UOp(Ops.SOURCE,arg=source),UOp(Ops.BINARY,arg=blob)))

def outputs(arm,boundary,q,k,v,os_,ls):
  from tinygrad import Tensor,dtypes
  batch=q.shape[0];assert batch in (1,2) and boundary in ('A','B')
  assert q.dtype==k.dtype==v.dtype==dtypes.bfloat16 and q.device==k.device==v.device==os_.device==ls.device
  assert os_.shape==(math.prod(shape(batch))+2*GUARD,) and ls.shape==(batch*16*720+2*GUARD,)
  if boundary=='B':
    assert q.shape==k.shape==v.shape==(batch,720,16,72)
    q,k,v=(x.transpose(1,2) for x in (q,k,v))
  assert q.shape==k.shape==v.shape==shape(batch)
  # Proven normalization: register offset-contiguous guarded views before callification.
  out=os_[GUARD:-GUARD].reshape(shape(batch)).contiguous()
  args=(out,)
  if arm=='candidate':
    assert ACTIVE
    lse=ls[GUARD:-GUARD].reshape(batch,16,720).contiguous();args+=(lse,)
  result=tuple(Tensor.custom_kernel(*args,q.contiguous(),k.contiguous(),v.contiguous(),
    fxn=partial(program,arm=arm,batch=batch))[:len(args)])
  if boundary=='B': result=(result[0].transpose(1,2).contiguous(),*result[1:])
  return result

def execute(arm,boundary,*args):
  result=outputs(arm,boundary,*args);result[0].realize(*result[1:]);return result

@contextmanager
def abi_scope(records=None):
  global ACTIVE
  from tinygrad import dtypes
  from tinygrad.runtime import ops_amd
  from tinygrad.runtime.support.hcq2 import layout_args,pack_args
  from tinygrad.engine.realize import get_call_kernel_args
  from tinygrad.uop.ops import UOp,Ops
  assert not ACTIVE
  old=ops_amd.AMDComputeQueue.kernargs
  def kernargs(queue,call,prg,data):
    if prg.src[0].arg.name!=NAME: return old(queue,call,prg,data)
    assert ACTIVE and prg.src[3].arg==artifacts('candidate')[1] and prg.src[2].arg==artifacts('candidate')[0]
    assert str(prg.arg.target)=='USB+AMD:LLVM:gfx1200'
    assert queue.dev.arch=='gfx1200' and tuple(queue.dev.target)==(12,0,0)
    batch=prg.arg.global_size[2];assert batch in (1,2)
    assert tuple(prg.arg.global_size)==(16,12,batch) and tuple(prg.arg.local_size)==(128,1,1)
    assert tuple(prg.arg.outs)==(0,1) and tuple(prg.arg.ins)==(2,3,4)
    assert (data.kernargs_segment_size,data.group_segment_size,data.private_segment_size)==(296,9216,96)
    assert data.wave32 and not data.enable_dispatch_ptr
    assert (data.rsrc1,data.rsrc2,data.rsrc3)==(1074528528,590725,0)
    values=get_call_kernel_args(call,prg,queue.devs);layout=layout_args(values)
    assert [off for off,_ in layout]==[0,8,16,24,32] and all(v.dtype==dtypes.uint64 for _,v in layout)
    if records is not None: records.append({'batch':batch,'argument_offsets':[off for off,_ in layout],
      'implicit_hex':implicit(batch).hex(),'source':SOURCE_SHA,'binary':ELF_SHA,'mode':'CU'})
    return UOp(Ops.LINEAR,src=tuple(pack_args(layout,40)+[UOp(Ops.BINARY,arg=implicit(batch))]),arg='kernargs').getaddr(queue.devs)
  ops_amd.AMDComputeQueue.kernargs,ACTIVE=kernargs,True
  try: yield
  finally:
    assert ops_amd.AMDComputeQueue.kernargs is kernargs
    ops_amd.AMDComputeQueue.kernargs,ACTIVE=old,False

def metrics(actual,ref):
  import numpy as np
  assert actual.shape==ref.shape and np.isfinite(actual).all()
  d=actual.astype(np.float64)-ref.astype(np.float64)
  return {'rmse':float(np.sqrt(np.mean(d*d))),'max_abs':float(np.max(np.abs(d))),
    'per_head_rmse':np.sqrt(np.mean(d*d,axis=(0,2,3))).tolist(),
    'last_row_max_abs':float(np.max(np.abs(d[:,:,719]))),'last_8_dims_max_abs':float(np.max(np.abs(d[:,:,:,-8:])))}

def numeric_gate(c,b,limits):
  for field in ('rmse','max_abs'):
    r=limits[field];assert c[field]<=max(r['floor'],r['relative']*b[field]),(field,c,b)
  r=limits['per_head_rmse']
  assert all(x<=max(r['floor'],r['relative']*y) for x,y in zip(c['per_head_rmse'],b['per_head_rmse'],strict=True)),(c,b)
