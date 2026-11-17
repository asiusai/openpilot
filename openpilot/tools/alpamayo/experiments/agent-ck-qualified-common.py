"""Fixed attention graphs and integer-only qualification utilities. No device opens on import."""
from functools import lru_cache
import hashlib
import importlib.util
from pathlib import Path

EXP=Path(__file__).parent
ROOT=Path('/mnt/nvme-500/tmp/alpamayo-speed-20261116/agent-ck-qualified-work')
OC,LC,G=4194304,32768,256
QSHAPE,KVSHAPE,LSHAPE=(1,16,2048,128),(1,8,2048,128),(1,16,2048)
CHANNELS=['all_bit_mismatch','guarded_nonfinite','guarded_poison','redzone_mismatch',
          'returned_bit_mismatch','returned_nonfinite','returned_poison']


@lru_cache(None)
def load(name):
  spec=importlib.util.spec_from_file_location(name.replace('-','_'),EXP/(name+'.py'))
  module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
  return module


def outputs(boundary,arm,q,k,v,ostore,lstore):
  if boundary=='B': q,k,v=q.transpose(1,2),k[:,:,:2048],v[:,:,:2048]
  result=load('agent-ck-output-common').outputs(arm,True,q,k,v,ostore,lstore)
  if boundary=='B': result=(result[0].transpose(1,2).reshape(1,2048,2048).contiguous(),*result[1:])
  return result


def execute(boundary,arm,*inputs):
  result=outputs(boundary,arm,*inputs);result[0].realize(*result[1:]);return result


def utility(dst,src=None,mode='copy'):
  """Explicit integer GPU store, with no host COPY operation or floating conversion."""
  from tinygrad import Tensor,dtypes
  from tinygrad.uop.ops import UOp,KernelInfo
  dt=dtypes.uint16 if dst.dtype.itemsize==2 else dtypes.uint32
  out=dst.bitcast(dt).flatten().contiguous()
  args=(out,) if src is None else (out,src.bitcast(dt).flatten().contiguous())
  def program(*xs):
    count=xs[0].numel();assert mode=='sentinel' or count%256==0
    i=UOp.special((count+255)//256,'gidx0')*256+UOp.special(256,'lidx0')
    if mode=='sentinel': value=UOp.const(0xffffffff,dt)
    elif mode=='poison':
      guard,poison=(0x5aa5,0x7fc1) if dt==dtypes.uint16 else (0x5aa55aa5,0x7fc12345)
      value=((i>=G)&(i<count-G)).where(poison,guard).cast(dt)
    else:
      j=i;valid=None
      if mode=='q_B': j=(i//128%16)*2048*128+(i//2048)*128+i%128
      elif mode=='kv_B':
        row=i//128%2304;valid=row<2048;j=(i//(2304*128))*2048*128+row*128+i%128
      else: assert mode=='copy'
      value=xs[1].index(j if valid is None else j.valid(valid)).load()
      if valid is not None: value=valid.where(value,UOp.const(0x7fc1,dt))
    return UOp.sink(xs[0].index(i.valid(i<count) if mode=='sentinel' else i).store(value),
                    arg=KernelInfo(name=f'ckq_{mode}_u{dt.itemsize*8}_{count}'))
  return Tensor.custom_kernel(*args,fxn=program)[0]


def validator(store,returned,reference,boundary='A'):
  """Seven independent counts per 256 words. Every bit and every word is compared."""
  from tinygrad import Tensor,dtypes
  dt=dtypes.uint16 if store.dtype.itemsize==2 else dtypes.uint32
  is_o=dt==dtypes.uint16
  if is_o and boundary=='B': returned=returned.reshape(1,2048,16,128).transpose(1,2)
  raw,ref=store.bitcast(dt).flatten(),reference.bitcast(dt).flatten()
  ret=returned.bitcast(dt).flatten().pad(((G,G),),value=0x5aa5 if is_o else 0x5aa55aa5)
  index=Tensor.arange(raw.numel(),dtype=dtypes.int32)
  interior=(index>=G)&(index<raw.numel()-G)
  mask,poison,guard=(0x7f80,0x7fc1,0x5aa5) if is_o else (0x7f800000,0x7fc12345,0x5aa55aa5)
  checks=[raw!=ref,interior&((raw&mask)==mask),interior&(raw==poison),(~interior)&(raw!=guard),
    interior&(ret!=raw),interior&((ret&mask)==mask),interior&(ret==poison)]
  return Tensor.stack(*[x.cast(dtypes.uint32).reshape(-1,256).sum(axis=1) for x in checks])


def tail_validator(k,v):
  from tinygrad import Tensor,dtypes
  return Tensor.stack(*[(x[:,:,2048:].bitcast(dtypes.uint16)!=0x7fc1).cast(dtypes.uint32).reshape(-1,256).sum(axis=1)
                        for x in (k,v)])


def cpu_flags(whole,returned,reference):
  """Independent NumPy implementation uses floating finiteness, not exponent-mask tests."""
  import numpy as np
  whole,returned,reference=(np.asarray(x).reshape(-1) for x in (whole,returned,reference))
  is_o=whole.dtype==np.uint16;assert whole.dtype in (np.uint16,np.uint32)
  guard,poison=(0x5aa5,0x7fc1) if is_o else (0x5aa55aa5,0x7fc12345)
  interior=np.zeros(whole.size,bool);interior[G:-G]=True
  ret=np.pad(returned,(G,G),constant_values=guard)
  def finite(bits):
    values=(bits.astype(np.uint32)<<16).view(np.float32) if is_o else bits.view(np.float32)
    return np.isfinite(values)
  flags=np.array([whole!=reference,interior&~finite(whole),interior&(whole==poison),~interior&(whole!=guard),
                  interior&(ret!=whole),interior&~finite(ret),interior&(ret==poison)],dtype=np.uint32)
  return flags.reshape(len(CHANNELS),-1,256).sum(axis=2,dtype=np.uint32)


def controls(whole,returned):
  """Each bit position has its own comparison group; NaNs, infinities, redzones, and end words are separate."""
  import numpy as np
  a,b=whole.copy().reshape(-1),returned.copy().reshape(-1)
  width=a.dtype.itemsize*8
  edits=[]
  def setword(target,index,value,label):
    arr=a if target=='guarded' else b
    arr[index]=value;edits.append({'target':target,'index':index,'value':int(value),'label':label})
  for bit in range(width):
    index=G+(bit+1)*256+8;setword('guarded',index,int(a[index])^(1<<bit),f'guarded_bit_{bit}')
    index=(bit+50)*256+12;setword('returned',index,int(b[index])^(1<<bit),f'returned_bit_{bit}')
    setword('guarded',bit,int(a[bit])^(1<<bit),f'redzone_bit_{bit}')
  specials=[0x7fc1,0x7f80,0xff80,0x7f81,0xffc1] if width==16 else [0x7fc12345,0x7f800000,0xff800000,0x7f800001,0xffc12345]
  for n,value in enumerate(specials):
    setword('guarded',G+(100+n)*256+16,value,f'guarded_special_{n}')
    setword('returned',(110+n)*256+20,value,f'returned_special_{n}')
  for target,index in [('guarded',a.size-G-1),('guarded',a.size-1),('returned',b.size-1)]:
    arr=a if target=='guarded' else b;setword(target,index,int(arr[index])^1,'last_'+target)
  return a,b,edits


def ranges(before,after,block=8192):
  """Coalesce changed blocks only; reconstruction must equal every byte of the target."""
  import numpy as np
  a,b=memoryview(before).cast('B'),memoryview(after).cast('B')
  assert len(a)==len(b) and len(b)%block==0
  changed=np.any(np.frombuffer(a,np.uint8).reshape(-1,block)!=np.frombuffer(b,np.uint8).reshape(-1,block),axis=1)
  spans=[]
  for index in np.flatnonzero(changed):
    start=int(index)*block
    if spans and spans[-1][0]+spans[-1][1]==start: spans[-1][1]+=block
    else: spans.append([start,block])
  rebuilt=bytearray(a)
  for offset,length in spans: rebuilt[offset:offset+length]=b[offset:offset+length]
  assert bytes(rebuilt)==bytes(b)
  return spans


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
