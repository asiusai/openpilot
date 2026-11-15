"""Experimental hipBLASLt kernels in the existing USB context; no host installation."""
from contextlib import contextmanager
from functools import lru_cache, partial
import hashlib
import json
import os
from pathlib import Path
import struct

from tinygrad import Tensor, dtypes
from tinygrad.uop.ops import UOp, Ops, KernelInfo
from tinygrad.runtime.support.elf import elf_loader

ROOT = Path('/tmp/opencode')
PROGRAMS = {}
ACTIVE = False


@lru_cache(None)
def selected(index):
  records = json.loads(Path(os.environ.get('ALPAMAYO_VENDOR_SOLUTIONS',ROOT/'alpamayo-vendor-selected.json')).read_text())
  record = next(r for r in records if r['solution']['index'] == index)
  solution, metadata = record['solution'], record['metadata']
  assert solution['internalArgsSupport']['version'] == 2
  assert solution['sizeMapping']['globalSplitU'] == 1 and solution['sizeMapping']['streamK'] == 0
  assert metadata['.private_segment_fixed_size'] == 0 and metadata['.kernarg_segment_size'] == 176
  assert metadata['.wavefront_size'] == 32
  return record


@lru_cache(None)
def binary(index):
  """Add a descriptor alias; preserve every loaded code and data byte."""
  record = selected(index)
  source = (ROOT/'alpamayo-vendor-fp8.hsaco').read_bytes()
  assert source[:6] == b'\x7fELF\x02\x01'
  shoff = struct.unpack_from('<Q',source,40)[0]
  entry_size,count,strings_index = struct.unpack_from('<HHH',source,58)
  assert entry_size == 64
  sections = [list(struct.unpack_from('<IIQQQQIIQQ',source,shoff+i*64)) for i in range(count)]
  strings = source[sections[strings_index][4]:sections[strings_index][4]+sections[strings_index][5]]
  def name(offset,table=strings): return table[offset:table.index(b'\0',offset)].decode()
  rodata = next(s for s in sections if name(s[0]) == '.rodata')
  symbols = []
  for section in sections:
    if section[1] not in (2,11): continue
    assert section[9] == 24
    st = sections[section[6]]
    names = source[st[4]:st[4]+st[5]]
    for off in range(section[4],section[4]+section[5],24):
      sn,info,other,si,value,size = struct.unpack_from('<IBBHQQ',source,off)
      if name(sn,names) == record['metadata']['.symbol']:
        symbols.append((value,size))
  assert symbols and len(set(symbols)) == 1
  address,size = symbols[0]
  assert size == 64 and rodata[3] <= address < rodata[3]+rodata[5]
  alias = rodata.copy()
  alias[3],alias[4],alias[5] = address,rodata[4]+address-rodata[3],64
  rodata[0] = len(strings)
  strings += b'.rodata.original\0'
  result = bytearray(source)
  sections[strings_index][4],sections[strings_index][5] = len(result),len(strings)
  result += strings
  result += bytes(-len(result)%8)
  struct.pack_into('<Q',result,40,len(result))
  struct.pack_into('<H',result,60,count+1)
  for section in [*sections,alias]: result += struct.pack('<IIQQQQIIQQ',*section)
  original_image,_,original_relocs = elf_loader(source)
  image,loaded,relocs = elf_loader(bytes(result))
  assert image == original_image and relocs == original_relocs
  assert next(s.header.sh_addr for s in loaded if s.name == '.rodata') == address
  return bytes(result)


def argument_values(index,shape,addresses,cu_count):
  rows,columns,width = shape
  out,x,w,xs,ws,ones,bias,zero = addresses
  mapping = selected(index)['solution']['sizeMapping']
  gx,gy = (columns+mapping['macroTile'][0]-1)//mapping['macroTile'][0],(rows+mapping['macroTile'][1]-1)//mapping['macroTile'][1]
  stagger = (mapping['staggerU']&255) | ((mapping['staggerStrideShift']<<8)&0x1f00) | (mapping['staggerUMapping']<<13)
  assert not mapping['globalSplitUCoalesced'] and not mapping['globalSplitUWorkGroupMappingRoundRobin']
  assert mapping['workGroupMappingXCC'] == 1 and mapping['workGroupMappingXCCGroup'] == -1 and 0 < cu_count < 1024
  return {'Gemm info':1,'kernel info0':1|(stagger<<16),
    'kernel info1':(cu_count<<22)|(1<<16)|(mapping['workGroupMapping']&65535),'numWG':gx*gy,
    'SizesFree0':columns,'SizesFree1':rows,'SizesFree2':1,'SizesSum0':width,
    'D':out,'C':zero,'A':w,'B':x,
    'strideD0':columns,'strideD1':rows*columns,'strideC0':columns,'strideC1':rows*columns,
    'strideA0':width,'strideA1':width*columns,'strideB0':width,'strideB1':width*rows,
    'alpha':1.0,'beta':0.0,'AddressScaleA':ws,'AddressScaleB':xs,
    'AddressScaleC':ones,'AddressScaleD':ones,'AddressScaleAlphaVec':ones,'bias':bias,
    'biasType':0,'StrideBias':0,'activationAlpha':0.0,'activationBeta':0.0,'activationType':0}


@contextmanager
def argument_adapter(device):
  """Scope the vendor ABI adapter to this process and this already-open device."""
  from tinygrad.runtime.ops_amd import AMDComputeQueue
  from tinygrad.engine.realize import get_call_kernel_args
  from tinygrad.runtime.support.hcq2 import pack_args
  global ACTIVE
  assert not ACTIVE and device.is_usb and device.arch == 'gfx1200'
  original = AMDComputeQueue.kernargs
  def kernargs(queue,call,prg,data):
    name = prg.src[0].arg.name
    if name not in PROGRAMS: return original(queue,call,prg,data)
    # Tensile assembly leaves descriptor kernarg_size zero; its ELF metadata supplies the 176-byte ABI.
    assert queue.dev is device and data.private_segment_size == 0 and data.kernargs_segment_size in (0,176) and not data.enable_dispatch_ptr
    index,shape = PROGRAMS[name]
    addresses = get_call_kernel_args(call,prg,queue.devs)
    assert len(addresses) == 8 and all(a.dtype == dtypes.uint64 for a in addresses)
    values = argument_values(index,shape,addresses,device.cu_cnt)
    entries = []
    for arg in selected(index)['metadata']['.args']:
      value = values.pop(arg['.name'])
      if not isinstance(value,UOp): value = UOp.const(value,dtypes.float32 if isinstance(value,float) else dtypes.uint32)
      assert value.dtype.itemsize == arg['.size']
      entries.append((arg['.offset'],value))
    assert not values
    return UOp(Ops.LINEAR,src=tuple(pack_args(entries,176)),arg='kernargs').getaddr(queue.devs)
  AMDComputeQueue.kernargs = kernargs
  ACTIVE = True
  try: yield
  finally:
    try: device.synchronize()
    finally:
      AMDComputeQueue.kernargs = original
      ACTIVE = False


def program(out,x,w,xs,ws,ones,bias,zero,*,shape,index):
  assert ACTIVE
  rows,columns,width = shape
  record = selected(index)
  mapping = record['solution']['sizeMapping']
  name = f'alpamayo_vendor_{index}_{rows}_{columns}_{width}'
  PROGRAMS[name] = index,shape
  # The placeholder declares all real buffer dependencies; the vendor binary performs the matrix operation.
  value = x.flatten()[0].load().float()+w.flatten()[0].load().float()
  for parameter in (xs,ws,ones,bias,zero): value = value+parameter.flatten()[0].load().float()
  sink = UOp.sink(out.flatten()[0].store(value.cast(dtypes.bfloat16)),
    UOp.special((columns+mapping['macroTile'][0]-1)//mapping['macroTile'][0],'gidx0'),
    UOp.special((rows+mapping['macroTile'][1]-1)//mapping['macroTile'][1],'gidx1'),
    UOp.special(record['metadata']['.max_flat_workgroup_size'],'lidx0'),arg=KernelInfo(name=name))
  return UOp(Ops.PROGRAM,src=(sink,UOp(Ops.LINEAR,src=tuple(sink.toposort())),
    UOp(Ops.SOURCE,arg=f'// Official hipBLASLt solution {index}; unchanged loaded image'),UOp(Ops.BINARY,arg=binary(index))))


def gemm(x,w,scales,ones,bias,zero,index=27955):
  assert ACTIVE and x.ndim == w.ndim == 2 and x.dtype == w.dtype == dtypes.fp8e4m3
  rows,width = x.shape
  columns,wwidth = w.shape
  assert width == wwidth and width >= 32 and width%16 == 0 and rows > 0 and columns%16 == 0
  assert all(t.dtype == dtypes.float32 and t.numel() == 1 for t in scales)
  assert ones.shape == bias.shape == (columns,) and ones.dtype == bias.dtype == dtypes.float32
  assert zero.shape == (rows,columns) and zero.dtype == dtypes.bfloat16
  output = Tensor.empty(rows,columns,device=x.device,dtype=dtypes.bfloat16)
  return Tensor.custom_kernel(output,x.contiguous(),w.contiguous(),*scales,ones,bias,zero,
    fxn=partial(program,shape=(rows,columns,width),index=index))[0]


if __name__ == '__main__':
  from types import SimpleNamespace
  from tinygrad.runtime.ops_amd import _amd_program_image
  reports = []
  for index in (27955,27854):
    blob = binary(index)
    data,image = _amd_program_image(SimpleNamespace(iface=SimpleNamespace(props={'lds_size_in_kb':64}),target=(12,0,0)),blob)
    assert data.private_segment_size == 0 and data.kernargs_segment_size in (0,176) and data.wave32
    report = dict(solution=index,descriptor=data.desc_offset,entry=data.entry_point_offset,
      binary_sha256=hashlib.sha256(blob).hexdigest(),loaded_image_sha256=hashlib.sha256(image).hexdigest(),
      image_bytes=len(image),lds=data.group_segment_size,descriptor_kernargs=data.kernargs_segment_size,
      metadata_kernargs=selected(index)['metadata']['.kernarg_segment_size'])
    reports.append(report)
  print(json.dumps(reports,indent=2))
