"""Run the existing complete CPU mock with genuine USB read alignment enforcement."""
import argparse
import ast
import importlib.util
import json
from pathlib import Path
from types import ModuleType,SimpleNamespace

EXP=Path(__file__).parent
s=importlib.util.spec_from_file_location('ck_aligned_overlay',EXP/'agent-ck-aligned-control-overlay.py')
N=importlib.util.module_from_spec(s);s.loader.exec_module(N)
proof={'strict_USB_reader_used_for_complete_runner':False,'old_six_byte_read_rejected':False,
  'read_calls':0,'read_bytes':0,'physical_word_cases':[],'negatives':[]}

def strict_install(io,raw,dev):
  import numpy as np
  from tinygrad.runtime.support.usb import USBMMIOInterface
  real_read=io.read
  def read(tensor,device):
    b=tensor.uop.buffer;base=b.base;address=int(base.get_storage().buf)
    data=raw(base,np.uint8) # validates this is the CPU mock's SimAllocator, never a GPU allocation
    class USB:
      def pcie_mem_read(self,addr,size):
        assert addr%4==0 and size%4==0,'Unaligned physical USB read'
        assert address<=addr and addr+size<=address+base.nbytes,'USB read outside base storage'
        return memoryview(data).cast('B')[addr-address:addr-address+size]
    class PCI:
      def map_bar(self,bar,off,size):
        assert bar==0 and address<=off and off+size<=address+base.nbytes
        return USBMMIOInterface(USB(),off,size,'B')
    # Only address backing is simulated. The original chunk traversal and actual USBMMIOInterface.__getitem__ execute.
    base_proxy=SimpleNamespace(get_storage=lambda:SimpleNamespace(meta=SimpleNamespace(mapping=SimpleNamespace(paddrs=[(address,base.nbytes)]))))
    proxy=SimpleNamespace(uop=SimpleNamespace(buffer=SimpleNamespace(offset=b.offset,base=base_proxy)),numel=tensor.numel,dtype=tensor.dtype)
    result=real_read(proxy,SimpleNamespace(iface=SimpleNamespace(pci_dev=PCI(),vram_bar=0)))
    proof['read_calls']+=1;proof['read_bytes']+=len(result)
    return result
  io.read=read;proof['strict_USB_reader_used_for_complete_runner']=True

def cases(common,io,raw,dev):
  import numpy as np
  from tinygrad import Tensor,dtypes
  from tinygrad.device import Buffer,BufferSpec
  from tinygrad.uop.ops import UOp
  H=common.load('agent-ck-integrated-hook')
  owner=SimpleNamespace(device=dev,io=io,report={})
  owner.download=lambda t:H.Integration.download(owner,t)
  def extract(source):
    node=next(n for n in ast.walk(ast.parse(source)) if isinstance(n,ast.FunctionDef) and n.name=='patch_word')
    ns={'owner':owner,'Tensor':Tensor,'np':np,'_aligned':N.A}
    exec(compile(ast.Module(body=[node],type_ignores=[]),'<actual-production-patch_word>','exec'),ns)
    return ns['patch_word']
  patched=extract(N.controls_source());old=extract((EXP/'agent-ck-integrated-controls.py').read_text())
  base=Buffer('AMD',64,options=BufferSpec(cpu_access=True)).ensure_allocated()
  raw(base,np.uint8)[:]=np.arange(64,dtype=np.uint8)
  try: old(Tensor(UOp.from_buffer(base,dtypes.bfloat16)),1,0x5aa5,dtypes.uint16)
  except AssertionError as e:
    assert 'off=0, sz=6' in str(e);proof['old_six_byte_read_rejected']=True
  else: raise AssertionError('Old six-byte read escaped the real USB guard')
  for dt,td in ((dtypes.uint16,dtypes.bfloat16),(dtypes.uint32,dtypes.float32)):
    size=dt.itemsize
    for host in (False,True):
      base=Buffer('AMD',64,options=BufferSpec(cpu_access=host)).ensure_allocated()
      for offset in (0,size,3*size):
        view=base.view(56,0).view(7*size,offset).ensure_allocated()
        tensor=Tensor(UOp.from_buffer(view,td))
        for index in (0,3,6):
          data=raw(base,np.uint8);data[:]=np.arange(64,dtype=np.uint8);expected=data.copy()
          value=0x5aa5 if size==2 else 0x5aa55aa5;at=offset+index*size
          expected[at:at+size]=np.frombuffer(value.to_bytes(size,'little'),np.uint8)
          row=patched(tensor,index,value,dt)
          np.testing.assert_array_equal(data,expected)
          assert row['address']==int(base.get_storage().buf)+at and row['offset_bytes']==at
          assert row['host_mapped']==host and row['read_region']['bytes']%4==0
          proof['physical_word_cases'].append({'size':size,'host_mapped':host,'view_offset':offset,'index':index,**row})
  def reject(name,fn):
    try: fn()
    except (AssertionError,ValueError): proof['negatives'].append(name)
    else: raise AssertionError(('Alignment negative accepted',name))
  base=Buffer('AMD',64).ensure_allocated();valid=base.view(14,2).ensure_allocated()
  for name,b,index,dt in (('negative_index',valid,-1,dtypes.uint16),('past_last_index',valid,7,dtypes.uint16),
    ('negative_offset',base.view(14,-2),1,dtypes.uint16),('unaligned_word_offset',base.view(14,1),1,dtypes.uint16),
    ('extent_overrun',base.view(12,60),1,dtypes.uint16),('partial_word_extent',base.view(13,0),1,dtypes.uint16),
    ('unsupported_dtype',valid,1,dtypes.uint8),('rounded_base_extent_overrun',Buffer('AMD',6).ensure_allocated(),2,dtypes.uint16)):
    reject(name,lambda b=b,index=index,dt=dt:N.A.neighbors(b,index,dt))
  proxy=SimpleNamespace(base=base,offset=2,nbytes=14,get_storage=lambda:SimpleNamespace(buf=int(base.get_storage().buf)+4))
  reject('pointer_offset_mismatch',lambda:N.A.neighbors(proxy,1,dtypes.uint16))
  badbase=SimpleNamespace(nbytes=64,get_storage=lambda:SimpleNamespace(buf=int(base.get_storage().buf)+2))
  proxy=SimpleNamespace(base=badbase,offset=0,nbytes=14,get_storage=badbase.get_storage)
  reject('physical_base_alignment',lambda:N.A.neighbors(proxy,1,dtypes.uint16))
  # Corrupt an extra alignment neighbor, outside the original three-word window. The full window must reject it.
  data=raw(base,np.uint8);data[:]=np.arange(64,dtype=np.uint8)
  tensor=Tensor(UOp.from_buffer(base.view(14,0).ensure_allocated(),dtypes.bfloat16));calls=[0]
  download=owner.download
  def corrupt_extra(t):
    calls[0]+=1
    if calls[0]==2: data[0]^=1
    return download(t)
  owner.download=corrupt_extra
  reject('extra_alignment_neighbor_corruption',lambda:patched(tensor,2,0x5aa5,dtypes.uint16))
  owner.download=download
  proof['all_base_bytes_only_target_changed']=True

def main(output):
  # Keep the production mock intact except its permissive raw read and these focused patch cases.
  path=EXP/'agent-ck-integrated-check.py'
  original=json.loads((N.O.OLD.PREPARED/'protocol.json').read_text())
  assert N.C.sha(path)==original['pins'][str(path)]
  source=path.read_text()
  edits=(('  io.read=lambda t,d:raw(t.uop.buffer,np.uint8).tobytes()','  _strict_install(io,raw,dev)'),
    ("      for arm in ('baseline','candidate'):","      _alignment_cases(C,io,raw,dev)\n      for arm in ('baseline','candidate'):"))
  for old,new in edits:
    assert source.count(old)==1;source=source.replace(old,new)
  m=ModuleType('ck_aligned_full_check');m.__file__=str(path);m._strict_install=strict_install;m._alignment_cases=cases
  exec(compile(source,str(path),'exec'),m.__dict__);m.R=N.R;m.C=N.C
  m.main(output)
  assert proof['old_six_byte_read_rejected'] and len(proof['physical_word_cases'])==36
  result=json.loads(output.read_text());result['aligned_control']=proof;result['baseline_lineage']=N.lineage()
  output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
  print(json.dumps({'completed_complete_CPU_mock':True,'aligned_word_cases':36,'negatives':len(proof['negatives']),'gpu_executed':False}))

if __name__=='__main__':
  p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();main(a.output)
