"""Exercise production patch operations with pinned Buffer APIs; never open or execute a GPU."""
import argparse
import ast
from contextlib import nullcontext
import ctypes
from dataclasses import replace
import importlib
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace

EXP=Path(__file__).parent
ROOT=Path('/mnt/nvme-500/tmp/alpamayo-speed-20261116/agent-ck-patch-work')
TINY=Path('/tmp/opencode/alpamayo-asius-20261114/tinygrad')


def main(out,catalog_path=None):
  out.mkdir(parents=True,exist_ok=False)
  os.environ.update(DEV='USB+AMD:LLVM',ALLOW_DEVICE_USAGE='0',PARALLEL='0',PROFILE='0',BEAM='0',JITBEAM='0',JIT='1',CAPTURING='1',
    LLVMOPT='1',GMMU='0',TC_OPT='2',DEBUG='0',TMPDIR=str(ROOT),CACHEDB=str(ROOT/'cpu-cache.db'),LRU='0')
  sys.path[:0]=[str(TINY),str(EXP)]
  import numpy as np
  from tinygrad import Tensor,dtypes
  from tinygrad.device import Buffer,BufferSpec,Device,HostAllocator,Compiled
  from tinygrad.uop.ops import UOp,Ops,graph_rewrite
  from tinygrad.engine import realize
  from tinygrad.runtime import ops_amd
  from tinygrad.runtime.support import hcq2
  from tinygrad.runtime.support.usb import USBMMIOInterface
  from tinygrad.runtime.support.amd import AMDIP,import_module
  from tinygrad.renderer.cstyle import ClangRenderer
  from tinygrad.renderer.llvmir import CPULLVMRenderer
  from tinygrad.runtime.autogen.am import navi_offsets
  transport=importlib.import_module('agent-ck-patch-transport');identity=importlib.import_module('agent-ck-patch-identity')
  common=importlib.import_module('agent-ck-qualified-common')
  physical_delta=0x10000000000000
  class SimulatedController:
    def __init__(self): self.regions={};self.written_bytes=0;self.write_calls=0
    def address(self,address,size):
      host=address-physical_delta
      assert any(base<=host and host+size<=base+length for base,length in self.regions.items())
      return host
    def pcie_mem_write(self,address,data):
      ctypes.memmove(self.address(address,len(data)),bytes(data),len(data))
      self.written_bytes+=len(data);self.write_calls+=1
    def pcie_mem_read(self,address,size): return memoryview(ctypes.string_at(self.address(address,size),size))
  controller=SimulatedController()
  class CapabilityAllocator(HostAllocator):
    def _alloc(self,size,options):
      store=super()._alloc(size,options)
      controller.regions[store.buf]=size
      return replace(store,host=USBMMIOInterface(controller,store.buf+physical_delta,size,'B') if options.cpu_access else None)
    def _free(self,storage,options): controller.regions.pop(storage.buf)
  class HostOnlyAMD:
    device='AMD';arch='gfx1200';target=(12,0,0);host='CPU';is_usb=True;has_copy_queue=True;peer_group='AMD';prof_enabled=False
    def __init__(self):
      self.allocator=CapabilityAllocator(self);self.syncs=0
      self.sdma=import_module('sdma',(6,0,0));self.max_copy_size=0x40000000
      self.nbio=AMDIP('nbif',(6,3,1),bases={i:tuple(getattr(navi_offsets,f'NBIO_BASE__INST{i}_SEG{s}',0) for s in range(9)) for i in range(6)})
      self.hdp_flush=(self.nbio.regBIF_BX_PF0_GPU_HDP_FLUSH_REQ,self.nbio.regBIF_BX_PF0_GPU_HDP_FLUSH_DONE)
      self.iface=SimpleNamespace(props={'lds_size_in_kb':64})
    def sdma_queue(self,index): assert index==0;return SimpleNamespace(ring=SimpleNamespace(nbytes=16<<20))
    def synchronize(self,*_):
      self.syncs+=1
      if transport.ACTIVE is not None:
        assert transport.ACTIVE['staging'].is_allocated() and transport.ACTIVE['target'].is_allocated()
  dev=HostOnlyAMD()
  cpu=Compiled.__new__(Compiled);cpu.remote=None
  Compiled.__init__(cpu,'CPU',HostAllocator(cpu),[ClangRenderer,CPULLVMRenderer],None,arch='x86_64,native')
  def forbidden_device(self,name):
    assert name in ('AMD','CPU'),('Device opening blocked',name)
    return dev if name=='AMD' else cpu
  Device.__class__.__getitem__=forbidden_device
  def audit(event,args):
    if event=='open' and isinstance(args[0],(str,bytes)):
      name=os.fsdecode(args[0]);assert not name.startswith(('/dev/bus/usb','/dev/dri','/dev/kfd')),name
  sys.addaudithook(audit)
  Compiled.wait_timeout_ms=10000
  copy_calls={};host_programs={};copies=[];base_queues=hcq2._build_queues
  frozen=json.loads(catalog_path.read_text()) if catalog_path else None
  def observed_queues(ctx):
    assert len(ctx.batch)==1 and ctx.batch[0][1:]==('AMD','COPY:0')
    item=identity.copy_call(ctx.batch[0][0]);copy_calls[identity.key(item)]=item
    return base_queues(ctx)
  hcq2._build_queues=observed_queues
  def compile_and_emulate(linear,*args,**kwargs):
    active=transport.ACTIVE
    assert active is not None and linear is active['linear'] and len(linear.src)==1 and linear.src[0].body.op is Ops.STORE
    assert not args and kwargs=={'update_stats':False}
    active['raw_checked']=True
    # Real CPU compilation of the full USB-aware HCQ submit graph. Never link or call it.
    linked=realize.compile_linear(linear,input_uops=[],cache=True)
    for program in [u for u in linked.toposort(enter_calls=True) if u.op is Ops.PROGRAM]:
      assert program.arg.target.device=='CPU'
      item=identity.host_program(program);pid=identity.key(item);host_programs[pid]=item
      (out/(pid+'.source')).write_text(program.src[2].arg)
      (out/(pid+'.elf')).write_bytes(program.src[3].arg)
      (out/(pid+'.ast')).write_text(str(program.src[0])+'\n')
    dst,src=active['target'],active['staging']
    assert dst.get_storage().host is None and src.get_storage().host is not None and dst.nbytes==src.nbytes==4
    # Capability simulation only: all addresses are owned anonymous host allocations.
    ctypes.memmove(dst.get_storage().buf,src.get_storage().buf,4)
    copies.append({'dst':transport.info(dst),'src':transport.info(src),'bytes':4})
  realize.run_linear=compile_and_emulate
  def tensor(raw,mapped=True,shape=None):
    dt=dtypes.bfloat16 if raw.dtype.itemsize==2 else dtypes.float32
    buf=Buffer('AMD',raw.nbytes,options=BufferSpec(cpu_access=mapped,nolru=True)).ensure_allocated()
    ctypes.memmove(buf.get_storage().buf,raw.tobytes(),raw.nbytes)
    return Tensor(UOp.from_buffer(buf,dt)).reshape(raw.shape if shape is None else shape)
  def read(t):
    b=t.uop.buffer;return ctypes.string_at(b.get_storage().buf,b.nbytes)
  report={'gpu_executed':False,'active_phase':'validator_all_bit_negative_controls_A_normalized_candidate',
    'io':{'host_write_bytes':0,'host_write_calls':0},'cases':[],'hcq_batches':[],'native_argument_records':[]}
  def write(t,data,offset=0): transport.write(t,data,offset,dev,report)
  tree=ast.parse((EXP/'agent-ck-qualified-run.py').read_text())
  patch=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='patch_words')
  ns=dict(common=common,write=write)
  exec(compile(ast.fix_missing_locations(ast.Module(body=[patch],type_ignores=[])),str(EXP/'agent-ck-qualified-run.py'),'exec'),ns)
  patch_words=ns['patch_words'];started=time.monotonic()
  if frozen is not None:
    transport.preflight(dev,frozen,report)
    guard=importlib.import_module('agent-ck-patch-guard')
    (out/'parent-programs').mkdir()
    runtime_plan={**frozen,'patch_output_root':str(out)}
    context=guard.bindings(common.load('agent-ck-output-common'),runtime_plan,report)
    context.__enter__()
  for count,dt,guard in ((4194304,np.uint16,0x5aa5),(32768,np.uint32,0x5aa55aa5)):
    # Finite independent test words; no claim that these are attention outputs.
    whole=np.full(count+512,0x3f80 if dt==np.uint16 else 0x3f800000,dt);whole[:256]=whole[-256:]=guard
    ret=whole[256:-256].copy();changed,changed_ret,_=common.controls(whole,ret)
    for mapped in (True,False):
      guarded=tensor(whole,True);returned=tensor(ret,mapped,common.QSHAPE if count==4194304 else common.LSHAPE)
      for label,t,before,after in [('guard_change',guarded,whole,changed),('return_change',returned,ret,changed_ret),
                                  ('guard_restore',guarded,changed,whole),('return_restore',returned,changed_ret,ret)]:
        patch_words(t,before,after)
        assert read(t)==after.tobytes(), ('Every byte including untouched neighbours must match',label,count,mapped)
        report['cases'].append({'label':label,'elements':count,'returned_host_mapped':mapped,'full_byte_match':True})
  for input_index in (1,2):
    original=np.full((1,8,2304,128),0x7fc1,np.uint16);original[:,:,:2048]=0x3f80
    changed=original.copy();tail=changed[:,:,2048:]
    for bit in range(16): tail[0,0,bit*2,8]^=np.uint16(1<<bit)
    for head in range(8): tail[0,head,-1,-1]^=np.uint16(1)
    t=tensor(original,True)
    patch_words(t,original,changed);assert read(t)==changed.tobytes()
    patch_words(t,changed,original);assert read(t)==original.tobytes()
    report['cases'].append({'label':'capacity_tail_change_restore','input':input_index,'full_byte_match':True})
  # Real Buffer.view composition, including nonzero tensor and patch offsets on the host-mapped path.
  raw=np.arange(128,dtype=np.uint32);base=tensor(raw,True);b=base.uop.buffer.view(128,64).ensure_allocated()
  t=Tensor(UOp.from_buffer(b,dtypes.uint32));before=read(base);write(t,b'\x12\x34\x56\x78',12)
  expected=bytearray(before);expected[76:80]=b'\x12\x34\x56\x78';assert read(base)==bytes(expected)
  if frozen is not None: transport.finalize(report,runtime_plan)
  report['negative_controls']=[]
  def rejected(name,fn,target):
    original=read(target)
    try: fn()
    except AssertionError: pass
    else: raise AssertionError(('Expected rejection',name))
    assert read(target)==original
    report['negative_controls'].append({'name':name,'rejected':True,'all_target_bytes_unchanged':True})
  rejected('unaligned_length',lambda:write(base,b'12',0),base)
  rejected('unaligned_offset',lambda:write(base,b'1234',2),base)
  rejected('negative_offset',lambda:write(base,b'1234',-4),base)
  rejected('past_buffer_end',lambda:write(base,b'1234',base.uop.buffer.nbytes),base)
  if frozen is not None:
    device_only=tensor(np.full(4194304,0x3f80,np.uint16),False,common.QSHAPE)
    rejected('original_host_property_assertion',lambda:device_only.uop.buffer.host,device_only)
    rejected('unknown_device_copy_range',lambda:write(device_only,b'1234',0),device_only)
    saved=runtime_plan['copy_host_programs'];runtime_plan['copy_host_programs']={}
    try: rejected('unprepared_copy_host_binary_source_AST_ABI',lambda:write(device_only,b'1234',25624),device_only)
    finally: runtime_plan['copy_host_programs']=saved
    for field in ('source_sha256','binary_sha256','ast_key','linear_key','target','params'):
      altered=json.loads(json.dumps(saved));pid=next(iter(altered));altered[pid][field]='deliberately_invalid'
      runtime_plan['copy_host_programs']=altered
      try: rejected('altered_host_program_'+field,lambda:write(device_only,b'1234',25624),device_only)
      finally: runtime_plan['copy_host_programs']=saved
  if frozen is not None:
    context.__exit__(None,None,None)
    assert report['copy_batches'] and len(report['copy_invocations'])>=120
  assert not Device._opened_devices
  assert controller.written_bytes==report['io']['host_write_bytes'] and controller.write_calls==report['io']['host_write_calls']
  report.update(opened_devices=[],seconds=time.monotonic()-started,synchronize_calls=dev.syncs,
    device_copy_count=len(copies),device_copy_bytes=len(copies)*4,copy_call_count=len(copy_calls),host_program_count=len(host_programs),
    simulation='Pinned Buffer.copy_from, Buffer.view, USBMMIOInterface, and full HCQ compilation. Anonymous host backing simulates PCIe and SDMA effects. No compiled program executed.',
    physical_host_address_distinct_from_gpu_address=True,simulated_controller_write_bytes=controller.written_bytes,
    hardware_compile_context={'gfx':'gfx1200','target':[12,0,0],'nbif':[6,3,1],'sdma_module':[6,0,0],
      'sdma_ring_bytes':16<<20,'max_copy_size':dev.max_copy_size,'host_target':str(cpu.renderer.target)},copies=copies)
  destinations=sorted({(r['logical_buffer']['bytes'],r['logical_buffer']['offset_bytes'],r['offset'],r['bytes'])
    for r in report['patch_writes'] if r['transport']=='Buffer.view.copy_from_staged_AMD_dword' and r['complete']})
  catalog={'copy_calls':copy_calls,'copy_host_programs':host_programs,'copy_destinations':[list(x) for x in destinations]}
  if frozen is not None: assert catalog==frozen
  report['strict_runtime_guard_exercised']=frozen is not None
  (out/'catalog.json').write_text(json.dumps(catalog,indent=2)+'\n')
  (out/'cpu-proof.json').write_text(json.dumps(report,indent=2)+'\n')
  print(json.dumps({k:report[k] for k in ('seconds','device_copy_count','copy_call_count','host_program_count','hardware_compile_context')}))


if __name__=='__main__':
  parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True);parser.add_argument('--catalog',type=Path);args=parser.parse_args()
  assert args.out.parent.resolve()==ROOT.resolve();main(args.out,args.catalog)
