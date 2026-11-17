"""CPU-only: real runner lifecycle, byte transport mocks, ABI, and negative controls."""
import contextlib, importlib, json, math, os, sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

EXP=Path(__file__).parent
ROOT=Path('/mnt/nvme-500/tmp/alpamayo-speed-20261116/agent-vision-attention-artifacts')
os.environ.update(DEV='CPU',CPU='1',AMD='0',ALLOW_DEVICE_USAGE='0',PARALLEL='0',PROFILE='0',
  CACHEDB=str(ROOT/'cpu-check-cache.db'),TMPDIR=str(ROOT),OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
sys.path.insert(0,'/tmp/opencode/alpamayo-asius-20261114/tinygrad');sys.path.insert(0,str(EXP))
C=importlib.import_module('agent-vision-attention-common')
R=importlib.import_module('agent-vision-attention-run')

def main():
  import numpy as np
  import tinygrad
  from tinygrad import Device,Tensor,dtypes
  from tinygrad.runtime import ops_amd
  from tinygrad.uop.ops import UOp
  from tinygrad.device import Buffer
  from tinygrad.codegen import to_program
  from tinygrad.helpers import Target
  from tinygrad.renderer.llvmir import AMDLLVMRenderer
  assert not Device._opened_devices
  plan=R.verify();cases={n:{k:np.load(v['path'],allow_pickle=False) for k,v in f.items()} for n,f in plan['fixtures'].items()}
  def reject(fn):
    try:fn()
    except (AssertionError,ValueError):return
    raise AssertionError('negative control was accepted')
  tests=[]
  # Exact host checker exercises ALL bits, output boundaries, dimensions, and both redzones.
  for b in (1,2):
    for kind in ('O','LSE'):
      dt=np.uint16 if kind=='O' else np.uint32;sentinel=0x7fc1 if kind=='O' else 0x7fc12345
      count=math.prod(C.shape(b)) if kind=='O' else b*16*720
      good=np.full(count,0x3f80 if kind=='O' else 0x3f800000,dt);guard=np.pad(good,(256,256),constant_values=sentinel)
      R.check_words(guard.tobytes(),good.tobytes(),b,kind,'A')
      for index in (0,count//2,count-1):
        for bit in range(16 if kind=='O' else 32):
          bad=good.copy();bad[index]^=1<<bit
          reject(lambda:R.check_words(guard.tobytes(),bad.tobytes(),b,kind,'A'))
      for index in (0,255,count+256,count+511):
        bad=guard.copy();bad[index]^=1
        reject(lambda:R.check_words(bad.tobytes(),good.tobytes(),b,kind,'A'))
      for value in (sentinel,0x7f80 if kind=='O' else 0x7f800000):
        bad=guard.copy();bad[256+count-1]=value;returned=good.copy();returned[-1]=value
        reject(lambda:R.check_words(bad.tobytes(),returned.tobytes(),b,kind,'A'))
      tests.append({'batch':b,'kind':kind,'negative_all_bit_positions':True,'redzones_poison_infinity_tail':True})
  # Actual program construction and strict exact association matching.
  I=C.load(EXP/'agent-ck-source-identity.py');renderer=AMDLLVMRenderer(Target('AMD','LLVM','gfx1200','USB'))
  def tensor(s,dt=dtypes.bfloat16):return Tensor(UOp.from_buffer(Buffer('AMD',math.prod(s)*dt.itemsize),dt)).reshape(s)
  abi=[]
  def no_open(*a,**k):raise AssertionError('unexpected device open')
  with patch.object(type(Device),'__getitem__',no_open):
    for b in (1,2):
      inputs=[tensor(C.shape(b)) for _ in range(3)]+[tensor((math.prod(C.shape(b))+512,)),tensor((b*16*720+512,),dtypes.float32)]
      reject(lambda:C.outputs('candidate','A',*inputs))
      with C.abi_scope():
        result=C.outputs('candidate','A',*inputs);raw,_=result[0].linear_with_vars(*result[1:])
        native=next(call for call in raw.src if call.body.op.name=='PROGRAM')
        p=to_program(native.body,renderer);I.authorize(p,plan)
        reject(lambda:I.authorize(p.replace(src=(*p.src[:2],p.src[2].replace(arg=p.src[2].arg+'\n'),p.src[3])),plan))
        data=SimpleNamespace(**json.loads((ROOT/'candidate/loader.json').read_text()))
        queue=SimpleNamespace(dev=SimpleNamespace(arch='gfx1200',target=(12,0,0)),devs={})
        addresses=[UOp.const(0x100000+i*0x100000,dtypes.uint64) for i in range(5)]
      with patch('tinygrad.engine.realize.get_call_kernel_args',return_value=addresses),patch.object(UOp,'getaddr',lambda self,*a:self),C.abi_scope(abi):
        words=ops_amd.AMDComputeQueue.kernargs(queue,native,p,data)
        assert words.src[-1].arg==C.implicit(b)
        bad=SimpleNamespace(**vars(data));bad.rsrc1^=1<<29
        reject(lambda:ops_amd.AMDComputeQueue.kernargs(queue,native,p,bad))
      assert not C.ACTIVE
      reject(lambda:C.outputs('candidate','A',inputs[0].reshape(b,16,360,144),*inputs[1:]))
  # Run the COMPLETE production driver. Only numerical execution and device allocation are simulated.
  # Actual checked host read/write code enforces the USB four-byte alignment rule.
  blocks={};counter=[0x10000000];calls=[]
  class Host:
    def __init__(self,b):self.b=b
    def view(self,offset=0,size=None,fmt='B'):
      assert fmt=='B';return memoryview(self.b.data)[self.b.offset+offset:self.b.offset+offset+size]
  class Buf:
    def __init__(self,size,base=None,offset=0):
      self.nbytes=size;self.offset=offset;self.base=base or self
      if base is None:
        self.address=counter[0];counter[0]+=((size+65535)//65536)*65536
        self.data=bytearray(size);blocks[self.address]=self
      else:self.address=base.address;self.data=base.data
      self.host=Host(self)
    def get_storage(self):return SimpleNamespace(buf=self.address+self.offset,meta=SimpleNamespace(mapping=SimpleNamespace(paddrs=[(self.address,self.base.nbytes)])))
  class FT:
    def __init__(self,s,dt,buffer=None):self.shape=tuple(s);self.dtype=dt;self.uop=SimpleNamespace(buffer=buffer or Buf(math.prod(s)*dt.itemsize))
    def numel(self):return math.prod(self.shape)
    def realize(self,*args):return self
  def bar(*,bar,off,size):
    assert off%4==0 and size%4==0 and size<=65536,(off,size)
    for address,b in blocks.items():
      if address<=off and off+size<=address+b.nbytes:return memoryview(b.data)[off-address:off-address+size]
    raise AssertionError('out-of-range physical BAR read')
  dev=SimpleNamespace(device='AMD',arch='gfx1200',target=(12,0,0),is_usb=True,prof_enabled=False,renderer=renderer,
    synchronize=lambda:None,iface=SimpleNamespace(vram_bar=0,pci_dev=SimpleNamespace(map_bar=bar)))
  fixtures={}
  def payload(x,b,boundary):return (x[:b] if boundary=='A' else x[:b].transpose(0,2,1,3).copy()).tobytes()
  for b in (1,2):
    for boundary in ('A','B'):
      for name,case in cases.items():fixtures[(b,boundary,tuple(payload(case[k],b,boundary) for k in ('q','k','v')))]=name
  def simulated(arm,boundary,q,k,v,os_,ls):
    assert C.ACTIVE;b=q.shape[0]
    key=(b,boundary,tuple(bytes(t.uop.buffer.data) for t in (q,k,v)));name=fixtures[key]
    f=cases[name];out=f['o'][:b].copy();u=out.view(np.uint32)
    bits=((u+np.uint32(0x7fff)+((u>>16)&1))>>16).astype(np.uint16)
    os_.uop.buffer.data[512:-512]=bits.tobytes()
    result=FT((b,720,16,72) if boundary=='B' else C.shape(b),dtypes.bfloat16)
    result.uop.buffer.data[:]=(bits.transpose(0,2,1,3).copy() if boundary=='B' else bits).tobytes()
    outputs=(result,)
    if arm=='candidate':
      ls.uop.buffer.data[1024:-1024]=f['lse'][:b].tobytes()
      l=FT((b,16,720),dtypes.float32);l.uop.buffer.data[:]=f['lse'][:b].tobytes();outputs+=(l,)
    calls.append((b,boundary,arm,name));return outputs
  class Jit:
    def __init__(self,fn):self.fn=fn;self.cnt=0
    def __call__(self,*args):self.cnt+=1;return self.fn(*args)
  io=C.load(EXP/'agent-ck-attention-v3-run.py')
  # Source/ABI guards were tested with real UOps above. No simulated arithmetic is represented as GPU evidence.
  with patch.object(type(Device),'__getitem__',lambda *a:dev),patch.object(tinygrad,'TinyJit',Jit),\
       patch.object(io,'allocate',lambda device,s,dt:FT(s,dt)),patch.object(C,'outputs',simulated),\
       patch.object(R,'bindings',lambda *a:contextlib.nullcontext()):
    R.run(plan,ROOT/'cpu-lifecycle-report.json')
  report=json.loads((ROOT/'cpu-lifecycle-report.json').read_text())
  assert report['completed'] and len(report['fixtures'])==7 and len(report['replays'])==12
  assert not C.ACTIVE and not Device._opened_devices
  proof={'gpu_executed':False,'opened_devices':[],'arithmetic_simulated_not_validated':True,
    'complete_production_driver_calls':len(calls),'actual_ABI_invocations':abi,'validator_negatives':tests,
    'production_host_transport_simulated_with_USB_four_byte_rule':True,'driver_cleanup':True,
    'simulated_report_sha256':C.sha(ROOT/'cpu-lifecycle-report.json'),'runner_sha256':C.sha(EXP/'agent-vision-attention-run.py')}
  # The production runner necessarily sets its GPU flag. Label this separate CPU artifact unambiguously.
  report.update(gpu_executed=False,simulation_only=True,numerical_micro_passed=False,speed_screen_passed=False)
  report['timings']={};report['speed']={};report['classification']='CPU lifecycle simulation; no numerical or speed evidence'
  (ROOT/'cpu-lifecycle-report.json').write_text(json.dumps(report,indent=2)+'\n')
  proof['simulated_report_sha256']=C.sha(ROOT/'cpu-lifecycle-report.json')
  (ROOT/'cpu-check.json').write_text(json.dumps(proof,indent=2)+'\n')
  print(json.dumps({k:v for k,v in proof.items() if k not in ('actual_ABI_invocations','validator_negatives')},indent=2))

if __name__=='__main__':main()
