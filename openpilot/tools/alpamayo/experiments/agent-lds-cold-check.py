"""CPU execution and offline gfx1200 compilation for the private rotating trial."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('agent_lds_cold_main',HERE/'agent-lds-cold-run.py')
S=importlib.util.module_from_spec(spec)
spec.loader.exec_module(S)


def rejected(fn):
  try: fn()
  except (AssertionError,IndexError): return True
  raise AssertionError('A deliberately invalid graph was accepted')


def numeric_program(fn,render):
  from tinygrad import dtypes
  from tinygrad.codegen import to_program
  from tinygrad.uop.ops import UOp
  m,n,k=S.SHAPE
  specs=[((m,n),dtypes.bfloat16),((m,k),dtypes.fp8e4m3),((n,k),dtypes.fp8e4m3),((1,),dtypes.float32),((1,),dtypes.float32)]
  return to_program(fn(*(UOp.placeholder(shape,dtype,i) for i,(shape,dtype) in enumerate(specs))),render),specs


def cpu_audit(render,binaries):
  from tinygrad import Device,dtypes
  from tinygrad.codegen import to_program
  from tinygrad.uop.ops import UOp
  assert not Device._opened_devices
  artifact=HERE/'agent-lds-cold-artifacts'
  artifact.mkdir(exist_ok=True)
  proofs={}
  for label,fn in S.numeric_functions(render).items():
    program,specs=numeric_program(fn,render)
    assert S.E.sha(program.src[3].arg)==binaries[label]
    out,a,_,sx,sw=[UOp.param(slot,dtype,int(np.prod(shape)),device='AMD') for slot,(shape,dtype) in zip((7,0,3,1,2),specs)]
    weights=[UOp.param(3+i,dtypes.fp8e4m3,S.SHAPE[1]*S.SHAPE[2],device='AMD') for i in range(4)]
    proofs[label]={}
    for workload,order in S.ORDERS.items():
      calls=[program.call(out,a,weights[index],sx,sw) for index in order]
      proofs[label][workload]=S.offline_queue(calls,order,binaries[label])
    order=S.ORDERS['rotating']
    calls=[program.call(out,a,weights[index],sx,sw) for index in order]
    bad_output=UOp.param(8,dtypes.bfloat16,S.SHAPE[0]*S.SHAPE[1],device='AMD')
    negatives={
      'dropped_call':rejected(lambda:S.offline_queue(calls[:-1],order,binaries[label])),
      'all_same_weight':rejected(lambda:S.offline_queue([program.call(out,a,weights[0],sx,sw)]*40,order,binaries[label])),
      'reordered_weights':rejected(lambda:S.offline_queue(calls[::-1],order,binaries[label])),
      'changed_output':rejected(lambda:S.offline_queue(calls[:-1]+[program.call(bad_output,a,weights[3],sx,sw)],order,binaries[label])),
      'changed_binary':rejected(lambda:S.offline_queue(calls,order,'0'*64)),
    }
    proofs[label]['negative_controls']=negatives
  validation={}
  size=S.SHAPE[0]*S.SHAPE[1]
  cases={
    'compare':(S.compare_kernel,[((size//1024,),dtypes.uint16),((size,),dtypes.bfloat16),((size,),dtypes.bfloat16)]),
    'fill':(S.fill_kernel,[((size,),dtypes.bfloat16),((2,),dtypes.uint32)]),
    'poke':(S.poke_kernel,[((size,),dtypes.bfloat16),((2,),dtypes.uint32)]),
  }
  for name,(fn,specs) in cases.items():
    program=to_program(fn(*(UOp.placeholder(shape,dt,i) for i,(shape,dt) in enumerate(specs))),render)
    source,binary=program.src[2].arg,program.src[3].arg
    resources=S.H.C.metadata(binary)
    assert resources['private_segment_fixed_size']==resources['vgpr_spill_count']==resources['sgpr_spill_count']==0
    assert resources['wavefront_size']==32 and resources['workgroup_processor_mode']==1
    (artifact/(name+'.ll')).write_text(source)
    (artifact/(name+'.hsaco')).write_bytes(binary)
    (artifact/(name+'.s')).write_text(S.H.C.disassemble(binary))
    validation[name]=dict(binary_sha256=S.E.sha(binary),source_sha256=S.E.sha(source.encode()),resources=resources,
      global_size=program.arg.global_size,local_size=program.arg.local_size)
  env=dict(os.environ,DEV='CPU',ALLOW_DEVICE_USAGE='1')
  child=artifact/'host.json'
  result=subprocess.run([sys.executable,str(Path(__file__)),'--host','--output',str(child)],env=env,capture_output=True,text=True,timeout=120)
  (artifact/'host.log').write_text(result.stdout+result.stderr)
  assert result.returncode==0, result.stdout+result.stderr
  host=json.loads(child.read_text())
  assert host['complete'] and not host['gpu_executed']
  assert not Device._opened_devices
  return dict(offline_rotation_queue_proofs=proofs,validator_compilation=validation,cpu_execution=host,
    planned_io=dict(operand_upload_bytes=180*1024**2,full_output_cpu_readbacks=0,flags_per_check=4096,bytes_per_check=8192),
    traffic_scope=dict(weights_bytes=72*1024**2,activation_bytes=18*1024**2,shared_output_bytes=8*1024**2,
      single_weight_A_W_output_bytes=44*1024**2,rotating_A_W_output_bytes=98*1024**2,
      nominal_operand_request_bytes=dict(current=1207959552,amd4=428212224),
      nominal_operand_request_reduction=1-428212224/1207959552,measured_dram_bytes=None,measured_cache_hit_rate=None,
      comment='Cache-footprint stress only. Four weights exceed the cited 32 MiB Infinity Cache; guaranteed coldness is not established.'))


def host():
  from tinygrad import Device,Tensor,TinyJit,dtypes
  from tinygrad.uop.ops import UOp,KernelInfo
  assert os.environ['DEV']=='CPU'
  def tensor(bits): return Tensor(np.array(bits,dtype=np.uint16),device='CPU').bitcast(dtypes.bfloat16).realize()
  def compare(a,b):
    flags=Tensor.full(((a.numel()+1023)//1024,),1,device='CPU',dtype=dtypes.uint16).contiguous().realize()
    return Tensor.custom_kernel(flags,a,b,fxn=S.compare_kernel)[0].realize()
  validation=[]
  for size in (1,31,32,1023,1024,1025,4097,S.SHAPE[0]*S.SHAPE[1]):
    rng=np.random.default_rng(size)
    a=rng.integers(0,65536,size,dtype=np.uint16)
    a[(a&0x7f80)==0x7f80]=0x3f80
    b=a.copy()
    for index in set((0,size//2,size-1,min(1023,size-1),min(1024,size-1))): b[index]^=1
    for name,aa,bb in (('equal',a,a),('changed',a,b),('poison',np.full(size,0x7fc1,np.uint16),np.full(size,0x7fc1,np.uint16))):
      actual=compare(tensor(aa),tensor(bb)).numpy()
      bad=(aa!=bb)|((aa&0x7f80)==0x7f80)|((bb&0x7f80)==0x7f80)
      expected=np.array([bad[i:i+1024].any() for i in range(0,size,1024)],np.uint16)
      np.testing.assert_array_equal(actual,expected)
      validation.append(dict(size=size,case=name,all_flags_exact=True))
  # Pin every BF16 bit, finite encoding, signed zero, and the entire tail domain.
  all_bits=np.arange(65536,dtype=np.uint16)
  for bit in range(16):
    a=np.full(1025,0x3f00,np.uint16); b=a.copy(); b[-1]^=np.uint16(1<<bit)
    np.testing.assert_array_equal(compare(tensor(a),tensor(b)).numpy(),[0,1])
  np.testing.assert_array_equal(compare(tensor(all_bits),tensor(all_bits)).numpy(),
    np.array([((all_bits[i:i+1024]&0x7f80)==0x7f80).any() for i in range(0,65536,1024)],np.uint16))
  output=tensor(np.zeros(1025,np.uint16))
  control=Tensor(np.array([0x7fc1,1024],np.uint32),device='CPU').realize()
  Tensor.custom_kernel(output,control,fxn=S.fill_kernel)[0].realize()
  np.testing.assert_array_equal(output.bitcast(dtypes.uint16).numpy(),np.full(1025,0x7fc1,np.uint16))
  control.uop.buffer.host[:]=np.array([0x8000,1024],np.uint32).tobytes()
  Tensor.custom_kernel(output,control,fxn=S.poke_kernel)[0].realize()
  expected=np.full(1025,0x7fc1,np.uint16); expected[-1]=0x8000
  np.testing.assert_array_equal(output.bitcast(dtypes.uint16).numpy(),expected)
  # Exercise both comparison capture signatures and changed inputs after capture.
  dual,single=TinyJit(compare),TinyJit(lambda x:compare(x,x))
  a,b=tensor(np.zeros(1025,np.uint16)),tensor(np.zeros(1025,np.uint16))
  for _ in range(3): assert not dual(a,b).numpy().any(); assert not single(a).numpy().any()
  b.uop.buffer.host[:]=expected.tobytes()
  assert dual(a,b).numpy().any() and single(b).numpy().any()

  def numeric(out,a,w,sx,sw):
    r=UOp.range(out.numel(),0)
    value=(a.flatten().index(0)+w.flatten().index(0)+sx.flatten().index(0)+sw.flatten().index(0)).cast(dtypes.bfloat16)
    return out.flatten().index(r).store(value).end(r).sink(arg=KernelInfo(name='agent_lds_cold_cpu_graph'))
  a=Tensor(np.array([1],np.float32),device='CPU').realize()
  sx=Tensor(np.array([2],np.float32),device='CPU').realize()
  sw=Tensor(np.array([3],np.float32),device='CPU').realize()
  weights=tuple(Tensor(np.array([10*(i+1)],np.float32),device='CPU').realize() for i in range(4))
  graphs={}
  for name,order in S.ORDERS.items():
    jit=S.RotatingJit(numeric,order,None,(2,3,4))
    for _ in range(3): result=jit(a,sx,sw,*weights)[0]
    np.testing.assert_array_equal(result.float().numpy(),np.full((2,3),6+10*(order[-1]+1),np.float32))
    calls=list(jit.captured._linear.src)
    proof=S.offline_queue(calls,order,device='CPU')
    for phase in range(4):
      permuted=weights[phase:]+weights[:phase]
      result=jit(a,sx,sw,*permuted)[0]
      np.testing.assert_array_equal(result.float().numpy(),np.full((2,3),6+10*((order[-1]+phase)%4+1),np.float32))
    last=order[-1]
    weights[last].uop.buffer.host[:]=np.array([70],np.float32).tobytes()
    np.testing.assert_array_equal(jit(a,sx,sw,*weights)[0].float().numpy(),np.full((2,3),76,np.float32))
    weights[last].uop.buffer.host[:]=np.array([10*(last+1)],np.float32).tobytes()
    assert rejected(lambda:jit(a,sx,sw,weights[0],weights[0],weights[2],weights[3]))
    graphs[name]=dict(proof=proof,rebinding_all_four_phases=True,inplace_final_weight_change=True,aliased_weights_rejected=True)
  data=S.H.fixture((17,48,256),4)
  known={key:S.E.sha(data[key].tobytes()) for key in S.KEYS}
  class FakeIO:
    def write(self,*_): pass
  changes=[]
  for state,expect in ((5,['a']),(5,[]),(6,['a','w','sx','sw']),(0,['a','w','sx','sw'])):
    changed=S.update_changed(FakeIO(),(None,)*4,S.H.fixture((17,48,256),state),known)
    assert changed==expect; changes.append(dict(state=state,changed=changed))
  arrays=S.weight_data(S.H.fixture(S.SHAPE,0))
  assert sum(w.nbytes for w in arrays)==72*1024**2
  assert all(np.isfinite(S.E.decode_fp8(w)).all() for w in arrays)
  assert len(set(S.E.sha(w.tobytes()) for w in arrays))==4
  assert all(name in ('CPU','NPY') for name in Device._opened_devices),Device._opened_devices
  return dict(complete=True,gpu_executed=False,opened_devices=list(Device._opened_devices),validator_cases=validation,
    every_bf16_bit_and_encoding=True,fill_poke_and_jit_replay=True,actual_cpu_jit_graphs=graphs,
    changed_only_uploads=changes,weight_sha256=[S.E.sha(w.tobytes()) for w in arrays],weights_total_bytes=sum(w.nbytes for w in arrays))


if __name__=='__main__':
  parser=argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--host',action='store_true',required=True)
  parser.add_argument('--output',type=Path,required=True)
  args=parser.parse_args()
  S.E.write_json(args.output,host())
