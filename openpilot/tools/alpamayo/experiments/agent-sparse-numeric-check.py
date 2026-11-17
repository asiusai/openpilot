#!/usr/bin/env python3
"""CPU-only operand proof, immutable capture references, and bounded compilation."""
import argparse,hashlib,importlib.util,json,os
from pathlib import Path
import time
import numpy as np

HERE=Path(__file__).resolve().parent
def local(name):
  spec=importlib.util.spec_from_file_location(name.replace('-','_'),HERE/(name+'.py'))
  m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


def mapping():
  from tinygrad.renderer.tc import amd_rdna4
  from tinygrad import dtypes
  tc=next(tc for tc in amd_rdna4 if tc.dtype_in==dtypes.fp8e4m3 and tc.dtype_out==dtypes.float32)
  a,b,c=tc.frag_coords();seen=[set(),set()];old_changed=0
  for lane in range(32):
    for e in range(8):
      original_k=(lane//16)*4+(e//4)*8+e%4
      previous_k=(lane//16)*8+(e//4)*4+e%4
      assert a[lane][e]==(lane%16,original_k)
      assert b[lane][e]==(original_k,lane%16)
      assert c[lane][e]==((lane//16)*8+e,lane%16)
      seen[0].add(a[lane][e]);seen[1].add(b[lane][e]);old_changed+=original_k!=previous_k
  assert len(seen[0])==len(seen[1])==256 and old_changed==128
  # Independent uint32 packed addresses for all K32 words, output fragments,
  # and both operands. New shuffled B16 must equal the pinned TC slots.
  rng=np.random.default_rng(171104);w=rng.integers(0,256,(64,64),dtype=np.uint8)
  C=local('agent-sparse-numeric-common');packed=C.old().fmt().pack(w).ravel()
  def word(row,k):
    address=(row//16*4+k//16)*256+(k%16//4)*64+row%16*4+k%4
    return int.from_bytes(packed[address:address+4].tobytes(),'little')
  checked=0
  for kt in range(2):
    for tile in range(4):
      b32=np.array([[word(tile*16+lane%16,kt*32+lane//16*16+4*j)for j in range(4)]for lane in range(32)],np.uint32)
      for lane in range(32):
        p=b32[lane^16]
        actual=((b32[lane,[0,2]],p[[0,2]])if lane<16 else(p[[1,3]],b32[lane,[1,3]]))
        for half in range(2):
          expected=[word(tile*16+lane%16,kt*32+half*16+lane//16*4+j*8)for j in range(2)]
          np.testing.assert_array_equal(actual[half],expected);checked+=8
  return dict(pinned_TC_all_ABC_slots_exact=True,K16_coordinate_pairs_per_operand=256,
    previous_K_positions_per_operand_changed=old_changed,B32_peer_reconstruction_bytes_exact=checked,
    per_instruction_K_sequence_original=[(lane//16)*4+(e//4)*8+e%4 for lane in (0,16)for e in range(8)],
    per_instruction_K_sequence_previous=[(lane//16)*8+(e//4)*4+e%4 for lane in (0,16)for e in range(8)],
    ascending_two_K16_preserved=True,original_C_map_unchanged=True,scale_association_unchanged=True)


def compile_all(C,kernel,out):
  old=C.load(C.EXP/'agent-sparse-compile.py','numeric_disassembly')
  prior=C.load(C.OLD/'agent-sparse-fc2-check.py','numeric_descriptor')
  results=[]
  for m,n,k in ((64,64,32),(64,64,64),(64,64,256),(64,64,9216),(2112,2048,9216)):
    for mode in ('dense','hybrid'):
      variants=((False,False),(True,False),(False,True),(True,True))if m==64 else((False,False),(True,False),(False,True))if mode=='hybrid'else((False,False),(True,False))
      for raw,initial in variants:
        text,binary=kernel.compiled(m,n,k,mode,raw,initial);stem=f'{m}-{n}-{k}-{mode}-raw{int(raw)}-C{int(initial)}'
        asm=old.disassemble(binary)
        for suffix,content in (('.cpp',text),('.s',asm)): (out/(stem+suffix)).write_text(content)
        (out/(stem+'.hsaco')).write_bytes(binary)
        resource=old.metadata(binary)
        assert resource['wavefront_size']==32 and resource['workgroup_processor_mode']==1
        assert 'v_wmma_f32_16x16x16_fp8_fp8'in asm
        if mode=='hybrid':assert 'v_swmmac_f32_16x16x32_fp8_fp8'in asm
        results.append(dict(name=stem,shape=[m,n,k],mode=mode,raw=raw,initial=initial,
          binary_sha256=hashlib.sha256(binary).hexdigest(),resources=resource,**prior.descriptor(binary)))
  from tinygrad import dtypes
  from tinygrad.uop.ops import UOp,Ops
  from tinygrad.helpers import Target
  from tinygrad.renderer.llvmir import AMDLLVMRenderer
  from tinygrad.runtime.support.compiler_llvm import LLVMCompiler
  from tinygrad.codegen import to_program
  renderer=AMDLLVMRenderer(Target('AMD','LLVM','gfx1200'));renderer.compiler=LLVMCompiler('AMDGPU','gfx1200','-cumode')
  for m,n,k in ((64,64,32),(64,64,64),(64,64,256),(64,64,9216),(2112,2048,9216)):
    for dtype in (dtypes.float32,dtypes.bfloat16):
      params=[UOp.placeholder(s,d,i)for i,(s,d)in enumerate((((m,n),dtype),((m,k),dtypes.fp8e4m3),((n,k),dtypes.fp8e4m3),((1,),dtypes.float32),((1,),dtypes.float32)))]
      program=to_program(kernel.original().original_program(*params,renderer=renderer),renderer)
      binary=next(s.arg for s in program.src if s.op is Ops.BINARY)
      llvm=next(s.arg for s in program.src if s.op is Ops.SOURCE)
      name=f'{m}-{n}-{k}-original-{dtype.name}';(out/(name+'.ll')).write_text(llvm)
      (out/(name+'.hsaco')).write_bytes(binary);(out/(name+'.s')).write_text(old.disassemble(binary))
      assert program.src[0].arg.name=='alpamayo_fp8_gemm'
      results.append(dict(name=name,binary_sha256=hashlib.sha256(binary).hexdigest(),resources=old.metadata(binary),**prior.descriptor(binary)))
  return results


def cpu_replay(C):
  from tinygrad import Tensor,TinyJit,dtypes
  from tinygrad.uop.ops import UOp,KernelInfo
  def pack(v,m,f,x):
    value=x.flatten().index(0).load()
    return UOp.sink(v.flatten().index(0).store(value),m.flatten().index(0).store(value+1),
      f.flatten().index(0).store(value+2),arg=KernelInfo(name='agent_sparse_fc2_compress'))
  def multiply(out,x,w,s,t,v,m,f,seed):
    value=sum(a.flatten().index(0).load()for a in(x,w,s,t,v,m,f,seed))
    return UOp.sink(out.flatten().index(0).store(value),arg=KernelInfo(name='agent_sparse_numeric_gemm'))
  def run(x,w,s,t,v,m,f,out,seed):
    v,m,f=Tensor.custom_kernel(v,m,f,x,fxn=pack)[:3]
    return Tensor.custom_kernel(out,x,w,s,t,v,m,f,seed,fxn=multiply)[0].realize()
  args=[Tensor([v],dtype=dtypes.uint32).contiguous().realize()for v in(1,2,3,4,0,0,0,0,5)]
  jit=TinyJit(run);jit(*args)
  serial=C.load(C.OLD/'agent-sparse-fc2-serial.py','numeric_cpu_scoped_HCQ');prove=serial.prove
  serial.prove=lambda batch,queues,phases,count=1:prove(batch,queues,phases,count=1)
  phases=('agent_sparse_fc2_compress','agent_sparse_numeric_gemm')
  with serial.observe(jit,phases):jit(*args)
  proof=serial.inspect(jit,phases)
  np.testing.assert_array_equal(jit(*args).numpy(),np.array([21],np.uint32))
  for slot in(0,1,2,3,8):args[slot].assign(Tensor([7],dtype=dtypes.uint32)).realize()
  args[7].assign(Tensor([999],dtype=dtypes.uint32)).realize()
  np.testing.assert_array_equal(jit(*args).numpy(),np.array([59],np.uint32))
  assert proof['captured_kernel_calls']==2 and proof['ordered_execution_barriers']==1
  return dict(**proof,all_current_X_W_scales_seed_mutations_exact=True,output_poison_replaced=True)


def main():
  p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--compile',action='store_true');args=p.parse_args()
  if os.environ.get('DEV')!='CPU':raise ValueError('DEV=CPU required')
  if os.environ.get('OPENBLAS_NUM_THREADS')!='4' or os.environ.get('OMP_NUM_THREADS')!='4':raise ValueError('Use four CPU BLAS/OpenMP threads')
  C=local('agent-sparse-numeric-common');kernel=local('agent-sparse-numeric-kernel')
  report=dict(cpu_only=True,all_pass=False,mapping=mapping(),controls=[],actual_references=[],compiler=[])
  try:
    old,plan,manifest,records,_,_=C.capture()
    report['capture_sha256']=C.sha(C.CAPTURE);report['old_stopped_report_sha256']=C.sha(C.FAILED)
    report['replay_cpu']=cpu_replay(C)
    exact=np.array([[0.,-0.,2.**-149,1.,2.]],np.float32)
    changed=exact.copy();changed[0,-1]=np.nextafter(np.float32(2),np.float32(3))
    metric=C.metrics(changed,exact)
    assert metric['differing_bits_elements']==1 and metric['maximum_ULP']==1 and not metric['bits_exact']
    dense_gate=local('agent-sparse-numeric-run');evidence=[]
    dense_gate.require_dense(evidence,exact,exact,'CPU-exact-positive-control')
    try:dense_gate.require_dense(evidence,changed,exact,'CPU-one-ULP-negative-control')
    except ArithmeticError:pass
    else:raise AssertionError('Aligned dense gate accepted one ULP mismatch')
    bf=C.metrics(np.array([[0x3f80,0x8000,1]],np.uint16),np.array([[0x4000,0,1]],np.uint16))
    assert bf['maximum_absolute_error']==1. and bf['differing_signed_zeros']==1 and bf['differing_bits_elements']==2
    zero=np.zeros((1,1),np.float64)
    assert C.consistency(np.ones((1,1),np.float32),zero,zero,32)['elements_outside_bound']==1
    C.digest(metric)
    report['independent_metric_and_bound_negative_controls']=True
    for fixture in C.fixtures():
      q=fixture['a'];v,m,f=old.fmt().compress(q);proof=C.preservation(q,v,m,f)
      dot,absolute,identity=C.exact_reference(q,fixture['w'])
      if fixture.get('integer_control')or fixture['name']=='integer':
        ints=old.fmt().decode(q).astype(np.int32)@old.fmt().decode(fixture['w']).astype(np.int32).T
        np.testing.assert_array_equal(dot,ints.astype(np.float64))
      if fixture['name']=='basis-every-K-lane':
        corrupted=v.copy();corrupted.ravel()[np.flatnonzero(corrupted.ravel()&127)[0]]=0
        try:C.preservation(q,corrupted,m,f)
        except AssertionError:pass
        else:raise AssertionError('Nonzero preservation gate accepted dropped data')
      report['controls'].append(dict(case=fixture['name'],shape=[64,64,q.shape[1]],**proof,**identity,
        conditional_bound=C.consistency(dot.astype(np.float32),dot,absolute,q.shape[1])))
    for record,data in records:
      start=time.monotonic();a=old.fmt().unpack(data['activation_packed']);w=old.fmt().unpack(data['weight_packed'])
      values,metadata,flags=old.fmt().compress(a);proof=C.preservation(a,values,metadata,flags)
      dot,absolute,identity=C.exact_reference(a,w)
      scale=np.float32(data['activation_scale'][0]*data['weight_scale'][0])
      ideal=old.bf16(dot.astype(np.float32)*scale)
      saved=np.asarray(data['original_bf16']);assert np.isfinite((saved.astype(np.uint32)<<16).view(np.float32)).all()
      report['actual_references'].append(dict(layer=record['layer'],shape=record['shape'],**identity,**proof,
        eligible_tiles=int(flags.sum()),total_tiles=int(flags.size),original_vs_ideal_bf16=old.compare(saved,ideal),
        seconds=time.monotonic()-start,exact_FP64_bound_in_2pow_minus18_units=a.shape[1]*448**2*2**18,
        independent_fixed_int64_dot_coordinates=16,quality_claim=False))
      print('CPU exact-dyadic reference complete: layer',record['layer'],flush=True)
    if args.compile:
      output=args.output.parent/(args.output.stem+'-compiled');output.mkdir(parents=True,exist_ok=True)
      report['compiler']=compile_all(C,kernel,output)
    from tinygrad import Device
    assert not any(d.startswith('AMD')for d in Device._opened_devices)
    report.update(all_pass=True,opened_devices=sorted(Device._opened_devices),source_hashes={p.name:C.sha(p)for p in HERE.glob('agent-sparse-numeric-*.py')})
  except BaseException as error:report['error']=dict(type=type(error).__name__,message=str(error));raise
  finally:C.write(args.output,report)
  print(json.dumps(dict(cpu_only=True,all_pass=True,controls=len(report['controls']),actual_layers=len(report['actual_references']),compiled=len(report['compiler']))))


if __name__=='__main__':main()
