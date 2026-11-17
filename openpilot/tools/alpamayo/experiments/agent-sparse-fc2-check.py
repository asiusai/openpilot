#!/usr/bin/env python3
"""CPU controls and offline gfx1200 resource/assembly analysis. Never opens AMD."""
import argparse
from collections import Counter
from functools import partial
import importlib.util
import json
import os
from pathlib import Path
import re
from types import SimpleNamespace
import numpy as np

HERE=Path(__file__).resolve().parent
def local(name):
  spec=importlib.util.spec_from_file_location(name.replace('-','_'),HERE/(name+'.py'))
  m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


def cpu_serial(serial):
  from tinygrad import Tensor,TinyJit,dtypes
  from tinygrad.uop.ops import UOp,Ops,KernelInfo
  from tinygrad.runtime.support import hcq2
  def compression(v,m,f,x):
    value=x.flatten().index(0).load()
    return UOp.sink(v.flatten().index(0).store(value),m.flatten().index(0).store(value+1),
      f.flatten().index(0).store(value+2),arg=KernelInfo(name='agent_sparse_fc2_compress'))
  def gemm(input0,out,v,m,f):
    value=sum(p.flatten().index(0).load()for p in (input0,v,m,f))
    return UOp.sink(out.flatten().index(0).store(value),arg=KernelInfo(name='agent_sparse_fc2_gemm'))
  def run(x,v,m,f,out):
    for _ in range(40):
      v,m,f=Tensor.custom_kernel(v,m,f,x,fxn=compression)[:3]
      out=Tensor.custom_kernel(x,out,v,m,f,fxn=gemm)[1].realize()
    return out
  tensors=[Tensor([n],dtype=dtypes.uint32).contiguous().realize()for n in (2,0,0,0,0)]
  jit=TinyJit(run)
  for _ in range(3):jit(*tensors)
  proof=serial.inspect(jit,('agent_sparse_fc2_compress','agent_sparse_fc2_gemm'))
  assert proof['captured_kernel_calls']==80 and proof['ordered_execution_barriers']==79
  assert proof['kernel_arguments'][1]['output_argument_indices']==[1]
  tensors[0].assign(Tensor([7],dtype=dtypes.uint32)).realize()
  np.testing.assert_array_equal(jit(*tensors).numpy(),np.array([31],np.uint32))
  calls=[c.without_after for c in jit.captured._linear.src]
  batch=[(c,'CPU','COMPUTE:0')for c in calls]
  ctx=SimpleNamespace(batch=batch,calls={('CPU','COMPUTE:0'):list(range(80))},
    peers={('CPU','COMPUTE:0'):set()},queues={'CPU':['COMPUTE:0']},stamps=lambda *_:())
  queues=hcq2._build_queues(ctx)
  wrapped=[(c.after(UOp.sink()),d,q)for c,d,q in batch]
  serial.prove(wrapped,queues,('agent_sparse_fc2_compress','agent_sparse_fc2_gemm'))
  independent=[]
  for i,(call,d,q)in enumerate(batch):
    if i%2:
      src=list(call.src);base,_,_=hcq2.unwrap_lane(src[2])
      src[2]=src[2].substitute({base:UOp.param(20+i,base.dtype,base.max_numel(),base.device)},walk=True)
      call=call.replace(src=tuple(src))
    independent.append((call,d,q))
  rejected=False
  try:serial.prove(independent,queues,('agent_sparse_fc2_compress','agent_sparse_fc2_gemm'))
  except AssertionError:rejected=True
  assert rejected,'Independent outputs must never qualify as serial timings'
  return dict(**proof,nonzero_output_argument_verified=True,AFTER_wrapping_verified=True,
    captured_input_mutation_exact=True,independent_outputs_rejected=True)


def compile_one(out,kernel,old,m,n,k,mode,raw=False,initial=False):
  text,binary=kernel.compiled(m,n,k,mode,raw,initial)
  name=f'{m}-{n}-{k}-{mode}-raw{int(raw)}-C{int(initial)}'
  (out/(name+'.cpp')).write_text(text);(out/(name+'.hsaco')).write_bytes(binary)
  asm=old.disassemble(binary);(out/(name+'.s')).write_text(asm)
  counts=Counter(re.findall(r'^\s*([a-z][a-z0-9_]+)\s',asm,re.M))
  resources=old.metadata(binary)
  if mode=='hybrid':assert counts['v_swmmac_f32_16x16x32_fp8_fp8']>0 and counts['v_wmma_f32_16x16x16_fp8_fp8']>0
  if mode=='dense':assert counts['v_swmmac_f32_16x16x32_fp8_fp8']==0 and counts['v_wmma_f32_16x16x16_fp8_fp8']>0
  assert resources['wavefront_size']==32 and resources['workgroup_processor_mode']==1
  return dict(name=name,shape=[m,n,k],mode=mode,raw=raw,initial_C=initial,resources=resources,
    **descriptor(binary),
    instructions={key:value for key,value in counts.items()if 'wmma'in key or 'bpermute'in key},
    binary_sha256=__import__('hashlib').sha256(binary).hexdigest(),binary_bytes=len(binary))


def descriptor(binary):
  import ctypes
  from tinygrad.runtime.support.elf import elf_loader
  from tinygrad.runtime.autogen import amdgpu_kd
  image,sections,_=elf_loader(binary);start=next(s.header.sh_addr for s in sections if s.name=='.rodata')
  data=amdgpu_kd.llvm_amdhsa_kernel_descriptor_t.from_buffer_copy(bytes(image[start:start+ctypes.sizeof(amdgpu_kd.llvm_amdhsa_kernel_descriptor_t)]))
  result=dict(fp32_denorm_mode=(data.compute_pgm_rsrc1>>16)&3,fp32_rounding_mode=(data.compute_pgm_rsrc1>>12)&3)
  assert result==dict(fp32_denorm_mode=3,fp32_rounding_mode=0),result
  return result


def main():
  p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
  p.add_argument('--compile',action='store_true');a=p.parse_args()
  if os.environ.get('DEV','CPU')!='CPU':raise ValueError('Run with DEV=CPU')
  C,kernel,serial=map(local,('agent-sparse-fc2-common','agent-sparse-fc2-kernel','agent-sparse-fc2-serial'))
  from tinygrad import Device,dtypes
  from tinygrad.uop.ops import UOp,Ops
  report=dict(cpu_only=True,complete=False,all_pass=False,checks=[],compiler=[])
  a.output.parent.mkdir(parents=True,exist_ok=True)
  try:
    rng=np.random.default_rng(166432)
    w=rng.integers(0,256,(64,64),dtype=np.uint8);packed=C.fmt().pack(w)
    def word(row,k):
      address=(row//16*4+k//16)*256+(k%16//4)*64+row%16*4+k%4
      return int.from_bytes(packed.ravel()[address:address+4].tobytes(),'little')
    # Verify EVERY byte of the new peer-lane B32 -> two K16 mapping independently.
    checked=0
    for kt in range(2):
      for nt in range(4):
        b=np.array([[word(nt*16+lane%16,kt*32+(lane//16)*16+j*4)for j in range(4)]for lane in range(32)],np.uint32)
        for lane in range(32):
          peer=b[lane^16]
          got=(b[lane,:2]if lane<16 else peer[2:],peer[:2]if lane<16 else b[lane,2:])
          for h in range(2):
            expected=[word(nt*16+lane%16,kt*32+h*16+(lane//16)*8+j*4)for j in range(2)]
            np.testing.assert_array_equal(got[h],expected);checked+=8
    report['checks'].append(dict(case='asymmetric-B-peer-lane-mapping',independent_bytes=checked,all_exact=True))
    for fixture in C.fixtures():
      q=fixture['a'];values,meta,flags=C.fmt().compress(q)
      expanded=C.fmt().expand(values,meta,flags,q)
      # Omitted signed-zero bit patterns can differ, but finite values cannot.
      np.testing.assert_array_equal(C.fmt().decode(expanded),C.fmt().decode(q))
      assert not np.isnan(C.fmt().decode(fixture['w'])).any()
      report['checks'].append(dict(case=fixture['name'],shape=[64,64,q.shape[1]],
        finite_representation_exact=True,eligible_tiles=int(flags.sum()),total_tiles=int(flags.size)))
    c=np.float32(2**24)
    sequential=np.float32(np.float32(c+np.float32(1))+np.float32(1))
    combined=np.float32(c+np.float32(2))
    assert sequential!=combined
    report['rounding_boundary_witness']=dict(C=float(c),K16_twice=float(sequential),K32_combined=float(combined),
      hardware_prediction=False,meaning='Different reduction boundaries CAN differ. ISA does not specify an equivalence guarantee; GPU controls decide.')
    expected=C.bf16(np.array([0.,-0.,2.**-133,2.**-126,1.,-1.,np.inf,-np.inf,np.nan],np.float32))
    assert expected[:8].tolist()==[0,32768,1,128,16256,49024,32640,65408]
    report['checks'].append(dict(case='BF16-RNE-signedzero-subnormal-special-values',all_exact=True))
    report['serial_cpu']=cpu_serial(serial)
    tiles=2112//16*(9216//32)
    storage=dict(values=tiles*32*2*4,metadata_uint32=tiles*32*4,eligibility_uint32=tiles*4)
    assert sum(storage.values())==14750208 and storage['metadata_uint32']==4866048
    report['actual_workspace_bytes']=dict(**storage,total=sum(storage.values()),
      note='Old format.audit extra_bytes undercounts padded uint32 metadata; use actual allocation sizes.')
    try:C.guard_disk(Path('/tmp/opencode/not-a-capture'))
    except ValueError:pass
    else:raise AssertionError('Capture root-filesystem fallback accepted')
    report['capture_root_filesystem_rejected']=True
    # Confirm native calls declare all three compressor writes and real GEMM IO.
    for mode in ('compress','hybrid'):
      shapes=(((8,32,2),dtypes.uint32),((8,32),dtypes.uint32),((8,),dtypes.uint32),((64,64),dtypes.uint8)) if mode=='compress' else (
        ((64,64),dtypes.bfloat16),((64,64),dtypes.uint8),((64,64),dtypes.uint8),((1,),dtypes.float32),((1,),dtypes.float32),
        ((8,32,2),dtypes.uint32),((8,32),dtypes.uint32),((8,),dtypes.uint32),((1,),dtypes.float32))
      params=[UOp.placeholder(shape,dt,i)for i,(shape,dt)in enumerate(shapes)]
      program=kernel.program(*params,m=64,n=64,k=64,mode=mode)
      assert program.arg.outs==((0,1,2)if mode=='compress'else(0,))
      assert program.arg.ins==((3,)if mode=='compress'else tuple(range(1,9)))
      report['checks'].append(dict(case='native-'+mode+'-actual-IO-metadata',outs=list(program.arg.outs),ins=list(program.arg.ins)))
    if a.compile:
      old=C.load(C.EXP/'agent-sparse-compile.py','sparse_fc2_cpu_disassembler')
      out=a.output.parent/(a.output.stem+'-compiled');out.mkdir(exist_ok=True)
      for m,n,k in ((64,64,32),(64,64,64),(64,64,256),(64,64,9216),(2112,2048,9216)):
        report['compiler'].append(compile_one(out,kernel,old,m,n,k,'compress'))
        for mode in ('dense','hybrid'):
          for raw,initial in ((False,False),(True,False),(False,True),(True,True))if m==64 else((False,False),):
            report['compiler'].append(compile_one(out,kernel,old,m,n,k,mode,raw,initial))
      from tinygrad.codegen import to_program
      from tinygrad.helpers import Target
      from tinygrad.renderer.llvmir import AMDLLVMRenderer
      from tinygrad.runtime.support.compiler_llvm import LLVMCompiler
      renderer=AMDLLVMRenderer(Target('AMD','LLVM','gfx1200'));renderer.compiler=LLVMCompiler('AMDGPU','gfx1200','-cumode')
      for m,n,k in ((64,64,32),(2112,2048,9216)):
        for dtype in (dtypes.float32,dtypes.bfloat16):
          shapes=(((m,n),dtype),((m,k),dtypes.fp8e4m3),((n,k),dtypes.fp8e4m3),((1,),dtypes.float32),((1,),dtypes.float32))
          params=[UOp.placeholder(shape,dt,i)for i,(shape,dt)in enumerate(shapes)]
          program=to_program(kernel.original_program(*params,renderer=renderer),renderer)
          assert program.src[0].arg.name=='alpamayo_fp8_gemm',program.src[0].arg.name
          binary=next(s.arg for s in program.src if s.op is Ops.BINARY)
          name=f'{m}-{n}-{k}-ORIGINAL-4x4-{dtype.name}'
          (out/(name+'.hsaco')).write_bytes(binary);(out/(name+'.s')).write_text(old.disassemble(binary))
          report['compiler'].append(dict(name=name,resources=old.metadata(binary),
            **descriptor(binary),
            globals=list(program.arg.globals),outs=list(program.arg.outs),ins=list(program.arg.ins),
            global_size=list(program.arg.global_size),local_size=list(program.arg.local_size)))
    assert not any(name.startswith('AMD')for name in Device._opened_devices)
    report.update(complete=True,all_pass=True,opened_devices=sorted(Device._opened_devices),
      source_hashes={p.name:C.sha(p)for p in HERE.glob('agent-sparse-fc2-*.py')},gpu_arithmetic_exactness=None)
  except BaseException as error:
    report['error']=dict(type=type(error).__name__,message=str(error));raise
  finally:C.write(a.output,report)
  print(json.dumps(dict(cpu_only=True,all_pass=True,checks=len(report['checks']),compiled=len(report['compiler']))))


if __name__=='__main__':main()
