"""CPU oracles and genuine gfx1200 compilation. Never opens an AMD device."""
import argparse
import ctypes
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import struct
import sys

HERE = Path(__file__).resolve().parent
TINY = Path('/tmp/opencode/alpamayo-asius-20261114/tinygrad')


def load(path):
  spec = importlib.util.spec_from_file_location(Path(path).stem.replace('-', '_'), path)
  result = importlib.util.module_from_spec(spec); spec.loader.exec_module(result)
  return result


def bf16(values):
  import numpy as np
  bits = np.asarray(values, np.float32).view(np.uint32)
  return np.where(((-bits)&0x7f800000)!=0,
                  bits+((bits>>16)&1)+0x7fff,
                  np.where((bits&0xffff)!=0,bits|0x10000,bits)).astype(np.uint32).__rshift__(16).astype(np.uint16)


def norm_b(values):
  import numpy as np
  bits = np.asarray(values, np.float32).view(np.uint32)
  return ((bits+0x7fff+((bits>>16)&1))>>16).astype(np.uint16)


def floats(values):
  import numpy as np
  return (np.asarray(values,np.uint16).astype(np.uint32)<<16).view(np.float32)


def packed(values):
  rows,width=values.shape
  return values.reshape(rows//16,16,width//16,4,4).transpose(0,2,3,1,4).copy().reshape(rows,width)


def fp8_nearest(values):
  """Independent nearest-neighbor E4M3FN oracle, including even ties and signed zero."""
  import numpy as np
  codes=np.arange(127,dtype=np.uint8)
  exponents=(codes>>3).astype(np.int32); mantissas=(codes&7).astype(np.float64)
  table=np.where(exponents==0,mantissas*2.**-9,(1+mantissas/8)*np.exp2(exponents-7))
  x=np.asarray(values,np.float32); signs=np.signbit(x).astype(np.uint8)*128
  magnitude=np.minimum(np.abs(x).astype(np.float64),448)
  high=np.searchsorted(table,magnitude,side='left').clip(0,126); low=np.maximum(0,high-1)
  dl,dh=magnitude-table[low],table[high]-magnitude
  result=np.where(dl<dh,low,np.where(dh<dl,high,np.where((low&1)==0,low,high))).astype(np.uint8)|signs
  return np.where(np.isnan(x),np.uint8(127)|signs,result).astype(np.uint8)


def oracle(a,other,weight,scale, *, inverse=None, reciprocal=None):
  """Frozen reduction emulation. Hardware rsqrt/rcp approximations require parent GPU checks."""
  import numpy as np
  residual=bf16(floats(a)+floats(other)); x=floats(residual)
  rows,width=x.shape
  thread=np.zeros((rows,256),np.float32)
  for offset in range(0,width,256):
    values=x[:,offset:offset+256]-np.float32(0)
    thread=np.float32(thread+np.float32(values*values))
  for shift in (16,8,4,2,1): thread=np.float32(thread+thread[:,np.arange(256)^shift])
  variance=np.zeros(rows,np.float32)
  for wave in range(8): variance=np.float32(variance+thread[:,wave*32])
  variance=np.float32(variance/np.float32(width))
  if inverse is None: inverse=np.float32(1/np.sqrt(np.float32(variance+np.float32(1e-5))))
  if reciprocal is None: reciprocal=np.float32(1/np.float32(scale))
  normalized=norm_b(np.float32((x-np.float32(0))*np.asarray(inverse)[:,None]))
  weighted=norm_b(np.float32(floats(normalized)*floats(weight)))
  scaled=np.float32(floats(weighted)*np.float32(reciprocal))
  scaled=np.where(scaled < -448,np.float32(-448),scaled)
  scaled=np.where(448 < scaled,np.float32(448),scaled)
  return residual,packed(fp8_nearest(scaled)),weighted,variance


def cpu_checks(kernel):
  import numpy as np
  import torch
  source=kernel.source(); original=kernel.frozen_norm_source()
  for statement in ('squared=sum(squared); if(lane==0) sums[wave]=squared; barrier();',
                    'float variance=0; for(int i=0;i<8;i++) variance+=sums[i]; variance/=WIDTH;',
                    'float inv=__builtin_amdgcn_rsqf(variance+EPSILON);',
                    'x+=shuffle<0x401f>(x); x+=shuffle<0x201f>(x); x+=shuffle<0x101f>(x);'):
    assert statement in source and statement in original
  assert '#pragma clang fp contract(off)' in source
  rng=np.random.default_rng(90602048)
  a=bf16(rng.normal(size=(16,2048)).astype(np.float32)); b=bf16(rng.normal(size=(16,2048)).astype(np.float32))
  w=bf16(rng.normal(1,.2,size=2048).astype(np.float32))
  a[0]=0; b[0]=0
  a[1]=0x3f80; b[1]=0x3b80  # 1 + 2^-8 is a BF16 tie.
  reference=oracle(a,b,w,np.float32(.0125))
  torch_bits=torch.from_numpy(floats(a)+floats(b)).to(torch.bfloat16).view(torch.uint16).numpy()
  np.testing.assert_array_equal(reference[0],torch_bits)
  assert np.all(reference[0][1]==0x3f80)
  normalized=np.array([1.00390625],np.float32)
  weight=np.array([1.0078125],np.float32)
  staged=norm_b(floats(norm_b(normalized))*weight)
  collapsed=norm_b(normalized*weight)
  assert int(staged[0])==0x3f81 and int(collapsed[0])==0x3f82
  torch_staged=(torch.from_numpy(normalized).to(torch.bfloat16).float()*torch.from_numpy(weight)).to(torch.bfloat16).view(torch.uint16).numpy()
  np.testing.assert_array_equal(staged,torch_staged)
  # Independent scalar grouping: eight per-thread products, then five XOR stages.
  row=floats(reference[0][2]); partial=[]
  for t in range(256):
    total=np.float32(0)
    for index in range(t,2048,256): total=np.float32(total+np.float32(row[index]*row[index]))
    partial.append(total)
  for bit in (16,8,4,2,1): partial=[np.float32(partial[i]+partial[i^bit]) for i in range(256)]
  total=np.float32(0)
  for wave in range(8): total=np.float32(total+partial[wave*32])
  assert np.float32(total/np.float32(2048)).view(np.uint32)==reference[3][2].view(np.uint32)
  # Bit codec is independent of the Tensor FP8 lowering. Exhaust every finite code and all midpoint ties.
  codes=np.arange(256,dtype=np.uint8)
  values=torch.from_numpy(codes).view(torch.float8_e4m3fn).float().numpy()
  finite=np.isfinite(values)
  np.testing.assert_array_equal(fp8_nearest(values[finite]),codes[finite])
  positive=values[:127]; mid=np.float32((positive[:-1].astype(np.float64)+positive[1:])/2)
  np.testing.assert_array_equal(fp8_nearest(mid),torch.from_numpy(mid).to(torch.float8_e4m3fn).view(torch.uint8).numpy())
  labels=np.arange(16*2048,dtype=np.int32).reshape(16,2048)
  independent=np.empty_like(labels)
  for r in range(16):
    for c in range(2048): independent.flat[((r//16)*128+c//16)*256+(c%16//4)*64+(r%16)*4+c%4]=labels[r,c]
  np.testing.assert_array_equal(packed(labels),independent)
  return dict(residual_ties=True, frozen_reduction_tree=True, independent_scalar_reduction=True,
              intermediate_bf16_threshold_bits=[hex(int(staged[0])),hex(int(collapsed[0]))],
              all_finite_fp8_codes=True, fp8_midpoint_even_ties=True, packing_bijection=True,
              hardware_rsqrt_and_reciprocal_not_emulated=True)


def disassemble(binary):
  from tinygrad.runtime.support.compiler_amd import comgr,check,_get_comgr_data
  check(comgr.amd_comgr_create_action_info(ctypes.byref(info:=comgr.amd_comgr_action_info_t())))
  check(comgr.amd_comgr_action_info_set_isa_name(info,b'amdgcn-amd-amdhsa--gfx1200'))
  check(comgr.amd_comgr_create_data_set(ctypes.byref(src:=comgr.amd_comgr_data_set_t())))
  check(comgr.amd_comgr_create_data_set(ctypes.byref(dst:=comgr.amd_comgr_data_set_t())))
  check(comgr.amd_comgr_create_data(comgr.AMD_COMGR_DATA_KIND_EXECUTABLE,ctypes.byref(data:=comgr.amd_comgr_data_t())))
  try:
    check(comgr.amd_comgr_set_data(data,len(binary),binary))
    check(comgr.amd_comgr_set_data_name(data,b'agent-residual-norm.hsaco'))
    check(comgr.amd_comgr_data_set_add(src,data))
    check(comgr.amd_comgr_do_action(comgr.AMD_COMGR_ACTION_DISASSEMBLE_EXECUTABLE_TO_SOURCE,info,src,dst))
    return _get_comgr_data(dst,comgr.AMD_COMGR_DATA_KIND_SOURCE).decode()
  finally:
    check(comgr.amd_comgr_release_data(data));check(comgr.amd_comgr_destroy_data_set(src))
    check(comgr.amd_comgr_destroy_data_set(dst));check(comgr.amd_comgr_destroy_action_info(info))


def compile_checks(kernel, artifacts):
  from tinygrad.runtime.support.compiler_amd import HIPCompiler
  from tinygrad.runtime.support.elf import elf_loader
  def modes(binary):
    image,sections,_=elf_loader(binary)
    address=next(section.header.sh_addr for section in sections if section.name=='.rodata')
    rsrc1=struct.unpack_from('<I',image,address+48)[0]
    result=dict(fp32_round_mode=(rsrc1>>12)&3,fp32_denorm_mode=(rsrc1>>16)&3)
    assert result==dict(fp32_round_mode=0,fp32_denorm_mode=3)
    return result
  rows={}
  for name,text in (('baseline',kernel.frozen_norm_source()),('fused',kernel.source())):
    binary=HIPCompiler('gfx1200').compile(text)
    assembly=disassemble(binary)
    notes=subprocess.check_output(['/usr/lib/llvm-18/bin/llvm-readelf','--notes','-'],input=binary).decode()
    resources={key:int(value) for key,value in re.findall(r'\.(group_segment_fixed_size|private_segment_fixed_size|sgpr_count|sgpr_spill_count|vgpr_count|vgpr_spill_count):\s*(\d+)',notes)}
    assert resources and resources['private_segment_fixed_size']==resources['vgpr_spill_count']==resources['sgpr_spill_count']==0
    stem=artifacts/('agent-residual-norm-'+name)
    for suffix,value in (('.cc',text),('.asm',assembly),('.notes',notes)): stem.with_suffix(suffix).write_text(value)
    stem.with_suffix('.hsaco').write_bytes(binary)
    rows[name]=dict(source_sha256=kernel.sha(text.encode()),binary_sha256=kernel.sha(binary),resources=resources,float_modes=modes(binary),
      instructions={op:len(re.findall(r'\b'+op+r'(?:_e32|_e64)?\b',assembly)) for op in
                    ('v_fma_f32','v_rsq_f32','v_rcp_f32','v_cvt_pk_fp8_f32','s_barrier_signal','s_barrier_wait')})
    assert rows[name]['instructions']['v_fma_f32']==0
    assert not re.search(r'\bv_(?:dual_)?fma\w*',assembly)
    assert rows[name]['instructions']['v_rsq_f32']>=1
  assert rows['fused']['instructions']['v_rcp_f32']>=1 and rows['fused']['instructions']['v_cvt_pk_fp8_f32']>=1
  # Generate the actual frozen packed quantizer's full M2048 graph offline.
  from unittest.mock import patch
  from tinygrad import Tensor,dtypes
  from tinygrad.helpers import Target
  from tinygrad.renderer.llvmir import AMDLLVMRenderer
  from tinygrad.runtime.support.compiler_llvm import LLVMCompiler
  from tinygrad.codegen import to_program
  quantizer=load(kernel.FROZEN/'quantize-packed.py')
  x=Tensor.empty(1,2048,2048,dtype=dtypes.bfloat16,device='NULL')
  scale=Tensor.empty((),dtype=dtypes.float32,device='NULL')
  with patch.object(Tensor,'realize',lambda self,*args:self): value,_=quantizer.quantize(x,scale)
  linear=value.schedule_linear()
  assert len(linear.src)==1
  renderer=AMDLLVMRenderer(Target('AMD','LLVM','gfx1200'))
  renderer.compiler=LLVMCompiler('AMDGPU','gfx1200','-cumode')
  program=to_program(linear.src[0].body,renderer)
  llvm,binary=program.src[2].arg,program.src[3].arg
  assembly=disassemble(binary)
  assert 'fdiv contract afn' in llvm and 'v_s_rcp_f32' in assembly
  assert not any(op in assembly for op in ('v_div_scale','v_div_fmas','v_div_fixup'))
  (artifacts/'agent-residual-norm-original-pack.ll').write_text(llvm)
  (artifacts/'agent-residual-norm-original-pack.asm').write_text(assembly)
  (artifacts/'agent-residual-norm-original-pack.hsaco').write_bytes(binary)
  rows['actual_frozen_pack']=dict(source_sha256=kernel.sha(llvm.encode()),binary_sha256=kernel.sha(binary),
    float_modes=modes(binary),
    association='One scalar reciprocal and rounded FP32 multiply; candidate uses the vector reciprocal instruction',
    scalar_vector_reciprocal_bit_equivalence_requires_gpu=True)
  return rows


def main():
  parser=argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--artifacts',type=Path,default=HERE/'agent-residual-norm-cpu-artifacts')
  args=parser.parse_args();args.artifacts.mkdir(parents=True,exist_ok=True)
  os.environ.update(DEV='NULL',PROFILE='0',CACHEDB=str(args.artifacts/'agent-residual-norm-cache.db'))
  sys.path[:0]=[str(TINY),str(HERE.parent)]
  kernel=load(HERE/'agent-residual-norm-kernel.py')
  result=dict(cpu=cpu_checks(kernel),compilation=compile_checks(kernel,args.artifacts),gpu_executed=False)
  from tinygrad import Device
  assert not any('AMD' in name or 'USB' in name for name in Device._opened_devices)
  (args.artifacts/'agent-residual-norm-report.json').write_text(json.dumps(result,indent=2)+'\n')
  print(json.dumps(result,indent=2))


if __name__=='__main__': main()
