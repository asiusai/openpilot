"""Compile new scaled INT4 kernels and independent FP8 controls, CPU only."""
import argparse
from collections import Counter
import ctypes
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import subprocess


def local(name):
  spec=importlib.util.spec_from_file_location(name.replace('-','_'),Path(__file__).with_name(name+'.py'))
  module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def descriptor(binary):
  from tinygrad.runtime.support.elf import elf_loader
  from tinygrad.runtime.autogen import amdgpu_kd
  image,sections,_=elf_loader(binary)
  start=next(s.header.sh_addr for s in sections if s.name=='.rodata')
  desc=amdgpu_kd.llvm_amdhsa_kernel_descriptor_t.from_buffer_copy(bytes(image[start:start+ctypes.sizeof(amdgpu_kd.llvm_amdhsa_kernel_descriptor_t)]))
  return dict(float_mode=(desc.compute_pgm_rsrc1>>12)&255,fp32_denorm_mode=(desc.compute_pgm_rsrc1>>16)&3,
              fp32_rounding_mode=(desc.compute_pgm_rsrc1>>12)&3,kernarg_bytes=desc.kernarg_size)


def fp8_program(check,m,n,k,bias=False,raw=False):
  from tinygrad import dtypes
  from tinygrad.codegen import to_program
  from tinygrad.helpers import Target
  from tinygrad.renderer.llvmir import AMDLLVMRenderer
  from tinygrad.runtime.support.compiler_llvm import LLVMCompiler
  from tinygrad.uop.ops import UOp
  renderer=AMDLLVMRenderer(Target('AMD','LLVM','gfx1200'));renderer.compiler=LLVMCompiler('AMDGPU','gfx1200','-cumode')
  shapes=[((m,n),dtypes.float32 if raw else dtypes.bfloat16),((((m+15)//16)*16,k),dtypes.fp8e4m3),
          ((((n+15)//16)*16,k),dtypes.fp8e4m3),((m,),dtypes.float32),((n,),dtypes.float32)]
  if bias and not raw:shapes.append(((n,),dtypes.bfloat16))
  args=[UOp.placeholder(shape,dt,i) for i,(shape,dt) in enumerate(shapes)]
  return to_program(check.reference(*args,raw=raw,renderer=renderer),renderer)


def main():
  p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
  args.output.mkdir(parents=True,exist_ok=False)
  kernel,check,old=local('agent-int4-scaled-kernel'),local('agent-int4-scaled-check'),local('agent-int4k32-compile')
  from tinygrad.uop.ops import Ops
  records=[]
  for m,n,k in ((17,48,32),(2112,2048,9216)):
    modes=[f'{kind}-{schedule}' for schedule in ('cu','wgp') for kind in ('quantize','integer','no-bias','bias')]
    for mode in (*modes,'fp8-raw','fp8-no-bias','fp8-bias'):
      info=None
      if mode.startswith('fp8'):
        program=fp8_program(check,m,n,k,bias=mode=='fp8-bias',raw=mode=='fp8-raw')
        binary=next(x.arg for x in program.src if x.op is Ops.BINARY)
        text=next(x.arg for x in program.src if x.op is Ops.SOURCE)
        info=dict(outs=list(program.arg.outs),ins=list(program.arg.ins),globals=list(program.arg.globals))
        assert program.arg.outs==(0,),info
      else:
        kind,schedule=mode.rsplit('-',1)
        if kind=='quantize':text,binary=kernel.compiled_quantize(m,k,wgp=schedule=='wgp')
        else:text,binary=kernel.compiled_gemm(m,n,k,bias=kind=='bias',integer=kind=='integer',wgp=schedule=='wgp')
      asm=old.disassemble(binary);counts=Counter(re.findall(r'^\s*([a-z][a-z0-9_]+)\s',asm,re.M))
      if not mode.startswith('fp8') and kind in ('integer','no-bias','bias'):
        assert counts['v_wmma_i32_16x16x32_iu4']>0 and counts['v_wmma_i32_16x16x16_iu4']==0
        wmma_lines=[line for line in asm.splitlines() if 'v_wmma_i32_16x16x32_iu4 ' in line]
        assert all('neg_lo:[0,1,0]' in line for line in wmma_lines),wmma_lines
      if mode=='fp8-bias' or (not mode.startswith('fp8') and kind=='bias'):
        assert any('fma' in name for name in counts),'Expected explicit scale+bias contraction'
      desc=descriptor(binary)
      resources=old.metadata(binary)
      if not mode.startswith('fp8'):
        assert desc['fp32_denorm_mode']==3 and desc['fp32_rounding_mode']==0,desc
        assert resources['workgroup_processor_mode']==int(schedule=='wgp'),resources
        assert resources['wavefront_size']==32
      stem=f'{m}-{n}-{k}-{mode}'
      (args.output/(stem+'.source')).write_text(text);(args.output/(stem+'.s')).write_text(asm)
      (args.output/(stem+'.hsaco')).write_bytes(binary)
      record=dict(shape=[m,n,k],mode=mode,binary_bytes=len(binary),resources=resources,descriptor=desc,metadata=info,
                  instructions={key:value for key,value in counts.items() if any(t in key for t in
                    ('wmma','global_load','global_store','fma','mul_f32','div','rcp','rndne','bpermute'))})
      records.append(record);print(json.dumps(record),flush=True)
  files=[Path(__file__),Path(__file__).with_name('agent-int4-scaled-kernel.py'),Path(__file__).with_name('agent-int4-scaled-check.py'),
         Path(__file__).with_name('agent-int4k32-kernel.py'),Path(__file__).with_name('agent-int4k32-compile.py')]
  revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd='/tmp/opencode/alpamayo-asius-20261114/tinygrad',text=True).strip()
  report=dict(cpu_only=True,tinygrad_revision=revision,source_hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},records=records)
  (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
