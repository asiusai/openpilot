"""CPU-only reconstruction of all ten actual GPU keys and their binary difference.

This records a diagnosis, never a qualification. Old reports and sources stay intact.
No pickle loading, device opening, GPU execution, or runtime patch survives the test.
"""
import argparse
import ctypes
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import struct
import sys
from unittest.mock import patch

os.environ['DEV']='AMD'
sys.dont_write_bytecode=True
from tinygrad import Device, dtypes
from tinygrad.codegen import do_to_program, to_program_cache, to_program_key
from tinygrad.helpers import Target
from tinygrad.renderer.llvmir import AMDLLVMRenderer
from tinygrad.runtime.autogen import amdgpu_kd as kd
from tinygrad.runtime.support.elf import elf_loader
from tinygrad.uop.ops import Ops

ROOT=Path(__file__).resolve().parent
WORK=Path('/mnt/nvme-500/tmp/alpamayo-speed-20261116')


def local(path):
  spec=importlib.util.spec_from_file_location(Path(path).stem.replace('-','_'),path)
  value=importlib.util.module_from_spec(spec);sys.modules[spec.name]=value;spec.loader.exec_module(value)
  return value


def sha(value): return hashlib.sha256(value).hexdigest()


def renderer():
  wgp=local(ROOT/'wgp_compiler.py')
  result=AMDLLVMRenderer(Target('AMD','LLVM','gfx1200','USB'))
  result.compiler=wgp.WGPCompiler('gfx1200')
  return result


def info(program):
  return dict(key=program.key.hex(),arg=repr(program.arg),parts=[dict(op=u.op.name,key=u.key.hex(),
    arg=repr(u.arg) if u.op not in (Ops.SOURCE,Ops.BINARY) else None,
    data_sha256=sha(u.arg.encode() if isinstance(u.arg,str) else u.arg) if u.op in (Ops.SOURCE,Ops.BINARY) else None)
    for u in program.src])


def leaves(a,b,path=''):
  if type(a)!=type(b): return [(path,a,b)]
  if isinstance(a,dict):
    if a.keys()!=b.keys(): return [(path+'.keys',list(a),list(b))]
    return [v for k in a for v in leaves(a[k],b[k],path+'.'+str(k))]
  if isinstance(a,list):
    if len(a)!=len(b): return [(path+'.length',len(a),len(b))]
    return [v for i,(x,y) in enumerate(zip(a,b)) for v in leaves(x,y,path+'.'+str(i))]
  return [] if a==b else [(path,a,b)]


def signature_digest(value):
  payload={k:value[k] for k in ('kernels','input_names','input_contract')}
  return sha(json.dumps(payload,sort_keys=True).encode())


def binary_proof(a,b):
  if len(a)!=len(b): raise AssertionError('Binary size differs')
  image_a,sections_a,relocs_a=elf_loader(a)
  image_b,sections_b,relocs_b=elf_loader(b)
  assert relocs_a==relocs_b
  assert [s.name for s in sections_a]==[s.name for s in sections_b]
  changes=[]
  for x,y in zip(sections_a,sections_b):
    assert bytes(x.header)==bytes(y.header)
    if x.content==y.content: continue
    offsets=[i for i,(v,w) in enumerate(zip(x.content,y.content)) if v!=w]
    assert len(x.content)==len(y.content) and len(offsets)==1
    if x.name=='.rodata':
      assert len(x.content)==ctypes.sizeof(kd.llvm_amdhsa_kernel_descriptor_t)==64
      dx=kd.llvm_amdhsa_kernel_descriptor_t.from_buffer_copy(x.content)
      dy=kd.llvm_amdhsa_kernel_descriptor_t.from_buffer_copy(y.content)
      assert dx.compute_pgm_rsrc1 ^ dy.compute_pgm_rsrc1 == kd.COMPUTE_PGM_RSRC1_GFX10_PLUS_WGP_MODE
      assert dx.compute_pgm_rsrc1 & kd.COMPUTE_PGM_RSRC1_GFX10_PLUS_WGP_MODE
      assert not dy.compute_pgm_rsrc1 & kd.COMPUTE_PGM_RSRC1_GFX10_PLUS_WGP_MODE
      label='COMPUTE_PGM_RSRC1_GFX10_PLUS_WGP_MODE: 1 -> 0'
    elif x.name=='.note':
      marker=b'\xb9.workgroup_processor_mode'
      assert x.content.count(marker)==y.content.count(marker)==1
      assert offsets==[x.content.index(marker)+len(marker)]
      assert x.content[offsets[0]]==1 and y.content[offsets[0]]==0
      label='AMDGPU metadata .workgroup_processor_mode: 1 -> 0'
    else: raise AssertionError('Unreviewed ELF difference: '+x.name)
    changes.append(dict(section=x.name,section_offset=offsets[0],file_offset=x.header.sh_offset+offsets[0],meaning=label))
  assert {c['section'] for c in changes}=={'.rodata','.note'}
  # Every byte outside those exact fields must match; this is diagnostic, never a new cache key.
  diffs=[i for i,(x,y) in enumerate(zip(a,b)) if x!=y]
  assert diffs==sorted(c['file_offset'] for c in changes)
  text=next(s.content for s in sections_a if s.name=='.text')
  return dict(changes=changes,binary_bytes=len(a),instruction_sha256=sha(text),relocations=relocs_a,
              loaded_image_byte_differences=sum(x!=y for x,y in zip(image_a,image_b)),
              equal_source_and_instructions=True,equal_relocations=True,scheduling_mode_equal=False)


def symbolic_contract(program,count):
  """Interpret pointer and vector instructions as symbols, independently of LLVM."""
  env={};written={};loads=[]
  for u in program.src[1].src:
    src=[env[s] for s in u.src]
    if u.op is Ops.CONST: result=u.arg
    elif u.op is Ops.PARAM:
      assert u.dtype==dtypes.float32 and u.arg.slot in range(count+1)
      assert src==[count if u.arg.slot==0 else 1]
      result=('pointer',u.arg.slot,0,src[0])
    elif u.op is Ops.CAST:
      assert u.dtype==dtypes.int32 and isinstance(src[0],int)
      result=src[0]
    elif u.op is Ops.INDEX:
      pointer,index=src
      assert pointer[0]=='pointer' and 0<=index<pointer[3]
      result=('pointer',pointer[1],pointer[2]+index,1)
    elif u.op is Ops.SHRINK:
      pointer,start,size=src
      assert pointer[0]=='pointer' and start>=0 and size>0 and start+size<=pointer[3]
      result=('pointer',pointer[1],pointer[2]+start,size)
    elif u.op is Ops.LOAD:
      pointer=src[0]
      assert pointer[1] in range(1,count+1) and pointer[2:]==(0,1)
      loads.append(pointer[1]);result=('input_bits',pointer[1])
    elif u.op is Ops.STACK: result=('vector',*src)
    elif u.op is Ops.STORE:
      pointer,value=src
      assert value[0] in ('vector','input_bits')
      values=value[1:] if value[0]=='vector' else (value,)
      assert pointer[1]==0 and pointer[3]==len(values)
      for i,item in enumerate(values,pointer[2]):
        assert i not in written
        written[i]=item
      result=None
    elif u.op is Ops.SINK: result=None
    else: raise AssertionError('Unexpected scalar concatenation instruction '+u.op.name)
    env[u]=result
  assert written=={i:('input_bits',i+1) for i in range(count)}
  assert sorted(loads)==list(range(1,count+1))
  assert program.arg.global_size==program.arg.local_size==(1,1,1)
  assert program.arg.outs==(0,) and program.arg.ins==tuple(range(1,count+1))
  # Float contents are opaque bit payloads. Include zeros, infinities, NaNs and changed token values.
  cases=[0,0x80000000,0x7f800000,0xff800000,0x7fc12345,0x00000001,0x3f800000,0x49800000]
  for state in range(len(cases)):
    inputs={slot:struct.pack('<I',cases[(slot+state)%len(cases)]) for slot in range(1,count+1)}
    actual=b''.join(inputs[written[i][1]] for i in range(count))
    assert actual==b''.join(inputs[i] for i in range(1,count+1))
  return dict(output_to_input_slots=[written[i][1] for i in range(count)],each_input_loaded_once=True,
              each_output_written_once=True,out_of_bounds_accesses=0,arithmetic_operations=0,bit_payload_controls=len(cases))


def dump_program(directory,count,label,program):
  prefix=directory/f'{count}-{label}'
  prefix.with_suffix('.ll').write_text(program.src[2].arg)
  prefix.with_suffix('.elf').write_bytes(program.src[3].arg)
  nodes=list(program.src[1].src);indices={u:i for i,u in enumerate(nodes)}
  value=dict(program=info(program),sink=repr(program.src[0]),linear=[dict(index=i,op=u.op.name,
    dtype=str(u.dtype),shape=repr(u._shape),arg=repr(u.arg),sources=[indices[s] for s in u.src]) for i,u in enumerate(nodes)])
  prefix.with_suffix('.json').write_text(json.dumps(value,indent=2)+'\n')


def main():
  parser=argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--ordinary-report',type=Path,default=WORK/'agent-startup-cache-trial1-ordinary/report.json')
  parser.add_argument('--candidate-report',type=Path,default=WORK/'agent-startup-cache-trial1-candidate/report.json')
  parser.add_argument('--output',type=Path,required=True)
  parser.add_argument('--artifacts',type=Path,required=True)
  args=parser.parse_args()
  if args.output.exists(): raise FileExistsError(args.output)
  args.artifacts.mkdir()
  helper=local(ROOT/'agent-startup-key-prime.py')
  ordinary=json.loads(args.ordinary_report.read_text())['startup_cache']
  candidate=json.loads(args.candidate_report.read_text())['startup_cache']
  before,after=ordinary['serial_signatures'],candidate['serial_signatures']
  differences=leaves(before,after)
  expected_paths={f'.{n}.digest' for n in range(2,7)}|{f'.{n}.kernels.{764*n}.program_key' for n in range(2,7)}
  assert {d[0] for d in differences}==expected_paths and len(differences)==10
  for n in range(1,7):
    for signatures in (before,after):
      assert signatures[str(n)]['digest']==signature_digest(signatures[str(n)])
      assert len(signatures[str(n)]['kernels'])==764*n+1
  rows=[]
  def forbidden(*args,**kwargs): raise AssertionError('CPU proof cannot open any device')
  with patch.object(type(Device),'__getitem__',forbidden):
    ren=renderer()
    constructor,arguments=ren.__reduce__()
    reconstructed=constructor(*arguments)  # Exact Renderer serialization protocol, no pickle needed.
    assert type(reconstructed.compiler).__name__=='AMDLLVMCompiler'
    for count in range(2,7):
      ast=helper.concat_ast(count)
      wgp=do_to_program(ast,ren);cu=do_to_program(ast,reconstructed)
      assert wgp.key.hex()==before[str(count)]['kernels'][-1]['program_key']==helper.ORDINARY[count]
      assert cu.key.hex()==after[str(count)]['kernels'][-1]['program_key']==helper.CANDIDATE[count]
      assert wgp.arg==cu.arg and wgp.src[:3]==cu.src[:3]
      contract=symbolic_contract(wgp,count)
      assert contract==symbolic_contract(cu,count)
      proof=binary_proof(wgp.src[3].arg,cu.src[3].arg)
      bad=bytearray(cu.src[3].arg);bad[256]^=1
      try: binary_proof(wgp.src[3].arg,bytes(bad))
      except AssertionError: pass
      else: raise AssertionError('Changed instruction bytes were accepted')
      for label,program in (('ordinary-wgp',wgp),('candidate-cu',cu)): dump_program(args.artifacts,count,label,program)
      rows.append(dict(count=count,kernel_index=764*count,ordinary=info(wgp),candidate=info(cu),
                       binary_proof=proof,symbolic_contract=contract,instruction_corruption_rejected=True))
    # Prove the helper uses ordinary compilation, then hits the same actual cache entries.
    first=helper.prime(ren);second=helper.prime(ren)
    assert all(not row['cache_hit'] for row in first['rows']) and all(row['cache_hit'] for row in second['rows'])
    # Explicitly test the existing compiler-key collision. A pre-existing CU result must fail closed.
    ast=helper.concat_ast(2);key=to_program_key(ast,ren);saved=to_program_cache[key]
    to_program_cache[key]=do_to_program(ast,reconstructed)
    try:
      try: helper.prime(ren)
      except AssertionError: pass
      else: raise AssertionError('Existing wrong-mode cache entry was accepted')
    finally: to_program_cache[key]=saved
  assert not Device._opened_devices
  result=dict(cpu_only=True,gpu_executed=False,devices=[],qualification=False,
    old_strict_rejection_retained=True,exact_actual_program_key_reproductions=10,
    report_differences=differences,rows=rows,prime_first=first,prime_repeat=second,
    wrong_mode_cache_rejected=True,reports={str(p):sha(p.read_bytes()) for p in (args.ordinary_report,args.candidate_report)},
    sources={str(p):sha(p.read_bytes()) for p in (Path(__file__),ROOT/'agent-startup-key-prime.py',ROOT/'wgp_compiler.py')})
  with args.output.open('x') as stream: json.dump(result,stream,indent=2);stream.write('\n')
  print(json.dumps(dict(exact_actual_program_keys=10,changed_binary_bytes_per_kernel=2,
                       strict_gate_unchanged=True,cpu_proof=str(args.output),gpu_executed=False),indent=2))


if __name__=='__main__': main()
