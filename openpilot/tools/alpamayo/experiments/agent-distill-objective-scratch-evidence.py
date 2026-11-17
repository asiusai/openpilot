"""Read the completed compiler cache and reproduce its largest private segment on CPU only."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import pickle
import re
import sqlite3
import struct
import sys
from types import SimpleNamespace

sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
WORK=Path('/mnt/nvme-500/tmp/alpamayo-speed-20261116')
ROOT=WORK/'agent-distill-objective-scratch-artifacts'
FAILED=WORK/'agent-distill-objective-qualify-artifacts/live-v1'
BASE=Path('/tmp/opencode/alpamayo-asius-20261114')


def sha(path):
  with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
def no_gpu(event,args):
  if event=='open' and isinstance(args[0],(str,bytes)) and os.fsdecode(args[0]).startswith(('/dev/dri','/dev/kfd','/dev/nvidia','/dev/bus/usb')):
    raise RuntimeError('Scratch evidence is compiler-only')
sys.addaudithook(no_gpu)


def note_integers(sections):
  """Read only unique integer metadata fields from this LLVM AMDGPU MessagePack note."""
  result={}
  for section in sections:
    if section.name!='.note':continue
    offset=0
    while offset<len(section.content):
      namesz,descsz,kind=struct.unpack_from('<III',section.content,offset);offset+=12
      name=section.content[offset:offset+namesz];offset+=(namesz+3)//4*4
      desc=section.content[offset:offset+descsz];offset+=(descsz+3)//4*4
      if kind!=32 or name!=b'AMDGPU\0':continue
      for field in ('private_segment_fixed_size','sgpr_count','vgpr_count','sgpr_spill_count','vgpr_spill_count','wavefront_size'):
        key=('.'+field).encode();assert desc.count(key)==1
        index=desc.index(key)+len(key);tag=desc[index]
        if tag<=0x7f:value=tag
        elif tag in (0xcc,0xcd,0xce,0xcf):value=struct.unpack_from({0xcc:'>B',0xcd:'>H',0xce:'>I',0xcf:'>Q'}[tag],desc,index+1)[0]
        else:raise ValueError('Unexpected integer metadata encoding')
        result[field]=value
  assert len(result)==6
  return result


def main():
  parser=argparse.ArgumentParser();parser.add_argument('--output-dir',type=Path,required=True);args=parser.parse_args()
  root=args.output_dir.resolve()
  if root.exists() or not root.is_relative_to(ROOT):raise ValueError('Fresh owned compiler output required')
  root.mkdir(parents=True)
  os.environ.update(DEV='CPU',TC='0',ALLOW_TF32='0',PYTHONDONTWRITEBYTECODE='1')
  sys.path.insert(0,str(BASE/'tinygrad'))
  from tinygrad import Device
  from tinygrad.runtime.ops_amd import _amd_program_image
  from tinygrad.runtime.support.elf import elf_loader
  from tinygrad.runtime.support.compiler_llvm import AMDLLVMCompiler
  db=sqlite3.connect(f'file:{FAILED}/compiler-cache.db?mode=ro',uri=True)
  dev=SimpleNamespace(iface=SimpleNamespace(props={'lds_size_in_kb':64}),target=(11,0,0))
  rows=[];maximum=None
  for source,stored in db.execute('SELECT key,val FROM "compile_llvm_gfx1100_+cumode_opt_24"'):
    binary=pickle.loads(stored);data,_=_amd_program_image(dev,binary)
    row=dict(name=re.search(r'define.*?@([^ (]+)',source).group(1),private_bytes=data.private_segment_size,
      group_bytes=data.group_segment_size,source_sha256=hashlib.sha256(source.encode()).hexdigest(),
      binary_sha256=hashlib.sha256(binary).hexdigest())
    rows.append(row)
    if maximum is None or data.private_segment_size>maximum[0]['private_bytes']:maximum=(row,source,binary,data)
  db.close();row,source,binary,data=maximum
  (root/'largest-private-kernel.ll').write_text(source);(root/'largest-private-kernel.o').write_bytes(binary)
  recompiled=AMDLLVMCompiler('gfx1100').compile(source)
  assert recompiled==binary,'Pinned compiler did not reproduce the cached binary'
  _,sections,_=elf_loader(binary);metadata=note_integers(sections)
  assert metadata['private_segment_fixed_size']==row['private_bytes']==1424
  props_path=Path('/sys/devices/virtual/kfd/kfd/topology/nodes/2/properties')
  props={line.split()[0]:int(line.split()[1]) for line in props_path.read_text().splitlines()}
  geometry={k:props.get(k,1) for k in ('simd_count','simd_per_cu','max_slots_scratch_cu','num_xcc','gfx_target_version')}
  assert geometry==dict(simd_count=192,simd_per_cu=2,max_slots_scratch_cu=32,num_xcc=1,gfx_target_version=110000)
  size=((max(128,row['private_bytes'])+3)//4*4)*64*geometry['max_slots_scratch_cu']*(geometry['simd_count']//geometry['simd_per_cu'])*geometry['num_xcc']
  assert size==279969792
  original=json.loads((FAILED/'report.json').read_text());wall=json.loads((FAILED/'wall.json').read_text())
  assert original['failure']==dict(type='RuntimeError',message='buffer of size 279.97MB is too large') and not wall['timed_out']
  assert json.loads((FAILED/'journal.json').read_text())==dict(committed=0,intended=None)
  result=dict(complete=True,gpu_executed=False,kernels_executed=0,optimizer_updates=0,
    failed_report_sha256=sha(FAILED/'report.json'),failed_cache_sha256=sha(FAILED/'compiler-cache.db'),
    source_sha256={str(Path(__file__)):sha(__file__)},compiler_cache_rows=len(rows),
    kernels_with_private_segments=sorted((r for r in rows if r['private_bytes']),key=lambda r:r['private_bytes'],reverse=True),
    maximum_kernel=dict(row,metadata=metadata,llvm_register_array_count=source.count('alloca ['),
      llvm_register_arrays=re.findall(r'alloca \[(\d+) x float\]',source),argument_count=len(re.findall(r'float\* noalias',source)),
      source_bytes=len(source.encode()),cached_binary_bytes=len(binary),recompiled_binary_exact=True),
    geometry=geometry,geometry_source=str(props_path),geometry_is_availability_review=False,
    scratch_formula='round_up(max(private_bytes,128),4) * 64 * max_slots_scratch_cu * cu_count * xccs',
    requested_scratch_bytes=size,old_cap=256*1024**2,new_cap=320*1024**2,
    over_old_cap_bytes=size-256*1024**2,new_cap_headroom_bytes=320*1024**2-size,
    failed_run_actual_allocation=original['allocation_budget'],failed_wall=wall,
    limits_unchanged=dict(live=12*1024**3,cumulative=256*1024**3,free_reserve=5*1024**3,timeout_seconds=900),
    graph_or_compiler_change=False,opened_devices=list(Device._opened_devices))
  assert not Device._opened_devices
  (root/'report.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
