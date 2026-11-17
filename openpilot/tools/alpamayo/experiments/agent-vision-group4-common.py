"""Bounded group4-vs2 experiment utilities. No device access on import."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

WORK=Path('/mnt/nvme-500/tmp/alpamayo-speed-20261116')
BASE=WORK/'agent-paced-dynamic-source-cpu02/alpamayo'
ROOT=WORK/'agent-vision-group4-protocol-v2'
SNAP=ROOT/'alpamayo'
EXP=SNAP/'experiments'
PLAN=ROOT/'agent-vision-group4-plan.json'
PY=Path('/tmp/opencode/alpamayo-asius-20261114/upstream/.venv/bin/python')
TINY=Path('/tmp/opencode/alpamayo-asius-20261114/tinygrad')
sys.dont_write_bytecode=True


def sha(path):
  with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
def read(path):return json.loads(Path(path).read_text())
def write(path,value):
  with Path(path).open('x') as stream:stream.write(json.dumps(value,indent=2,allow_nan=False)+'\n')
def module(path,name):
  s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m
def set_option(command,name,value):
  flag='--'+name
  if flag in command:command[command.index(flag)+1]=str(value)
  else:command.extend((flag,str(value)))
def require(value,message):
  if not value:raise ValueError(message)


def resources(binary):
  """Read integer AMD note fields; no MessagePack dependency or device access."""
  from tinygrad.runtime.support.elf import elf_loader
  _,sections,_=elf_loader(binary)
  note=next(section.content for section in sections if section.name=='.note')
  out={}
  for key in ('vgpr_count','sgpr_count','vgpr_spill_count','sgpr_spill_count','private_segment_fixed_size',
              'group_segment_fixed_size','wavefront_size','workgroup_processor_mode'):
    field=('.'+key).encode();pos=note.index(field)+len(field);tag=note[pos]
    if tag<128:out[key]=tag
    elif tag in (204,205,206,207):out[key]=int.from_bytes(note[pos+1:pos+1+(1 << (tag-204))],'big')
    else:raise ValueError((key,tag))
  return out
