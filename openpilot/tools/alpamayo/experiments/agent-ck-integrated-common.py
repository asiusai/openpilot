"""Private paths and a single module owner for the optional CK screen."""
from functools import lru_cache
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess

EXP=Path(__file__).parent
WORK=Path('/mnt/nvme-500/tmp/alpamayo-speed-20261116')
ROOT=WORK/'agent-ck-integrated-work'
BASE=WORK/'padding-source/alpamayo'
TINY=Path('/tmp/opencode/alpamayo-asius-20261114/tinygrad')
PREPARED=ROOT/'prepared'
INDICES=[0,8,16,24,32,40,48,56]
LIVE=WORK/'agent-ck-lifetime-parent/report.json'
LIVE_SHA='da3472003336419c4f76145b662f4d90b2540635a83b215e8f139f252b475499'
MICRO=WORK/'agent-ck-patch-parent/report.json'
MICRO_SHA='5be762ce7d963e54e1e87dd82d88c1ff4cda6797996089d73990f5bed80840ff'

def sha(path):
  h=hashlib.sha256()
  with Path(path).open('rb') as f:
    while block:=f.read(8*1024*1024): h.update(block)
  return h.hexdigest()

@lru_cache(None)
def load(name,directory=EXP):
  spec=importlib.util.spec_from_file_location(name.replace('-','_'),Path(directory)/(name+'.py'))
  module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
  return module

def write(path,value):
  with Path(path).open('x') as f: f.write(json.dumps(value,indent=2,allow_nan=False)+'\n')

def verify():
  p=json.loads((PREPARED/'protocol.json').read_text())
  assert p['scope']=='opt_in_all28_fresh_paired_screen8' and p['adoption_allowed'] is False
  assert p['indices']==INDICES and p['timeout_seconds_per_arm']==600
  assert subprocess.check_output(['git','-C',str(TINY),'rev-parse','HEAD'],text=True).strip()==p['tinygrad_commit']
  for file,digest in p['pins'].items(): assert sha(file)==digest,('Changed pinned input',file)
  assert sha(LIVE)==LIVE_SHA and sha(MICRO)==MICRO_SHA
  assert json.loads(LIVE.read_text())['live_first_layer_micro_passed']
  assert json.loads(MICRO.read_text())['numerical_micro_gates_passed']
  for record in json.loads((PREPARED/'inputs.json').read_text()):
    for path,digest in record['files'].items(): assert sha(path)==digest,('Changed sample',path)
  return p

def storage_gate(rows,sizes):
  spans=[]
  for row,size in zip(rows,sizes,strict=True):
    assert row['bytes']==size and 0<=row['offset_bytes']<=row['base_bytes']-size
    assert row['gpu_address']==row['base_gpu_address']+row['offset_bytes'] and row['gpu_address']%16==0
    a,b=row['gpu_address'],row['gpu_address']+size
    assert all(b<=c or a>=d for c,d in spans),'Aliased native arguments'
    spans.append((a,b))
  return True
