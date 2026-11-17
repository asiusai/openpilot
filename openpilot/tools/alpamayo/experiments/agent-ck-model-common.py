"""Pinned paths and CPU-only source verification for the live-layout prerequisite."""
import hashlib
import importlib.util
import json
from pathlib import Path

EXP = Path(__file__).parent
WORK = Path('/mnt/nvme-500/tmp/alpamayo-speed-20261116')
ROOT = WORK/'agent-ck-model-work'
BASE = WORK/'padding-source/alpamayo'
TINY = Path('/tmp/opencode/alpamayo-asius-20261114/tinygrad')
CACHE = WORK/'agent-expert-combined-cache-artifacts/fp8-cache'
CHECKPOINT = Path('/tmp/opencode/alpamayo-asius-20261114/model')
OLD = WORK/'agent-ck-patch-work/prepared-final/protocol.json'
MICRO = WORK/'agent-ck-patch-parent/report.json'
PREPARED = ROOT/'prepared-final'
PROTOCOL = PREPARED/'protocol.json'
INDICES = [0,8,16,24,32,40,48,56]

def sha(path):
  h = hashlib.sha256()
  with Path(path).open('rb') as stream:
    while block := stream.read(8*1024*1024): h.update(block)
  return h.hexdigest()

def load(name, directory=EXP):
  spec = importlib.util.spec_from_file_location(name.replace('-','_'), directory/(name+'.py'))
  value = importlib.util.module_from_spec(spec); spec.loader.exec_module(value)
  return value

def write_json(path, value):
  with Path(path).open('x') as stream: stream.write(json.dumps(value,indent=2,allow_nan=False)+'\n')

def storage_gate(rows, expected_bytes):
  assert len(rows)==len(expected_bytes)
  spans=[]
  for row,size in zip(rows,expected_bytes,strict=True):
    assert row['bytes']==size and row['offset']>=0 and row['offset']+size<=row['base_bytes']
    assert row['gpu_address']==row['base_gpu_address']+row['offset'] and row['gpu_address']%16==0
    span=(row['gpu_address'],row['gpu_address']+size)
    assert all(span[1]<=a or span[0]>=b for a,b in spans), 'Aliased kernel inputs'
    spans.append(span)
  return True

def verify():
  plan = json.loads(PROTOCOL.read_text())
  assert plan['scope'] == 'first_actual_language_layer_layout_and_paired_micro_only'
  assert plan['timeout_seconds'] == 180 and plan['adoption_allowed'] is False
  for path, digest in plan['source_sha256'].items():
    assert sha(path) == digest, f'Changed source/evidence: {path}'
  assert sha(MICRO) == '5be762ce7d963e54e1e87dd82d88c1ff4cda6797996089d73990f5bed80840ff'
  micro = json.loads(MICRO.read_text())
  for key in ('completed','gpu_executed','numerical_micro_gates_passed','validator_qualified',
              'causal_gqa_mutations_passed','speed_screen_passed'): assert micro[key] is True
  assert micro['protocol_sha256'] == sha(OLD)
  assert plan['limits'] == json.loads(OLD.read_text())['limits']
  return plan
