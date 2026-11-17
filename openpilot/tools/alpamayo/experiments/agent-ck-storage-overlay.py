"""Narrow metadata-contract overlay. The original runner and numerical sources stay frozen."""
import ast
import copy
import hashlib
import importlib.util
import inspect
import json
from pathlib import Path
import runpy
from types import SimpleNamespace

EXP = Path(__file__).parent
WORK = Path('/mnt/nvme-500/tmp/alpamayo-speed-20261116')
ROOT = WORK/'agent-ck-storage-work'
PREPARED = ROOT/'prepared'
PROTOCOL = PREPARED/'protocol.json'
ORIGINAL_ROOT = WORK/'agent-ck-model-work'
ORIGINAL_RUNNER = EXP/'agent-ck-model-run.py'
ORIGINAL_PROTOCOL = ORIGINAL_ROOT/'prepared-final/protocol.json'
PINS = {
  ORIGINAL_RUNNER:'2e725bb47be492f414a4761605b032d127bfae17e529a02bf767fa7a79fe1b05',
  EXP/'agent-ck-model-common.py':'7c57b3b99e85591bee4fc8257755ee99781b1c51fed41324a1704a8bb8d57324',
  EXP/'agent-ck-output-common.py':'fa34d16edfc26d12ac005c0ef3f02651e345cb889b42c5019d85aebad2ca8493',
  ORIGINAL_PROTOCOL:'3f1ab36c32f0cca3ea9468e4745c5f2f258375061688e0df0023233f6e589f48',
  WORK/'agent-ck-model-parent/report.json':'74fd2ae084fb2e3d5c1179ce81a8b5c511a0aa0ebb0b953d05f4e4b8425030dc',
  ORIGINAL_ROOT/'parent-result.json':'eaabcd2b1a71e0bb7e96d5375fcc57d267fc24f44848a90aa8fdaab2b2810a6a',
  WORK/'agent-ck-model-parent/run.log':'3c51616d2bfbc20c05b54de10e200d57c598c725bf97d23e5b2047e3e08ea682',
}

def sha(path):
  h=hashlib.sha256()
  with Path(path).open('rb') as stream:
    while block:=stream.read(8*1024*1024): h.update(block)
  return h.hexdigest()

def load(name):
  spec=importlib.util.spec_from_file_location(name.replace('-','_'),EXP/(name+'.py'))
  value=importlib.util.module_from_spec(spec);spec.loader.exec_module(value);return value

def storage_gate(rows, expected_bytes):
  assert len(rows)==len(expected_bytes)
  spans=[]
  for row,size in zip(rows,expected_bytes,strict=True):
    assert row['bytes']==size and row['offset_bytes']>=0 and row['offset_bytes']+size<=row['base_bytes']
    assert row['gpu_address']==row['base_gpu_address']+row['offset_bytes'] and row['gpu_address']%16==0
    span=(row['gpu_address'],row['gpu_address']+size)
    assert all(span[1]<=a or span[0]>=b for a,b in spans), 'Aliased kernel inputs'
    spans.append(span)
  return True

def check_delta():
  old=load('agent-ck-model-common')
  before=ast.parse(inspect.getsource(old.storage_gate))
  changed=copy.deepcopy(before);count=0
  for node in ast.walk(changed):
    if isinstance(node,ast.Subscript) and isinstance(node.slice,ast.Constant) and node.slice.value=='offset':
      node.slice.value='offset_bytes';count+=1
  after=ast.parse(inspect.getsource(storage_gate))
  assert count==3 and ast.dump(changed)==ast.dump(after)
  return {'replacement':"row['offset'] -> row['offset_bytes']",'occurrences':count,'all_other_gate_AST_equal':True}

def derived_invocation():
  invocation=json.loads((ORIGINAL_ROOT/'prepared-final/invocation.json').read_text())
  command=invocation['command'];command[command.index('--output')+1]=str(ROOT/'benchmark-unused.json')
  invocation['environment']['CACHEDB']=str(ROOT/'cache.db')
  invocation['environment']['TMPDIR']=str(ROOT/'tmp')
  return invocation

def derived_plan():
  plan=json.loads(ORIGINAL_PROTOCOL.read_text())
  command=plan['launcher_command']
  command[command.index('--output')+1]=str(ROOT/'benchmark-unused.json')
  command[command.index('--invocation')+1]=str(PREPARED/'invocation.json')
  return plan

def verify():
  assert all(sha(path)==digest for path,digest in PINS.items()), 'Changed original source or failure evidence'
  plan=json.loads(PROTOCOL.read_text());expected=derived_plan()
  assert plan['storage_overlay']['gate_delta']==check_delta()
  for key,value in expected.items():
    if key=='source_sha256':
      assert all(plan[key].get(path)==digest for path,digest in value.items())
    else: assert plan[key]==value, ('Changed original protocol field',key)
  assert json.loads((PREPARED/'invocation.json').read_text())==derived_invocation()
  for name in ('catalog.json','producer-catalog.json'):
    assert (PREPARED/name).read_bytes()==(ORIGINAL_ROOT/'prepared-final'/name).read_bytes()
  for path,digest in plan['source_sha256'].items(): assert sha(path)==digest, f'Changed bound source: {path}'
  assert plan['timeout_seconds']==180 and plan['internal_timeout_seconds']==170
  assert plan['adoption_allowed'] is False
  return plan

def load_runner():
  """Reuse every original function/code object. Replace only its private common-module binding."""
  assert all(sha(path)==digest for path,digest in PINS.items())
  check_delta()
  namespace=runpy.run_path(str(ORIGINAL_RUNNER),run_name='agent_ck_storage_frozen_runner')
  globals_=namespace['run'].__globals__
  assert namespace['main'].__globals__ is globals_ and namespace['structural_graph'].__globals__ is globals_
  original=globals_['C']
  overlay=SimpleNamespace(**vars(original))
  overlay.ROOT,overlay.PREPARED,overlay.PROTOCOL=ROOT,PREPARED,PROTOCOL
  overlay.storage_gate,overlay.verify=storage_gate,verify
  globals_['C']=overlay
  assert all(getattr(overlay,name) is getattr(original,name) for name in vars(original)
             if name not in ('ROOT','PREPARED','PROTOCOL','storage_gate','verify'))
  return namespace
