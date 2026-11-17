"""One source-bound mutation-mask correction; frozen reload helpers stay intact."""
import ast
from contextlib import contextmanager
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

EXP=Path(__file__).resolve().parent
WORK=Path('/mnt/nvme-500/tmp/alpamayo-speed-20261116')
BASE=WORK/'agent-plan-reload-protocol2.json'
BASE_SHA='831d39314038999032b080ca77ff4bb5cf0fc95ded5c34a4993f8ade3a567a58'
OLD="Tensor.arange(2112,device='AMD')"
NEW="Tensor.arange(2112).to(first.device)"

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def local(path):
  spec=importlib.util.spec_from_file_location('agent_plan_mutation_'+Path(path).stem.replace('-','_'),path)
  module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
  return module

def source():
  if sha(BASE)!=BASE_SHA: raise ValueError('Original protocol2 changed')
  protocol=json.loads(BASE.read_text());path=EXP/'agent-plan-reload-run.py'
  if sha(path)!=protocol['reload_sources'][path.name]: raise ValueError('Frozen reload source changed')
  text=path.read_text()
  if text.count(OLD)!=1 or NEW in text: raise ValueError('Expected exactly one original mask construction')
  changed=text.replace(OLD,NEW)
  before=ast.parse(text);after=ast.parse(changed)
  capture=next(n for n in after.body if isinstance(n,ast.FunctionDef) and n.name=='capture_one')
  original=next(n for n in before.body if isinstance(n,ast.FunctionDef) and n.name=='capture_one')
  # The entire source must return byte-for-byte to the frozen source by reversing this edit.
  if changed.replace(NEW,OLD)!=text: raise AssertionError('More than mask construction changed')
  return original,capture,dict(base_protocol_sha256=BASE_SHA,old=OLD,new=NEW,replacements=1,
    frozen_source_sha256=sha(path),adapted_source_sha256=hashlib.sha256(changed.encode()).hexdigest(),
    capture_ast_sha256=hashlib.sha256(ast.dump(capture,include_attributes=False).encode()).hexdigest())

def verify_base():
  _,_,record=source()
  local(EXP/'agent-plan-reload-loader.py').source_gate(json.loads(BASE.read_text()))
  return record

@contextmanager
def adapted(module):
  _,capture,record=source();original=module.capture_one
  try:
    exec(compile(ast.Module(body=[capture],type_ignores=[]),str(EXP/'agent-plan-mutation-adapter.py')+' [capture_one]','exec'),module.__dict__)
    yield record
  finally: module.capture_one=original
