"""Compose two pinned-API corrections without changing frozen plan sources."""
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
MASK_OLD="Tensor.arange(2112,device='AMD')"
MASK_NEW="Tensor.arange(2112).to(first.device)"
COPY_OLD='buffer.ensure_allocated().copyin(memoryview(bytes([pattern])*buffer.nbytes))'
COPY_NEW='plan_poison_copy(buffer,pattern)'
SOURCES=('agent-plan-poison-copy.py','agent-plan-mutation-v2-adapter.py','agent-plan-mutation-v2-check.py',
         'agent-plan-mutation-v2-run.py','agent-plan-mutation-v2-parent.py')

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def local(path):
  spec=importlib.util.spec_from_file_location('agent_plan_mutation_v2_'+Path(path).stem.replace('-','_'),path)
  module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
  return module

def derive(text):
  if sha(BASE)!=BASE_SHA: raise ValueError('Frozen protocol2 changed')
  expected=json.loads(BASE.read_text())['reload_sources']['agent-plan-reload-run.py']
  if hashlib.sha256(text.encode()).hexdigest()!=expected: raise ValueError('Frozen capture function source changed')
  changed=text
  for old,new in ((MASK_OLD,MASK_NEW),(COPY_OLD,COPY_NEW)):
    if changed.count(old)!=1 or new in changed: raise ValueError('Expected one exact API expression')
    changed=changed.replace(old,new)
  if changed.replace(COPY_NEW,COPY_OLD).replace(MASK_NEW,MASK_OLD)!=text:
    raise AssertionError('Changes extend beyond the two API expressions')
  before=next(n for n in ast.parse(text).body if isinstance(n,ast.FunctionDef) and n.name=='capture_one')
  after=next(n for n in ast.parse(changed).body if isinstance(n,ast.FunctionDef) and n.name=='capture_one')
  # Numerical loader/proof, all predicates, six cases, and finally remain the
  # original AST. The only edits above are the two source-bound expressions.
  return before,after,dict(base_protocol_sha256=BASE_SHA,frozen_source_sha256=expected,
    replacements=[dict(old=MASK_OLD,new=MASK_NEW),dict(old=COPY_OLD,new=COPY_NEW)],
    adapted_source_sha256=hashlib.sha256(changed.encode()).hexdigest(),
    capture_ast_sha256=hashlib.sha256(ast.dump(after,include_attributes=False).encode()).hexdigest())

def source(): return derive((EXP/'agent-plan-reload-run.py').read_text())

def verify_base():
  _,_,change=source()
  local(EXP/'agent-plan-reload-loader.py').source_gate(json.loads(BASE.read_text()))
  return change

@contextmanager
def adapted(module):
  _,capture,change=source();original=module.capture_one
  if 'plan_poison_copy' in module.__dict__: raise ValueError('Unexpected existing transport hook')
  try:
    module.plan_poison_copy=local(EXP/'agent-plan-poison-copy.py').poison_copy
    exec(compile(ast.Module(body=[capture],type_ignores=[]),str(EXP/'agent-plan-mutation-v2-adapter.py')+' [capture_one]','exec'),module.__dict__)
    yield change
  finally:
    module.capture_one=original
    module.__dict__.pop('plan_poison_copy',None)
