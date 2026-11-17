"""Specialize the qualified count1 loader to sealed actual family contracts."""
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

EXP=Path(__file__).resolve().parent

def local(path):
  spec=importlib.util.spec_from_file_location('family_loader_'+Path(path).stem.replace('-','_'),path)
  module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
  return module

contract=local(EXP/'agent-plan-family-reload-contract.py')
base=local(EXP/'agent-plan-reload-loader.py')
family=contract.family()

def validate(document,programs,proof,count,protocol):
  expected,_,sealed,views=contract.read_member(count,protocol)
  if document!=expected or proof!=sealed: raise ValueError('Only the sealed actual member and proof are admitted')
  if set(programs)!=set(document['programs']) or any(p.key.hex()!=key for key,p in programs.items()):
    raise ValueError('Original numerical PROGRAM identity changed')
  return views

def constructor():
  """Only replace count1-specific guards, output layout, arena extent, and bounds."""
  path=EXP/'agent-plan-reload-loader.py';text=path.read_text()
  frozen=json.loads((contract.WORK/'agent-plan-reload-protocol2.json').read_text())['reload_sources'][path.name]
  if hashlib.sha256(text.encode()).hexdigest()!=frozen: raise ValueError('Qualified constructor source changed')
  cls=next(n for n in ast.parse(text).body if isinstance(n,ast.ClassDef) and n.name=='Prepared')
  method=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='__init__')
  source=ast.get_source_segment(text,method)
  replacements=[
    ('*,bulk_index,vocab,checkpoint):','*,bulk_index,vocab,checkpoint,count,protocol):'),
    ("    inspect=local(EXP/'agent-plan-reload-inspect.py')\n    if sha(inspect.CAPTURE)!=inspect.CAPTURE_SHA or document!=json.loads(inspect.CAPTURE.read_text()):\n      raise ValueError('Only the exact reviewed actual plan is accepted')\n    if set(programs)!=set(document['programs']) or any(p.key.hex()!=key for key,p in programs.items()):\n      raise ValueError('PROGRAM identity changed')\n    if not proof['proved'] or proof['calls']!=765: raise ValueError('No complete scratch proof')\n    if len(document['calls'])!=765 or len(programs)!=29 or document['registry']['unresolved']:\n      raise ValueError('Unexpected actual plan')",
     "    views=family_validate(document,programs,proof,count,protocol)\n    self.count=count"),
    ("    self.ret=tuple(Tensor(UOp.new_buffer('AMD',prod(row['shape']),getattr(dtypes,row['dtype']))).reshape(row['shape'])\n                   for row in document['registry']['owners'] if row['name'] in ('return/logits','return/token'))\n    # Original _greedy_decode returns token shape (1,1); BlockDecoder reshapes it\n    # to (1,). Preserve that exact recorded view chain, including its layout key.\n    self.ret=(self.ret[0],self.ret[1].reshape(1,1).reshape(1))",
     "    output_rows={row['name']:row for row in document['registry']['owners']}\n    self.ret=tuple(family.rebuild_view(output_rows[name],views[name]) for name in ('return/logits','return/token'))"),
    ("spec['nbytes']!=10499328", "spec['nbytes']!=proof['scratch_bytes']"),
    ("{'block_position':{'dtype':'i32','min':0,'max':2303},'block_step':{'dtype':'i32','min':0,'max':63}}", "family.bounds(count)"),
  ]
  changed=source
  for old,new in replacements:
    if changed.count(old)!=1: raise ValueError('Frozen constructor anchor differs: '+old[:70])
    changed=changed.replace(old,new)
  restored=changed
  for old,new in reversed(replacements): restored=restored.replace(new,old)
  if restored!=source: raise AssertionError('Constructor changes extend beyond reviewed specializations')
  namespace=dict(vars(base),family=family,family_validate=validate)
  exec(compile(changed,str(EXP/'agent-plan-family-reload-loader.py')+' [qualified-constructor]','exec'),namespace)
  return namespace['__init__'],dict(frozen_source_sha256=frozen,
    constructor_sha256=hashlib.sha256(source.encode()).hexdigest(),adapted_constructor_sha256=hashlib.sha256(changed.encode()).hexdigest(),
    replacements=[dict(old=a,new=b) for a,b in replacements])

class Prepared(base.Prepared):
  __init__,derivation=constructor()

  def __call__(self,logits,cos,sin,position,step):
    from tinygrad.uop.ops import UOp
    if self.jit is None: raise ValueError('No parent-authorized runtime plan')
    self.guard();ranges=family.bounds(self.count)
    if type(position) is not int or not 0<=position<=ranges['block_position']['max']:
      raise ValueError('Decoder position outside original member range')
    if type(step) is not int or not 0<=step<=ranges['block_step']['max']:
      raise ValueError('Decoder rotary step outside original member range')
    return self.jit(logits,cos,sin,UOp.variable('block_position',0,ranges['block_position']['max']).bind(position),
                    UOp.variable('block_step',0,ranges['block_step']['max']).bind(step))

def build(count,model,inputs,protocol,*,bulk_index,vocab,checkpoint):
  doc,programs,proof,_=contract.read_member(count,protocol)
  return Prepared(doc,programs,proof,model,inputs,bulk_index=bulk_index,vocab=vocab,checkpoint=checkpoint,count=count,protocol=protocol)
