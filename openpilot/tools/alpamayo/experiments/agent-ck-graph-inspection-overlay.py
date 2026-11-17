"""Two exact hook edits; the frozen runner and all model/control code stay unchanged."""
from functools import lru_cache
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace,ModuleType

EXP=Path(__file__).parent
def module(name):
  spec=importlib.util.spec_from_file_location(name.replace('-','_'),EXP/(name+'.py'))
  value=importlib.util.module_from_spec(spec);spec.loader.exec_module(value);return value

OLD=module('agent-ck-integrated-common')
ROOT=OLD.WORK/'agent-ck-graph-inspection-work'
PREPARED=ROOT/'prepared'
OLD_PROTOCOL_SHA='1b8797fc5596a6eeff39c17e028fde513a31d211678220ea171999fdc5c2da8e'
HOOK_SHA='f1e03b529cf27cde24edbe76b17963fcf6dd3f07e130fa3c62bd94022ad01ad7'
GRAPH=module('agent-ck-graph-inspection')
BEFORE="""      from tinygrad.runtime.support.hcq2 import HCQInfo
      from tinygrad.uop.ops import Ops
      for call in captured._linear.src:
        if isinstance(call.arg.aux,HCQInfo): native+=sum(k[1]=='alpamayo_ck_d128' for k in call.arg.aux.kernels)
        elif call.body.op is Ops.PROGRAM: native+=call.body.src[0].arg.name=='alpamayo_ck_d128'
      assert native==(28 if self.arm=='candidate' else 0),('Captured CK coverage',native)
"""
AFTER="""      native=_graph.inspect_captured(captured,self.common,self.report,self.arm,phase)
"""
CHECK="    assert identity.identity(p)==expected,'Changed CK source/AST/ELF/target/ABI association'\n"
REGISTER="""    keys=report.setdefault('checked_native_program_keys',[])
    if p.key.hex() not in keys: keys.append(p.key.hex())
"""

def hook_source():
  path=EXP/'agent-ck-integrated-hook.py';assert OLD.sha(path)==HOOK_SHA
  source=path.read_text();assert source.count(BEFORE)==source.count(CHECK)==1
  return source.replace(BEFORE,AFTER).replace(CHECK,CHECK+REGISTER)

@lru_cache(None)
def load(name,directory=EXP):
  if name!='agent-ck-integrated-hook' or Path(directory)!=EXP: return OLD.load(name,directory)
  value=ModuleType('ck_graph_inspection_hook');value.__file__=str(EXP/'agent-ck-integrated-hook.py');value._graph=GRAPH
  exec(compile(hook_source(),value.__file__,'exec'),value.__dict__)
  return value

def verify():
  assert OLD.sha(OLD.PREPARED/'protocol.json')==OLD_PROTOCOL_SHA
  original=OLD.verify()
  protocol=json.loads((PREPARED/'protocol.json').read_text())
  assert protocol['previous_protocol_sha256']==OLD_PROTOCOL_SHA and protocol['timeout_seconds_per_arm']==600
  for path,digest in protocol['pins'].items(): assert OLD.sha(path)==digest,('Changed retry source',path)
  assert OLD.sha(PREPARED/'effective-hook.py')==protocol['effective_hook_sha256']
  assert (PREPARED/'effective-hook.py').read_text()==hook_source()
  return {**original,'graph_inspection_retry':protocol}

C=SimpleNamespace(**{k:v for k,v in vars(OLD).items() if not k.startswith('__')})
C.ROOT=ROOT;C.PREPARED=PREPARED;C.load=load;C.verify=verify
R=module('agent-ck-integrated-run');R.C=C

def comparator():
  value=module('agent-ck-integrated-compare');value.C=C;return value
