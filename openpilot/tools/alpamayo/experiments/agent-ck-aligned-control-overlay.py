"""One candidate-only diagnostic change; retain the passed baseline by exact lineage."""
import ast
import copy
from functools import lru_cache
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace,ModuleType

EXP=Path(__file__).parent
def module(name):
  s=importlib.util.spec_from_file_location(name.replace('-','_'),EXP/(name+'.py'))
  m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
O=module('agent-ck-graph-inspection-overlay')
A=module('agent-ck-aligned-control')
ROOT=O.C.WORK/'agent-ck-aligned-control-work';PREPARED=ROOT/'prepared'
BASELINE=O.C.WORK/'agent-ck-graph-inspection-parent-baseline-v1/report.json'
BASELINE_SHA='6ea1376ec59497175780a080bc17a8c7501775a12050f17c8dc4b5147c7ef590'
BASE_PIPELINE_SHA='17cff641291bb8ab88be6b67a35a9a00391247a1d09efc5c08578a9fa16fdc14'
BASE_PROTOCOL_SHA='b092bf51d6f1b5b655aa153020889932bb796613ae204cc4ff734b3be8c818a2'
FAILED=O.C.WORK/'agent-ck-graph-inspection-parent-candidate-v1/report.json'
FAILED_SHA='b8ce47aa829455ef89e394c8a5b1194b3c7a8a35d098cfc0dfc9cb76b3f9a38b'
OLD_CONTROLS_SHA='89adfeb562dd402f385489feb51c48ea60bb85726c5cfa84d0ea9a342e46b531'
OLD_WINDOW="""    lo,hi=max(0,index-1),min(count,index+2)
    neighbors=Tensor(UOp.from_buffer(buffer.view((hi-lo)*size,lo*size).ensure_allocated(),dt))
"""

def controls_source():
  p=EXP/'agent-ck-integrated-controls.py';assert O.C.sha(p)==OLD_CONTROLS_SHA
  source=p.read_text()
  edits=((OLD_WINDOW,'    neighbors,relative,region=_aligned.neighbors(buffer,index,dt)\n'),
    ('expected[index-lo]=value','expected[relative]=value'),
    ("'bytes':size,'neighbors_exact':True,","'bytes':size,'neighbors_exact':True,'read_region':region,"))
  for old,new in edits:
    assert source.count(old)==1;source=source.replace(old,new)
  return source

@lru_cache(None)
def load(name,directory=EXP):
  if name!='agent-ck-integrated-controls' or Path(directory)!=EXP: return O.load(name,directory)
  m=ModuleType('ck_aligned_controls');m.__file__=str(EXP/'agent-ck-integrated-controls.py');m._aligned=A
  exec(compile(controls_source(),m.__file__,'exec'),m.__dict__);return m

def lineage():
  assert O.C.sha(O.PREPARED/'protocol.json')==BASE_PROTOCOL_SHA
  assert O.C.sha(BASELINE)==BASELINE_SHA and O.C.sha(FAILED)==FAILED_SHA
  b=json.loads(BASELINE.read_text());assert b['completed'] and b['gpu_executed'] and b['arm']=='baseline'
  assert b['pipeline_sha256']==BASE_PIPELINE_SHA==O.C.sha(b['pipeline'])
  assert b['source_protocol_sha256']==BASE_PROTOCOL_SHA and b['indices']==O.C.INDICES and b['initial_decoder_history']==[]
  assert b['hooks_restored'] and b['model_renderer_unchanged'] and len(b['prefills'])==len(b['decoder_histories'])==8
  assert [r['phase'] for r in b['prefills']]==[f'screen_{i}_original64_{n}' for i,n in enumerate(O.C.INDICES)]
  assert all(not any(v for row in r['all28_audit_counts'] for v in row) for r in b['prefills'])
  assert all(g['native']==0 for row in b['capture_inspections'] for g in row['graphs'])
  # Prove all code outside the local patch_word diagnostic is byte-for-byte/AST identical.
  old=ast.parse((EXP/'agent-ck-integrated-controls.py').read_text());new=ast.parse(controls_source())
  for tree in (old,new):
    patch=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='patch_word')
    patch.body=[ast.Pass()]
  assert ast.dump(old)==ast.dump(new)
  assert O.hook_source()==(O.PREPARED/'effective-hook.py').read_text()
  for name in ('samples.json','inputs.json','ck-identity.json'):
    assert (PREPARED/name).read_bytes()==(O.PREPARED/name).read_bytes()
  def recipe(path):
    value=json.loads(path.read_text());value['command'][value['command'].index('--output')+1]='OUTPUT'
    for key in ('CACHEDB','TMPDIR'): value['environment'][key]=key
    return value
  reference=recipe(O.PREPARED/'baseline-invocation.json')
  assert reference==recipe(O.PREPARED/'candidate-invocation.json')
  assert reference==recipe(PREPARED/'candidate-invocation.json')==recipe(PREPARED/'baseline-invocation.json')
  pipeline=json.loads(Path(b['pipeline']).read_text())
  assert [r['sample'] for r in pipeline['runs']]==json.loads((PREPARED/'samples.json').read_text())
  return {'baseline_report':str(BASELINE),'baseline_report_sha256':BASELINE_SHA,'baseline_pipeline_sha256':BASE_PIPELINE_SHA,
    'baseline_protocol_sha256':BASE_PROTOCOL_SHA,'same_model_native_hook_and_ABI_sources':True,
    'same_numerical_recipe_inputs_order_and_initial_history':True,'only_change':'Candidate negative-control aligned neighbor read and exact extra-neighbor comparison'}

def verify():
  O.verify();evidence=lineage();p=json.loads((PREPARED/'protocol.json').read_text())
  assert p['baseline_lineage']==evidence and p['timeout_seconds']==600
  for path,digest in p['pins'].items(): assert O.C.sha(path)==digest,('Changed aligned-control source',path)
  assert (PREPARED/'effective-controls.py').read_text()==controls_source()
  return p

C=SimpleNamespace(**vars(O.C));C.ROOT=ROOT;C.PREPARED=PREPARED;C.load=load;C.verify=verify
R=O.R;R.C=C

def comparator_source():
  source=(EXP/'agent-ck-integrated-compare.py').read_text()
  old="r['source_protocol_sha256']==C.sha(C.PREPARED/'protocol.json')"
  assert source.count(old)==1
  return source.replace(old,"r['source_protocol_sha256']==(BASE_PROTOCOL_SHA if arm=='baseline' else C.sha(C.PREPARED/'protocol.json'))")

def compare(candidate,output):
  C.verify()
  m=ModuleType('ck_aligned_comparison');m.__file__=str(EXP/'agent-ck-integrated-compare.py')
  exec(compile(comparator_source(),m.__file__,'exec'),m.__dict__)
  m.C=C;m.BASE_PROTOCOL_SHA=BASE_PROTOCOL_SHA
  return m.compare(BASELINE,candidate,output)
