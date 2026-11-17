"""Source-bound serial decoder family contracts. No hardware access or code generation."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

EXP=Path(__file__).resolve().parent
WORK=Path('/mnt/nvme-500/tmp/alpamayo-speed-20261116')
TG=Path('/tmp/opencode/alpamayo-asius-20261114/tinygrad')
SOURCE=WORK/'agent-paced-dynamic-source-cpu02/alpamayo'
BASE=WORK/'agent-plan-mutation-v2-protocol.json'
BASE_SHA='c739c74b0eb3b366f366063643dd645efea0cef699a1fd8be99f1963358f974b'
SUCCESS=WORK/'agent-plan-mutation-retry1/report.json'
QUALIFIED=WORK/'agent-startup-key-fix1-replay/report.json'
SOURCES=('agent-plan-family-contract.py','agent-plan-family-capture.py','agent-plan-family-offline.py',
         'agent-plan-family-run.py','agent-plan-family-parent.py','agent-plan-family-check.py')
COUNTS=(2,3,4,5,6)
LIMITS=dict(child_seconds=300,start_next_count_before_seconds=225,artifact_bytes=256*1024**2,
            scratch_bytes_per_count=64*1024**2,live_tensor_bytes=7*1024**3,work_reserve_bytes=5*1024**3)

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def local(path):
  spec=importlib.util.spec_from_file_location('agent_plan_family_'+Path(path).stem.replace('-','_'),path)
  module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
  return module

def fresh(path):
  path=Path(path)
  if not path.resolve().is_relative_to(WORK) or path.exists(): raise ValueError('Use a new output under WORK')
  return path

def bounds(count):
  if type(count) is not int or count not in range(1,7): raise ValueError('Only serial lengths 1 through 6 are supported')
  return {'block_position':{'min':0,'max':2304-count,'dtype':'i32'},
          'block_step':{'min':0,'max':64-count,'dtype':'i32'}}

def signatures(): return json.loads(QUALIFIED.read_text())['startup_cache']['serial_signatures']

def validate_success():
  report=json.loads(SUCCESS.read_text());result=report['plan_reload']
  required=('qualified','fresh_process_original_vs_reloaded_exact','loaded_before_original_count1',
    'full_KV_restored','first_logits_restored','target_payload_restored','current_owner_retained',
    'allocation_identity_retained','generation_history_unchanged')
  if any(result.get(k) is not True for k in required) or len(result['cases'])!=6: raise ValueError('Count1 evidence incomplete')
  if result['numerical_PROGRAMs_exact']!=765 or result['original_alias_variants']!=2 or result['reloaded_alias_variants']!=2:
    raise ValueError('Count1 contract differs')
  for row in result['cases']:
    if not row['prediction_bits_equal'] or not row['all28_KV_bits_equal']: raise ValueError('Count1 mutation failed')
    if row['original_poison']['pattern']!=0xA5 or row['reload_poison']['pattern']!=0x5A: raise ValueError('Poison contract differs')
  if report['qualified'] is not False or result['ready_for_live_use'] is not False: raise ValueError('Scope of prior qualification changed')
  return report

def source_gate(protocol=None):
  if sha(BASE)!=BASE_SHA: raise ValueError('Frozen successful API protocol changed')
  local(EXP/'agent-plan-mutation-v2-run.py').verify(json.loads(BASE.read_text()))
  validate_success()
  if protocol is not None:
    if protocol['base']!={'path':str(BASE),'sha256':BASE_SHA}: raise ValueError('Base link changed')
    if protocol['sources']!={name:sha(EXP/name) for name in SOURCES}: raise ValueError('Family source changed')
    if protocol['limits']!=LIMITS or protocol['target']!='USB+AMD:LLVM:gfx1200': raise ValueError('Bound or target changed')
    if protocol['counts'] not in [list(COUNTS),[6]]: raise ValueError('Only the full missing family or count6 is authorized')
    for path,digest in protocol['files'].items():
      if sha(path)!=digest: raise ValueError('Family evidence changed: '+path)

def view_recipe(tensor):
  """Serialize only exact output RESHAPE nodes over its new allocation."""
  from tinygrad.uop.ops import Ops
  base=tensor.uop.base;u=tensor.uop;rows=[]
  while u is not base:
    if (u.op is not Ops.RESHAPE or len(u.src)!=2 or u.arg is not None or u.src[1].op not in (Ops.STACK,Ops.CONST) or
        any(s.op not in (Ops.STACK,Ops.CONST) for s in u.src[1].toposort()) or any(type(n) is not int or n<=0 for n in u.marg)):
      raise ValueError('Unreviewed output view; capture must preserve evidence before reload')
    rows.append(dict(op='RESHAPE',shape=list(u.marg)));u=u.src[0]
  return list(reversed(rows))

def rebuild_view(row,recipe):
  from tinygrad import Tensor,dtypes
  from tinygrad.uop.ops import UOp,Ops
  dtype=getattr(dtypes,row['base_dtype'])
  if row['offset_bytes']!=0 or row['nbytes']!=row['storage_bytes'] or row['nbytes']%dtype.itemsize:
    raise ValueError('Output must own its entire allocation')
  u=UOp.new_buffer(row['device'],row['nbytes']//dtype.itemsize,dtype)
  for entry in recipe:
    if set(entry)!={'op','shape'} or entry['op']!='RESHAPE' or any(type(n) is not int or n<=0 for n in entry['shape']):
      raise ValueError('Unreviewed output recipe')
    u=u.reshape(tuple(entry['shape']))
  result=Tensor(u)
  if (list(result.shape)!=row['shape'] or result.dtype.name!=row['dtype'] or
      result.uop.substitute({result.uop.base:UOp(Ops.NOOP)}).key.hex()!=row['layout_key']):
    raise ValueError('Output view or dtype differs')
  return result

def validate_document(document,count,expected,views):
  """Check each original family member independently. No repeated-graph inference."""
  from collections import Counter
  if document['qualified_numerical_identities_equal'] is not True or document['registry']['unresolved']:
    raise ValueError('Incomplete original numerical/ownership capture')
  calls=document['calls']
  actual=[dict(device='AMD',name=c['name'],program_key=c['program_key'],output_indices=c['outputs'],input_indices=c['inputs']) for c in calls]
  if actual!=expected['kernels'] or len(calls)!=764*count+1 or any(c['index']!=i or c['ordered_dependencies'] for i,c in enumerate(calls)):
    raise ValueError('Original numerical PROGRAM sequence or ordering changed')
  if document['input_contract']!=expected['input_contract'] or document['input_names']!=[0,1,2]: raise ValueError('Input ABI changed')
  if sorted((len(a['inputs']),a['staging_count']) for a in document['aliases'])!=[(0,0),(1,1)]:
    raise ValueError('Both original alias contracts are required')
  aliased=next(a['inputs'][0] for a in document['aliases'] if a['inputs'])
  if aliased['owners']!=['return/logits']: raise ValueError('Alias is not the output-logits allocation')
  expected_roles={'immutable_weight':227,'immutable_weight_scale':169,'immutable_token_mapping':1,
                  'caller_mutable_kv':28,'caller_current_input':3,'plan_output':2}
  if Counter(r['role'] for r in document['registry']['owners'])!=expected_roles: raise ValueError('Explicit owner registry changed')
  if document['variables'] not in ({},bounds(count)): raise ValueError('Unexpected explicit scalar bindings')
  arena,=document['registry']['scratch']
  if (arena['name'],arena['device'],arena['dtype'])!=('scratch.0','AMD','i8') or not 0<arena['nbytes']<=LIMITS['scratch_bytes_per_count']:
    raise ValueError('Unreviewed arena contract')
  rows={r['name']:r for r in document['registry']['owners']}
  if rows['return/logits']['shape']!=[1,1,2112] or rows['return/token']['shape']!=[count]: raise ValueError('Original output shapes changed')
  if set(views)!={'return/logits','return/token'}: raise ValueError('Missing exact return view recipes')
  for name,recipe in views.items(): rebuild_view(rows[name],recipe)
  return dict(count=count,calls=len(calls),programs=len(document['programs']),scratch_bytes=arena['nbytes'],
    token_shape=rows['return/token']['shape'],variables=bounds(count),aliases=2,owners=len(rows))
