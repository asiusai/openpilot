"""Frozen evidence, storage and exact two-query startup comparison."""
import copy
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import struct
import sys

EXP=Path(__file__).resolve().parent
WORK=Path('/mnt/nvme-500/tmp/alpamayo-speed-20261116')
ROOT=WORK/'agent-paced-dynamic-source-cpu02/alpamayo'
STARTUP=WORK/'startup-spec-frozen-source'
TG=Path('/tmp/opencode/alpamayo-asius-20261114/tinygrad')
PY=Path('/tmp/opencode/alpamayo-asius-20261114/upstream/.venv/bin/python')
INVOCATION=ROOT.parent/'pipeline-invocation.json'
FP8=WORK/'agent-expert-combined-cache-artifacts/fp8-cache'
FAMILY=WORK/'agent-plan-family-reload-protocol1.json'
INDICES=(0,16)
PINS={
 str(FAMILY):'0c71a5a9236ed59b9fb25c67753a8f776d3ce11a259be53cc1e0dedb0d3f239b',
 str(WORK/'agent-plan-family-reload-trial1/report.json'):'5259b3465289c4941987a0ff6cf5225a4c40be11f5d6d821a5eb0ff7da590931',
 str(WORK/'agent-plan-mutation-retry1/report.json'):'135f3e6f6c80df75ff3d2e7329901f2b7aa085a76342dbd8db9c3263f27bc55d',
 str(STARTUP/'agent-startup-spec-warm.py'):'3b7f2d7d5e1888243271fa34b7bb8f91d7b7f7c5dd3789503bec1071a3e2900c',
 str(FP8/'index.json'):'c4366c5b7e92b4cd5e2d826881b99b9a80361ec171ebd4a50d51b0f718dffa3e'}

def load(path):
  name='startup_persist_'+Path(path).stem.replace('-','_')
  spec=importlib.util.spec_from_file_location(name,path)
  module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
  return module

def sha(path):
  with Path(path).open('rb') as stream: return hashlib.file_digest(stream,'sha256').hexdigest()
def read(path): return json.loads(Path(path).read_text())
def save(path,value):
  with Path(path).open('x') as stream: json.dump(value,stream,indent=2,allow_nan=False);stream.write('\n')
def sources(): return {str(p):sha(p) for p in sorted(EXP.glob('agent-startup-persist-*.py'))}
def equal(a,b,label='value'):
  if type(a) is not type(b): raise AssertionError(label+': type')
  if isinstance(a,dict):
    if set(a)!=set(b): raise AssertionError(label+': fields')
    for k in a: equal(a[k],b[k],label+'.'+str(k))
  elif isinstance(a,list):
    if len(a)!=len(b): raise AssertionError(label+': length')
    for i,(x,y) in enumerate(zip(a,b,strict=True)): equal(x,y,label+'.'+str(i))
  elif isinstance(a,float):
    if not math.isfinite(a) or not math.isfinite(b) or struct.pack('!d',a)!=struct.pack('!d',b):
      raise AssertionError(label+': finite float bits')
  elif a!=b: raise AssertionError(label+': differs')

def storage(path):
  path=Path(path).resolve()
  if not path.is_relative_to(WORK) or WORK.stat().st_dev==Path('/').stat().st_dev:
    raise ValueError('Writable storage must be on the mounted 500GB WORK filesystem')
  return path

def evidence():
  for p,h in PINS.items():
    if sha(p)!=h: raise ValueError('Qualified immutable evidence changed: '+p)
  family=load(EXP/'agent-plan-family-reload-contract.py');family.gate(read(FAMILY))
  a=read(WORK/'agent-plan-family-reload-trial1/report.json')['plan_family_reload']
  b=read(WORK/'agent-plan-mutation-retry1/report.json')['plan_reload']
  if not a['whole_family_2_through_6_passed'] or not a['qualified'] or not b['qualified']:
    raise ValueError('All six original persistent plans require GPU qualification')
  policy=load(ROOT/'experiments/agent-paced-dynamic-policy.py');policy.verify_snapshot()
  old=read(WORK/'agent-startup-spec-candidate/report.json')
  if [old['rows'][i]['speculation']['accepted'] for i in INDICES]!=[True,False]: raise ValueError('Selection changed')
  equal(old['rows'][0]['history_after'],old['rows'][16]['history_before'],'selected history boundary')
  return dict(pins=PINS,selected_samples=[old['rows'][i]['sample'] for i in INDICES],indices=list(INDICES),
    target='USB+AMD:LLVM:gfx1200',qualified_counts=list(range(1,7)),fresh_pair_required=True)

def verify(manifest):
  if manifest['schema']!=1 or manifest['sources']!=sources(): raise ValueError('Startup integration source changed')
  for p,h in manifest['files'].items():
    if sha(p)!=h: raise ValueError('Sealed dependency changed: '+p)
  equal(evidence(),manifest['evidence'],'evidence')
  if manifest['child_seconds']!=450 or manifest['outer_timeout_seconds']!=0: raise ValueError('Bound changed')

def cache_check():
  storage(FP8);index=read(FP8/'index.json');blob=FP8/'weights.bin'
  if sha(FP8/'index.json')!=PINS[str(FP8/'index.json')] or blob.is_symlink(): raise ValueError('FP8 index changed')
  if blob.stat().st_size!=index['nbytes'] or sha(blob)!=index['sha256']: raise ValueError('FP8 resident source blob differs')
  return dict(index=str(FP8/'index.json'),index_sha256=sha(FP8/'index.json'),blob=str(blob),
    bytes=blob.stat().st_size,blob_sha256=index['sha256'],streamed_full_blob=True)

def selected(full):
  result=copy.copy(full)
  result.rows=[full.rows[i] for i in INDICES];result.samples=[full.samples[i] for i in INDICES];result.inputs=[full.inputs[i] for i in INDICES]
  result.provenance=dict(full.provenance,count=2,original_count=64,selected_indices=list(INDICES),
    inputs_sha256=hashlib.sha256(json.dumps(result.inputs,sort_keys=True,allow_nan=False).encode()).hexdigest(),
    metadata_sha256=hashlib.sha256(json.dumps(result.rows,sort_keys=True,allow_nan=False).encode()).hexdigest(),
    history='Fresh empty history; original row zero then original row sixteen')
  return result

def projected_provenance(report,path):
  value=copy.deepcopy(report['consumer_provenance']);factory=value['initializer']['factory']
  folder=Path(factory['source_snapshot']);expected=Path(path).with_suffix('.benchmark-sources')
  if not folder.is_absolute() or folder!=expected.resolve() or not folder.is_relative_to(WORK) or folder.is_symlink():
    raise ValueError('Unverified factory snapshot location')
  files=report['source_sha256'];equal(files,report['initialized']['source_sha256'],'snapshot inventory')
  if len(files)!=253 or {p.name for p in folder.iterdir()}!=set(files): raise ValueError('Incomplete snapshot inventory')
  for name,h in files.items():
    p=folder/name
    if not p.is_file() or p.is_symlink() or sha(p)!=h: raise ValueError('Factory snapshot content changed')
  ms=factory['model_load_ms']
  if type(ms) not in (int,float) or not math.isfinite(ms) or ms<0: raise ValueError('Invalid factory duration')
  del factory['model_load_ms']
  factory['source_snapshot']={'complete_file_sha256':dict(sorted(files.items()))}
  return value

def compare(reference,candidate,reference_path,candidate_path):
  for k in ('protocol','decoder','prefill_attention','initialized','source_sha256'):
    equal(reference[k],candidate[k],k)
  equal(projected_provenance(reference,reference_path),projected_provenance(candidate,candidate_path),'consumer_provenance')
  for report,role in ((reference,'ordinary'),(candidate,'persisted')):
    if report['mode']!='consumer' or not report['gpu_executed'] or len(report['rows'])!=2: raise ValueError('Fresh two Consumer predictions required')
    state=report['startup_persist']
    if not state['completed'] or state['role']!=role or not state['source_before_after_exact'] or not state['hooks_restored']:
      raise ValueError('Incomplete source-bound startup arm')
    warm=report['startup_spec']
    for k in ('ready','full_kv_bits_restored','actual_first_logits_bits_restored','target_payload_bits_restored','warmed_owner_retained'):
      if warm.get(k) is not True: raise ValueError('Missing startup restoration: '+k)
    if warm['current_inputs']!=1 or warm['future_inputs'] or warm['predictions_published'] or warm['forced_tokens']:
      raise ValueError('Startup input scope changed')
    if report['rows'][0]['history_before']!={'history':[],'prefill_draft_old_history':[]}:
      raise ValueError('Fresh empty history required')
    if [r['speculation']['accepted'] for r in report['rows']]!=[True,False]: raise ValueError('Actual acceptance/fallback not covered')
    if any(not r['completed'] or not r['consumer_protocol']['input_reused'] or len(r['defined_kv_prefix_bits'])!=28 for r in report['rows']):
      raise ValueError('Owner/completion/KV-prefix coverage incomplete')
  for k in ('decoder_plan','graph_counts'):
    equal(reference['startup_spec'][k],candidate['startup_spec'][k],k)
  equal(reference['startup_persist']['ready_signatures'],candidate['startup_persist']['ready_signatures'],'actual ready plans')
  for i,(a,b) in enumerate(zip(reference['rows'],candidate['rows'],strict=True)):
    a,b=copy.deepcopy(a),copy.deepcopy(b)
    # Existing parity treats measured prediction durations as observations. Every other field stays exact.
    a['prediction'].pop('timings_ms',None);b['prediction'].pop('timings_ms',None)
    equal(a,b,'query.'+str(i))
  return dict(exact=True,queries=2,accepted=1,fallback=1,all_prediction_float_bits=True,all28_defined_KV_prefix_bits=True,
    all_other_canonical_fields=True,actual_ready_signatures_exact=True,ready_for_live_use=False,full64_still_required=True)
