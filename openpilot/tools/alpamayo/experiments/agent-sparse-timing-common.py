"""Frozen numerical-output templates and strict source/capture gates."""
import hashlib,importlib.util,json
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
EXP=Path('/tmp/opencode/openpilot-alpamayo/openpilot/tools/alpamayo/experiments')
WORK=Path('/mnt/nvme-500/tmp/alpamayo-speed-20261116')
NUMERIC=EXP/'agent-sparse-numeric-source'
EVIDENCE=WORK/'agent-sparse-numeric-actual/report.json'
CALLS=40
LAYERS=(0,14,27)
SHAPE=(2112,2048,9216)


def load(path,name):
  spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
def local(name):return load(HERE/(name+'.py'),name.replace('-','_'))
def sha(path):
  h=hashlib.sha256()
  with Path(path).open('rb')as f:
    while data:=f.read(8*1024*1024):h.update(data)
  return h.hexdigest()
def array_sha(value):return hashlib.sha256(np.ascontiguousarray(value).view(np.uint8)).hexdigest()
def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def write(path,value):
  path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def candidate_expected(original,entry):
  """Reconstruct ALL previous BF16 candidate bits, with no tolerance.

  The completed full-array comparison counted every differing element. Its
  worst-values list contains every difference for these three fixed layers.
  The separately captured full-output SHA256 must certify the reconstruction.
  """
  original=np.asarray(original)
  assert original.dtype==np.uint16 and original.shape==(2112,2048)
  metric=entry['hybrid_vs_original_BF16']
  assert metric['elements']==original.size and metric['all_finite']
  assert not metric['actual_nonfinite']and not metric['reference_nonfinite']
  changed=[x for x in metric['worst_values']if x['actual_bits']!=x['reference_bits']]
  assert len(changed)==metric['differing_bits_elements'],'Incomplete prior numerical-output inventory'
  assert len({tuple(x['coordinate'])for x in changed})==len(changed),'Repeated difference coordinate'
  expected=original.copy()
  for value in metric['worst_values']:
    row,col=value['coordinate'];assert 0<=row<2112 and 0<=col<2048
    assert int(original[row,col])==int(value['reference_bits'],16)
    if value in changed:expected[row,col]=int(value['actual_bits'],16)
  assert np.isfinite((expected.astype(np.uint32)<<16).view(np.float32)).all()
  states=entry['replay']['replays'];first=next(x for x in states if x['change']=='unmodified')
  restored=next(x for x in states if x['change']=='restore')
  assert first['output_sha256']==restored['output_sha256']==array_sha(expected),'Prior complete-output certificate mismatch'
  assert all(x['all_bits_equal_fresh_same_candidate']and x['current_format_exact']and x['poison_replaced']for x in states)
  return expected


def numeric_inputs():
  N=load(NUMERIC/'agent-sparse-numeric-common.py','timing_frozen_numeric_common')
  plan=json.loads((NUMERIC/'agent-sparse-numeric-plan.json').read_text())
  old,capture,records,runner,root=N.verify(plan)
  evidence=json.loads(EVIDENCE.read_text())
  assert evidence['complete']and evidence['status']=='DENSE_ALIGNMENT_QUALIFIED_SPARSE_NUMERICAL_CHARACTERIZED'
  assert evidence['classification']=='NUMERICAL_DIAGNOSTIC_ONLY'and evidence['timings']==[]
  assert evidence['plan_identity']==plan['identity_sha256']and evidence['source_capture_sha256']==N.sha(N.CAPTURE)
  assert not evidence['exact_adoption_claim']and not evidence['quality_claim']and not evidence['route_executed']
  assert len(evidence['controls'])==11 and [x['layer']for x in evidence['actual']]==list(LAYERS)
  assert all(x['bits_exact']and x['all_finite']for family in evidence['controls']for x in family['dense_original_checks'])
  expected=[]
  for record,data in records:
    entry=next(x for x in evidence['actual']if x['layer']==record['layer'])
    assert entry['shape']==list(SHAPE)and entry['actual_prompt_rows']==2027
    assert all(x['bits_exact']and x['all_finite']for x in entry['dense_original_checks'])
    assert entry['hybrid_BF16_vs_own_raw_RNE']['bits_exact']and entry['replay']['input_and_weight_restored']
    assert not entry['hybrid_consistency']['elements_outside_bound']and entry['hybrid_consistency']['all_finite']
    expected.append((record,data,candidate_expected(data['original_bf16'],entry),entry))
  return N,old,plan,capture,expected,runner,root,evidence


def dependencies():
  paths=[*HERE.glob('agent-sparse-timing-*.py'),HERE/'agent-sparse-timing-cpu.json',EVIDENCE,
    WORK/'agent-sparse-numeric-actual/invocation.json',NUMERIC/'agent-sparse-numeric-plan.json']
  return {str(p):sha(p)for p in sorted(paths)}


def prepare():
  cpu=json.loads((HERE/'agent-sparse-timing-cpu.json').read_text())
  assert cpu['cpu_only']and cpu['all_pass']
  assert cpu['source_hashes']=={p.name:sha(p)for p in HERE.glob('agent-sparse-timing-*.py')}
  _,_,numeric,_,records,_,_,_=numeric_inputs()
  result=dict(schema='agent-sparse-timing-v1',sources=dependencies(),numeric_plan=numeric,
    numeric_evidence_sha256=sha(EVIDENCE),classification='NUMERICAL_DIAGNOSTIC_ONLY',calls=CALLS,rounds=3,
    orders=[['original','hybrid'],['hybrid','original'],['original','hybrid']],layers=list(LAYERS),shape=list(SHAPE),
    expected=[dict(layer=r['layer'],original_sha256=array_sha(d['original_bf16']),candidate_sha256=array_sha(expected))for r,d,expected,_ in records],
    boundary='Existing packed FP8 activations/weights -> compression/metadata/eligibility+dense fallback+GEMM/scales/BF16. Common quantization/packing and static weight preparation excluded.',
    route_or_model_integration=False,numerical_tolerance=False)
  result['identity_sha256']=digest(result);return result


def verify(plan):
  assert plan['schema']=='agent-sparse-timing-v1'
  assert digest({k:v for k,v in plan.items()if k!='identity_sha256'})==plan['identity_sha256']
  for path,expected in plan['sources'].items():assert sha(path)==expected,'Frozen timing dependency changed: '+path
  assert str(Path(__file__).resolve())in plan['sources'],'Run the frozen timing source'
  data=numeric_inputs();assert data[2]==plan['numeric_plan']and sha(EVIDENCE)==plan['numeric_evidence_sha256']
  assert plan['calls']==40 and plan['rounds']==3 and plan['shape']==list(SHAPE)and plan['layers']==list(LAYERS)
  assert plan['orders']==[['original','hybrid'],['hybrid','original'],['original','hybrid']]
  assert plan['classification']=='NUMERICAL_DIAGNOSTIC_ONLY'and not plan['numerical_tolerance']
  for (record,arrays,expected,_),qualified in zip(data[4],plan['expected'],strict=True):
    assert qualified==dict(layer=record['layer'],original_sha256=array_sha(arrays['original_bf16']),candidate_sha256=array_sha(expected))
  return data
