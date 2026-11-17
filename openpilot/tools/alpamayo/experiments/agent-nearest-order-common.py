"""Independent same-order investigation. Every earlier artifact stays immutable."""
import copy,hashlib,importlib.util,inspect,json,os,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
WORK=Path('/mnt/nvme-500/tmp/alpamayo-speed-20261116')
LIVE_PLAN=WORK/'agent-nearest-live-data/agent-nearest-live-plan.json'
SAVED=WORK/'agent-nearest-live-parity4/report.json'
BASE=WORK/'agent-paced-dynamic-source-cpu02/alpamayo'
FROZEN=WORK/'agent-nearest-live-source'
PROTOCOL=HERE/'agent-nearest-order-protocol.json'
PINS={str(LIVE_PLAN):'56443231afbf58b7ae5f65317b2f6f310c521affe5574ff4c8ecb982384dd311',
  str(SAVED):'3201f34b37b31c604ce0970bd3e134d9387951b0432200c3b0ce298beb25c205',
  str(SAVED.with_name('run.log')):'cbe52fd419d4b75d2c1a1fd3bc5328c0df5f9d06146d602595e33b50962250dc',
  str(BASE/'experiments/benchmark-pipeline.py'):'bcf890f8321dc0b0ff499380e01a42956f3a2395015efe091648a2a04e04cf56'}
RESTORE=('ready','full_kv_bits_restored','actual_first_logits_bits_restored','target_payload_bits_restored','warmed_owner_retained')
sys.dont_write_bytecode=True

def sha(path):
  with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
def read(path):return json.loads(Path(path).read_text())
def save(path,value):
  with Path(path).open('x') as stream:stream.write(json.dumps(value,indent=2,allow_nan=False)+'\n')
def load(path,name=None):
  path=Path(path);name=name or 'agent_nearest_order_'+path.stem.replace('-','_')
  spec=importlib.util.spec_from_file_location(name,path);value=importlib.util.module_from_spec(spec);sys.modules[name]=value;spec.loader.exec_module(value);return value
def local(name):return load(BASE/'experiments'/(name+'.py'))
def own_sources():return {p.name:sha(p) for p in sorted(HERE.glob('agent-nearest-order-*.py'))}
def setup_cpu():
  os.environ.update(DEV='CPU',CUDA_VISIBLE_DEVICES='',HIP_VISIBLE_DEVICES='',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
    TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',CACHEDB=str(HERE/'agent-nearest-order-cpu-cache.db'),
    XDG_CACHE_HOME=str(HERE/'agent-nearest-order-cpu-cache'))
  sys.path[:0]=['/tmp/opencode/alpamayo-asius-20261114/tinygrad',str(BASE),str(BASE/'experiments'),
    '/tmp/opencode/openpilot-alpamayo','/home/john/asius/openpilot/opendbc_repo']
def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,allow_nan=False).encode()).hexdigest()
def remember(history,tokens):return [old for old in history if old!=tokens][-63:]+[tokens[:]]

def verify_assets():
  for path,expected in PINS.items():
    if sha(path)!=expected:raise ValueError('Frozen evidence changed: '+path)
  plan=read(LIVE_PLAN);old=load(FROZEN/'agent-nearest-live-run.py');policy=load(FROZEN/'agent-nearest-live-policy.py')
  root,startup,dynamic=old.verify(plan,policy)
  assert root==BASE
  return plan,policy,startup,dynamic

def validate_saved(plan,startup,report=None):
  """Validate reusable evidence without changing its failed comparison status."""
  import numpy as np
  saved=read(SAVED) if report is None else report;qualified=read(plan['startup_candidate'])
  if not (saved['mode']=='nearest-live-parity4' and saved['gpu_executed'] is True and saved['queries']==4 and
          saved['predictions_qualified'] is False and saved['comparison']['exact'] is False):raise ValueError('Preserve the actual failed4 evidence')
  if saved['plan_sha256']!=sha(LIVE_PLAN) or saved['adapter_sha256']!=sha(FROZEN/'agent-nearest-live-policy.py'):
    raise ValueError('Saved Consumer policy provenance differs')
  if saved['startup_sources']!=startup.own_sources() or saved['qualified_factory_protocol']!=qualified['protocol']:
    raise ValueError('Saved Consumer startup/factory provenance differs')
  for key in ('source_sha256','settings','runtime_environment'):
    if saved['initialized'][key]!=qualified['initialized'][key]:raise ValueError('Saved initialized source/settings differ: '+key)
  warm=saved['startup_spec']
  if not all(warm.get(key) is True for key in RESTORE):raise ValueError('Missing bit-exact startup restoration')
  if warm['forced_tokens']!=0 or warm['future_inputs']!=0 or warm['current_inputs']!=1 or warm['predictions_published']!=0:
    raise ValueError('Startup used output or future-input conditioning')
  if warm['source_sha256']['agent-startup-spec-warm.py']!=sha(Path(plan['startup_runner']).with_name('agent-startup-spec-warm.py')):
    raise ValueError('Startup helper source changed')
  history=[]
  if len(saved['rows'])!=4:raise ValueError('Four complete Consumer rows required')
  for case,row in zip(plan['cases'],saved['rows'],strict=True):
    if row['sample']!=case['sample'] or row['query_ns']!=case['query_ns'] or not row['completed']:raise ValueError('Saved completion/order mismatch')
    for key,value in case['expected_cpu_hashes'].items():
      if row[key]!=value:raise ValueError('Saved Consumer input hash differs: '+key)
    if row['history_before']!={'history':history,'prefill_draft_old_history':[]}:raise ValueError('Unexpected prior history')
    history=remember(history,row['prediction']['generated_token_ids'])
    if row['history_after']!={'history':history,'prefill_draft_old_history':[]}:raise ValueError('Unexpected next history')
    spec=row['speculation'];prediction=row['prediction']
    if (spec['accepted']!=case['expected_speculative_accept'] or not spec['eligible'] or not spec['all_targets_finite'] or
        spec['generated']!=prediction['generated_token_ids'] or spec['final_cache_length']!=spec['actual_prompt']+len(spec['generated']) or
        spec['action_rotary_offset']!=len(spec['generated'])):raise ValueError('Saved target verification/cache contract differs')
    if row['decoder_statistics']['prefill_draft']!=spec:raise ValueError('Speculation record mismatch')
    if row['seed']!=0 or row['noise']!=qualified['initialized']['settings']['initial_noise']:raise ValueError('Noise/seed changed')
    for key,shape in (('xyz',(1,64,3)),('rot',(1,64,3,3))):
      value=np.asarray(prediction[key]);assert value.shape==shape and np.isfinite(value).all()
    protocol=row['consumer_protocol']
    if protocol['input_reused'] is not True or protocol['owned_input_objects']!=1:raise ValueError('Warmed Consumer owner was replaced')
    if protocol['source_capture_timestamps_ns']!=[f['capture_ns'] for f in case['metadata']['frames']]:raise ValueError('Capture provenance mismatch')
  return saved

def transform_main(pipeline):
  """One reviewed ownership substitution; original main still performs preparation/run."""
  if sha(pipeline.__file__)!=PINS[str(BASE/'experiments/benchmark-pipeline.py')]:raise ValueError('Canonical benchmark source changed')
  source=inspect.getsource(pipeline.main)
  before='  inputs = None\n  validation = '
  after='  inputs = components.inputs\n  validation = '
  if source.count(before)!=1:raise ValueError('Ambiguous warmed-owner substitution')
  modified=source.replace(before,after,1);namespace=dict(pipeline.__dict__)
  exec(compile(modified,str(HERE/'agent-nearest-order-run.py')+':canonical-main','exec'),namespace)
  pipeline.main=namespace['main']
  return dict(original_sha256=hashlib.sha256(source.encode()).hexdigest(),adapted_sha256=hashlib.sha256(modified.encode()).hexdigest(),
    substitutions=[dict(before=before,after=after)])

def compare_rows(reference,saved):
  """Every original audit field and prediction bit is mandatory. No tolerance."""
  import numpy as np
  if len(reference)!=4 or len(saved)!=4:raise ValueError('Four rows required')
  differences=[]
  for index,(a,b) in enumerate(zip(reference,saved,strict=True)):
    if not a['completed'] or not b['completed']:raise ValueError('Incomplete prediction')
    if set(a)!=set(b)-{'consumer_protocol'}:raise ValueError('Audit schema differs')
    changed=[]
    for key in sorted(a):
      av,bv=a[key],b[key]
      if key=='prediction':
        av,bv=({k:v for k,v in p.items() if k!='timings_ms'} for p in (av,bv))
        for field,shape in (('xyz',(1,64,3)),('rot',(1,64,3,3))):
          aa,bb=(np.asarray(p[field],dtype=np.float64) for p in (av,bv))
          if aa.shape!=shape or bb.shape!=shape or not np.isfinite(aa).all() or not np.isfinite(bb).all():raise ValueError('Invalid full prediction')
          if not np.array_equal(aa.view(np.uint64),bb.view(np.uint64)):changed.append('prediction.'+field+'.bits')
      if digest(av)!=digest(bv):changed.append(key)
    if changed:differences.append(dict(index=index,query_ns=a['query_ns'],fields=changed))
  return dict(exact=not differences,queries=4,differences=differences,
    scope='all audit fields, input buffers, histories, statistics, speculation, full finite non-timing predictions and XYZ/rotation bits')
