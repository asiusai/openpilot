"""Independent saved-live and exact PV composition admission."""
from functools import lru_cache
import hashlib
import importlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
from types import CodeType

WORK = Path('/mnt/nvme-500/tmp/alpamayo-speed-20261116')
EXP = Path('/tmp/opencode/openpilot-alpamayo/openpilot/tools/alpamayo/experiments')
ROOT = WORK/'agent-expert-live-source'
PROTOCOL = WORK/'agent-expert-live-protocol.json'
ARTIFACTS = WORK/'agent-expert-live-artifacts'
LIVE = WORK/'agent-paced-dynamic-source-cpu02/alpamayo'
MANIFEST = EXP/'agent-exact-live-protocol-v1/manifest.json'
MANIFEST_SHA = '6b0eb22cf950b77ec921d39175694b259b7657acd2c7e33c5e1b6622cc9150de'
REFERENCE = WORK/'agent-exact-live-reference/report.json'
REFERENCE_SHA = '20b4c093f7c4e2fe77807471fc9c9f83e977e57d99354930807d9a69c8f15c6c'
CONSUMER = WORK/'agent-exact-live-consumer/report.json'
CONSUMER_SHA = '1dc782153a1a3d4691564d57830eb3920ff8594dc082989701cee4e9841240e5'
PACED = WORK/'agent-exact-live-paced/report.json'
PACED_SHA = 'f576f750210ed77bdc89aeb035e9309abeb594417546eb39e78173aa78e48291'
PV_ROOT = WORK/'agent-expert-pv-qualified-source'
PV_PROTOCOL = WORK/'agent-expert-pv-qualified-protocol.json'
PV_SHA = '658bcb7d828a975e69f55cdf221a3308d30ce6f2a499a4ecc5ed53c1bee59a42'
CACHE_ROOT = WORK/'agent-expert-combined-cache-source'
CACHE_PROTOCOL = WORK/'agent-expert-combined-cache-protocol.json'
CACHE_SHA = '723432734e33005b7a16c6756cb5ee64d2b88b692520ed9b63be6395537172fd'
WEIGHTS = WORK/'agent-expert-combined-cache-artifacts/fp8-cache'
TINY = Path('/tmp/opencode/alpamayo-asius-20261114/tinygrad')
PY = '/tmp/opencode/alpamayo-asius-20261114/upstream/.venv/bin/python'
WRAPPER = Path('/tmp/opencode/run-alpamayo-speed-trial.py')
COMPOSED = {'quality-reference':'accd373fd69d8ec03ddf7525257b11ed915831121a8ba5cfdc26d0aa76c4944d',
 'quality-candidate':'877aa639fc9e9327375306eda5671a469948d1e7b107acc6baf1afe2e28cc43c',
 'native-reference':'ac3800eaf82939780fef6f951ca334352a42aa93b996465c655f36b1f6c8b7bc',
 'native-candidate':'1be981a6770543a74a72d0fac955353cf2d11f829cf7da7ce020149bbee63142'}
CORE = {'model.py':'fce24a07fdd48bd24c05e67518c986bcb462d09dfb75b47daf4140dc4e67f5fb',
 'amd_attention.py':'fa9d2f01c6557c97541f8a62e4e02ee7cf2608584787cad7d42b9922a00866f4',
 'experiments/expert-kernels.py':'4676128ef21bd42f851311b3b5dba0563ab6cd1e54d367a3bca2e377f5a53b8b',
 'experiments/wgp_compiler.py':'ccd7427848f13a8032e8af3c305fc8b2db3150242b3ff67a63f25acd52da8890'}


def sha(path):
  with Path(path).open('rb') as stream: return hashlib.file_digest(stream,'sha256').hexdigest()
def read(path): return json.loads(Path(path).read_text())
def write(path,value,*,new=False):
  path = Path(path); path.parent.mkdir(parents=True,exist_ok=True)
  with path.open('x' if new else 'w') as stream: stream.write(json.dumps(value,indent=2,allow_nan=False)+'\n')
def load(path,name):
  spec = importlib.util.spec_from_file_location(name,path)
  module = importlib.util.module_from_spec(spec); sys.modules[name] = module; spec.loader.exec_module(module)
  return module


def reserve(path,upcoming=64*1024**2):
  path = Path(path).resolve(); mount = Path('/mnt/nvme-500')
  assert mount.is_mount() and path.is_relative_to(WORK)
  existing = path
  while not existing.exists(): existing = existing.parent
  assert existing.stat().st_dev == mount.stat().st_dev != Path('/').stat().st_dev
  usage = shutil.disk_usage(existing)
  assert usage.free-upcoming >= max(usage.total*.1,5*1024**3), 'Retain original 10% / 5 GiB reserve'
  return {'path':str(path),'device':existing.stat().st_dev,'root_device':Path('/').stat().st_dev,
    'total_bytes':usage.total,'free_bytes':usage.free,'upcoming_bytes':upcoming,'reserve_passed':True}


def storage():
  return {'CACHEDB':str(ARTIFACTS/'compiler-cache.db'),'TMPDIR':str(ARTIFACTS/'tmp'),
    'XDG_CACHE_HOME':str(ARTIFACTS/'cache'),'HF_HOME':str(ARTIFACTS/'cache/huggingface'),
    'TORCH_HOME':str(ARTIFACTS/'cache/torch'),'ALPAMAYO_FP8_CACHE':str(WEIGHTS)}


def configure():
  proof = reserve(ARTIFACTS)
  for key,value in storage().items():
    if key not in ('CACHEDB','ALPAMAYO_FP8_CACHE'): Path(value).mkdir(parents=True,exist_ok=True)
  os.environ.update(storage(),PYTHONDONTWRITEBYTECODE='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false')
  tempfile.tempdir = None
  if 'tinygrad.helpers' in sys.modules:
    assert sys.modules['tinygrad.helpers'].CACHEDB == storage()['CACHEDB'], 'Set storage before tinygrad import'
  return proof


def paths():
  old = WORK/'agent-expert-pv-route-source/alpamayo'
  for rel,digest in CORE.items(): assert sha(LIVE/rel) == sha(old/rel) == digest
  sys.path[:0] = [str(TINY),str(LIVE/'experiments'),str(LIVE),'/tmp/opencode/openpilot-alpamayo','/home/john/asius/openpilot/opendbc_repo']
  for name in ('amd_attention','amd_gemm','wgp_compiler'):
    module = importlib.import_module(name)
    expected = LIVE/('experiments/wgp_compiler.py' if name == 'wgp_compiler' else name+'.py')
    assert Path(module.__file__).resolve() == expected, 'Foreign numerical import: '+name
  module = sys.modules['wgp_compiler']
  def codes(code):
    yield code
    for child in code.co_consts:
      if isinstance(child,CodeType): yield from codes(child)
  for name in ('__init__','__reduce__'):
    fn = getattr(module.WGPCompiler,name)
    expected = next(c for c in codes(compile(Path(module.__file__).read_bytes(),fn.__code__.co_filename,'exec')) if c.co_qualname == fn.__code__.co_qualname)
    assert fn.__code__ == expected
  return module


@lru_cache(None)
def live():
  paths()
  assert sha(MANIFEST) == MANIFEST_SHA
  manifest = read(MANIFEST)
  for name in ('common','install','run'):
    path = EXP/f'agent-exact-live-{name}.py'; assert sha(path) == manifest['files'][str(path)]
  m = load(EXP/'agent-exact-live-common.py','expert_live_original_common')
  m.verify(manifest)
  # Byte-identical installer copy and sampler artifacts reside on the 500 GB drive.
  m.HERE = ARTIFACTS/'hooks'
  assert sha(m.HERE/'agent-exact-combined-install.py') == manifest['files'][str(EXP/'agent-exact-combined-install.py')]
  return m,manifest


@lru_cache(None)
def pv():
  paths()
  assert sha(PV_PROTOCOL) == PV_SHA
  for path,digest in read(PV_PROTOCOL)['files'].items(): assert sha(path) == digest
  P = load(PV_ROOT/'agent-expert-pv-qualified-common.py','expert_live_pv_common')
  _,evidence = P.protocol(PV_PROTOCOL,PV_SHA)
  P.SNAP,P.paths = LIVE,paths  # Explicit independent byte-identity relocation.
  paths()
  return P,evidence


def validate_pv(info,candidate,queries,evidence):
  R = load(WORK/'agent-expert-combined-source/agent-expert-combined-run.py','expert_live_qualified_pv_validator')
  R.validate_pv(info,candidate,evidence,queries)
  for cap in info['captures']:
    for name,digest in {'alpamayo_attention_scores':'7b216e173c90c31e1be68cd78f1d2dfc8a8ee2b9b59be59dff3a23f5d6735110',
      'alpamayo_attention_softmax':'36a2f4de58bbae9c26029990d4af467467c0be3fff41501a1a207bf9dace692d'}.items():
      assert cap['binaries'][name] == [digest], 'Non-PV program changed'


def evidence():
  m,manifest = live(); P,binding = pv()
  assert sha(REFERENCE) == REFERENCE_SHA and sha(CONSUMER) == CONSUMER_SHA
  assert sha(PACED) == PACED_SHA
  reference,consumer = read(REFERENCE),read(CONSUMER)
  previous = m.qualify(reference,consumer,manifest,MANIFEST_SHA)
  for report in (reference,consumer):
    path = REFERENCE if report['mode'] == 'reference' else CONSUMER
    m.equal(report['agent_exact_live'],read(path.with_suffix('.agent-exact-live.json')),'prior.live.sidecar')
    wall = read(path.parent/'wall.json'); assert wall['returncode'] == 0 and not wall['timed_out']
  assert sha(CACHE_PROTOCOL) == CACHE_SHA
  cache = read(CACHE_PROTOCOL)
  # Keep all original source and report pins; no reconstruction or restamping.
  for path,digest in cache['files'].items(): assert sha(path) == digest,path
  summaries = {}
  for key,digest in COMPOSED.items():
    path = WORK/f'agent-expert-combined-cache-{key}/report.json'; assert sha(path) == digest
    result = read(path); status = result['agent_expert_combined']; candidate = key.endswith('candidate')
    assert status == read(path.with_suffix('.agent-expert-combined.json'))
    assert status['protocol_sha256'] == CACHE_SHA and status['passed'] and status['coverage_validated'] and status['hooks_restored']
    count = 4 if key.startswith('quality') else 36
    assert len(result['warmup'])+len(result['runs']) == count
    validate_pv(status['pv'],candidate,count,binding)
    if candidate: assert status['paired_exact']['all_exact'] and status['paired_exact']['samples'] == count
    summaries[key] = {'sha256':digest,'queries':count,'candidate':candidate,'pv_program':binding['candidate' if candidate else 'current']['program_key']}
  source = {p.name:sha(p) for p in [*(LIVE/'experiments').glob('*.py'),*(LIVE/'experiments').glob('*.c'),LIVE/'model.py']}
  assert source == reference['initialized']['source_sha256'] == consumer['initialized']['source_sha256']
  assert source == reference['source_sha256'] == consumer['source_sha256']
  return {'saved_pair':previous,'reference_sha256':REFERENCE_SHA,'consumer_sha256':CONSUMER_SHA,
    'live_manifest_sha256':MANIFEST_SHA,'composed_cache_protocol_sha256':CACHE_SHA,'composed':summaries,
    'pv':binding,'core_relocations':CORE,'source_set_identical_to_saved_pair':True}


def validate_startup(report,*,paced=False):
  m,_ = live(); old = read(PACED if paced else CONSUMER)['startup_spec']; warm = report['startup_spec']
  for key in ('ready','full_kv_bits_restored','actual_first_logits_bits_restored','target_payload_bits_restored','warmed_owner_retained'):
    assert warm[key] is True,key
  for key in ('source_sha256','decoder_plan','graph_counts','snapshot_bytes','current_inputs','future_inputs','predictions_published',
              'forced_tokens','memory_limit_bytes','stage_limit_seconds','accepted_suffix_rows'):
    m.equal(warm[key],old[key],'full_startup.'+key)
  assert [s['name'] for s in warm['stages']] == [s['name'] for s in old['stages']], 'Full original startup stages required'


def qualify(candidate,data,digest):
  m,manifest = live(); reference,old = read(REFERENCE),read(CONSUMER)
  comparison = m.compare(reference,candidate)
  assert candidate['comparison'] == comparison and candidate['predictions_qualified'] and candidate['gpu_executed']
  status = candidate['agent_expert_live']
  assert status['protocol_sha256'] == digest and status['mode'] == 'consumer' and status['passed'] and status['hooks_restored']
  assert status['reference_sha256'] == REFERENCE_SHA and status['saved_reference_admission'] == data['reference_admission']
  assert [r['sample'] for r in candidate['rows']] == manifest['selected_samples']
  assert [r['speculation']['accepted'] for r in candidate['rows']] == [True,False]
  assert candidate['rows'][0]['history_before'] == {'history':[],'prefill_draft_old_history':[]}
  validate_startup(candidate)
  for i,(before,after) in enumerate(zip(old['rows'],candidate['rows'],strict=True)):
    m.equal(before['consumer_protocol'],after['consumer_protocol'],f'owner.{i}')
    assert len(after['defined_kv_prefix_bits']) == 28
  m.equal(candidate['source_sha256'],old['source_sha256'],'all_model_sources')
  m.equal(candidate['consumer_provenance'],old['consumer_provenance'],'consumer_implementation_and_ownership_policy')
  assert status['transforms'] == {k:v for k,v in manifest['expected_transforms'].items() if k != 'paced'}
  validate_pv(status['pv'],True,5,data['evidence']['pv'])  # Three original startup predictions plus both audited queries.
  assert status['coverage']['consumer_bindings'] == 1 and status['coverage']['live_sampler_calls'] == 0
  return comparison


def verify(path,digest):
  assert sha(path) == digest
  data = read(path)
  assert data['base_manifest_sha256'] == MANIFEST_SHA and data['reference_sha256'] == REFERENCE_SHA
  for file,expected in data['files'].items(): assert sha(file) == expected,file
  m,manifest = live(); m.verify(manifest)
  m.equal(evidence(),data['evidence'],'independent_qualifications')
  assert data['pace'] == manifest['pace'] and not data['automatic_paced']
  assert data['reference_admission']['source_protocol_inputs_order_empty_history_and_ownership_proven']
  paths()
  return data
