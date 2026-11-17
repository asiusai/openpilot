"""Independent exact admission for ONE nearest recorded paced replay. CPU by default."""
import hashlib, importlib.util, json, os, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
WORK = Path('/mnt/nvme-500/tmp/alpamayo-speed-20261116')
REFERENCE = HERE/'agent-nearest-order-parent-reference/report.json'
PARENT_REFERENCE = WORK/'agent-nearest-order-reference4/report.json'
PROTOCOL = HERE/'agent-nearest-paced-protocol.json'
PINS = {
  HERE/'agent-nearest-order-common.py': 'de7a1f7c017c58d9b3178e10e86b75bcb0c903f20d93f53347687b3ae4ab4144',
  HERE/'agent-nearest-order-run.py': 'be021b264e0d999d555c2b92365d496264ae74bb5f62185d56ec803d07abe006',
  HERE/'agent-nearest-order-protocol.json': '88dd9b7ae6104f4be4e8ad51e0fa72af55be37f052733ccd928e769b08305f33',
  REFERENCE: '1ba713b7c3893aedb79a1c492164737bc31686e21422ff3cb2c65ff010012596',
  PARENT_REFERENCE: '6132e74c97bbf8206e36ebc41eaa67c806517463cab14e0b0686401588cdcab6',
  Path('/tmp/opencode/run-alpamayo-nearest-order.py'): '6f0b6bde89f2903d12271cbe0380aa37201e9305ac3db4867b08c4c9f07d9a34',
  WORK/'agent-paced-replay-cache-35-43/paced-cache.json': 'd0a26d5bf9ec0c9b455b974a0669ac38a63ea5249db4adb29dd40bd4487429df',
  WORK/'agent-startup-spec-paced/report.json': '2fcf7c3a08736843947c1827670f4061123d976bd78b9fddf3d37bd572d9865e',
}
sys.dont_write_bytecode = True

def sha(path):
  with Path(path).open('rb') as stream: return hashlib.file_digest(stream, 'sha256').hexdigest()

def read(path): return json.loads(Path(path).read_text())
def save(path, value):
  with Path(path).open('x') as stream: stream.write(json.dumps(value, indent=2, allow_nan=False)+'\n')
def load(path):
  spec = importlib.util.spec_from_file_location('nearest_paced_'+Path(path).stem.replace('-', '_'), path)
  value = importlib.util.module_from_spec(spec); sys.modules[spec.name] = value; spec.loader.exec_module(value)
  return value
def sources(): return {p.name: sha(p) for p in sorted(HERE.glob('agent-nearest-paced-*.py'))}
def setup_cpu():
  os.environ.update(DEV='CPU', CUDA_VISIBLE_DEVICES='', HIP_VISIBLE_DEVICES='', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
    TOKENIZERS_PARALLELISM='false', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
    CACHEDB=str(HERE/'agent-nearest-paced-cpu-cache.db'), XDG_CACHE_HOME=str(HERE/'agent-nearest-paced-cpu-cache'))
  sys.path[:0] = ['/tmp/opencode/alpamayo-asius-20261114/tinygrad', '/tmp/opencode/openpilot-alpamayo',
    '/home/john/asius/openpilot/opendbc_repo']

def check_pins(pins):
  for path, expected in pins.items():
    if sha(path) != expected: raise ValueError('Pinned evidence/source changed: '+str(path))

def pair(c, plan, startup, protocol, reference, consumer):
  """Recompute the exact gate; stored success booleans never replace comparisons."""
  c.validate_saved(plan, startup, consumer)  # Deliberately still requires the OLD failed status.
  if (reference.get('mode') != 'nearest-order-canonical-reference4' or reference.get('gpu_executed') is not True or
      reference.get('queries') != 4 or reference.get('same_order_exact') is not True or
      reference.get('ownership_exact') is not True or reference.get('old_comparison_still_failed') is not True or
      reference.get('automatic_pace') is not False or reference.get('automatic_adoption') is not False or
      reference.get('conditioned_on_saved_outputs') is not False or 'error' in reference):
    raise ValueError('Completed independent real-GPU same-order reference required')
  if (reference['source_sha256'] != c.own_sources() or reference['protocol_sha256'] != sha(c.PROTOCOL) or
      reference['saved_consumer_sha256'] != sha(c.SAVED) or reference['canonical_main_transform'] != protocol['canonical_main_transform'] or
      reference['loop_and_cache_location_changes'] != protocol['bounded_recipe_changes']):
    raise ValueError('Canonical reference source/protocol changed')
  if reference['initialized'] != consumer['initialized'] or reference['qualified_factory_protocol'] != consumer['qualified_factory_protocol']:
    raise ValueError('Initial source/settings/environment/factory mismatch')
  rw, cw = reference['startup_spec'], consumer['startup_spec']
  if not all(rw.get(key) is True for key in c.RESTORE): raise ValueError('Reference startup restoration failed')
  for key in ('source_sha256', 'graph_counts', 'decoder_plan', 'snapshot_bytes', 'actual_prompt', 'accepted_suffix_rows'):
    if rw[key] != cw[key]: raise ValueError('Initial warmed-state contract differs: '+key)
  if (rw.get('current_inputs') != 1 or rw.get('future_inputs') != 0 or rw.get('predictions_published') != 0 or
      rw.get('forced_tokens') != 0 or rw.get('initialization_only') is not True):
    raise ValueError('Reference startup used future inputs or published/forced outputs')
  comparison = c.compare_rows(reference['rows'], consumer['rows'])
  if comparison != reference['comparison'] or not comparison['exact']: raise ValueError('Independent exact row comparison failed')
  if reference['reference_ownership'] != [r['consumer_protocol'] for r in consumer['rows']]:
    raise ValueError('Per-query ownership/input/cache bounds differ')
  return dict(exact=True, queries=4, accepted=2, fallback=2, comparison=comparison, ownership_exact=True,
    original_failed_status_preserved=True, initial_state_matched=True, startup_restoration_exact=True,
    saved_per_query_kv_bytes_available=False,
    kv_scope='Existing full startup bit restoration and per-query KV allocation ownership; no invented cross-run KV dumps')

def verify():
  check_pins(PINS)
  c = load(HERE/'agent-nearest-order-common.py'); order = load(HERE/'agent-nearest-order-run.py')
  plan, policy, startup, dynamic = c.verify_assets(); protocol = order.verify_protocol()
  reference, consumer, parent = read(REFERENCE), read(c.SAVED), read(PARENT_REFERENCE)
  bridge = parent.pop('parent_bridge')
  if parent != reference or bridge != dict(original_report=str(REFERENCE), original_sha256=sha(REFERENCE),
      adapter_sha256=PINS[Path('/tmp/opencode/run-alpamayo-nearest-order.py')], unchanged_original_gates=True):
    raise ValueError('Parent copy no longer matches original reference and bridge provenance')
  proof = pair(c, plan, startup, protocol, reference, consumer)
  inventory = read(Path(plan['raw_frames']).parent/'paced-cache.json')
  if inventory['frames'] != 450 or len(inventory['files']) != 450: raise ValueError('Require the existing450 RGB cache')
  if inventory['trace']['source_end_ns']-inventory['trace']['source_start_ns'] != 8_000_000_000:
    raise ValueError('Wrong raw-cache source window')
  admission = dict(schema=1, mode='independent-nearest-same-order-admission', gpu_evidence_recomputed=True,
    canonical_reference=dict(path=str(REFERENCE), sha256=sha(REFERENCE)),
    parent_reference=dict(path=str(PARENT_REFERENCE), sha256=sha(PARENT_REFERENCE)),
    saved_consumer=dict(path=str(c.SAVED), sha256=sha(c.SAVED)),
    original_order_protocol_sha256=sha(c.PROTOCOL), nearest_plan_sha256=sha(c.LIVE_PLAN),
    nearest_policy_sha256=sha(c.FROZEN/'agent-nearest-live-policy.py'), startup_sources=startup.own_sources(),
    startup64_qualification=plan['startup_qualification'], proof=proof,
    nearest_policy=policy.POLICY, raw_inventory_sha256=sha(Path(plan['raw_frames']).parent/'paced-cache.json'),
    scope='Admission to one recorded35..43s replay only; no general trajectory-quality, device, or sub100ms qualification')
  return c, plan, policy, startup, dynamic, protocol, reference, consumer, admission
