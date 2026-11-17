"""Source gates and two-query parity for the qualified exact live composition."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
WORK = Path('/mnt/nvme-500/tmp/alpamayo-speed-20261116')
ROOT = WORK/'agent-paced-dynamic-source-cpu02/alpamayo'
STARTUP = WORK/'startup-spec-frozen-source'
COMBINED = HERE/'agent-exact-combined-protocol-v1/manifest.json'
INVOCATION = ROOT.parent/'pipeline-invocation.json'
RGB = WORK/'agent-paced-replay-cache-35-43/rgb'
INDICES = (0,16)


def load(path, name):
  spec = importlib.util.spec_from_file_location(name, path)
  module = importlib.util.module_from_spec(spec)
  sys.modules[name] = module
  spec.loader.exec_module(module)
  return module


c = load(HERE/'agent-exact-combined-common.py', 'exact_live_combined_common')
sha, read, equal, write_new = c.sha, c.read, c.equal, c.write_new


def evidence():
  c.verify(read(COMBINED))
  for stage in ('quality','native'):
    report = read(WORK/f'agent-exact-combined-{stage}-candidate/report.agent-exact-combined.json')
    if not all(report[k] for k in ('passed','hooks_restored','combined_gpu_qualified')):
      raise ValueError('Require completed exact combined GPU qualification')
    expected = 64 if stage == 'quality' else 36
    if report['paired_exact']['predictions'] != expected or not report['paired_exact']['all_prediction_float_bits']:
      raise ValueError('Combined prediction coverage changed')
  policy = load(ROOT/'experiments/agent-paced-dynamic-policy.py', 'exact_live_original_policy')
  policy.verify_snapshot()
  startup = load(STARTUP/'agent-startup-spec-run.py', 'exact_live_original_startup')
  old = read(WORK/'agent-startup-spec-candidate/report.json')
  proof = startup.qualify(read(WORK/'agent-startup-spec-reference/report.json'), old, policy)
  if old['rows'][0]['history_after'] != old['rows'][16]['history_before']:
    raise ValueError('Selected accepted/fallback history boundary changed')
  if [old['rows'][i]['speculation']['accepted'] for i in INDICES] != [True,False]:
    raise ValueError('Predeclared acceptance and fallback changed')
  return dict(combined_manifest_sha256=sha(COMBINED), original_startup=proof,
              original_startup_sources=startup.own_sources(), indices=list(INDICES))


def verify(manifest):
  if manifest['schema'] != 1: raise ValueError('Unsupported exact live manifest')
  for path, digest in manifest['files'].items():
    if sha(path) != digest: raise ValueError(f'Frozen exact live bytes changed: {path}')
  equal(evidence(), manifest['evidence'], 'qualification_evidence')
  for entry in manifest['identical_foundations']:
    if sha(entry['live']) != entry['sha256'] or sha(entry['qualified']) != entry['sha256']:
      raise ValueError('Qualified foundation relocation lost byte identity')


def selected(full):
  result = copy.copy(full)
  result.rows = [full.rows[i] for i in INDICES]
  result.samples = [full.samples[i] for i in INDICES]
  result.inputs = [full.inputs[i] for i in INDICES]
  result.provenance = dict(full.provenance, count=2, original_count=64, selected_indices=list(INDICES),
    inputs_sha256=hashlib.sha256(json.dumps(result.inputs,sort_keys=True,allow_nan=False).encode()).hexdigest(),
    metadata_sha256=hashlib.sha256(json.dumps(result.rows,sort_keys=True,allow_nan=False).encode()).hexdigest(),
    history='Fresh empty lookup history; accepted row zero followed by original fallback row sixteen')
  return result


def compare(reference, candidate):
  for key in ('protocol','decoder','prefill_attention','initialized'):
    equal(reference[key], candidate[key], key)
  if reference['mode'] != 'reference' or candidate['mode'] != 'consumer': raise ValueError('Wrong paired roles')
  if len(reference['rows']) != 2 or len(candidate['rows']) != 2: raise ValueError('Require both predeclared queries')
  for i,(old,new) in enumerate(zip(reference['rows'],candidate['rows'],strict=True)):
    if not old['completed'] or not new['completed']: raise ValueError('Missing completion barrier')
    a,b = (copy.deepcopy(row) for row in (old,new))
    a.pop('consumer_protocol',None); b.pop('consumer_protocol',None)
    a['prediction'].pop('timings_ms',None); b['prediction'].pop('timings_ms',None)
    equal(a,b,f'query.{i}')
  return dict(exact=True,queries=2,differences=[],all_prediction_float_bits=True,
    scope='All audited inputs, token buffers, lookup history, defined KV prefix bytes, outputs, and speculative records')


def qualify(reference, candidate, manifest, manifest_sha):
  result = compare(reference,candidate)
  if not candidate['predictions_qualified'] or candidate['comparison'] != result:
    raise ValueError('Consumer parity did not pass its complete comparison')
  for value,role in ((reference,'reference'),(candidate,'consumer')):
    status = value['agent_exact_live']
    if not status['passed'] or status['mode'] != role or status['manifest_sha256'] != manifest_sha:
      raise ValueError('Fresh pair has different source or role')
    if not status['hooks_restored'] or not value['gpu_executed']: raise ValueError('Incomplete fresh GPU pair')
  equal([r['sample'] for r in candidate['rows']],manifest['selected_samples'],'selected_order')
  equal([r['speculation']['accepted'] for r in candidate['rows']],[True,False],'accepted_and_fallback')
  warm = candidate['startup_spec']
  for key in ('ready','full_kv_bits_restored','actual_first_logits_bits_restored','target_payload_bits_restored','warmed_owner_retained'):
    if warm.get(key) is not True: raise ValueError(f'Missing original startup proof: {key}')
  if candidate['rows'][0]['consumer_protocol']['input_reused'] is not True: raise ValueError('Warmed input owner was replaced')
  if candidate['rows'][0]['history_before'] != {'history':[],'prefill_draft_old_history':[]}:
    raise ValueError('Fresh parity did not start with empty lookup history')
  return result
