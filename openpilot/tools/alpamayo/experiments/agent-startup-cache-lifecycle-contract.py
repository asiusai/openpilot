"""Source-bound overlay evidence and strict reuse of the completed ordinary arm."""
import importlib.util
from pathlib import Path
import sys

EXP=Path(__file__).resolve().parent;WORK=Path('/mnt/nvme-500/tmp/alpamayo-speed-20261116')
BASE=WORK/'agent-startup-persist-protocol2.json'
REFERENCE=WORK/'agent-startup-persist-trial1-ordinary/report.json'
SOURCES=tuple('agent-startup-cache-lifecycle-'+s+'.py' for s in ('overlay','contract','run','check','parent'))
PINS={str(BASE):'42adfb2ca105a0c52f72eb8aabe1ddea3011c13365479bb05414fc8e6db9bc02',
 str(REFERENCE):'90c4ce7ae9826a9701af2784a4c0eabb7abfcbda13bab9cf81f49ddac8491a39',
 str(REFERENCE.with_suffix('.startup-persist.json')):'bedcc46f9a9d438ad0240af3c26720bc65b4326824ecd5a433a1f01100781b90',
 str(WORK/'agent-startup-persist-trial1-persisted/report.startup-persist.json'):'c09a2bb5aa3cb4d829d819cea8df6cb92940e01950a78814811e3dc73259f8f1',
 str(WORK/'agent-startup-persist-trial1-storage/ordinary/parent-command.json'):'98b5743960b26c34426741996fddb7f2e47e17ef70f9935c81a4955055f7119d',
 str(WORK/'agent-startup-persist-trial1-persisted/run.log'):'a0ea4f9441855a719ba5acda6a19293055b2c3133881aedc3c0507d2fc5189f2'}

def local(path):
  name='startup_lifecycle_'+Path(path).stem.replace('-','_');spec=importlib.util.spec_from_file_location(name,path)
  module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module);return module
m=local(EXP/'agent-startup-persist-common.py')

def sources(): return {str(EXP/name):m.sha(EXP/name) for name in SOURCES}

def reuse():
  for p,h in PINS.items():
    if m.sha(p)!=h: raise ValueError('Immutable baseline/failure evidence changed: '+p)
  m.verify(m.read(BASE));r=m.read(REFERENCE);s=r['startup_persist'];side=m.read(REFERENCE.with_suffix('.startup-persist.json'))
  if s!=side['status']: raise ValueError('Ordinary report and completion sidecar differ')
  if any(s.get(k) is not True for k in ('completed','gpu_executed','source_before_after_exact','hooks_restored','live_plan_signatures_unchanged')):
    raise ValueError('Ordinary arm incomplete')
  if s['manifest_sha256']!=PINS[str(BASE)] or s['role']!='ordinary' or s['factory_loads']!=1 or s['consumer_bindings']!=1:
    raise ValueError('Ordinary protocol, load count, or role changed')
  if r['mode']!='consumer' or [v['speculation']['accepted'] for v in r['rows']]!=[True,False]: raise ValueError('Require actual accepted/fallback pair')
  if [v['sample'] for v in r['rows']]!=m.read(BASE)['selected_samples']: raise ValueError('Original selected order differs')
  if r['rows'][0]['history_before']!={'history':[],'prefill_draft_old_history':[]}: raise ValueError('Ordinary initial history differs')
  for key in ('ready','full_kv_bits_restored','actual_first_logits_bits_restored','target_payload_bits_restored','warmed_owner_retained'):
    if r['startup_spec'].get(key) is not True: raise ValueError('Ordinary restoration incomplete')
  for row in r['rows']:
    if not row['completed'] or not row['consumer_protocol']['input_reused'] or len(row['defined_kv_prefix_bits'])!=28:
      raise ValueError('Ordinary complete KV/owner coverage missing')
  m.projected_provenance(r,REFERENCE)
  return dict(reference=str(REFERENCE),sha256=PINS[str(REFERENCE)],base_sha256=PINS[str(BASE)],
    source_unchanged=True,ordinary_branch_untouched=True,accepted_and_fallback=True,all28_prefixes=True,
    use='Exactness baseline only; noncontemporaneous timing is informational',fresh_ordinary_execution=False)

def gate(protocol):
  if protocol['schema']!=1 or protocol['base']!={'path':str(BASE),'sha256':PINS[str(BASE)]}: raise ValueError('Overlay base changed')
  if protocol['sources']!=sources() or protocol['pins']!=PINS: raise ValueError('Overlay source/evidence changed')
  m.equal(reuse(),protocol['ordinary_reuse'],'ordinary reuse')
  if protocol['child_seconds']!=450 or protocol['outer_timeout_seconds']!=0 or protocol['role']!='persisted': raise ValueError('Retry scope changed')
  if len(protocol['cpu_evidence'])!=1: raise ValueError('One final CPU evidence record required')
  for p,h in protocol['cpu_evidence'].items():
    if m.sha(p)!=h: raise ValueError('CPU evidence changed')
    checked=m.read(p)
    if not checked['passed'] or checked['sources']!=sources() or checked['gpu_executed']: raise ValueError('CPU validation incomplete')

def compare(protocol,candidate,protocol_sha256):
  gate(protocol);c=m.read(candidate);side=m.read(Path(candidate).with_suffix('.cache-lifecycle.json'))
  if not side['completed'] or not side['source_before_after_exact'] or not side['hooks_restored']:
    raise ValueError('Lifecycle retry did not complete')
  if (side['protocol_sha256']!=protocol_sha256 or side['sources']!=sources() or side['base_sha256']!=PINS[str(BASE)] or
      side['lifecycle']['KV_bindings']!=[1]):
    raise ValueError('Lifecycle source or binding scope differs')
  if c['startup_persist']['manifest_sha256']!=PINS[str(BASE)]: raise ValueError('Underlying persistence protocol changed')
  result=m.compare(m.read(REFERENCE),c,REFERENCE,candidate)
  result.update(ordinary_reused_sha256=PINS[str(REFERENCE)],lifecycle_source_bound=True,
    startup_speed_claim=False,full64_still_required=True,ready_for_live_use=False)
  return result
