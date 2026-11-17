"""Parent-only exact-key repair composed with the unchanged frozen startup.

--audit-only stops after full current-input startup and strict plan comparison.
Without that flag, the unchanged original Consumer runs the full64 parity gate.
Neither mode qualifies the old CU-mode candidate or enables paced execution.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parent
WORK=Path('/mnt/nvme-500/tmp/alpamayo-speed-20261116')
FROZEN=WORK/'startup-cache-parent-source'
TG=Path('/tmp/opencode/alpamayo-asius-20261114/tinygrad')
FROZEN_PINS={
 'agent-startup-cache-capture.py':'be34672068c417ccff1cc4bfb139edc26b32ae57f0b466be7e32ad18fef8f62b',
 'agent-startup-cache-run.py':'9249c75cd72b780b592117b7fd3aed3494990370de0c8a0bdfc26f451d43b3f9',
 'agent-startup-spec-run.py':'7e22277322166f6938f28c700ade7ac62196260c3310d90d92f6c59ba15c36f0',
 'agent-startup-spec-warm.py':'3b7f2d7d5e1888243271fa34b7bb8f91d7b7f7c5dd3789503bec1071a3e2900c',
}
TG_PINS={
 'tinygrad/codegen/__init__.py':'703eeb27b62c2d254a05a32580d04cc8e82c709efa637616d1bc470dfb0d6fbb',
 'tinygrad/renderer/__init__.py':'2c8bebcc45bee76543bc7ce91c0246f841ed3d900aec703853cc6915a4a9ab9f',
 'tinygrad/renderer/llvmir.py':'b10ca67a45991a642aaec705877ef6adc6da9e40b51d1ffdbfc7830675fed75d',
 'tinygrad/engine/worker.py':'8be2d2199e3d5eee3f6d907281ce97791b7d644e218f1a22b05a428094ab65c6',
 'tinygrad/uop/ops.py':'357ce1fd8b9cc3d04789aacbed3480944aeb11e8b280c5fd80cf57839a8cb16d',
 'tinygrad/schedule/__init__.py':'d8a492fd742e4d201486e34c208acfb73cb7ee4a88feb7cd2a8d84944c111d42',
 'tinygrad/runtime/support/compiler_llvm.py':'78ba30c79a1e0b93a189dfc4023db2f1fec0accaa26ad72310b9571287b988a2',
}


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def local(path):
  spec=importlib.util.spec_from_file_location(Path(path).stem.replace('-','_'),path)
  value=importlib.util.module_from_spec(spec);sys.modules[spec.name]=value;spec.loader.exec_module(value)
  return value


def sources(): return {name:sha(ROOT/name) for name in ('agent-startup-key-run.py','agent-startup-key-prime.py')}


def validate():
  for root,pins in ((FROZEN,FROZEN_PINS),(TG,TG_PINS)):
    for name,expected in pins.items():
      if sha(root/name)!=expected: raise ValueError('Reviewed source changed: '+str(root/name))
  frozen=local(FROZEN/'agent-startup-cache-run.py')
  prototype=local(FROZEN/'agent-startup-cache-capture.py')
  frozen.validate_pins(prototype)
  return frozen,prototype


class DiagnosticReady(BaseException): pass


def initialized_renderer():
  from tinygrad import Device
  # The factory must own the existing device. This helper never initializes one.
  name=Device.DEFAULT
  if name not in Device._opened_devices: raise ValueError('The original factory has not initialized its device')
  return Device[name].renderer


def check_startup(startup):
  fields=('ready','full_kv_bits_restored','actual_first_logits_bits_restored','target_payload_bits_restored','warmed_owner_retained')
  if not all(startup.get(field) is True for field in fields):
    raise AssertionError('Original current-input/KV/payload startup proof is incomplete')


def main(argv=None):
  parser=argparse.ArgumentParser(add_help=False)
  parser.add_argument('--audit-only',action='store_true')
  parser.add_argument('--ordinary-report',type=Path,default=WORK/'agent-startup-cache-trial1-ordinary/report.json')
  parser.add_argument('--key-proof',type=Path,default=WORK/'agent-startup-key-proof.json')
  cli,options=parser.parse_known_args(argv)
  sys.dont_write_bytecode=True
  def option(name,default=None): return options[options.index(name)+1] if name in options else default
  if option('--task')!='replay' or option('--mode')!='consumer':
    raise ValueError('This reviewed helper only supports current-input startup audit or full64 Consumer replay')
  if '--execute-gpu' not in options:
    raise ValueError('Only the parent may execute this helper, with explicit --execute-gpu')
  output=Path(option('--output'))
  if output.exists(): raise FileExistsError(output)
  frozen,prototype=validate()
  proof=json.loads(cli.key_proof.read_text())
  if proof.get('exact_actual_program_key_reproductions')!=10 or proof.get('old_strict_rejection_retained') is not True:
    raise ValueError('Require the completed CPU program reconstruction')
  primer_sources=[value for path,value in proof['sources'].items() if Path(path).name=='agent-startup-key-prime.py']
  if primer_sources!=[sha(ROOT/'agent-startup-key-prime.py')]:
    raise ValueError('The primer differs from its CPU proof')
  if proof['reports'].get(str(cli.ordinary_report))!=sha(cli.ordinary_report):
    raise ValueError('Ordinary GPU evidence differs from the CPU proof')
  ordinary=json.loads(cli.ordinary_report.read_text())
  if ordinary.get('startup_cache_sources')!=frozen.sources(prototype):
    raise ValueError('Ordinary GPU evidence does not use the exact frozen implementation')
  helper=local(ROOT/'agent-startup-key-prime.py')
  base=local(FROZEN/'agent-startup-spec-run.py')
  original_local=base.local
  ledger=dict(diagnostic_only=cli.audit_only,qualified=False,sources=sources(),ordinary_report_sha256=sha(cli.ordinary_report),
              cpu_proof_sha256=sha(cli.key_proof),frozen_sources=FROZEN_PINS,runtime_sources=TG_PINS)
  decoders=[];restores=[]
  def hooked_local(path):
    value=original_local(path)
    if Path(path).resolve()!=FROZEN/'agent-startup-spec-warm.py': return value
    original_initialize=value.initialize
    def initialize(components,*args,**kwargs):
      start=time.perf_counter()
      ledger['prime']=helper.prime(initialized_renderer())
      decoders.append(components.decoder)
      with prototype.install(components.decoder,True,value.ClosedGraphs) as result:
        ledger['startup_cache']=result
        startup=original_initialize(components,*args,**kwargs)
      result.update(ready=True,ready_with_plan_audit_ms=(time.perf_counter()-start)*1000,
        frozen_initializer_ms=startup['initialization_ms'],graph_counts=dict(startup['graph_counts']),
        decoder_plan=dict(startup['decoder_plan']))
      check_startup(startup)
      ledger['startup_spec']=startup
      ledger['strict_plan_proof']=prototype.compare_signatures(ordinary['startup_cache'],result)
      ledger['startup_ready']=True
      print(json.dumps({'startup_key_ready':dict(ready_ms=result['ready_with_plan_audit_ms'],
        prime_host_ms=ledger['prime']['host_ms'],strict_plan_proof=ledger['strict_plan_proof'])}),flush=True)
      if cli.audit_only: raise DiagnosticReady()
      return startup
    value.initialize=initialize
    restores.append((value,original_initialize))
    return value
  base.local=hooked_local
  try:
    try: base.main(options)
    except DiagnosticReady:
      if not cli.audit_only or ledger.get('startup_ready') is not True: raise
      ledger.update(full64_replayed=False,consumer_bound=False,
        note='Startup-only diagnostic. Exact current-input restoration and strict plans passed; no qualification or paced authorization.')
      with output.open('x') as stream: json.dump(ledger,stream,indent=2);stream.write('\n')
      return ledger
    after={str(key[1]):prototype.graph_signature(jit) for key,jit in decoders[0].jits.items()}
    if after!=ledger['startup_cache']['serial_signatures']:
      raise AssertionError('Consumer execution changed a warmed serial plan or alias coverage')
    ledger['startup_cache']['live_plan_signatures_unchanged']=True
    report=json.loads(output.read_text())
    reference=json.loads(Path(option('--compare-reference')).read_text())
    policy=base.local(Path(option('--base-source',str(base.DEFAULT)))/'experiments/agent-paced-dynamic-policy.py')
    policy.verify_snapshot()
    ledger['prediction_parity']=base.qualify(reference,report,policy)
    if report['protocol']!=ordinary['protocol'] or report['initialized']!=ordinary['initialized']:
      raise AssertionError('Actual replay recipe differs from the ordinary GPU evidence')
    ledger.update(full64_replayed=True,consumer_bound=True,qualified=True)
    report.update(startup_cache=ledger['startup_cache'],startup_cache_sources=frozen.sources(prototype),startup_key=ledger)
    output.write_text(json.dumps(report,indent=2)+'\n')
    return ledger
  except BaseException as error:
    ledger['error']=repr(error)
    raise
  finally:
    base.local=original_local
    for value,original_initialize in restores: value.initialize=original_initialize
    with output.with_suffix('.startup-key.json').open('x') as stream: json.dump(ledger,stream,indent=2);stream.write('\n')


if __name__=='__main__': main()
