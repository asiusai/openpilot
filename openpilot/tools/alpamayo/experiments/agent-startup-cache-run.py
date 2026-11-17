"""Compose the frozen qualified startup with one instance-only capture experiment."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parent
TG = Path('/tmp/opencode/alpamayo-asius-20261114/tinygrad')
PINS = {
  ROOT/'agent-startup-spec-run.py':'7e22277322166f6938f28c700ade7ac62196260c3310d90d92f6c59ba15c36f0',
  ROOT/'agent-startup-spec-warm.py':'3b7f2d7d5e1888243271fa34b7bb8f91d7b7f7c5dd3789503bec1071a3e2900c',
  TG/'tinygrad/engine/jit.py':'bdf5f3029e9e98faaddb0e83a51296189e7cf6ebf845d84fb46b8414f2a9424b',
  TG/'tinygrad/engine/realize.py':'d8229adf90f86e8f21f2e9b6081d6eb0604c42058faa52cde4826c55cf08292f',
  TG/'tinygrad/runtime/support/hcq2.py':'dbab7aef4f8c226d43ca5d138381d5d757a9d8683dfca381ac32e6d826734480',
}


def local(path):
  spec = importlib.util.spec_from_file_location(Path(path).stem.replace('-','_'),path)
  value = importlib.util.module_from_spec(spec)
  sys.modules[spec.name] = value
  spec.loader.exec_module(value)
  return value


def sources(prototype):
  return {name:prototype.sha(ROOT/name) for name in ('agent-startup-cache-run.py','agent-startup-cache-capture.py')}


def validate_pins(prototype):
  for path, expected in PINS.items():
    if prototype.sha(path)!=expected:
      raise ValueError('Pinned startup/runtime source changed: '+str(path))


def qualify(base, prototype, root, reference, baseline, candidate):
  policy = base.local(root/'experiments/agent-paced-dynamic-policy.py')
  policy.verify_snapshot()
  for report in (baseline,candidate):
    base.qualify(reference,report,policy)
    if report.get('startup_cache_sources')!=sources(prototype):
      raise ValueError('The new startup sources changed after the paired GPU runs')
    if not report.get('startup_cache',{}).get('ready'):
      raise ValueError('The startup plan audit did not complete')
    if report['startup_cache'].get('live_plan_signatures_unchanged') is not True:
      raise ValueError('Real Consumer execution changed a captured serial plan or alias coverage')
  if baseline['protocol']!=candidate['protocol'] or baseline['initialized']!=candidate['initialized']:
    raise ValueError('Paired startup recipes differ')
  proof = prototype.compare_signatures(baseline['startup_cache'],candidate['startup_cache'])
  return dict(**proof, both_exact64=True, acceptance_and_fallback=True,
    baseline_startup_ms=baseline['startup_cache']['ready_with_plan_audit_ms'],
    candidate_startup_ms=candidate['startup_cache']['ready_with_plan_audit_ms'],
    timing_scope='one cold process per variant with the existing local binary cache; repeat matched runs before performance adoption')


def main(argv=None):
  parser = argparse.ArgumentParser(add_help=False)
  parser.add_argument('--variant',choices=('ordinary','serial-first-capture'),required=True)
  parser.add_argument('--baseline-report',type=Path)
  cli, options = parser.parse_known_args(argv)
  sys.dont_write_bytecode = True
  prototype = local(ROOT/'agent-startup-cache-capture.py')
  validate_pins(prototype)
  base = local(ROOT/'agent-startup-spec-run.py')
  def option(name,default=None): return options[options.index(name)+1] if name in options else default
  task = option('--task')
  root = Path(option('--base-source',str(base.DEFAULT))).resolve()
  if task in ('qualify','paced'):
    if cli.variant!='serial-first-capture' or cli.baseline_report is None:
      raise ValueError('Candidate qualification needs its matched ordinary-startup report')
    ref = json.loads(Path(option('--reference-report')).read_text())
    candidate_path = option('--candidate-report') if task=='qualify' else option('--consumer-parity-report')
    proof = qualify(base,prototype,root,ref,json.loads(cli.baseline_report.read_text()),json.loads(Path(candidate_path).read_text()))
    if task=='qualify':
      print(json.dumps(proof,indent=2))
      return proof
  if task not in ('replay','paced') or (task=='replay' and option('--mode')!='consumer'):
    raise ValueError('Both variants use the real Consumer and the complete qualified startup')
  output = Path(option('--output'))
  ledger = {}
  original_local = base.local
  def hooked_local(path):
    value = original_local(path)
    if Path(path).resolve()!=ROOT/'agent-startup-spec-warm.py': return value
    original_initialize = value.initialize
    def initialize(components,*args,**kwargs):
      start = time.perf_counter()
      ledger['decoder'] = components.decoder
      try:
        with prototype.install(components.decoder,cli.variant=='serial-first-capture',value.ClosedGraphs) as result:
          ledger['report'] = result
          startup = original_initialize(components,*args,**kwargs)
        result.update(ready=True,ready_with_plan_audit_ms=(time.perf_counter()-start)*1000,
                      frozen_initializer_ms=startup['initialization_ms'],
                      graph_counts=dict(startup['graph_counts']),
                      decoder_plan=dict(startup['decoder_plan']))
        print(json.dumps({'startup_cache_ready':{key:result[key] for key in
          ('variant','ready_with_plan_audit_ms','frozen_initializer_ms','phase_host_ms','skipped_eager_lengths')}}),flush=True)
        return startup
      except BaseException as error:
        ledger.setdefault('report',{}).update(ready=False,error=repr(error))
        raise
    value.initialize = initialize
    return value
  base.local = hooked_local
  try:
    base.main(options)
    after = {str(key[1]):prototype.graph_signature(jit) for key,jit in ledger['decoder'].jits.items()}
    if after!=ledger['report']['serial_signatures']:
      raise AssertionError('Consumer execution added or changed a serial execution-plan link')
    ledger['report']['live_plan_signatures_unchanged'] = True
    report = json.loads(output.read_text())
    report['startup_cache'] = ledger['report']
    report['startup_cache_sources'] = sources(prototype)
    output.write_text(json.dumps(report,indent=2)+'\n')
  finally:
    base.local = original_local
    with output.with_suffix('.startup-cache.json').open('x') as stream:
      json.dump(dict(report=ledger.get('report'),sources=sources(prototype),
                     pinned_sources={str(p):h for p,h in PINS.items()}),stream,indent=2)
      stream.write('\n')


if __name__=='__main__': main()
