"""Parent-only original family capture using the unchanged frozen factory and input owner."""
import argparse
import json
import os
from pathlib import Path
import sys
import time
from unittest.mock import patch

EXP=Path(__file__).resolve().parent
WORK=Path('/mnt/nvme-500/tmp/alpamayo-speed-20261116')

def local(path):
  import importlib.util
  spec=importlib.util.spec_from_file_location('family_run_'+Path(path).stem.replace('-','_'),path)
  module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
  return module

def main():
  parser=argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--output',type=Path,required=True);parser.add_argument('--cache-db',type=Path,required=True)
  parser.add_argument('--family-protocol',type=Path,required=True)
  parser.add_argument('--execute-gpu',action='store_true');parser.add_argument('--preflight',action='store_true');args=parser.parse_args()
  for path in (args.output,args.cache_db,Path(os.environ.get('TMPDIR','')),Path(os.environ.get('XDG_CACHE_HOME',''))):
    if not path.resolve().is_relative_to(WORK): raise ValueError('All writable paths must remain under WORK')
  if args.output.exists() or args.cache_db.resolve()!=Path(os.environ['CACHEDB']).resolve(): raise ValueError('Output exists or cache differs')
  if args.preflight==args.execute_gpu: raise ValueError('Select exactly one of CPU preflight or explicit parent GPU capture')
  environment=json.loads((WORK/'agent-plan-bind-capture1/report.json').read_text())['protocol']['environment']
  for key,value in environment.items():
    if key in ('CACHEDB','TMPDIR','PYTHONDONTWRITEBYTECODE'): continue
    if value is None: os.environ.pop(key,None)
    else: os.environ[key]=value
  sys.dont_write_bytecode=True
  contract=local(EXP/'agent-plan-family-contract.py');protocol=json.loads(args.family_protocol.read_text());contract.source_gate(protocol)
  if args.preflight:
    # This is the same production runpy/factory import boundary that passed for count1.
    previous=local(EXP/'agent-plan-mutation-v2-run.py')
    argv=[str(previous.__file__),'--output',str(args.output),'--cache-db',str(args.cache_db),'--preflight']
    with patch.object(sys,'argv',argv): previous.main()
    value=json.loads(args.output.read_text());value['family_protocol_sha256']=contract.sha(args.family_protocol)
    value['family_counts']=protocol['counts'];args.output.write_text(json.dumps(value,indent=2)+'\n')
    return
  started=time.perf_counter();frozen=local(EXP/'agent-plan-bind-run.py')
  capture=local(EXP/'agent-plan-family-capture.py');derivation=capture.install(frozen,contract,protocol,started)
  original=frozen.capture_one
  def invoke(*values):
    values[3]['family_protocol_sha256']=contract.sha(args.family_protocol)
    values[3]['family_setup_derivation']=derivation
    return original(*values)
  frozen.capture_one=invoke
  try: frozen.main(['--output',str(args.output),'--cache-db',str(args.cache_db),'--execute-gpu'])
  finally:
    if args.output.exists():
      report=json.loads(args.output.read_text())
      report.update(qualified=False,executable_plan=False,ready_for_live_use=False,family_protocol_sha256=contract.sha(args.family_protocol),
        family_sources=protocol['sources'],family_setup_derivation=derivation)
      if 'plan_family' in report:
        keys=('full_kv_bits_restored','actual_first_logits_bits_restored','target_payload_bits_restored',
              'current_owner_retained','generation_history_unchanged')
        report['plan_family']['restored']={key:report.get(key,False) for key in keys}
      # Correct the inherited factory diagnostic label only in this new report.
      for stage in report['stages']:
        if stage['name']=='count1_captured_and_real_prefill_restored': stage['name']='family_captured_and_real_prefill_restored'
      args.output.write_text(json.dumps(report,indent=2)+'\n')

if __name__=='__main__': main()
