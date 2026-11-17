"""Parent-only sealed family reload with unchanged factory/input restoration."""
import argparse
import json
import os
from pathlib import Path
import sys
import time
from unittest.mock import patch

EXP=Path(__file__).resolve().parent;WORK=Path('/mnt/nvme-500/tmp/alpamayo-speed-20261116')

def local(path):
  import importlib.util
  spec=importlib.util.spec_from_file_location('family_reload_run_'+Path(path).stem.replace('-','_'),path)
  module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
  return module

def main():
  parser=argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--output',type=Path,required=True);parser.add_argument('--cache-db',type=Path,required=True)
  parser.add_argument('--reload-protocol',type=Path,required=True)
  parser.add_argument('--preflight',action='store_true');parser.add_argument('--execute-gpu',action='store_true');args=parser.parse_args()
  for path in (args.output,args.cache_db,Path(os.environ.get('TMPDIR','')),Path(os.environ.get('XDG_CACHE_HOME',''))):
    if not path.resolve().is_relative_to(WORK): raise ValueError('All writable storage must remain under WORK')
  if args.output.exists() or args.cache_db.resolve()!=Path(os.environ['CACHEDB']).resolve(): raise ValueError('Fresh output and matching cache required')
  if args.preflight==args.execute_gpu: raise ValueError('Select CPU preflight or explicit parent GPU execution')
  environment=json.loads((WORK/'agent-plan-bind-capture1/report.json').read_text())['protocol']['environment']
  for key,value in environment.items():
    if key in ('CACHEDB','TMPDIR','PYTHONDONTWRITEBYTECODE'): continue
    if value is None: os.environ.pop(key,None)
    else: os.environ[key]=value
  sys.dont_write_bytecode=True
  contract=local(EXP/'agent-plan-family-reload-contract.py');protocol=json.loads(args.reload_protocol.read_text())
  protocol_sha=contract.sha(args.reload_protocol);contract.gate(protocol)
  # Import/derive the actual production constructor even in CPU preflight.
  loader=contract.local(EXP/'agent-plan-family-reload-loader.py')
  if args.preflight:
    previous=local(EXP/'agent-plan-family-run.py')
    command=[str(previous.__file__),'--preflight','--output',str(args.output),'--cache-db',str(args.cache_db),
             '--family-protocol',str(contract.BASE)]
    with patch.object(sys,'argv',command): previous.main()
    contract.gate(protocol)
    if contract.sha(args.reload_protocol)!=protocol_sha: raise ValueError('Sealed protocol changed during preflight')
    data=json.loads(args.output.read_text());data.update(family_reload_protocol_sha256=protocol_sha,
      reload_counts=protocol['counts'],constructor_derivation=loader.Prepared.derivation,source_before_after_exact=True)
    args.output.write_text(json.dumps(data,indent=2)+'\n');return
  started=time.perf_counter();frozen=local(EXP/'agent-plan-bind-run.py')
  prefix=local(EXP/'agent-plan-family-capture.py');diagnostic=local(EXP/'agent-plan-family-reload-diagnostic.py')
  derivation=prefix.install(frozen,contract.family(),protocol,started)
  owner_state={}
  def session(context):
    context['report']['family_reload_protocol_sha256']=protocol_sha
    owners=diagnostic.ClosedOwners(context['model'])
    owner_state.update(owners=owners,report=context['report'])
    try: diagnostic.session(context,contract,protocol,started)
    finally: owners.guard()
  frozen.family_capture_session=session
  original_capture=frozen.capture_one
  def capture(*values):
    try: return original_capture(*values)
    finally:
      if owner_state:
        owner_state['owners'].guard()
        owner_state['report']['all_model_owner_allocations_and_views_retained']=True
  frozen.capture_one=capture
  failure=None
  try: frozen.main(['--output',str(args.output),'--cache-db',str(args.cache_db),'--execute-gpu'])
  except BaseException as error: failure=error;raise
  finally:
    gate_error=None
    try:
      contract.gate(protocol)
      if contract.sha(args.reload_protocol)!=protocol_sha: raise ValueError('Sealed protocol changed during diagnostic')
    except BaseException as error: gate_error=error
    if args.output.exists():
      report=json.loads(args.output.read_text());result=report.pop('plan_family',{})
      keys=('full_kv_bits_restored','actual_first_logits_bits_restored','target_payload_bits_restored',
            'current_owner_retained','generation_history_unchanged')
      result['restored']={key:report.get(key,False) for key in keys}
      result['qualified']=(failure is None and gate_error is None and result.get('all_requested_members_passed',False)
        and all(result['restored'].values()) and report.get('all_model_owner_allocations_and_views_retained',False))
      result['whole_family_2_through_6_passed']=result['qualified'] and protocol['counts']==[2,3,4,5,6]
      report.update(plan_family_reload=result,qualified=False,ready_for_live_use=False,executable_plan=False,
        family_reload_protocol_sha256=protocol_sha,family_reload_sources=protocol['sources'],setup_derivation=derivation,
        source_before_after_exact=gate_error is None,source_binding_error=repr(gate_error) if gate_error else None)
      for stage in report['stages']:
        if stage['name']=='count1_captured_and_real_prefill_restored': stage['name']='family_reload_diagnostic_and_real_prefill_restored'
      args.output.write_text(json.dumps(report,indent=2)+'\n')
    if gate_error is not None and failure is None: raise gate_error

if __name__=='__main__': main()
