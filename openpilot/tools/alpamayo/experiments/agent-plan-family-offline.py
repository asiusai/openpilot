"""Independently prove every captured family arena. No device open, link, or execution."""
import argparse
import json
import os
from pathlib import Path
import sys
import time
from unittest.mock import patch

EXP=Path(__file__).resolve().parent
WORK=Path('/mnt/nvme-500/tmp/alpamayo-speed-20261116')
TG=Path('/tmp/opencode/alpamayo-asius-20261114/tinygrad')
sys.dont_write_bytecode=True
sys.path.insert(0,str(TG))

def local(path):
  import importlib.util
  spec=importlib.util.spec_from_file_location('family_offline_'+Path(path).stem.replace('-','_'),path)
  module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
  return module

def inspect_member(contract,summary,count,expected):
  audit_path=Path(summary['audit']);sidepath=Path(summary['sidecar'])
  for path,key in ((audit_path,'audit_sha256'),(sidepath,'sidecar_sha256')):
    if path.is_symlink() or not path.resolve().is_relative_to(WORK) or contract.sha(path)!=summary[key]:
      raise ValueError('Captured family artifact identity changed')
  doc=json.loads(audit_path.read_text());side=json.loads(sidepath.read_text())
  if side['count']!=count or side['audit_sha256']!=summary['audit_sha256']: raise ValueError('Member identity differs')
  member=contract.validate_document(doc,count,expected,side['output_views'])
  if member!=side['contract'] or member!=summary['contract']: raise ValueError('Recorded member contract differs')
  codec=contract.local(EXP/'agent-plan-bind-codec.py')
  programs={key:codec.load_offline(audit_path.parent/'programs',row) for key,row in doc['programs'].items()}
  if set(programs)!={c['program_key'] for c in doc['calls']}: raise ValueError('PROGRAM set differs from actual calls')
  variables={}
  from tinygrad.dtype import AddrSpace
  for program in programs.values():
    for p in program.arg.params:
      if p.addrspace is not AddrSpace.ALU: continue
      if not p.is_variable or p.is_bound_var: raise ValueError('Unreviewed scalar ABI')
      value=dict(min=p.vmin,max=p.vmax,dtype=p.dtype.name)
      if p.expr in variables and variables[p.expr]!=value: raise ValueError('Conflicting variable binding')
      variables[p.expr]=value
  if variables!=contract.bounds(count): raise ValueError('Original implicit variable ranges differ')
  proof=contract.local(EXP/'agent-plan-reload-proof.py').prove(doc,programs)
  if not proof['proved'] or proof['calls']!=len(doc['calls']): raise AssertionError('Incomplete independent arena proof')
  return doc,programs,proof,member

def main():
  parser=argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--report',type=Path,required=True);parser.add_argument('--protocol',type=Path,required=True)
  parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
  contract=local(EXP/'agent-plan-family-contract.py');contract.fresh(args.output)
  for name in ('TMPDIR','CACHEDB','XDG_CACHE_HOME'):
    if not Path(os.environ.get(name,'')).resolve().is_relative_to(WORK): raise ValueError('Caches must remain under WORK')
  from tinygrad import Device
  def forbidden(*args,**kwargs): raise AssertionError('Offline family inspection attempted device access')
  result=dict(schema=1,gpu_executed=False,executable=False,qualified=False,ready_for_live_use=False,plans={})
  start=time.perf_counter()
  try:
    with patch.object(type(Device),'__getitem__',forbidden):
      protocol=json.loads(args.protocol.read_text());contract.source_gate(protocol)
      report=json.loads(args.report.read_text());family=report['plan_family']
      if report['family_protocol_sha256']!=contract.sha(args.protocol) or family['counts']!=protocol['counts']:
        raise ValueError('Capture protocol or counts differ')
      if not family['capture_completed'] or not all(family['restored'].values()): raise ValueError('Capture or restoration incomplete')
      if set(family['plans'])!={str(n) for n in protocol['counts']}: raise ValueError('Incomplete family coverage')
      proof_dir=args.output.with_suffix('.proofs');proof_dir.mkdir()
      for count in protocol['counts']:
        doc,programs,proof,member=inspect_member(contract,family['plans'][str(count)],count,contract.signatures()[str(count)])
        path=proof_dir/f'count{count}.json';path.write_text(json.dumps(proof,indent=2)+'\n')
        result['plans'][str(count)]=dict(contract=member,proof=str(path),proof_sha256=contract.sha(path),
          capture=family['plans'][str(count)],scratch_initialization_proved=True,
          initialized_bytes=proof['initialized_bytes'],scratch_read_checks=len(proof['scratch_read_checks']),
          full_output_writes=proof['full_output_writes'],fresh_process_reload_qualified=False)
        print(json.dumps({'family_offline_proved':count,'calls':len(doc['calls']),'programs':len(programs)}),flush=True)
      result.update(all_requested_scratch_proofs_passed=True,devices=list(Device._opened_devices),
        original_count1_reused_without_gpu_reexecution=True,
        next_required='Separate source-bound fresh-process reload protocol after reviewing these actual per-count artifacts')
  except BaseException as error:
    result['error']=repr(error);raise
  finally:
    result.update(elapsed_ms=(time.perf_counter()-start)*1000,protocol_sha256=contract.sha(args.protocol),capture_report_sha256=contract.sha(args.report))
    args.output.write_text(json.dumps(result,indent=2)+'\n')

if __name__=='__main__': main()
