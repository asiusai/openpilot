"""Freeze one repaired source catalog around all original qualified protocol gates."""
import difflib
import hashlib
import json
from pathlib import Path

EXP=Path(__file__).parent
WORK=Path('/mnt/nvme-500/tmp/alpamayo-speed-20261116')
ROOT=WORK/'agent-ck-source-work'
TARGET=ROOT/'prepared-final'
OLD=WORK/'agent-ck-qualified-work/prepared-final/protocol.json'


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
  assert not (TARGET/'protocol.json').exists()
  old=json.loads(OLD.read_text());catalog=json.loads((TARGET/'catalog.json').read_text())
  proof=json.loads((TARGET/'cpu-preflight.json').read_text())
  controls=json.loads((ROOT/'negative-controls-final.json').read_text())
  assert len(catalog['programs'])==28 and len(catalog['associations'])==30 and len(catalog['batches'])==19
  assert proof['real_HCQ_command_and_kernarg_construction_checked'] and not proof['gpu_executed']
  assert len(controls['negative_controls'])==10 and all(x['rejected'] for x in controls['negative_controls'])
  assert controls['catalog_sha256']==sha(TARGET/'catalog.json')
  assert set(catalog['programs'])==set(old['programs'])
  for digest in catalog['programs']:
    assert sha(TARGET/(digest+'.hsaco'))==sha(OLD.parent/(digest+'.hsaco'))==digest
  failure=WORK/'agent-ck-qualified-parent/report.json'
  prior=json.loads(failure.read_text())
  assert not prior['completed'] and prior['active_phase']=='resident_fixture_bank'
  assert prior['io']['host_write_bytes']==33554432 and prior['io']['host_read_bytes']==0
  digest='f20c5cb2dba4794debe9e4ebf61c272cae4d92685b2b028bafae8f7cbc0902d2'
  stored=OLD.parent/(digest+'.source');actual=WORK/'agent-ck-qualified-work/parent-programs'/(digest+'.source')
  old_sha=sha(stored);actual_sha=sha(actual)
  assert old_sha=='7d912a4fa561f4b089333eb5387fe9e21246cc4425feef8922f175a66bfe3382'
  assert actual_sha=='1c3179d7ca637c28373a267323bfd1361c214760948dbb7a98b89b5491f3d050'
  assert {catalog['associations'][a]['source_sha256'] for a in catalog['programs'][digest]['associations']}=={old_sha,actual_sha}
  assert stored.read_text().replace('bfloat','i16')==actual.read_text()
  (TARGET/'failure-source.diff').write_text(''.join(difflib.unified_diff(stored.read_text().splitlines(True),actual.read_text().splitlines(True),
    fromfile=str(stored),tofile=str(actual))))
  diagnosis={'failure_report_sha256':sha(failure),'classification':'Inventory collision; attention never executed in this failed trial.',
    'cause':'The binary-keyed preparation stored only the first source and silently discarded another exact source/AST/ABI association.',
    'shared_binary':digest,'first_source_sha256':old_sha,'actual_failed_source_sha256':actual_sha,
    'exact_diff':'Every difference is bfloat -> i16 in a plain 16-bit return copy. Identical complete ELF bytes.',
    'binary_collisions':{b:[catalog['associations'][a] for a in v['associations']] for b,v in catalog['programs'].items() if len(v['associations'])>1},
    'CPU_preflight':{'full_bank_buffers':len(proof['bank_hashes']),'full_bank_all_bytes_match':all(a==b for a,b in proof['bank_hashes'].items()),
      'fixture_upload_bytes':proof['io']['host_write_bytes'],'fixture_upload_calls':proof['io']['host_write_calls'],
      'catalog_shape_cases':19,'real_HCQ_and_kernargs':True,'gpu_executed':False},
    'repair':'Allow only independently compiled exact source/AST/target/binary/ABI associations and exact per-call view/dependency batches.',
    'unchanged':'All 28 binaries, baseline/CK arithmetic, seven full numerical fixtures, validators, 40 ordered mutable replays, and both fair timings.'}
  (TARGET/'diagnosis.json').write_text(json.dumps(diagnosis,indent=2)+'\n')
  extra=[EXP/(name+'.py') for name in ('agent-ck-source-identity','agent-ck-source-offline','agent-ck-source-guard',
    'agent-ck-source-run','agent-ck-source-check','agent-ck-source-freeze')]
  extra+=[OLD,failure,WORK/'agent-ck-qualified-parent/run.log',WORK/'agent-ck-qualified-work/parent-result.json',actual,
    WORK/'agent-ck-qualified-work/parent-programs'/(digest+'.hsaco'),ROOT/'negative-controls-final.json']
  extra+=list(TARGET.rglob('*'))
  plan=dict(old)
  plan.update(catalog)
  plan['source_sha256']={**old['source_sha256'],**{str(p):sha(p) for p in extra if p.is_file()}}
  plan['source_catalog_repair']={'original_protocol':str(OLD),'original_protocol_sha256':sha(OLD),
    'original_runner':str(EXP/'agent-ck-qualified-run.py'),'original_runner_sha256':sha(EXP/'agent-ck-qualified-run.py'),
    'changed_observers':['source/AST/target/binary/ABI authorization','association-aware artifact recording','exact call/view/dependency authorization'],
    'private_module_global_redirects':['ROOT','PROTOCOL','load(agent-ck-output-run)','load(agent-ck-attention-v3-run).record_programs'],
    'numerical_replay_timing_run_function_unchanged':True,'new_gpu_execution':False,
    'preflight_seconds':proof['seconds'],'no_timeout_extension':True}
  (TARGET/'protocol.json').write_text(json.dumps(plan,indent=2)+'\n')
  print(json.dumps({'protocol':str(TARGET/'protocol.json'),'sha256':sha(TARGET/'protocol.json'),
    'source_files_bound':len(plan['source_sha256']),'budget':plan['budget']},indent=2))


if __name__=='__main__': main()
