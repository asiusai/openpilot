"""Validate frozen binding qualification and independent seven-fixture provenance."""
import importlib.util
import json
from pathlib import Path

EXP=Path(__file__).parent
WORK=Path('/mnt/nvme-500/tmp/alpamayo-speed-20261116')


def load(name):
  spec=importlib.util.spec_from_file_location(name.replace('-','_'),EXP/(name+'.py'))
  module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
  return module


def verify():
  common=load('agent-ck-qualified-common');old=load('agent-ck-output-finish-evidence')
  _,_,prior=old.verify()
  paths={WORK/'agent-ck-output-finish-parent/report.json':'26e95ea054cd198f036ecd9df652a14d6ce4a2433eb03fe5b5dfc4ba43732980',
    WORK/'agent-ck-output-finish-work/parent-result.json':'bf8706f161de8119732e4e9a895d0af661e34c3544179594ce33469fe9af5891',
    WORK/'agent-ck-output-finish-work/prepared/protocol.json':'8d71e6c08eccb6b1f7a3eec8cfa0ca88777bb0a07e58753209714d752756e4da',
    Path('/tmp/opencode/run-alpamayo-ck-output-finish.py'):'fceb96b3bf5fbe9859d6de43901ed98772aedada1728c8300f4cfc89dd42cb4f'}
  for path,digest in paths.items(): assert common.sha(path)==digest,str(path)
  report=json.loads((WORK/'agent-ck-output-finish-parent/report.json').read_text())
  original=json.loads((WORK/'agent-ck-output-finish-work/parent-result.json').read_text())
  plan=json.loads((WORK/'agent-ck-output-finish-work/prepared/protocol.json').read_text())
  assert {k:v for k,v in report.items() if k!='parent_bridge'}==original
  assert report['parent_bridge']['original_gates_unchanged'] is True
  assert report['completed'] and report['gpu_executed'] and report['binding_proven'] and not report['adoption_allowed']
  assert report['protocol_sha256']==paths[WORK/'agent-ck-output-finish-work/prepared/protocol.json']
  assert report['reuse']==prior and report['io_completed']==plan['io_budget']
  assert len(report['phases'])==4
  for spec,row in zip(plan['phases'],report['phases'],strict=True):
    assert spec['name']==row['name'] and row['phase_gates_passed'] and row['stage']=='complete'
    oh,lh=old.component(row,'O',common.OC),old.component(row,'LSE',common.LC)
    old.error_gate(row['fp32_error'],prior['baseline_errors'][row['fixture']],plan['limits'])
    assert row['lse_error']['rmse']<=plan['limits']['lse_rmse_abs'] and row['lse_error']['max_abs']<=plan['limits']['lse_max_abs']
    if row['fixture']=='a': assert oh==prior['candidate_A_output_sha256'] and lh==prior['candidate_A_lse_sha256']
    else:
      assert row['mutation_changed_outside_causal_GQA_support']==row['mutation_LSE_mismatch_count']==0
      assert row['mutation_max_abs_change']>plan['limits']['mutation_min_output_max_abs_change']
  assert len(report['replay_dispatches'])==3 and len({r['linked_linear_key'] for r in report['replay_dispatches']})==1
  assert set(report['actual_programs'])==set(plan['programs'])
  for row in report['native_argument_records']:
    assert row['pointer_offsets']==[0,8,16,24,32] and row['kernarg_bytes']==296
    assert row['dispatch_pointer']==row['CU_mode_bit']==0
    assert [v['offset_bytes'] for v in row['arguments']]==[512,1024,0,0,0]
  bound=dict(plan['source_sha256']);bound.update({str(p):s for p,s in paths.items()})
  v3=WORK/'agent-ck-attention-v3/prepared-final/protocol.json'
  original_plan=json.loads(v3.read_text());bound.update(original_plan['source_sha256'])
  oracle=Path(original_plan['oracle']);manifest=json.loads((oracle/'manifest.json').read_text())
  assert manifest['limits']==original_plan['limits']==plan['limits']
  assert manifest['replay_order']==original_plan['replay_order'] and len(manifest['replay_order'])==40
  for fixture in manifest['fixtures']:
    for entry in fixture['files'].values(): bound[str(oracle/entry['path'])]=entry['sha256']
  for path,digest in bound.items(): assert common.sha(path)==digest,str(path)
  reuse={'completed_binding_report':str(WORK/'agent-ck-output-finish-parent/report.json'),
    'report_sha256':paths[WORK/'agent-ck-output-finish-parent/report.json'],
    'original_elapsed_seconds':report['elapsed_seconds'],'binding_proven':True,
    'candidate_A_output_sha256':prior['candidate_A_output_sha256'],'candidate_A_lse_sha256':prior['candidate_A_lse_sha256'],
    'baseline_A_output_sha256':prior['baseline_A_output_sha256'],'prior_claim_is_binding_only':True}
  return original_plan,manifest,bound,reuse
