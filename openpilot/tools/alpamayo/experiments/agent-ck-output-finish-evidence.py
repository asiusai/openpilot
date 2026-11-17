"""Read-only validation of completed phases in the interrupted binding diagnostic."""
import hashlib
import json
import math
from pathlib import Path

BASE = Path('/mnt/nvme-500/tmp/alpamayo-speed-20261116')
REPORT = BASE/'agent-ck-output-parent/report.json'
ORIGINAL = BASE/'agent-ck-output/parent-result.json'
PREVIOUS = BASE/'agent-ck-output/prepared-final/protocol.json'
REPORT_SHA = 'd641cc3cd048a04aee0e31b77bfdffc4e91361d8f9a46f3dad0577ca9cb93e75'
ORIGINAL_SHA = '7d6b5eba3d49756d5adebc34a777ca1b2dcc2dba519052707ea48bc55eedf4c3'
PROTOCOL_SHA = 'd03493d88c0faf87524230fd4717730ab2befa28e1a1f470743717124f2ce454'


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def component(row, key, count):
  item = row[key]
  assert item['guard_mismatch_count']==item['guarded_returned_mismatch_count']==0
  for kind in ('guarded','returned'):
    stats=item[kind]
    assert stats['elements']==stats['finite_count']==count
    assert stats['nonfinite_count']==stats['poison_count']==0
    assert len(stats['sha256'])==64
  assert item['guarded']['sha256']==item['returned']['sha256']
  return item['guarded']['sha256']


def error_gate(candidate, baseline, limits):
  for rule, field in [('output_rmse','rmse'),('output_max_abs','max_abs')]:
    c,b=candidate[field],baseline[field]
    assert math.isfinite(c) and math.isfinite(b) and c>=0 and b>=0
    assert c<=max(limits[rule]['absolute_floor'],limits[rule]['baseline_multiplier']*b)
  assert len(candidate['per_head_rmse'])==len(baseline['per_head_rmse'])==16
  rule=limits['per_head_rmse']
  assert all(math.isfinite(c) and math.isfinite(b) and c>=0 and b>=0 and
             c<=max(rule['absolute_floor'],rule['baseline_multiplier']*b)
             for c,b in zip(candidate['per_head_rmse'],baseline['per_head_rmse'],strict=True))


def claims(report, plan):
  assert report['protocol_sha256']==PROTOCOL_SHA
  assert report['completed'] is False and report['binding_proven'] is False and report['adoption_allowed'] is False
  assert report['failure']=={'phase':'candidate_normalized_capture_A','type':'TimeoutError',
                            'message':'110-second binding diagnostic deadline'}
  assert report['active_phase']=='candidate_normalized_capture_A'
  phases={p['name']:p for p in report['phases']}
  expected=['baseline_original_first_A','baseline_normalized_first_A','baseline_normalized_capture_A',
    'baseline_normalized_replay_mutate_v','baseline_normalized_replay_A','candidate_normalized_first_A']
  assert list(phases)==expected
  assert phases[expected[0]]['original_guard_binding_failed'] is True
  baseline_hash=None
  input_keys=None
  for name in expected[1:5]:
    row=phases[name]
    assert row['phase_gates_passed'] is True and row['arm']=='baseline' and row['normalize'] is True
    digest=component(row,'O',4194304)
    error_gate(row['fp32_error'],row['fp32_error'],plan['limits'])
    if name==expected[1]: baseline_hash=digest
    elif name!=expected[3]: assert digest==baseline_hash and row['same_A_output_mismatch_count']==0
    else:
      assert digest!=baseline_hash and row['mutation_max_abs_change']>plan['limits']['mutation_min_output_max_abs_change']
    if 'capture' in row:
      capture=row['capture']
      assert capture['expected_names']==[0,1,2,3,4] and all(not m for m in capture['staging_maps'])
      if input_keys is None: input_keys=capture['input_buffer_keys']
      assert input_keys==capture['input_buffer_keys']
      assert row['jit_count_after']==row['jit_count_before']+1
      if row['kind']=='replay': assert row['jit_count_before']>=2
  row=phases[expected[-1]]
  assert row['phase_gates_passed'] is True and row['arm']=='candidate' and row['normalize'] is True
  ohash,lhash=component(row,'O',4194304),component(row,'LSE',32768)
  error_gate(row['fp32_error'],phases[expected[1]]['fp32_error'],plan['limits'])
  assert row['lse_error']['rmse']<=plan['limits']['lse_rmse_abs']
  assert row['lse_error']['max_abs']<=plan['limits']['lse_max_abs']
  native=[r for r in report['native_argument_records'] if r['name']=='alpamayo_ck_d128']
  assert len(native)==1 and native[0]['phase']==expected[-1]
  record=native[0]
  assert record['pointer_offsets']==[0,8,16,24,32] and record['kernarg_bytes']==296
  assert record['dispatch_pointer']==record['CU_mode_bit']==0
  assert [a['offset_bytes'] for a in record['arguments'][:2]]==[512,1024]
  return {'reused_completed_baseline_phases':expected[1:5], 'reused_candidate_first_phase':expected[-1],
    'baseline_errors':{'a':phases[expected[1]]['fp32_error'],'mutate_v':phases[expected[3]]['fp32_error']},
    'baseline_A_output_sha256':baseline_hash,'candidate_A_output_sha256':ohash,'candidate_A_lse_sha256':lhash,
    'candidate_A_errors':row['fp32_error'],'candidate_A_lse_errors':row['lse_error'],
    'prior_capture_not_completed':True,'prior_failure_is_deadline_not_numerical':True}


def verify():
  assert sha(REPORT)==REPORT_SHA and sha(ORIGINAL)==ORIGINAL_SHA and sha(PREVIOUS)==PROTOCOL_SHA
  report=json.loads(REPORT.read_text()); original=json.loads(ORIGINAL.read_text());plan=json.loads(PREVIOUS.read_text())
  bridge=report['parent_bridge']
  assert bridge['original_report']==str(ORIGINAL) and bridge['original_sha256']==ORIGINAL_SHA
  assert bridge['unchanged_original_gates'] is True
  assert {k:v for k,v in report.items() if k!='parent_bridge'}==original
  assert sha('/tmp/opencode/run-alpamayo-ck-output.py')==bridge['adapter_sha256']
  for path,expected in plan['source_sha256'].items(): assert sha(path)==expected, f'Changed previous source: {path}'
  return plan,report,claims(report,plan)


def fixture_proof(oracle):
  """Prove that one bounded V write produces the complete frozen second fixture."""
  import numpy as np
  arrays={name:{key:np.load(Path(oracle)/name/(key+'.npy'),allow_pickle=False)
    for key in ('q','k','v','output_fp32','lse_fp32')} for name in ('a','mutate_v')}
  for name,data in arrays.items():
    for key,shape in [('q',(1,16,2048,128)),('k',(1,8,2048,128)),('v',(1,8,2048,128))]:
      assert data[key].shape==shape and data[key].dtype==np.uint16 and data[key].flags.c_contiguous
      assert np.isfinite((data[key].astype(np.uint32)<<16).view(np.float32)).all()
    assert data['output_fp32'].shape==(1,16,2048,128) and data['output_fp32'].dtype==np.float32
    assert data['lse_fp32'].shape==(1,16,2048) and data['lse_fp32'].dtype==np.float32
    assert np.isfinite(data['output_fp32']).all() and np.isfinite(data['lse_fp32']).all()
  assert all(np.array_equal(arrays['a'][k],arrays['mutate_v'][k]) for k in ('q','k','lse_fp32'))
  a,b=(arrays[name]['v'].tobytes() for name in ('a','mutate_v'))
  offset,length=(2*2048+128)*128*2,32*128*2
  assert offset==1081344 and length==8192
  assert a[:offset]+b[offset:offset+length]+a[offset+length:]==b
  assert b[:offset]+a[offset:offset+length]+b[offset+length:]==a
  assert a!=b
  proof={'buffer':'V','byte_offset':offset,'byte_length':length,'kv_head':2,'row_start':128,'row_stop':160,
    'full_fixture_reconstruction_verified':True,'Q_K_unchanged':True,'LSE_reference_unchanged':True,
    'payload_sha256':{name:{k:hashlib.sha256(data[k].tobytes()).hexdigest() for k in ('q','k','v')}
      for name,data in arrays.items()},
    'patch_sha256':{name:hashlib.sha256(data['v'].tobytes()[offset:offset+length]).hexdigest() for name,data in arrays.items()}}
  return arrays,proof
