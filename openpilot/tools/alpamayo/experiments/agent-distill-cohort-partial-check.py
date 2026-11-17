"""Validate a finished partial handoff, actual CPU inputs/mRoPE, and full finalize entrypoint."""
import argparse
import copy
import importlib.util
import json
import os
from pathlib import Path
import runpy
import sys

sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('agent_distill_cohort_common_partial_check',HERE/'agent-distill-cohort-common.py')
C=importlib.util.module_from_spec(spec);sys.modules[spec.name]=C;spec.loader.exec_module(C)

def validate(value,recover):
  expected={row['id']:row for row in C.coverage()['anchors']};rows=value['samples'];ids=[row['id'] for row in rows]
  assert value['expected_scene_ids']==list(expected) and value['completed_scene_ids']==ids and len(ids)==len(set(ids))
  assert value['completed_conditioned_inputs']==len(rows) and value['full_cohort_complete']==(len(rows)==24)
  assert value['training_scenes']==sum(row['split']=='train' for row in rows) and value['validation_scenes']==sum(row['split']=='validation' for row in rows)
  assert set(ids).isdisjoint(value['missing_scene_ids']) and set(ids+value['missing_scene_ids'])==set(expected)
  assert not value['gpu_executed'] and not value['training_executed'] and not value['teacher_inference_executed']
  for row in rows:recover.validate_record(row,expected[row['id']])
  for receipt in value['scenes']:
    row=receipt['sample'];assert row['id'] in ids and receipt['coverage_sha256']==C.COVERAGE_SHA and receipt['CPU_only']
    assert C.sha(receipt['receipt_path'])==receipt['receipt_sha256'] and receipt['native_reference_exact']
    assert len(receipt['native_frames'])==len(receipt['audit']['frames'])==10
    for frame in receipt['audit']['frames']:
      assert frame['capture_ns']<=frame['requested_ns']<=row['query_ns']
      assert 0<=frame['capture_ns']-frame['measured_calibration']['timestamp_ns']<=1000000000
    assert receipt['audit']['input_measurement_timestamps_max_ns']<=row['query_ns']
  return ids

def main():
  parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--manifest',type=Path,required=True);parser.add_argument('--output-dir',type=Path,required=True)
  args=parser.parse_args();root=args.output_dir.resolve();assert root.is_relative_to(C.ROOT) and not root.exists();C.storage(root.parent);root.mkdir()
  sys.addaudithook(C.cpu_guard)
  recovery=HERE/'agent-distill-cohort-recover.py';recover=C.module('cohort_partial_production_recovery',recovery)
  manifest=C.read(args.manifest);ids=validate(manifest,recover)
  assert ids,'No complete inputs available for actual preprocessing checks'
  before={path:C.sha(path) for receipt in manifest['scenes'] for path in [receipt['receipt_path'],*[str(Path(receipt['sample']['sample'])/name) for name in receipt['sample']['input_file_sha256']]]}
  # Real complete CLI via runpy; finalize-only performs no decode or preprocessing/model work.
  final=root/'agent-distill-cohort-finalize-entrypoint.json';previous=sys.argv
  sys.argv=[str(recovery),'--finish-existing','--manifest-output',str(final)]
  try:runpy.run_path(str(recovery),run_name='__main__')
  finally:sys.argv=previous
  repeated=C.read(final);assert repeated['completed_scene_ids']==ids and repeated['samples']==manifest['samples'];validate(repeated,recover)
  original=C.module('cohort_partial_qualified_CPU_adapter',HERE/'agent-distill-command-capture.py')
  plan=C.read(C.OLD_PLAN);ready=original.prepared_inputs(plan,manifest)
  import numpy as np
  policy=C.module('cohort_partial_original_padded_policy',Path(plan['runtime_root'])/'experiments/agent-speculation-policy.py')
  checks=[]
  for row,prepared,packed in ready:
    actual=row['actual_prompt_tokens'];assert actual==2027
    padded_prepared,padded=policy.padded_inputs(prepared,packed,2048,policy.CHECKPOINT_DRAFT_IDS,C.read(C.BASE/'model/config.json')['traj_ids']['future_start'])
    assert C.fingerprint(padded)==row['padded_packed'] and padded_prepared['prefill_draft_eligible']
    assert padded[0].shape==(1,2048) and padded[1].shape==(3,1,2048)
    np.testing.assert_array_equal(padded[0][:,:actual],packed[0]);np.testing.assert_array_equal(padded[1][:,:,:actual],packed[1])
    np.testing.assert_array_equal(padded[0][0,actual:actual+6],[124549,11135,3174,4286,1561,135084])
    np.testing.assert_array_equal(padded[1][:,:,actual:actual+6],np.broadcast_to(actual+int(packed[2])+np.arange(6),(3,1,6)))
    assert not padded[0][:,actual+6:].any() and not padded[1][:,:,actual+6:].any()
    checks.append(dict(id=row['id'],prepared_fingerprint_exact=True,packed_fingerprint_exact=True,actual_prompt_tokens=actual,
      full2048_mrope_exact=True,actual_real_prefix_unchanged=True,six_nominal_draft_and_END_padding_exact=True))
  negatives=[]
  for name,mutate in [
    ('partial relabeled24complete',lambda value:value.update(full_cohort_complete=True,completed_conditioned_inputs=24)),
    ('protected route',lambda value:value['samples'][0].update(route_id=C.PROTECTED)),
    ('sealed drive',lambda value:value['samples'][0].update(route_id=C.EXCLUDED[1])),
    ('wrong drive split',lambda value:value['samples'][0].update(split='validation')),
    ('changed command',lambda value:value['samples'][0]['conditioning'].update(text='Turn left.')),
    ('changed NPY checksum',lambda value:value['samples'][0]['input_file_sha256'].update(**{'image_frames.npy':'0'*64})),
    ('future calibration',lambda value:value['scenes'][0]['audit']['frames'][0]['measured_calibration'].update(timestamp_ns=10**18))]:
    changed=copy.deepcopy(manifest);mutate(changed)
    try:validate(changed,recover)
    except (AssertionError,ValueError,KeyError):negatives.append(name)
    else:raise AssertionError('Accepted negative: '+name)
  for path,digest in before.items():assert C.sha(path)==digest
  for path,digest in C.coverage()['source_sha256'].items():assert C.sha(path)==digest,path
  C.assert_no_devices()
  result=dict(complete=True,all_passed=True,partial_manifest_sha256=C.sha(args.manifest),actual_complete_input_checks=checks,
    completed_scene_ids=ids,missing_scene_ids=manifest['missing_scene_ids'],full_cohort_complete=False,
    actual_runpy_finalize_entrypoint_passed=True,bounded_recovery_skips_valid_inputs=True,
    original_receipts_inputs_and_recording_sources_unchanged=True,negative_controls_rejected=negatives,
    two12_prefill_GPU_mock_jobs_executed=False,GPU_mock_blocker='Only the admitted partial scenes exist; the strict24-scene adapter remains blocked.',
    gpu_executed=False,training_executed=False,teacher_inference_executed=False,sealed_or_protected_data_opened=False,
    source_sha256={str(path):C.sha(path) for path in [Path(__file__),recovery,HERE/'agent-distill-cohort-common.py',args.manifest]})
  C.write(root/'agent-distill-cohort-partial-check.json',result);print(json.dumps(result,indent=2))

if __name__=='__main__':main()
