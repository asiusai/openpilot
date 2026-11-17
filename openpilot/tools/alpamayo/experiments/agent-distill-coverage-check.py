"""Independent metadata checks. No image, target, active-run, or device reads."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import sys

sys.dont_write_bytecode=True
WORK=Path('/mnt/nvme-500/tmp/alpamayo-speed-20261116')
ROOT=WORK/'agent-distill-coverage-artifacts'
EXCLUDED=('0000008e--321b2ddc03','00000090--1cf3f7d926')
EXPECTED={'train':('00000071--858232e52d','f9ad9a96:2026-11-11:2505913223557',16,20),
  'validation':('00000073--41474e402a','f9ad9a96:2026-11-12:179462008389',8,60)}

def sha(path):
  with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
def validate(value):
  assert value['gpu_executed'] is value['training_executed'] is value['model_imported'] is False
  assert value['independent_drives']==2 and value['commands_per_anchor']==1 and value['video_decode_count']==0
  assert not value['active_continuation_inspected'] and not value['sealed_drive_opened'] and not value['protected_route_opened']
  assert len(value['anchors'])==24
  ids=[]
  for split,(route,drive,count,spacing) in EXPECTED.items():
    rows=[item for item in value['anchors'] if item['split']==split]
    assert len(rows)==count and [row['bin'] for row in rows]==list(range(count))
    assert all(b['query_ns']-a['query_ns']>=spacing*1000000000 for a,b in zip(rows,rows[1:]))
    drive_meta=next(item for item in value['drives'] if item['split']==split)
    for row in rows:
      ids.append(row['id']);q=row['query_ns']
      assert row['drive_id']==drive and row['route_id']==route and abs(q-drive_meta['previous_anchor_excluded_ns'])>=10000000000
      assert row['all_input_measurements_le_query'] and row['causal_pose_constructor_executed']
      assert not row['pixels_decoded'] and not row['complete_model_input_prepared'] and not row['admissible_for_teacher_capture_now']
      command=row['conditioning'];assert command['text']=='Follow the road.' and command['command_index']==0
      assert command['basis']=='experimental command' and command['recorded_available_ns'] is None
      assert command['recorded_navigation_source'] is None and not command['recorded_driver_intent'] and not command['future_motion_inference']
      assert not command['route_availability_or_safety_claim']
      pose=row['causal_pose_check'];assert pose['xyz_shape']==[1,20,3] and pose['rot_shape']==[1,20,3,3]
      assert not pose['future_returned'] and not pose['pose_arrays_exported']
      assert row['max_car_state_age_ns']<=150000000 and row['max_vehicle_parameters_age_ns']<=1000000000 and row['max_motion_history_gap_ns']<=150000000
      assert len(row['frames'])==10
      for view in (1,6):
        frames=[frame for frame in row['frames'] if frame['view']==view]
        assert len(frames)==5 and len({frame['capture_ns'] for frame in frames})==5
        assert [frame['requested_ns']-q for frame in frames]==[-2000000000,-1000000000,-500000000,-100000000,0]
        for frame in frames:
          assert frame['capture_ns']<=frame['requested_ns']<=q and frame['requested_ns']-frame['capture_ns']<=150000000
          assert 0<=frame['capture_ns']-frame['prior_extrinsics_ns']<=1000000000
          encoding=frame['encoding'];assert encoding['capture_ns']==frame['capture_ns'] and encoding['eof_ns']==frame['eof_ns'] and encoding['segment']==row['segment']
          video=next(video for video in row['source_videos'] if Path(video['path']).name==('ecamera.mp4' if view==1 else 'fcamera.mp4'))
          assert encoding['video_index']<int(video['header']['nb_read_packets'])
      assert len(row['source_videos'])==2
      for source in row['state_logs']+row['source_videos']:
        assert route in source['path'] and not any(route2 in source['path'] for route2 in EXCLUDED)
        assert value['source_sha256'][source['path']]==source['sha256']
  assert len(ids)==len(set(ids))
  jobs=value['capture_plan'];assert jobs['jobs']==2 and jobs['max_prefills_per_job']==12 and jobs['parent_hard_timeout_seconds_each']==300
  assert len(jobs['job_0_ids'])==len(jobs['job_1_ids'])==12 and set(jobs['job_0_ids']).isdisjoint(jobs['job_1_ids'])
  assert set(jobs['job_0_ids']+jobs['job_1_ids'])==set(ids)
  assert jobs['no_current_continuation_change'] and jobs['no_automatic_capture'] and not jobs['source_graph_replacement']

def main():
  parser=argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--manifest',type=Path,default=ROOT/'metadata-v1/agent-distill-coverage-manifest.json')
  parser.add_argument('--output',type=Path,default=ROOT/'agent-distill-coverage-check-v1.json');args=parser.parse_args()
  assert args.output.resolve().is_relative_to(ROOT) and not args.output.exists()
  def guard(event,values):
    if event=='open' and isinstance(values[0],(str,bytes)):
      path=os.fsdecode(values[0]);assert not path.startswith(('/dev/dri','/dev/kfd','/dev/nvidia','/dev/bus/usb'))
      assert not any(route in path for route in EXCLUDED)
      assert not ('sealed_test' in path and path.endswith('.npz'))
      assert 'agent-distill-continue-artifacts/pilot-' not in path
  sys.addaudithook(guard)
  manifest=json.loads(args.manifest.read_text());validate(manifest)
  for path,digest in manifest['source_sha256'].items():assert sha(path)==digest,path
  negatives=[]
  for name,mutate in [
    ('sealed drive',lambda value:value['anchors'][0].update(route_id=EXCLUDED[1])),
    ('protected route',lambda value:value['anchors'][0].update(route_id=EXCLUDED[0])),
    ('cross-drive split',lambda value:value['anchors'][0].update(drive_id=EXPECTED['validation'][1])),
    ('future frame',lambda value:value['anchors'][0]['frames'][0].update(capture_ns=value['anchors'][0]['query_ns']+1)),
    ('stale measured calibration',lambda value:value['anchors'][0]['frames'][0].update(prior_extrinsics_ns=1)),
    ('fabricated recorded intent',lambda value:value['anchors'][0]['conditioning'].update(recorded_driver_intent=True)),
    ('unprepared capture admission',lambda value:value['anchors'][0].update(admissible_for_teacher_capture_now=True)),
    ('duplicate capture job',lambda value:value['capture_plan'].update(job_1_ids=value['capture_plan']['job_0_ids'])),
    ('scene variants counted as drives',lambda value:value.update(independent_drives=24))]:
    changed=copy.deepcopy(manifest);mutate(changed)
    try:validate(changed)
    except (AssertionError,KeyError,StopIteration):negatives.append(name)
    else:raise AssertionError('Accepted negative: '+name)
  result=dict(complete=True,all_passed=True,gpu_executed=False,training_executed=False,image_decode_executed=False,
    model_imported=False,active_continuation_inspected=False,protected_route_opened=False,sealed_drive_opened=False,
    manifest_sha256=sha(args.manifest),verified_source_hashes=len(manifest['source_sha256']),
    verified_scene_anchors=24,verified_causal_native_frame_associations=240,negative_controls_rejected=negatives,
    helper_sha256=sha(Path(__file__)))
  with args.output.open('x') as stream:stream.write(json.dumps(result,indent=2)+'\n')
  print(json.dumps(result,indent=2))

if __name__=='__main__':main()
