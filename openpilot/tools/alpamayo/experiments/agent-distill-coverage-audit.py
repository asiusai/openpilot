"""Metadata-only coverage expansion on the two admitted drives. No image decode or model imports."""
import argparse
from bisect import bisect_right
from collections import Counter
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import sys

sys.dont_write_bytecode=True
EXP=Path(__file__).resolve().parent
WORK=Path('/mnt/nvme-500/tmp/alpamayo-speed-20261116')
ROOT=WORK/'agent-distill-coverage-artifacts'
ARCHIVE=Path('/mnt/nvme-256/footage/f9ad9a96/realdata')
EXCLUDED=('0000008e--321b2ddc03','00000090--1cf3f7d926')
DRIVES={'train':('00000071--858232e52d','f9ad9a96:2026-11-11:2505913223557',16,20),
  'validation':('00000073--41474e402a','f9ad9a96:2026-11-12:179462008389',8,60)}
OFFSETS=(-2000000000,-1000000000,-500000000,-100000000,0)
POLICY=dict(schema=1,counts={'train':16,'validation':8},command='Follow the road.',command_index=0,
  sampling='One anchor per chronological equal-duration bin. Preferred past/current-state class cycles stop,steering-turn,moving,chronological. Choose nearest bin center, then earlier timestamp.',
  candidate_grid_seconds=1,minimum_spacing_seconds={'train':20,'validation':60},
  minimum_within_segment_seconds=4,maximum_frame_floor_gap_ns=150000000,
  exclude_previous_anchor_radius_seconds=10,
  phase_rules={'stop':'Current vEgo<0.5m/s and standstill; all available past1s carState vEgo<0.5m/s.',
    'steering-turn':'Current vEgo>=2m/s and abs(steeringAngleDeg)>=20 degrees. Proxy only; recorded turn intent is not inferred.',
    'moving':'Current vEgo>=2m/s and abs(steeringAngleDeg)<20 degrees.'},
  selection_inputs='Input availability and observed carState only. No images, model predictions, targets, losses, future trajectories, navigation inference, or protected/sealed content.',
  split='Preserve existing full physical-drive assignments; scene anchors within a drive are correlated, not independent drives.',
  no_training_protocol_change=True,metadata_only=True)

def sha(path):
  with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
def read(path):return json.loads(Path(path).read_text())
def write(path,value):
  with Path(path).open('x') as stream:stream.write(json.dumps(value,indent=2,allow_nan=False)+'\n')
def module(name,path):
  spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
def permitted(path):
  path=Path(path).resolve();assert not any(route in str(path) for route in EXCLUDED)
  if path.is_relative_to(ARCHIVE):assert any(route in str(path) for route,_,_,_ in DRIVES.values())
  return path

def scan_drive(split,inventory,old_anchor,audit,capture):
  import numpy as np
  route,drive_id,count,spacing=DRIVES[split]
  states={kind:{} for kind in capture.bridge.STATE_LIMITS};sources={};segments=[];car_meta={};started=set();gps_dates=set()
  for number in inventory['available_complete_segments']:
    path=permitted(ARCHIVE/(route+'--'+str(number))/'rlog.zst')
    events,size,status=audit.load_events(path,128*1024**2)
    local_states={kind:{} for kind in states};frames={1:[],6:[]};encodes={1:{},6:{}};meta={};channels=Counter();stamps=[]
    for event in events:
      kind=event.which()
      if kind not in (*states,'wideRoadCameraState','narrowRoadCameraState','wideRoadEncodeIdx','narrowRoadEncodeIdx','deviceState','gpsLocation','gpsLocationExternal'):continue
      stamp=int(event.logMonoTime);stamps.append(stamp);channels[kind]+=1;m=getattr(event,kind)
      if kind in states:
        stamp=int(m.timestamp) if kind=='deviceMotion' else stamp
        if stamp<=0:continue
        value,valid=capture.state_value(kind,m) if event.valid else (None,False)
        if valid:
          try:json.dumps(value,allow_nan=False)
          except ValueError:valid=False
        local_states[kind][stamp]=(value if valid else None,int(event.logMonoTime),number)
        if kind=='carState' and valid:
          meta[stamp]=dict(speed_m_s=float(m.vEgo),steering_degrees=float(m.steeringAngleDeg),standstill=bool(m.standstill),
            left_blinker=bool(m.leftBlinker),right_blinker=bool(m.rightBlinker))
      elif event.valid and kind in ('wideRoadCameraState','narrowRoadCameraState'):
        view=1 if kind=='wideRoadCameraState' else 6
        frames[view].append(dict(view=view,frame_id=int(m.frameId),capture_ns=int(m.timestampSof),eof_ns=int(m.timestampEof),
          published_ns=int(event.logMonoTime),sensor=str(m.sensor),segment=number))
      elif event.valid and kind in ('wideRoadEncodeIdx','narrowRoadEncodeIdx'):
        view=1 if kind=='wideRoadEncodeIdx' else 6
        encodes[view][int(m.frameId)]=dict(segment=int(m.segmentNum),video_index=int(m.segmentId),capture_ns=int(m.timestampSof),eof_ns=int(m.timestampEof))
      elif event.valid and kind=='deviceState' and m.started:
        started.add(int(m.startedMonoTime))
      elif event.valid and kind in ('gpsLocation','gpsLocationExternal') and m.hasFix and m.unixTimestampMillis:
        from datetime import datetime,timezone
        gps_dates.add(datetime.fromtimestamp(m.unixTimestampMillis/1000,timezone.utc).date().isoformat())
    sources[str(path)]=sha(path)
    row=dict(number=number,rlog_path=str(path),rlog_sha256=sources[str(path)],uncompressed_bytes=size,parse_status=status,
      valid_input_channels=dict(channels),pixel_decode_executed=False)
    for view in (1,6):frames[view].sort(key=lambda item:item['capture_ns'])
    row['frame_counts']={str(view):len(frames[view]) for view in (1,6)}
    row['front_camera_ranges_ns']={str(view):[frames[view][0]['capture_ns'],frames[view][-1]['capture_ns']] if frames[view] else None for view in (1,6)}
    row['eligible_complete_log_and_fronts']=bool(status['complete'] and all(frames.values()))
    if row['eligible_complete_log_and_fronts']:
      for kind in states:
        for stamp,value in local_states[kind].items():
          if stamp in states[kind]:assert states[kind][stamp][:2]==value[:2]
          states[kind][stamp]=value
      car_meta.update(meta)
    segments.append((row,frames,encodes))
    print(f'Metadata {split} segment {number}: fronts={row["frame_counts"]}, complete={status["complete"]}',flush=True)
  assert started=={int(drive_id.rsplit(':',1)[1])} and gps_dates=={drive_id.split(':')[1]},(started,gps_dates)
  arrays={kind:(np.array(sorted(values),np.int64),[values[t][0] for t in sorted(values)]) for kind,values in states.items()}
  return dict(route=route,drive_id=drive_id,split=split,count=count,spacing=spacing,states=states,arrays=arrays,car_meta=car_meta,
    sources=sources,segments=segments,old_anchor=old_anchor,started_mono_ns=list(started)[0],dates=sorted(gps_dates))

def input_metadata(drive,segment,q,capture):
  import numpy as np
  from calibration_trials import asof
  row,frames,encodes=segment;arrays=drive['arrays'];states=drive['states']
  def required(kind,stamp):
    value,selected=asof(arrays[kind],stamp,capture.bridge.STATE_LIMITS[kind])
    if value is None:raise ValueError('Invalid latest '+kind)
    return value,selected
  chosen=[];state_refs={kind:set() for kind in states};ages={}
  for kind in states:
    value,stamp=required(kind,q);state_refs[kind].add(stamp)
  for view in (1,6):
    times=[frame['capture_ns'] for frame in frames[view]]
    for offset in OFFSETS:
      desired=q+offset;i=bisect_right(times,desired)-1
      if i<0 or not 0<=desired-times[i]<=150000000:raise ValueError('Causal five-image history unavailable')
      frame=dict(frames[view][i],requested_ns=desired)
      encoded=encodes[view].get(frame['frame_id'])
      if not encoded or encoded['capture_ns']!=frame['capture_ns'] or encoded['eof_ns']!=frame['eof_ns'] or encoded['segment']!=row['number']:
        raise ValueError('Frame/encoding association unavailable')
      calibration,stamp=required('extrinsicsCalibration',frame['capture_ns']);state_refs['extrinsicsCalibration'].add(stamp)
      frame.update(encoding=encoded,prior_extrinsics_ns=stamp,calibration_sha256=hashlib.sha256(json.dumps(calibration,sort_keys=True).encode()).hexdigest())
      chosen.append(frame)
    if len({frame['capture_ns'] for frame in chosen if frame['view']==view})!=5:raise ValueError('Duplicate history frame')
  for kind in ('carState','vehicleParameters'):
    elapsed=[]
    for stamp in q+np.arange(-38,1,dtype=np.int64)*50000000:
      _,selected=required(kind,int(stamp));elapsed.append(int(stamp)-selected);state_refs[kind].add(selected)
    ages[kind]=max(elapsed)
  mt=arrays['deviceMotion'][0];last=bisect_right(mt,q);first=max(0,bisect_right(mt,q-1900000000)-1)
  if last-first<2 or mt[first]>q-1900000000 or q-mt[last-1]>100000000 or np.diff(mt[first:last]).max()>150000000:
    raise ValueError('Measured causal history gap')
  for stamp in mt[first:last]:
    if states['deviceMotion'][int(stamp)][0] is None:raise ValueError('Invalid motion history')
    state_refs['deviceMotion'].add(int(stamp))
  cs_times=arrays['carState'][0];ci=bisect_right(cs_times,q)-1;current=drive['car_meta'].get(int(cs_times[ci]))
  if current is None:raise ValueError('Missing valid current carState')
  prior_times=cs_times[max(0,bisect_right(cs_times,q-1000000000)-1):ci+1]
  prior=[drive['car_meta'].get(int(stamp)) for stamp in prior_times]
  stop=current['standstill'] and current['speed_m_s']<.5 and prior and all(item and item['speed_m_s']<.5 for item in prior)
  phase='stop' if stop else 'steering-turn' if current['speed_m_s']>=2 and abs(current['steering_degrees'])>=20 else 'moving' if current['speed_m_s']>=2 else 'slow'
  used_segments=sorted({states[kind][stamp][2] for kind in state_refs for stamp in state_refs[kind]})
  return dict(query_ns=q,segment=row['number'],frames=chosen,phase=phase,observed_car_state=current,
    state_source_segments=used_segments,state_refs={kind:sorted(values) for kind,values in state_refs.items()},
    max_car_state_age_ns=ages['carState'],max_vehicle_parameters_age_ns=ages['vehicleParameters'],
    max_motion_history_gap_ns=int(np.diff(mt[first:last]).max()),
    all_input_measurements_le_query=all(stamp<=q for values in state_refs.values() for stamp in values),
    maximum_input_publication_ns=max(states[kind][stamp][1] for kind in state_refs for stamp in state_refs[kind]),
    causal_pose_constructor_executed=False,pixels_decoded=False,complete_model_input_prepared=False)

def choose(drive,capture):
  eligible=[];rejected=Counter();intervals=[]
  for segment in drive['segments']:
    row,frames,_=segment
    if not row['eligible_complete_log_and_fronts']:continue
    start=max(frames[1][0]['capture_ns'],frames[6][0]['capture_ns'])+4000000000
    end=min(frames[1][-1]['capture_ns'],frames[6][-1]['capture_ns'])
    if start>end:continue
    intervals.append([start,end]);times=[frame['capture_ns'] for frame in frames[6]];previous=None
    for desired in range(start,end+1,1000000000):
      index=bisect_right(times,desired)-1;q=times[index]
      if q==previous:continue
      previous=q
      if abs(q-drive['old_anchor'])<10000000000:continue
      try:eligible.append(input_metadata(drive,segment,q,capture))
      except ValueError as error:rejected[str(error)]+=1
  assert eligible,drive['split'];eligible.sort(key=lambda item:item['query_ns'])
  start,end=eligible[0]['query_ns'],eligible[-1]['query_ns'];selected=[];preferences=('stop','steering-turn','moving','chronological')
  for index in range(drive['count']):
    low=start+(end-start)*index//drive['count'];high=start+(end-start)*(index+1)//drive['count']+1
    candidates=[item for item in eligible if low<=item['query_ns']<high and (not selected or item['query_ns']-selected[-1]['query_ns']>=drive['spacing']*1000000000)]
    assert candidates,(drive['split'],index,'empty chronological bin')
    preferred=preferences[index%4];center=(low+high)//2
    item=min(candidates,key=lambda item:(preferred!='chronological' and item['phase']!=preferred,abs(item['query_ns']-center),item['query_ns']))
    selected.append(dict(item,bin=index,preferred_phase=preferred,preference_available=any(v['phase']==preferred for v in candidates)))
  return selected,dict(candidate_count=len(eligible),one_second_grid_valid_counts_by_phase=dict(Counter(item['phase'] for item in eligible)),
    rejected_candidate_reasons=dict(rejected),front_segment_history_safe_intervals_ns=intervals,
    front_segment_history_safe_seconds=sum((b-a)/1e9 for a,b in intervals),
    valid_candidate_grid_seconds=len(eligible),grid_duration_is_sampled_not_continuous_proof=True,
    selected_phases=dict(Counter(item['phase'] for item in selected)),minimum_selected_spacing_seconds=min((b['query_ns']-a['query_ns'])/1e9 for a,b in zip(selected,selected[1:])))

def finish_anchor(drive,item,capture,video_cache):
  import numpy as np
  from calibration_trials import asof,vehicle_history
  from camera_profiles import replay_module
  q=item['query_ns'];causal={}
  for kind,(times,values) in drive['arrays'].items():
    end=bisect_right(times,q);causal[kind]=(times[:end],values[:end])
  # Keep only actual causal motion values. Motion.sample must return no future trajectory.
  times,values=causal['deviceMotion'];motion={int(t):v for t,v in zip(times,values,strict=True) if v is not None}
  calibration,_=asof(causal['extrinsicsCalibration'],q,1000000000)
  xyz,rot,future,age=replay_module().Motion(motion).sample(q,calibration['rpyCalib']);assert future is None
  xyz,rot,detail=vehicle_history(dict(query_ns=q,ego_history_xyz=xyz,ego_history_rot=rot),causal)
  assert xyz.shape==(1,20,3) and rot.shape==(1,20,3,3) and np.isfinite(xyz).all() and np.isfinite(rot).all()
  pose=dict(xyz_shape=list(xyz.shape),rot_shape=list(rot.shape),xyz_sha256=hashlib.sha256(xyz.tobytes()).hexdigest(),
    rot_sha256=hashlib.sha256(rot.tobytes()).hexdigest(),future_returned=False,extrapolation_ms=age,vehicle_history_audit=detail,pose_arrays_exported=False)
  videos=[]
  for view,name in ((1,'ecamera.mp4'),(6,'fcamera.mp4')):
    path=permitted(ARCHIVE/(drive['route']+'--'+str(item['segment']))/name)
    if str(path) not in video_cache:
      probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0','-count_packets',
        '-show_entries','stream=codec_name,width,height,pix_fmt,duration,nb_read_packets:stream_side_data=rotation','-of','json',str(path)],text=True,timeout=15))['streams'][0]
      assert (probe['codec_name'],probe['width'],probe['height'],probe['pix_fmt'])==('hevc',1344,760,'yuv420p')
      video_cache[str(path)]=dict(path=str(path),sha256=sha(path),bytes=path.stat().st_size,header=probe,pixel_decode_executed=False)
    pin=video_cache[str(path)];assert max(frame['encoding']['video_index'] for frame in item['frames'] if frame['view']==view)<int(pin['header']['nb_read_packets'])
    videos.append(pin)
  row=dict(item);row.pop('state_refs');row.update(id=drive['split']+'-scene-'+f'{item["bin"]:02d}',split=drive['split'],route_id=drive['route'],drive_id=drive['drive_id'],
    source_videos=videos,state_logs=[dict(path=str(ARCHIVE/(drive['route']+'--'+str(number))/'rlog.zst'),sha256=drive['sources'][str(ARCHIVE/(drive['route']+'--'+str(number))/'rlog.zst')]) for number in item['state_source_segments']],
    conditioning=dict(text=POLICY['command'],command_index=0,basis='experimental command',recorded_available_ns=None,
      recorded_navigation_source=None,recorded_driver_intent=False,future_motion_inference=False,route_availability_or_safety_claim=False),
    causal_pose_constructor_executed=True,causal_pose_check=pose,
    metadata_ready_for_CPU_input_preparation=True,admissible_for_teacher_capture_now=False)
  return row

def main():
  parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output-dir',type=Path,default=ROOT/'metadata-v1');args=parser.parse_args()
  output=args.output_dir.resolve();assert output.is_relative_to(ROOT) and not output.exists();ROOT.mkdir(exist_ok=True)
  assert subprocess.check_output(['findmnt','-n','-o','UUID','-T',str(ARCHIVE)],text=True).strip()=='aac28ecf-db48-4b4f-8a99-bde6a3726ab9'
  assert subprocess.check_output(['findmnt','-n','-o','TARGET','-T',str(ROOT)],text=True).strip()=='/mnt/nvme-500'
  def guard(event,values):
    if event=='open' and isinstance(values[0],(str,bytes)):
      path=os.fsdecode(values[0]);assert not path.startswith(('/dev/dri','/dev/kfd','/dev/nvidia','/dev/bus/usb'))
      assert not any(route in path for route in EXCLUDED)
      assert not ('sealed_test' in path and path.endswith('.npz'))
      if path.startswith(str(ARCHIVE)):permitted(path)
  sys.addaudithook(guard);output.mkdir();write(output/'agent-distill-coverage-policy.json',POLICY)
  sys.path[:0]=[str(EXP),'/tmp/opencode/openpilot-alpamayo','/home/john/asius/openpilot/opendbc_repo']
  audit=module('distill_coverage_existing_audit',EXP/'agent-distill-data-audit.py');capture=module('distill_coverage_existing_capture',EXP/'agent-live-capture.py')
  inventory_path=WORK/'agent-distill-data-artifacts/agent-distill-data-audit.json';inventory=read(inventory_path)
  prior_path=WORK/'agent-distill-data-artifacts/agent-distill-data-cohort.json';prior=read(prior_path)
  anchors=[];drives=[];videos={};source_hashes={str(path):sha(path) for path in [Path(__file__),inventory_path,prior_path,
    EXP/'agent-distill-data-audit.py',EXP/'agent-live-capture.py',EXP/'agent-live-bridge.py',EXP/'calibration_trials.py',EXP/'camera_profiles.py',EXP/'replay-route.py',EXP/'native_warp.py',EXP/'native-warp.c',EXP/'fast-inputs.py']}
  for split in ('train','validation'):
    route,identity,_,_=DRIVES[split];old=next(row for row in prior['records'] if row['provisional_split']==split)
    assert old['route']==route and old['drive_id']==identity
    inv=next(row for row in inventory['routes'] if row['route']==route)
    summary_path=permitted(inv['summary_path']);assert sha(summary_path)==inv['summary_sha256'];summary=read(summary_path)
    drive=scan_drive(split,inv,old['anchor_query_ns'],audit,capture);selected,coverage=choose(drive,capture)
    for item in selected:anchors.append(finish_anchor(drive,item,capture,videos))
    source_hashes.update(drive['sources']);source_hashes[str(summary_path)]=sha(summary_path)
    drives.append(dict(split=split,route_id=route,drive_id=identity,evidenced_UTC_dates=drive['dates'],started_mono_ns=drive['started_mono_ns'],
      summary_duration_seconds=summary['durationSeconds'],summary_distance_meters=summary['distanceMeters'],
      previous_anchor_excluded_ns=old['anchor_query_ns'],segments=[row for row,_,_ in drive['segments']],coverage=coverage))
  source_hashes.update({path:pin['sha256'] for path,pin in videos.items()})
  capture_plan=WORK/'agent-distill-command-artifacts/capture-v1/agent-distill-command-capture-plan.json'
  frozen_capture=read(capture_plan);source_hashes[str(capture_plan)]=sha(capture_plan)
  for path in [Path(frozen_capture['runtime_root'])/'experiments'/name for name in ('agent-speculation-run.py','benchmark-pipeline.py','padded-prefill.py','quantize-packed.py')]:
    source_hashes[str(path)]=sha(path)
  result=dict(schema=1,status='Metadata-only 24-scene expansion; CPU pixels/preprocessing and future parent capture not executed.',
    policy_sha256=sha(output/'agent-distill-coverage-policy.json'),gpu_executed=False,training_executed=False,
    model_imported=False,video_decode_count=0,sealed_drive_opened=False,protected_route_opened=False,active_continuation_inspected=False,
    independent_drives=2,training_anchors=16,validation_anchors=8,commands_per_anchor=1,drive_splits_unchanged=True,
    drives=drives,anchors=anchors,source_sha256=source_hashes,
    teacher_recipe=dict(frozen_capture_plan_path=str(capture_plan),sha256=sha(capture_plan),
      selected_precision='Original selected FP8 weights/dynamic vision/fixed language; BF16 attention and residuals.',pad_prompt_to=2048,
      prefill_attention='shuffle-fma',full_images=10,history_frames_per_front_camera=5,history_offsets_ns=list(OFFSETS),
      teacher_group=[12,13,14,15],targets=['h_enter12','h_leave15','post_RoPE_K12','V12','cos','sin'],
      expert_steps_preserved=3,expert_and_driving_outputs_executed=False,
      preexisting_language_scale_calibration='32 original protected-route calibration windows were used previously; no new protected-route fitting or stopping.'),
    next_stage='CPU-only exact selected/reference preparation for these 24 scene IDs; verify arrays and inputs before parent-only teacher capture.',
    capture_plan=dict(status='Plan only, not a runnable capture manifest. Existing nine-example capture helper must not accept this metadata manifest.',
      jobs=2,max_prefills_per_job=12,parent_hard_timeout_seconds_each=300,job_0_ids=[row['id'] for row in anchors[:8]+anchors[16:20]],
      job_1_ids=[row['id'] for row in anchors[8:16]+anchors[20:24]],
      runtime='Reuse exact already-frozen canonical command capture source/installer order; minimal cohort/count adapter only after complete CPU inputs.',
      no_current_continuation_change=True,no_automatic_capture=True,source_graph_replacement=False),
    estimated_artifacts=dict(native_RGB_frames=240,native_RGB_bytes=240*3*760*1344,prepared_image_bytes=24*10*3*432*768,
      teacher_target_files=24,teacher_target_bytes=24*26215834,target_payload_bytes=24*25*1024**2,
      total_native_prepared_target_bytes=240*3*760*1344+24*10*3*432*768+24*26215834,
      budget_basis='Known native/prepared shapes and measured previous per-example NPZ size, not measured decode/capture runtime.'),
    limitations=['Two physical drives, not 24 independent drives.','Observed steering-turn is not verified intersection intent.',
      'No recorded navigation claim, trajectory labels, image-content selection, or loss-based selection.',
      'No claim of valid pixels or exact prepared arrays before subsequent CPU decode/preparation.',
      'This does not modify the active 25-to-100 protocol or qualify a student, driving quality, or latency.'])
  assert len(anchors)==24 and all(row['all_input_measurements_le_query'] for row in anchors)
  assert not any(name=='torch' or name.startswith('tinygrad') for name in sys.modules)
  write(output/'agent-distill-coverage-manifest.json',result)
  print(json.dumps(dict(output=str(output/'agent-distill-coverage-manifest.json'),sha256=sha(output/'agent-distill-coverage-manifest.json'),
    scenes=24,drives=[dict(split=d['split'],duration_seconds=d['summary_duration_seconds'],coverage=d['coverage']) for d in drives],gpu_executed=False),indent=2))

if __name__=='__main__':main()
