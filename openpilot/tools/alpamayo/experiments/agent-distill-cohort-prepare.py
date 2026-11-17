"""Exact CPU preparation of the unchanged 24-scene coverage cohort. No model inference."""
import argparse
from bisect import bisect_right
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('agent_distill_cohort_common_prepare',HERE/'agent-distill-cohort-common.py')
c=importlib.util.module_from_spec(spec);sys.modules[spec.name]=c;spec.loader.exec_module(c)

def main():
  parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output-dir',type=Path,default=c.ROOT/'inputs-v1');args=parser.parse_args()
  root=args.output_dir.resolve();assert root==c.ROOT/'inputs-v1' and not root.exists()
  c.source_mount();c.ROOT.mkdir(exist_ok=True);c.storage(c.ROOT,1024**3);sys.addaudithook(c.cpu_guard)
  source=c.coverage()
  for path,digest in source['source_sha256'].items():assert c.sha(path)==digest,path
  root.mkdir();(root/'samples').mkdir();(root/'native-rgb').mkdir();(root/'extract-sources').mkdir()
  os.environ.update(XDG_CACHE_HOME=str(root/'cpu-cache'),TMPDIR=str(root/'cpu-cache'),PYTHONDONTWRITEBYTECODE='1')
  (root/'cpu-cache').mkdir();c.write(root/'agent-distill-cohort-protocol.json',c.protocol())
  torch,helper=c.cpu_imports()
  import numpy as np
  from alpamayo2_edge.chat_template import build_conversation
  from alpamayo2_edge.sample import CAMERA_PIXEL_BOUNDS
  sys.path[:0]=['/tmp/opencode/openpilot-alpamayo','/home/john/asius/openpilot/opendbc_repo']
  audit=c.module('cohort_existing_log_audit',HERE/'agent-distill-data-audit.py')
  camera=c.module('cohort_existing_camera',HERE/'camera_profiles.py')
  capture=c.module('cohort_existing_capture',HERE/'agent-live-capture.py')
  native=c.module('cohort_existing_native',HERE/'native_warp.py')
  from calibration_trials import calibrated_image
  runtime=Path(c.read(c.OLD_PLAN)['runtime_root']);sys.path[:0]=[str(runtime/'experiments'),str(runtime)]
  fast=c.module('cohort_frozen_fast',runtime/'experiments/fast-inputs.py')
  assert Path(sys.modules['inputs'].__file__).resolve()==runtime/'inputs.py'
  policy=c.module('cohort_frozen_padding_policy',runtime/'experiments/agent-speculation-policy.py')
  adapter=fast.FastInputs(c.BASE/'model',camera_profile='nvidia',allow_missing_cameras=True,direct_byte_inputs=True,cache_resize=True)
  assert adapter.tokenize.__code__==helper._tokenize_messages.__code__
  geometry=c.read(c.GEOMETRY);native_refs=[];raw_lookup={};scene_audits=[];records=[]
  groups={}
  for row in source['anchors']:groups.setdefault((row['route_id'],row['segment']),[]).append(row)
  for (route,segment),rows in groups.items():
    c.source_mount();refs=[dict(frame,video_index=frame['encoding']['video_index']) for row in rows for frame in row['frames']]
    assert len({(frame['view'],frame['video_index']) for frame in refs})==len(refs)
    c.storage(root,len(refs)*(3*760*1344+128));key=route+'--'+str(segment)
    extract=root/'extract-sources'/key;extract.mkdir();raw=root/'native-rgb'/key
    cameras=[]
    for view,name in ((1,'ecamera.mp4'),(6,'fcamera.mp4')):
      pin=next(pin for pin in rows[0]['source_videos'] if Path(pin['path']).name==name)
      assert c.sha(pin['path'])==pin['sha256']
      rotations=[item['rotation'] for item in pin['header'].get('side_data_list',[]) if 'rotation' in item]
      assert rotations in ([],[0],[180],[-180])
      cameras.append(dict(view=view,video=pin['path'],video_pixels_rotated=False,video_display_rotation_degrees=abs(rotations[0]) if rotations else 0))
    c.write(extract/'manifest.json',dict(export='/',camera_manifest=dict(cameras=cameras),schedule=[dict(tick=0,frames=refs)]))
    camera.extract(SimpleNamespace(source=extract,output=raw,ticks=None,recording_uuid=None))
    for frame in refs:
      path=raw/str(frame['view'])/(str(frame['video_index'])+'.npy');array=np.load(path,allow_pickle=False)
      assert array.shape==(3,760,1344) and array.dtype==np.uint8
      digest=hashlib.sha256(array.tobytes()).hexdigest();identity=(route,segment,frame['view'],frame['video_index'])
      raw_lookup[identity]=path;native_refs.append(dict(route_id=route,segment=segment,view=frame['view'],video_index=frame['video_index'],
        frame_id=frame['frame_id'],capture_ns=frame['capture_ns'],file=str(path),file_sha256=c.sha(path),native_rgb_sha256=digest))
  assert len(native_refs)==240
  for row in source['anchors']:
    q=row['query_ns'];states={kind:[] for kind in capture.bridge.STATE_LIMITS}
    for pin in row['state_logs']:
      assert c.sha(pin['path'])==pin['sha256'];events,_,status=audit.load_events(Path(pin['path']),128*1024**2)
      for event in events:
        kind=event.which()
        if kind not in states:continue
        message=getattr(event,kind);stamp=int(message.timestamp) if kind=='deviceMotion' else int(event.logMonoTime)
        if not 0<stamp<=q:continue
        value,extra=capture.state_value(kind,message) if event.valid else (None,False)
        states[kind].append(capture.bridge.State(stamp,int(event.logMonoTime),value,bool(event.valid and extra)))
      assert status['complete']
    for values in states.values():
      by_stamp={}
      for state in sorted(values,key=lambda item:item.stamp):
        if state.stamp in by_stamp:assert state==by_stamp[state.stamp]
        by_stamp[state.stamp]=state
      values[:]=by_stamp.values()
    frames=[];source_inputs=[]
    for frame in row['frames']:
      identity=(row['route_id'],row['segment'],frame['view'],frame['encoding']['video_index']);path=raw_lookup[identity]
      image=np.ascontiguousarray(np.load(path,allow_pickle=False));image.flags.writeable=False
      received=max(frame['published_ns'],frame['eof_ns'])
      frames.append(capture.bridge.Frame(frame['view'],frame['frame_id'],frame['capture_ns'],frame['eof_ns'],received,received,image))
      cal_times=[value.stamp for value in states['extrinsicsCalibration']];cal=states['extrinsicsCalibration'][bisect_right(cal_times,frame['capture_ns'])-1]
      assert cal.valid and 0<=frame['capture_ns']-cal.stamp<=1000000000
      source_inputs.append(dict(frame,native_rgb_sha256=hashlib.sha256(image.tobytes()).hexdigest(),measured_calibration=dict(timestamp_ns=cal.stamp,value=cal.value)))
    selected_ns=max([state.received for values in states.values() for state in values]+[frame.ready_ns for frame in frames])
    selection=capture.bridge.Selection(q,selected_ns,tuple(frames),states,0,None)
    reference=capture.bridge.Preparer(geometry,{},warp=calibrated_image,cache_limit=0)
    accelerated=capture.bridge.Preparer(geometry,{},warp=native.calibrated_image,cache_limit=0)
    arrays,detail=reference.prepare(selection);actual,actual_detail=accelerated.prepare(selection)
    assert set(arrays)==set(actual)=={'image_frames','camera_indices','absolute_timestamps','ego_history_xyz','ego_history_rot'}
    for name in arrays:np.testing.assert_array_equal(arrays[name],actual[name],err_msg=row['id']+': '+name)
    assert arrays['image_frames'].shape==(10,1,3,432,768) and arrays['camera_indices'].tolist()==[1]*5+[6]*5
    c.storage(root,sum(value.nbytes+128 for value in arrays.values())+4096)
    sample=root/'samples'/row['id'];sample.mkdir()
    for name,value in arrays.items():np.save(sample/(name+'.npy'),value,allow_pickle=False)
    np.save(sample/'nav_text.npy',np.array(c.COMMANDS[0]),allow_pickle=False)
    data={name:torch.from_numpy(value.copy()) for name,value in arrays.items()};data['nav_text']=[c.COMMANDS[0]]
    fast.validate_available_sample(data);_,_,prepared=adapter.prepare(data)
    official=helper._tokenize_messages(build_conversation(dict(data,camera_pixel_bounds=CAMERA_PIXEL_BOUNDS)),adapter.processor,include_closed_thinking=True)
    for name,value in official.items():
      if name=='pixel_values':assert torch.equal(value,helper._normalize_patches(prepared['tokenized_data'][name].clone(),adapter.processor.image_processor))
      else:assert torch.equal(value,prepared['tokenized_data'][name]),name
    packed=adapter.native_inputs(prepared);actual_tokens=packed[0].shape[1];assert 1800<actual_tokens<=2042
    assert prepared['tokenized_data']['image_grid_thw'].tolist()==[[1,20,36]]*10 and prepared['tokenized_data']['pixel_values'].shape==(7200,768)
    padded_prepared,padded=policy.padded_inputs(prepared,packed,2048,policy.CHECKPOINT_DRAFT_IDS,adapter.config.traj_ids['future_start'])
    assert padded_prepared['actual_prompt_length']==actual_tokens and padded_prepared['prefill_draft_eligible']
    assert padded[0].shape==(1,2048) and padded[1].shape==(3,1,2048)
    np.testing.assert_array_equal(padded[0][:,:actual_tokens],packed[0]);np.testing.assert_array_equal(padded[1][:,:,:actual_tokens],packed[1])
    try:helper.prepare_model_inputs(data,adapter.config,adapter.tokenizer)
    except ValueError as error:assert '22' in str(error)
    else:raise AssertionError('Official full-camera applicability changed')
    record={key:row[key] for key in ('id','split','route_id','drive_id','query_ns','conditioning')}
    record.update(sample=str(sample),command_index=0,input_file_sha256={path.name:c.sha(path) for path in sorted(sample.glob('*.npy'))},
      actual_prompt_tokens=actual_tokens,prepared=c.fingerprint(prepared),packed=c.fingerprint(packed),padded_packed=c.fingerprint(padded),
      official_selected_tokens_and_image_normalization_exact=True,complete_input=True,source_scene_record_index=len(scene_audits))
    records.append(record);scene_audits.append(dict(id=row['id'],frames=source_inputs,state_sources=row['state_logs'],
      native_reference_image_bits_equal=True,input_measurement_timestamps_max_ns=max(value.stamp for values in states.values() for value in values),
      preparation_audit={key:detail[key] for key in ('vehicle_history_audit','motion_extrapolation_ms','camera_profile','history_policy')}))
    print(f'Prepared {row["id"]}: {actual_tokens} prompt tokens, exact 2048 padding/mRoPE',flush=True)
  sources=[Path(__file__),HERE/'agent-distill-cohort-common.py',c.GEOMETRY,HERE/'agent-distill-command-common.py',
    HERE/'camera_profiles.py',HERE/'agent-live-capture.py',HERE/'agent-live-bridge.py',HERE/'calibration_trials.py',HERE/'native_warp.py',HERE/'native-warp.c',
    runtime/'experiments/fast-inputs.py',runtime/'inputs.py',runtime/'experiments/agent-speculation-policy.py',
    c.BASE/'upstream/src/alpamayo2_edge/helper.py',c.BASE/'upstream/src/alpamayo2_edge/chat_template.py',c.BASE/'model/config.json',c.BASE/'model/model.safetensors.index.json']
  manifest=dict(schema=1,protocol=c.protocol(),admissible_for_internal_layer_capture=True,admissible_for_driving_quality_validation=False,
    gpu_executed=False,training_executed=False,completed_native_frames=240,completed_conditioned_inputs=24,independent_drives=2,
    samples=records,scenes=scene_audits,native_frames=native_refs,source_sha256={str(path):c.sha(path) for path in sources},
    geometry_limit='Exact selected nominal source optics/120-and-30-degree virtual projection, recorded measured extrinsics per frame; no new lens-distortion claim.',
    original_recordings_unchanged=True,protected_and_sealed_drives_opened=False,teacher_target_arrays_opened=False)
  c.validate_rows(manifest);c.assert_no_devices();c.write(c.INPUTS,manifest)
  files=[path for path in root.rglob('*') if path.is_file()]
  c.write(root/'agent-distill-cohort-artifact-bytes.json',dict(files=len(files),total_file_bytes=sum(path.stat().st_size for path in files),
    native_frame_file_bytes=sum(Path(row['file']).stat().st_size for row in native_refs),
    sample_file_bytes=sum(path.stat().st_size for row in records for path in Path(row['sample']).glob('*.npy'))))
  print(json.dumps(dict(manifest=str(c.INPUTS),sha256=c.sha(c.INPUTS),scenes=24,native_frames=240,gpu_executed=False),indent=2))

if __name__=='__main__':main()
