"""Bounded incremental recovery. Reuse completed frames/inputs; never relaunch the original decoder."""
import argparse
import copy
import hashlib
import importlib.util
import inspect
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('agent_distill_cohort_common_recover',HERE/'agent-distill-cohort-common.py')
C=importlib.util.module_from_spec(spec);sys.modules[spec.name]=C;spec.loader.exec_module(C)
RECEIPT='agent-distill-cohort-scene.json'

def durable(path,value):
  with Path(path).open('x') as stream:
    stream.write(json.dumps(value,indent=2,allow_nan=False)+'\n');stream.flush();os.fsync(stream.fileno())
  descriptor=os.open(Path(path).parent,os.O_RDONLY)
  try:os.fsync(descriptor)
  finally:os.close(descriptor)

def raw_frames(coverage):
  import numpy as np
  lookup={};bad=[]
  bases=[C.ROOT/'inputs-v1/native-rgb',*sorted((C.ROOT/'decoded-next').glob('attempt-*'))] if (C.ROOT/'decoded-next').exists() else [C.ROOT/'inputs-v1/native-rgb']
  for row in coverage['anchors']:
    for frame in row['frames']:
      identity=(row['route_id'],row['segment'],frame['view'],frame['encoding']['video_index'])
      relative=Path(row['route_id']+'--'+str(row['segment']))/str(frame['view'])/(str(frame['encoding']['video_index'])+'.npy')
      paths=[base/relative for base in bases]
      for path in paths:
        if not path.is_file():continue
        try:
          value=np.load(path,mmap_mode='r',allow_pickle=False)
          assert value.dtype==np.uint8 and value.shape==(3,760,1344)
          lookup[identity]=path;break
        except (ValueError,AssertionError,OSError,EOFError) as error:bad.append(dict(path=str(path),error=type(error).__name__+': '+str(error)))
  return lookup,bad

def validate_record(row,expected):
  import numpy as np
  assert all(row[key]==expected[key] for key in ('id','split','drive_id','route_id','query_ns','conditioning'))
  assert row['complete_input'] and row['official_selected_tokens_and_image_normalization_exact'] and 1800<row['actual_prompt_tokens']<=2042
  sample=Path(row['sample']).resolve();assert sample.is_relative_to(C.ROOT) and sample.name==row['id']
  shapes=dict(image_frames=(10,1,3,432,768),camera_indices=(10,),absolute_timestamps=(10,),ego_history_xyz=(1,20,3),ego_history_rot=(1,20,3,3),nav_text=())
  assert set(row['input_file_sha256'])=={key+'.npy' for key in shapes}
  for key,shape in shapes.items():
    path=sample/(key+'.npy');assert C.sha(path)==row['input_file_sha256'][path.name]
    value=np.load(path,mmap_mode='r',allow_pickle=False);assert value.shape==shape
    if key=='image_frames':assert value.dtype==np.uint8
    elif key in ('camera_indices','absolute_timestamps'):assert value.dtype==np.int64
    elif key=='nav_text':assert value.item()==C.COMMANDS[0]
    else:assert value.dtype==np.float32 and np.isfinite(value).all()
  times=np.load(sample/'absolute_timestamps.npy',allow_pickle=False);assert np.all(times<=row['query_ns']//1000)
  assert all(np.all(np.diff(times[index:index+5])>0) for index in (0,5))
  np.testing.assert_array_equal(np.load(sample/'camera_indices.npy',allow_pickle=False),[1]*5+[6]*5)

def receipts(coverage):
  expected={row['id']:row for row in coverage['anchors']};result={};invalid=[]
  for path in sorted(C.ROOT.glob('recovery-*/samples/*/'+RECEIPT)):
    try:
      item=C.read(path);row=item['sample'];validate_record(row,expected[row['id']])
      assert item['CPU_only'] and item['native_reference_exact'] and len(item['native_frames'])==10
      item['receipt_path']=str(path);item['receipt_sha256']=C.sha(path)
      if row['id'] in result:assert result[row['id']]['sample']['input_file_sha256']==row['input_file_sha256']
      else:result[row['id']]=item
    except (ValueError,AssertionError,KeyError,OSError,EOFError) as error:invalid.append(dict(path=str(path),error=type(error).__name__+': '+str(error)))
  return result,invalid

def receipt(sample,record,audit,native):
  validate_record(record,next(row for row in C.coverage()['anchors'] if row['id']==record['id']))
  selected=[row for row in native if row['route_id']==record['route_id'] and row['capture_ns'] in {frame['capture_ns'] for frame in audit['frames']}]
  assert len(selected)==10
  durable(sample/RECEIPT,dict(schema=1,sample=record,audit=audit,native_frames=selected,CPU_only=True,native_reference_exact=True,
    coverage_sha256=C.COVERAGE_SHA,preparer_sha256=C.sha(HERE/'agent-distill-cohort-prepare.py'),recovery_helper_sha256=C.sha(Path(__file__))))

def finish_available(root,coverage,lookup,selected):
  original=C.module('agent_distill_cohort_original_prepare_recovery',HERE/'agent-distill-cohort-prepare.py')
  private=SimpleNamespace(**{key:getattr(C,key) for key in dir(C) if not key.startswith('__')})
  private.INPUTS=root/'agent-distill-cohort-local-prepared.json'
  private.coverage=lambda:dict(coverage,anchors=[row for row in coverage['anchors'] if row['id'] in selected])
  private.validate_rows=lambda manifest:all(validate_record(row,next(item for item in coverage['anchors'] if item['id']==row['id'])) is None for row in manifest['samples'])
  text=inspect.getsource(original.main)
  def replace(before,after):
    nonlocal text
    assert text.count(before)==1,(before[:100],text.count(before));text=text.replace(before,after)
  replace("root==c.ROOT/'inputs-v1'","root.is_relative_to(c.ROOT) and root.name.startswith('recovery-')")
  replace("root.mkdir();(root/'samples').mkdir();(root/'native-rgb').mkdir();(root/'extract-sources').mkdir()","root.mkdir();(root/'samples').mkdir()")
  begin=text.index('  groups={}\n');end=text.index("  for row in source['anchors']:\n    q=row['query_ns']",begin)
  text=text[:begin]+'''  for row in source['anchors']:
    for frame in row['frames']:
      identity=(row['route_id'],row['segment'],frame['view'],frame['encoding']['video_index']);path=RECOVERY_LOOKUP[identity]
      value=np.load(path,allow_pickle=False);assert value.dtype==np.uint8 and value.shape==(3,760,1344)
      raw_lookup[identity]=path
      native_refs.append(dict(route_id=row['route_id'],segment=row['segment'],view=frame['view'],video_index=frame['encoding']['video_index'],
        frame_id=frame['frame_id'],capture_ns=frame['capture_ns'],file=str(path),file_sha256=c.sha(path),
        native_rgb_sha256=hashlib.sha256(value.tobytes()).hexdigest()))
  assert len(native_refs)==10*len(source['anchors'])
'''+text[end:]
  replace("    print(f'Prepared {row[\"id\"]}: {actual_tokens} prompt tokens, exact 2048 padding/mRoPE',flush=True)",
    "    RECOVERY_RECEIPT(sample,record,scene_audits[-1],native_refs)\n    print(f'Prepared {row[\"id\"]}: {actual_tokens} prompt tokens, exact 2048 padding/mRoPE',flush=True)")
  replace('completed_native_frames=240,completed_conditioned_inputs=24,independent_drives=2',
    "completed_native_frames=len(native_refs),completed_conditioned_inputs=len(records),independent_drives=len({row['drive_id'] for row in records}),full_cohort_complete=False")
  replace('scenes=24,native_frames=240','scenes=len(records),native_frames=len(native_refs)')
  namespace=dict(original.__dict__);namespace.update(c=private,__file__=str(Path(__file__).resolve()),RECOVERY_LOOKUP=lookup,RECOVERY_RECEIPT=receipt)
  exec(compile(text,str(Path(__file__).resolve())+':reuse-qualified-CPU-preparation','exec'),namespace)
  previous=sys.argv;sys.argv=[str(Path(__file__).resolve()),'--output-dir',str(root)]
  try:namespace['main']()
  finally:sys.argv=previous

def decode_scene(identity,coverage,lookup):
  """One bounded caller-owned decode attempt; skip every already valid frame."""
  import time
  row=next(row for row in coverage['anchors'] if row['id']==identity)
  missing=[dict(frame,video_index=frame['encoding']['video_index']) for frame in row['frames']
    if (row['route_id'],row['segment'],frame['view'],frame['encoding']['video_index']) not in lookup]
  if not missing:return
  C.source_mount();C.storage(C.ROOT,len(missing)*(3*760*1344+128)+4096)
  attempt=C.ROOT/'decoded-next'/('attempt-'+str(time.monotonic_ns()));attempt.mkdir(parents=True)
  source=attempt/'source';source.mkdir();output=attempt/(row['route_id']+'--'+str(row['segment']))
  cameras=[]
  for view,name in ((1,'ecamera.mp4'),(6,'fcamera.mp4')):
    pin=next(pin for pin in row['source_videos'] if Path(pin['path']).name==name);assert C.sha(pin['path'])==pin['sha256']
    rotation=[value['rotation'] for value in pin['header'].get('side_data_list',[]) if 'rotation' in value]
    assert rotation in ([],[0],[180],[-180])
    cameras.append(dict(view=view,video=pin['path'],video_pixels_rotated=False,video_display_rotation_degrees=abs(rotation[0]) if rotation else 0))
  C.write(source/'manifest.json',dict(export='/',camera_manifest=dict(cameras=cameras),schedule=[dict(tick=0,frames=missing)]))
  camera=C.module('agent_distill_cohort_original_incremental_extractor',HERE/'camera_profiles.py')
  camera.extract(SimpleNamespace(source=source,output=output,ticks=None,recording_uuid=None))

def finalize(output,coverage):
  lookup,bad_frames=raw_frames(coverage);completed,invalid_receipts=receipts(coverage)
  statuses=[];samples=[];audits=[]
  for expected in coverage['anchors']:
    identity=expected['id'];missing=[]
    for frame in expected['frames']:
      key=(expected['route_id'],expected['segment'],frame['view'],frame['encoding']['video_index'])
      if key not in lookup:missing.append(dict(view=frame['view'],video_index=frame['encoding']['video_index']))
    if identity in completed:
      item=completed[identity];samples.append(item['sample']);audits.append(item)
      status='complete';action=None
    elif missing:status='missing';action='Bounded single-scene --decode-scene '+identity+', preserving existing frames; then prepare from verified completed frames.'
    else:status='raw-complete-input-missing';action='Bounded --finish-existing run; no further decoding needed.'
    statuses.append(dict(id=identity,split=expected['split'],status=status,valid_native_frames=10-len(missing),missing_native_frames=missing,needed_action=action))
  value=dict(schema=1,protocol=C.protocol(),status='complete' if len(samples)==24 else 'partial',full_cohort_complete=len(samples)==24,
    expected_scene_ids=[row['id'] for row in coverage['anchors']],completed_scene_ids=[row['id'] for row in samples],
    missing_scene_ids=[row['id'] for row in statuses if row['status']!='complete'],failed_scene_ids=[],scene_status=statuses,
    completed_conditioned_inputs=len(samples),training_scenes=sum(row['split']=='train' for row in samples),validation_scenes=sum(row['split']=='validation' for row in samples),
    declared_drive_splits_unchanged=True,declared_training_scenes=16,declared_validation_scenes=8,
    samples=samples,scenes=audits,existing_valid_native_frame_count=len(lookup),invalid_native_files=bad_frames,invalid_or_incomplete_receipts=invalid_receipts,
    source_sha256={str(path):C.sha(path) for path in [Path(__file__),HERE/'agent-distill-cohort-common.py',HERE/'agent-distill-cohort-prepare.py',C.COVERAGE]},
    admissible_for_internal_layer_capture=bool(samples),admissible_for_driving_quality_validation=False,
    GPU_capture_adapter_ready=False,GPU_capture_blocker='Existing two12-prefill adapter requires all24 prepared inputs; this manifest exposes only verified complete scenes.',
    original_job_termination='Original decode-first process received SIGTERM at changed deadline. Completed and partial original frames remain untouched.',
    gpu_executed=False,training_executed=False,teacher_inference_executed=False,protected_or_sealed_drive_opened=False,
    current_fitting_protocol_changed=False,model_quality_or_latency_qualified=False)
  durable(output,value)
  print(json.dumps(dict(manifest=str(output),sha256=C.sha(output),status=value['status'],completed=value['completed_scene_ids'],missing=value['missing_scene_ids'],valid_native_frames=len(lookup),gpu_executed=False),indent=2))

def main():
  parser=argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--finish-existing',action='store_true');parser.add_argument('--decode-scene')
  parser.add_argument('--output-dir',type=Path,default=C.ROOT/'recovery-v1');parser.add_argument('--manifest-output',type=Path,required=True)
  parser.add_argument('--max-scenes',type=int,default=9);args=parser.parse_args()
  assert 1<=args.max_scenes<=24 and args.manifest_output.resolve().is_relative_to(C.ROOT) and not args.manifest_output.exists()
  C.storage(C.ROOT);C.source_mount();sys.addaudithook(C.cpu_guard);coverage=C.coverage()
  for path,digest in coverage['source_sha256'].items():assert C.sha(path)==digest,path
  sys.path[:0]=[str(HERE),'/tmp/opencode/openpilot-alpamayo','/home/john/asius/openpilot/opendbc_repo']
  lookup,_=raw_frames(coverage)
  if args.decode_scene:
    assert args.decode_scene in {row['id'] for row in coverage['anchors']};decode_scene(args.decode_scene,coverage,lookup)
    lookup,_=raw_frames(coverage)
  if args.finish_existing:
    known,_=receipts(coverage);selected=[]
    for row in coverage['anchors']:
      if row['id'] in known:continue
      if all((row['route_id'],row['segment'],frame['view'],frame['encoding']['video_index']) in lookup for frame in row['frames']):selected.append(row['id'])
    selected=selected[:args.max_scenes]
    if selected:finish_available(args.output_dir.resolve(),coverage,lookup,selected)
  finalize(args.manifest_output.resolve(),coverage);C.assert_no_devices()

if __name__=='__main__':main()
