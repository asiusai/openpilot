"""CPU-only nearest-native-history64 preparation, controls, and frozen parent commands."""
import argparse
import bisect
from collections import Counter, OrderedDict
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
WORK = Path('/mnt/nvme-500/tmp/alpamayo-speed-20261116')
OUT = WORK/'agent-history-nearest-data'
SOURCE = Path('/mnt/nvme-256/footage/f9ad9a96/exports/0000008e--321b2ddc03/alpamayo-4hz-calibrated-two-six')
RAW = Path('/mnt/nvme-500/tmp/alpamayo-route-4hz-20261116/native-frames')
RECIPE = WORK/'padding-source/alpamayo'
PY = Path('/tmp/opencode/alpamayo-asius-20261114/upstream/.venv/bin/python')
sys.dont_write_bytecode = True


def sha(path):
  with Path(path).open('rb') as stream: return hashlib.file_digest(stream,'sha256').hexdigest()


def read(path): return json.loads(Path(path).read_text())


def save(path,value):
  with Path(path).open('x') as stream: stream.write(json.dumps(value,indent=2,allow_nan=False)+'\n')


def module(path,name):
  spec = importlib.util.spec_from_file_location(name,path)
  result = importlib.util.module_from_spec(spec);sys.modules[name]=result;spec.loader.exec_module(result)
  return result


def copy_exact(source,target,hashes):
  target.parent.mkdir(parents=True,exist_ok=True)
  if target.exists():
    if sha(target) != sha(source): raise ValueError('Snapshot collision')
  else: shutil.copy2(source,target)
  if sha(source) != sha(target): raise ValueError('Source copy changed bytes')
  hashes[str(target)] = sha(target)


def controls(policy):
  rng = np.random.default_rng(171117)
  cases = 0
  for _ in range(500):
    times = np.cumsum(rng.integers(40_000_000,60_000_000,size=100)).tolist()
    query = times[70]+int(rng.integers(0,10_000_000))
    desired = query-int(rng.integers(0,2_000_000_000))
    eligible = [(abs(t-desired),t,i) for i,t in enumerate(times) if t <= query]
    expected = min(eligible)[2]
    assert policy.select_index(times,desired,query) == expected
    future = times[:71]+[t+1_000_000_000_000 for t in times[71:]]
    assert policy.select_index(future,desired,query) == expected
    cases += 1
  assert policy.select_index([0,100_000_000,200_000_000],50_000_000,200_000_000) == 0
  assert policy.select_index([0,100_000_000,200_000_000],190_000_000,199_000_000) == 1
  try: policy.select_index([100],200,100)
  except ValueError: pass
  else: raise AssertionError('Future desired input accepted')
  return dict(independent_nearest_cases=cases,ties_choose_past=True,unavailable_future_mutation_ignored=True,
              no_capture_after_query=True,no_output_or_gt_selection=True)


def decode_missing(missing,source,local_cache,raw_records):
  import av
  written,decode_controls = [], []
  for camera,indices in missing.items():
    if not indices: continue
    record = next(item for item in source['camera_manifest']['cameras'] if item['view'] == camera)
    if record['video_display_rotation_degrees'] != 0: raise ValueError('Unexpected front-camera orientation')
    video = Path(source['export'])/record['video']
    target = local_cache/str(camera);target.mkdir(parents=True,exist_ok=True)
    wanted = set(indices)
    # Sequential dependency decoding preserves video-index identity despite variable PTS.
    # Only the two missing required captures are persisted; no full route is generated.
    with av.open(str(video)) as container:
      stream = container.streams.video[0];stream.thread_type='AUTO';stream.thread_count=2
      for index,frame in enumerate(container.decode(stream)):
        if index in wanted or index in {i-1 for i in wanted}:
          image = np.ascontiguousarray(frame.to_ndarray(format='rgb24').transpose(2,0,1))
          previous = RAW/str(camera)/f'{index}.npy'
          if previous.exists():
            np.testing.assert_array_equal(image,np.load(previous,allow_pickle=False,mmap_mode='r'))
            decode_controls.append(dict(camera=camera,video_index=index,existing_native_bytes_exact=True))
          if index in wanted:
            path = target/f'{index}.npy';np.save(path,image,allow_pickle=False)
            native = raw_records[camera][index]
            written.append(dict(camera=camera,video_index=index,frame_id=native['frame_id'],capture_ns=native['capture_ns'],
                                file=str(path),sha256=sha(path),source_video=str(video)))
        if index >= max(wanted): break
    if any(not(target/f'{i}.npy').exists() for i in wanted): raise ValueError('Missing decoded native frame')
  return written,decode_controls


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--prepare',action='store_true',help='Authorized CPU decoding/calibration of this one64-input policy')
  args = parser.parse_args()
  if not args.prepare: parser.error('Use --prepare for CPU-only preparation; this tool cannot run GPU work')
  if OUT.exists(): raise FileExistsError('Preserve existing prepared evidence')
  if not OUT.is_relative_to(WORK) or str(OUT).startswith('/mnt/nvme-256/'): raise ValueError('Writes must stay on500GB WORK')
  os.environ.update(DEV='CPU',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONDONTWRITEBYTECODE='1')
  sys.path.insert(0,str(HERE));sys.path.insert(0,str(HERE.parent))
  policy = module(HERE/'agent-history-nearest-policy.py','agent_history_nearest_policy')
  interpolation = module(HERE/'calibration_trials.py','agent_history_nearest_original_calibration')
  old_inv = read(WORK/'full-padded-quality/invocation.json')
  command = old_inv['command'];begin,end=command.index('--samples')+1,command.index('--initial-noise')
  paths = list(map(Path,command[begin:end]));assert len(paths)==64 and len(set(paths))==64
  source = read(SOURCE/'manifest.json')
  rows = {Path(row['sample']).name:row for row in source['schedule']}
  raw_path = Path(source['export'])/'clips/native-timestamps.json'
  native = read(raw_path)['views'];native={c:native[str(c)] for c in (1,6)}
  times = {c:[row['capture_ns'] for row in native[c]] for c in (1,6)}
  assert all(all(a<b for a,b in zip(ts,ts[1:])) for ts in times.values())
  calibrations = read(SOURCE/'render-calibrations.json')
  cal_records = np.array([row['ns'] for row in calibrations],dtype=np.int64),calibrations
  old_ring = read(WORK/'stream-10hz/ring/manifest.json')
  ring_by = {Path(row['sample']).name:row for row in old_ring['schedule']}
  ring_report = read(WORK/'ring-10hz-full/report.json')
  ring_entries = ring_report['warmup']+ring_report['runs']
  ring_names = {Path(entry['sample']).name for entry in ring_entries}
  schedules,needed,changes,ring_differences = [],{1:set(),6:set()},[],[]
  for path in paths:
    original = rows[path.name];query=int(original['query_ns'])
    assert query//1000==int(path.name.removeprefix('sample-'))
    assert tuple(frame['view'] for frame in original['frames'])==(1,)*5+(6,)*5
    selected=[]
    for slot,frame in enumerate(original['frames']):
      camera=frame['view'];desired=query+policy.OFFSETS_NS[slot%5]
      floor=bisect.bisect_right(times[camera],desired)-1
      assert floor==frame['video_index'] and times[camera][floor]==frame['capture_ns']
      index=policy.select_index(times[camera],desired,query)
      capture=int(times[camera][index]);_,cal_ns=interpolation.asof(cal_records,capture,1_000_000_000)
      entry=dict(view=camera,video_index=index,frame_id=native[camera][index]['frame_id'],capture_ns=capture,
                 requested_ns=desired,image_calibration_ns=cal_ns)
      assert capture<=query
      if slot%5==4:assert entry['video_index']==frame['video_index']
      if index!=floor:
        changes.append(dict(sample=path.name,query_ns=query,slot=slot,history_offset_ns=policy.OFFSETS_NS[slot%5],
                            original=frame,nearest=entry,nearest_error_ns=capture-desired,
                            floor_error_ns=frame['capture_ns']-desired,causal_guard_passed=capture<=query))
      ring_frame=ring_by[path.name]['frames'][slot]
      if ring_frame['video_index']!=index:ring_differences.append(dict(sample=path.name,slot=slot,nearest=entry,old_ring=ring_frame))
      needed[camera].add(index);selected.append(entry)
    schedules.append((path,original,selected))
  assert len(changes)==192 and len(ring_differences)==2  # Bounded known64 fixture, not a tunable selector.
  OUT.mkdir(parents=True)
  local_cache=WORK/'agent-history-nearest-raw-cache'
  missing={c:sorted(i for i in ids if not(RAW/str(c)/f'{i}.npy').exists()) for c,ids in needed.items()}
  assert sum(map(len,missing.values()))==2
  written,decode_controls=decode_missing(missing,source,local_cache,native)
  cache=OrderedDict();raw_hashes={};candidate_paths=[];input_records=[];manifest_rows=[];floor_pixel_controls=0
  def calibrated(frame):
    camera,index=frame['view'],frame['video_index']
    calibration,stamp=interpolation.asof(cal_records,frame['capture_ns'],1_000_000_000)
    assert stamp==frame['image_calibration_ns']
    key=(camera,index,stamp)
    if key not in cache:
      raw=RAW/str(camera)/f'{index}.npy'
      if not raw.exists():raw=local_cache/str(camera)/f'{index}.npy'
      pixels=np.load(raw,allow_pickle=False,mmap_mode='r');raw_hashes[str(raw)]=sha(raw)
      cache[key]=interpolation.calibrated_image(pixels,camera,source['source_camera_geometry'],source['target_camera_geometry'],calibration)
    cache.move_to_end(key)
    while len(cache)>160:cache.popitem(last=False)
    return cache[key]
  for sequence,(path,original,selected) in enumerate(schedules):
    old_images=np.load(path/'image_frames.npy',allow_pickle=False,mmap_mode='r')
    assert old_images.shape==(10,1,3,432,768) and old_images.dtype==np.uint8
    old_stamps=np.load(path/'absolute_timestamps.npy',allow_pickle=False)
    np.testing.assert_array_equal(old_stamps,[frame['capture_ns']//1000 for frame in original['frames']])
    images=[];per_slot=[]
    for slot,(before,after) in enumerate(zip(original['frames'],selected,strict=True)):
      control=calibrated(before)
      np.testing.assert_array_equal(control,old_images[slot,0]);floor_pixel_controls+=1
      image=control if before['video_index']==after['video_index'] else calibrated(after)
      if slot%5==4:np.testing.assert_array_equal(image,old_images[slot,0])
      images.append(image[None]);per_slot.append(dict(slot=slot,camera=after['view'],video_index=after['video_index'],
        frame_id=after['frame_id'],capture_ns=after['capture_ns'],image_calibration_ns=after['image_calibration_ns'],
        capture_not_after_query=after['capture_ns']<=original['query_ns'],
        floor_pixel_sha256=hashlib.sha256(old_images[slot,0].tobytes()).hexdigest(),nearest_pixel_sha256=hashlib.sha256(image.tobytes()).hexdigest()))
    target=OUT/'samples'/path.name;target.mkdir(parents=True)
    for name in ('camera_indices','ego_history_xyz','ego_history_rot','nav_text'):
      shutil.copy2(path/(name+'.npy'),target/(name+'.npy'))
      assert sha(path/(name+'.npy'))==sha(target/(name+'.npy'))
    np.save(target/'image_frames.npy',np.stack(images),allow_pickle=False)
    np.save(target/'absolute_timestamps.npy',np.array([frame['capture_ns']//1000 for frame in selected],dtype=np.int64),allow_pickle=False)
    record=dict(sample=str(target),original_sample=str(path),query_ns=original['query_ns'],frames=per_slot,
      nonimage_arrays_exact=True,current_images_exact=True,
      file_sha256={p.name:sha(p) for p in target.glob('*.npy')},original_file_sha256={p.name:sha(p) for p in path.glob('*.npy')})
    input_records.append(record);candidate_paths.append(str(target))
    manifest_rows.append(dict(sample=str(target),query_ns=original['query_ns'],display_time_s=original['display_time_s'],
      recorded_motion_future_xyz=original['recorded_motion_future_xyz']))
    print(json.dumps(dict(prepared_cpu_input=sequence+1,total=64)),flush=True)
  # CPU tokenization checks use actual captures and original full-resolution images.
  import torch
  torch.set_num_threads(1)
  fast=module(RECIPE/'experiments/fast-inputs.py','agent_history_nearest_frozen_fast_inputs')
  adapter=fast.FastInputs(option(command,'checkpoint'),camera_profile='nvidia',allow_missing_cameras=True,cache_resize=True,direct_byte_inputs=True)
  prompts={'floor':[],'nearest':[]}
  for record in input_records:
    for kind,folder in (('floor',Path(record['original_sample'])),('nearest',Path(record['sample']))):
      data={p.stem:[str(np.load(p,allow_pickle=False).item())] if p.stem=='nav_text' else torch.from_numpy(np.load(p,allow_pickle=False,mmap_mode='c')) for p in folder.glob('*.npy')}
      config,_,prepared=adapter.prepare(data);packed=adapter.native_inputs(prepared)
      assert prepared['tokenized_data']['image_grid_thw'].tolist()==[[1,20,36]]*10
      assert int((packed[0]==config.vlm_config.image_token_id).sum())==1800
      assert packed[0].shape[1]+6<=2048
      prompts[kind].append(packed[0].shape[1])
  comparison_manifest=OUT/'agent-history-nearest-comparison-manifest.json'
  save(comparison_manifest,dict(schedule=manifest_rows,classification='Comparison-only recorded motion; never used for frame selection'))
  hashes={};snap=WORK/'agent-history-nearest-source/alpamayo';fexp=snap/'experiments';fexp.mkdir(parents=True)
  for origin,destination in ((RECIPE,snap),(RECIPE/'experiments',fexp)):
    for path in origin.iterdir():
      if path.is_file() and path.suffix in ('.py','.c','.json') and not path.name.startswith('agent-'):
        copy_exact(path,destination/path.name,hashes)
  for name in ('agent-speculation-run.py','agent-speculation-policy.py','agent-speculation-prefill.py','agent-attention-rowsoftmax.py'):
    copy_exact(RECIPE/'experiments'/name,fexp/name,hashes)
  for path in HERE.glob('agent-history-nearest-*.py'):copy_exact(path,fexp/path.name,hashes)
  recipe=read(WORK/'padding-source/padding2048-quality.json')
  args_command=recipe['command']
  vision=args_command.index('--vision-calibration');del args_command[vision:vision+2]
  args_command[1]=str(fexp/'benchmark-pipeline.py')
  args_command[args_command.index('--reduced-vocab')+1]=str(fexp/'coc_vocab.json')
  args_command[args_command.index('--pad-prompt-to')+1]='2048'
  recipe['environment']['PYTHONPATH']=str(Path('/tmp/opencode/alpamayo-asius-20261114/tinygrad'))+':'+str(snap)
  quality_commands={};invocations={}
  for kind,samples in (('floor',list(map(str,paths))),('nearest',candidate_paths)):
    invocation=copy.deepcopy(recipe)
    begin,end=invocation['command'].index('--samples')+1,invocation['command'].index('--initial-noise')
    invocation['command'][begin:end]=samples
    invocation['command'][invocation['command'].index('--output')+1]=str(WORK/('agent-history-nearest-quality-'+kind)/'report.json')
    inv=OUT/f'agent-history-nearest-{kind}-invocation.json';save(inv,invocation);hashes[str(inv)]=sha(inv);invocations[kind]=str(inv)
    quality_commands[kind]=[str(PY),str(fexp/'agent-speculation-run.py'),'--output',str(WORK/('agent-history-nearest-quality-'+kind)/'report.json'),
       '--invocation',str(inv),'--mode','candidate','--prefill-attention','shuffle-fma','--pad-prompt-to','2048','--warmup','2','--runs','62']
  for record in input_records:
    for name,digest in record['file_sha256'].items():hashes[str(Path(record['sample'])/name)]=digest
    for name,digest in record['original_file_sha256'].items():hashes[str(Path(record['original_sample'])/name)]=digest
  for path in (SOURCE/'manifest.json',SOURCE/'render-calibrations.json',raw_path,HERE/'calibration_trials.py',HERE/'camera_profiles.py',HERE/'prepare-stream.py',
               Path('/tmp/opencode/alpamayo-camera-cadence-audit.py'),WORK/'camera-cadence-audit.json',Path(option(command,'initial-noise')),
               Path(option(command,'prefill-calibration')),comparison_manifest):hashes[str(path)]=sha(path)
  hashes.update(raw_hashes)
  ring_comparison=read(WORK/'ring-10hz-full/comparison-floor.json')
  audit=dict(classification='NUMERICAL INPUT POLICY',cpu_only=True,gpu_executed=False,quality_qualified=False,queries=64,
    one_policy='nearest native frame per original historical target; all selected captures<=query; ties prefer past',
    offsets_ns=list(policy.OFFSETS_NS),changed_frames=changes,changed_frame_count=len(changes),full_floor_pixel_controls=floor_pixel_controls,
    current_image_controls=128,exact_nonimage_samples=64,native_timestamp_sha256=sha(raw_path),
    calibration_helper_sha256=sha(HERE/'calibration_trials.py'),unit_controls=controls(policy),decoded_missing=written,decoder_native_byte_controls=decode_controls,
    original_ring_evidence=dict(report_sha256=sha(WORK/'ring-10hz-full/report.json'),queries=len(ring_entries),
      overlapping_original64=len(ring_names&{path.name for path in paths}),all_four_windows_tested=False,
      comparison=ring_comparison['matched'],nearest_native_differences_from_prepared_ring=ring_differences),
    samples=input_records,real_cpu_prompt_lengths=prompts,
    recipe='Dynamic vision, fixed language,2048 prompt, genuine target-verified NOMINAL speculation, shuffle-FMA attention; fresh paired64 reference required',
    measurement='Input preparation/calibration excluded from these prepared-input quality trial timings',
    exact_same_query_and_learned_history_and_navigation=True,no_gt_or_output_based_selection=True)
  save(OUT/'agent-history-nearest-audit.json',audit)
  protocol=dict(classification='NUMERICAL INPUT POLICY',queries=64,quality_qualified=False,frozen_sha256=hashes,
    quality_commands=quality_commands,sample_paths=dict(floor=list(map(str,paths)),nearest=candidate_paths),
    automatic_speed_or_native_benchmark=False,audit=str(OUT/'agent-history-nearest-audit.json'),comparison_manifest=str(comparison_manifest))
  protocol_path=OUT/'agent-history-nearest-protocol.json';save(protocol_path,protocol)
  parent=[]
  for condition in ('floor','nearest'):
    name='agent-history-nearest-quality-'+condition
    parent.append([str(PY),'/tmp/opencode/run-alpamayo-speed-trial.py',name,'--program',str(fexp/'agent-history-nearest-run.py'),
      '--timeout-seconds','900','--set','condition',condition,'--set','protocol',str(protocol_path),'--flag','execute-gpu',
      '--env','PYTHONPATH',recipe['environment']['PYTHONPATH'],'--env','PYTHONDONTWRITEBYTECODE','1'])
  comparison=[str(PY),str(fexp/'compare-speed.py'),'--reference',str(WORK/'agent-history-nearest-quality-floor/report.json'),
     '--report',str(WORK/'agent-history-nearest-quality-nearest/report.json'),'--manifest',str(comparison_manifest),
     '--output',str(WORK/'agent-history-nearest-quality-nearest/comparison-floor.json')]
  for a,b in ((35,43),(112,120),(164,172),(228,236)):comparison+=['--window',str(a),str(b)]
  save(OUT/'agent-history-nearest-parent-plan.json',dict(gpu_executed=False,commands=parent,comparison_command=comparison,
    quality_gate='At most5% ADE and FDE growth in every window; shape/speed/slowing/reasoning review; no automatic adoption/speed test'))
  print(json.dumps(dict(output=str(OUT),changed_frame_slots=len(changes),raw_frames_decoded=len(written),
                       full_floor_pixel_controls=floor_pixel_controls,gpu_executed=False),indent=2))


def option(command,name):return command[command.index('--'+name)+1]


if __name__=='__main__':main()
