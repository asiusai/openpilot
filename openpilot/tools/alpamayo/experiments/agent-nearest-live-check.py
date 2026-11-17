"""Bounded CPU audit: actual64 availability, native calibration, full arrays and Consumer packing."""
import argparse
from collections import OrderedDict
import copy
from dataclasses import replace
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

HERE=Path(__file__).resolve().parent
WORK=Path('/mnt/nvme-500/tmp/alpamayo-speed-20261116')
OUT=WORK/'agent-nearest-live-data'
BASE=WORK/'agent-paced-dynamic-source-cpu02/alpamayo'
EXP=BASE/'experiments'
SOURCE=Path('/mnt/nvme-256/footage/f9ad9a96/exports/0000008e--321b2ddc03/alpamayo-4hz-calibrated-two-six')
RAW=Path('/mnt/nvme-500/tmp/alpamayo-route-4hz-20261116/native-frames')
CASES=(35.,116.5,167.,228.)  # Fixed before this audit; no ratio/error-based search.
sys.dont_write_bytecode=True


def sha(path):
  with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def read(path):return json.loads(Path(path).read_text())
def save(path,value):
  with Path(path).open('x') as f:f.write(json.dumps(value,indent=2,allow_nan=False)+'\n')
def load(path,name):
  s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m
def local(name):return load(EXP/(name+'.py'),'agent_nearest_live_'+name.replace('-','_'))


def independent(policy,bridge,Store):
  rng=np.random.default_rng(17112026)
  for _ in range(500):
    times=np.cumsum(rng.integers(40_000_000,60_000_000,100)).tolist()
    q=times[75];wanted=q-int(rng.integers(0,2_000_000_000))
    expected=min((abs(t-wanted),t,i) for i,t in enumerate(times) if t<=q)[2]
    assert policy.nearest_index(times,wanted,q)==expected
  assert policy.nearest_index([100,200,300],150,300)==0
  assert policy.nearest_index([100,200,300],300,299)==-1
  pixels=np.zeros((3,1,1),np.uint8);pixels.flags.writeable=False
  q=10_000_000_000;now=q+20_000_000
  store=Store(byte_limit=384*1024**2)
  for camera in (1,6):
    for i,t in enumerate(range(q-2_200_000_000,q+1,50_000_000)):
      # The closest -2s observation is captured, but NOT ready/received by dispatch.
      delayed=camera==6 and t==q-2_000_000_000
      ready=now+1 if delayed else t+10_000_000
      assert store.add_frame(bridge.Frame(camera,i,t,t+5_000_000,ready,ready,pixels))
  selected=store.select(now,0,{},250_000_000,False)
  assert selected.frames[5].capture_ns==q-2_050_000_000
  assert all(f.ready_ns<=now and f.capture_ns<=q for f in selected.frames)
  assert selected.frames[4].capture_ns==q and selected.frames[9].capture_ns==q
  for args,expected in (((now,q,{},250_000_000,False),'no_new_query'),
                        ((now,0,{1:q},250_000_000,False),'stale_current_camera'),
                        ((q+300_000_000,0,{},250_000_000,False),'stale_query'),
                        ((now,0,{},250_000_000,True),'missing_matching_small_model')):
    try:store.select(*args)
    except bridge.Unavailable as e:assert str(e)==expected
    else:raise AssertionError('Original guard was bypassed')
  current=selected.frames[-1]
  store.add_state('modelV2',q,now,dict(frameId=current.frame_id,timestampEof=current.eof_ns))
  matched=store.select(now,0,{},250_000_000,True)
  assert matched.small_model==dict(frameId=current.frame_id,timestampEof=current.eof_ns)
  return dict(independent_nearest_cases=500,tie_past=True,unreceived_capture_excluded=True,current_unchanged=True,
    no_new_query_guard=True,stale_current_guard=True,stale_query_guard=True,matching_small_model_guard=True)


def main():
  parser=argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--full64',action='store_true',help='Also reconstruct all64 calibrated inputs and exercise actual Consumer packing')
  args=parser.parse_args()
  if os.environ.get('DEV')!='CPU':raise ValueError('Set DEV=CPU explicitly')
  os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',XDG_CACHE_HOME=str(OUT/'agent-nearest-live-cpu-cache'))
  sys.path[:0]=[str(BASE),str(EXP),'/tmp/opencode/openpilot-alpamayo','/home/john/asius/openpilot/opendbc_repo']
  OUT.mkdir(exist_ok=True)
  target=OUT/('agent-nearest-live-full64.json' if args.full64 else 'agent-nearest-live-metadata64.json')
  if target.exists():raise FileExistsError('CPU evidence is immutable')
  policy=load(HERE/'agent-nearest-live-policy.py','agent_nearest_live_owned_policy')
  dynamic=local('agent-paced-dynamic-policy');dynamic.verify_snapshot()
  paced=local('agent-paced-replay');bridge=paced.bridge;live=local('agent-live-consumer')
  classes=policy.store_classes(bridge);Consumer=policy.consumer_class(live)
  originals=(bridge.Store.select,bridge.Preparer.prepare,live.Consumer._data,live.Consumer.predict,live.Consumer.synchronize)
  controls=independent(policy,bridge,classes.Store)
  audit=read(WORK/'agent-history-nearest-data/agent-history-nearest-audit.json')
  original=read(SOURCE/'manifest.json')
  manifest={k:original[k] for k in ('source_camera_geometry','target_camera_geometry')}
  source_rows={Path(row['sample']).name:{k:row[k] for k in ('query_ns','display_time_s','navigation_text')} for row in original['schedule'] if row.get('sample')}
  del original
  samples=[dict(record,**source_rows[Path(record['sample']).name]) for record in audit['samples']]
  records=[];input_files={};raw_files={};case_rows=[]
  zero=np.zeros((3,1,1),np.uint8);zero.flags.writeable=False
  if args.full64:
    import torch
    torch.set_num_threads(1)
    from tinygrad import Device
    assert Device.DEFAULT=='CPU' and not Device._opened_devices
    cpu=local('agent-live-consumer-check');replay=local('agent-live-replay')
    fast=local('fast-inputs');compact=local('compact-inputs').CompactInputs;Vision=local('vision-cache').VisionCache
    spec=local('agent-speculation-policy')
    from native_warp import calibrated_image,library
    library()
    clock=cpu.Clock();model=cpu.fake_model(clock)
    model.padded_prefill.last_targets=None
    adapter=fast.FastInputs('/tmp/opencode/alpamayo-asius-20261114/model',camera_profile='nvidia',allow_missing_cameras=True,cache_resize=True,direct_byte_inputs=True)
    direct=fast.FastInputs('/tmp/opencode/alpamayo-asius-20261114/model',camera_profile='nvidia',allow_missing_cameras=True,cache_resize=True,direct_byte_inputs=True)
    flags=read(WORK/'agent-startup-spec-candidate/report.json')['protocol']['flags']
    settings=SimpleNamespace(max_tokens=flags['max_tokens'],vision_batch=1,byte_pixels=True,cache_rotary_inputs=flags['cache_rotary_inputs'],
      coarse_timing=flags['coarse_timing'],greedy=True,diffusion_steps=3,initial_noise=np.load(flags['initial_noise'],allow_pickle=False))
    padding=lambda prepared,packed,length:spec.padded_inputs(prepared,packed,length,spec.CHECKPOINT_DRAFT_IDS,adapter.config.traj_ids['future_start'])
    vision=Vision(model,capacity=48)
    class Decoder:
      def __init__(self):
        self.model=model;self.prefill_draft_records=[];self.history=[];self.statistics=[]
        self.jits={};self.verifiers={};self.select_jits={};self.warmup_graphs=False;self.before_diffusion=lambda:None
      def generate(self,*a,**kw):raise AssertionError('No model inference in CPU packing checks')
      def run(self,m,args,config,tokenizer,prepared,packed,seed,inputs):
        ep,ek,eu,es=self.expected
        cpu.exact(prepared,ep)
        for a,b in zip(packed,ek,strict=True):cpu.exact(a,b)
        for a,b in zip(inputs.cpu,eu,strict=True):cpu.exact(a.value,b)
        assert inputs._signature==es
        self.capture=dict(prepared=replay.tree(prepared),packed=replay.tree(packed),cpu_buffers=replay.cpu_buffers(inputs),vision_keys=replay.tree(vision.keys))
        tokens=list(spec.CHECKPOINT_DRAFT_IDS)
        self.prefill_draft_records.append(dict(actual_prompt=prepared['actual_prompt_length'],generated=tokens,eligible=True,accepted=True,targets=tokens,all_targets_finite=True))
        return dict(xyz=np.zeros((1,64,3)).tolist(),rot=np.broadcast_to(np.eye(3),(1,64,3,3)).tolist(),generated_token_ids=tokens)
    decoder=Decoder();Inputs=cpu.cpu_inputs_class(compact)
    consumer=Consumer(adapter=adapter,model=model,runner=decoder.run,settings=settings,config=adapter.config,input_class=Inputs,
      vision_cache=vision,pad_inputs=padding,pad_length=2048,initializer_provenance={'CPU_fixture':True},decoder=decoder,
      speculation_policy=spec,speculation_draft=spec.CHECKPOINT_DRAFT_IDS,clock=clock)
  for start in (35.,112.,164.,228.):
    trace=paced.Trace(SOURCE/'manifest.json',paced.LOGS,start,8.)
    store=classes.Store(frame_limit=64,byte_limit=384*1024**2)
    lookup={(ref['view'],ref['frame_id']):ref for ref in trace.frames}
    by_query={ref['capture_ns']:ref for ref in trace.frames if ref['view']==6}
    position=0;full_frames=OrderedDict()
    if args.full64:preparer=classes.Preparer(manifest['source_camera_geometry'],manifest['target_camera_geometry'],warp=calibrated_image,clock=clock)
    for row in [r for r in samples if start<=r['display_time_s']<start+8]:
      query=row['query_ns'];now=by_query[query]['release_ns']  # Fixed earliest recorded release; never adjusted to force parity.
      while position<len(trace.events) and trace.events[position].release_ns<=now:
        event=trace.events[position];position+=1
        if event.kind=='frame':
          ref=event.value
          assert store.add_frame(bridge.Frame(ref['view'],ref['frame_id'],ref['capture_ns'],ref['eof_ns'],event.release_ns,event.release_ns,zero))
        else:store.add_state(event.kind,event.stamp,event.release_ns,event.value,event.valid)
      record=dict(sample=row['sample'],query_ns=query,display_time_s=row['display_time_s'],selected_ns=now,
        release_model='earliest original road publication release; receipt/IO latency unknown; metadata-only admission simulation')
      try:
        selection=store.select(now,0,{},250_000_000,False)
        assert selection.query_ns==query
        baseline=bridge.Store.select(store,now,0,{},250_000_000,False)
        assert selection.frames[4] is baseline.frames[4] and selection.frames[9] is baseline.frames[9]
        assert selection.small_model is baseline.small_model and selection.ingest_cpu_ns==baseline.ingest_cpu_ns
        assert all(all(a is b for a,b in zip(selection.states[k],baseline.states[k],strict=True)) for k in selection.states)
        available=[r for r in trace.frames if r['release_ns']<=now and r['capture_ns']<=query]
        independent_frames=[]
        for camera in (1,6):
          choices=[r for r in available if r['view']==camera]
          for offset in bridge.OFFSETS_NS:
            desired=query+offset
            expected=(max((r for r in choices if r['capture_ns']<=desired),key=lambda r:r['capture_ns']) if offset==0 else
              min(choices,key=lambda r:(abs(r['capture_ns']-desired),r['capture_ns'])))
            independent_frames.append(expected)
        assert [f.frame_id for f in selection.frames]==[r['frame_id'] for r in independent_frames]
        wanted=row['frames'];selected=[lookup[(f.camera,f.frame_id)] for f in selection.frames]
        differences=[dict(slot=i,offline=w,available=dict(frame_id=f.frame_id,capture_ns=f.capture_ns))
          for i,(w,f) in enumerate(zip(wanted,selection.frames,strict=True)) if (w['frame_id'],w['capture_ns'])!=(f.frame_id,f.capture_ns)]
        record.update(available_selection_completed=True,offline_nearest_frame_differences=differences,
          frames=[dict(f.reference(),requested_ns=query+bridge.OFFSETS_NS[i%5],video_index=selected[i]['video_index']) for i,f in enumerate(selection.frames)])
        if args.full64:
          actual=[]
          for frame,ref in zip(selection.frames,selected,strict=True):
            key=(frame.camera,frame.frame_id)
            if key not in full_frames:
              path=RAW/str(frame.camera)/f'{ref["video_index"]}.npy'
              if not path.exists():path=WORK/'agent-history-nearest-raw-cache'/str(frame.camera)/f'{ref["video_index"]}.npy'
              if not path.exists():path=WORK/'agent-paced-replay-cache-35-43/rgb'/str(frame.camera)/f'{ref["video_index"]}.npy'
              pixels=np.ascontiguousarray(np.load(path,allow_pickle=False));pixels.flags.writeable=False
              assert pixels.shape==(3,760,1344) and pixels.dtype==np.uint8
              raw_files[str(path)]=sha(path);full_frames[key]=replace(frame,pixels=pixels)
            full_frames.move_to_end(key);actual.append(full_frames[key])
            while len(full_frames)>64:full_frames.popitem(last=False)
          clock.now=now
          arrays,detail=preparer.prepare(replace(selection,frames=tuple(actual)))
          arrays['nav_text']=[row['navigation_text']]
          metadata=dict(query_ns=query,selected_ns=now,prepared_ns=clock(),navigation_text=row['navigation_text'],**detail)
          expected=cpu.load_case(dict(sample=row['sample']))[0]
          changed=[k for k in arrays if replay.digest(arrays[k])!=replay.digest(expected[k])]
          calibration_differences=[i for i,(ref,want) in enumerate(zip(detail['frames'],wanted,strict=True)) if ref['image_calibration_ns']!=want['image_calibration_ns']]
          decoder.expected=cpu.prepare_reference(direct,arrays,settings,compact,padding,2048)
          output=consumer.predict(arrays,copy.deepcopy(metadata));consumer.synchronize()
          # No fixture output is persisted or counted as a model prediction.
          record.update(full_arrays_changed=changed,calibration_differences=calibration_differences,
            prepared_consumer_hashes=decoder.capture,consumer_input_reused=output['live_consumer']['input_reused'],
            full_arrays_exact_saved_nearest=not changed and not calibration_differences)
          if changed or calibration_differences:
            folder=OUT/'agent-nearest-live-available-only'/Path(row['sample']).name;folder.mkdir(parents=True,exist_ok=True)
            for key,value in arrays.items():
              path=folder/(key+'.npy');value=np.asarray(value[0]) if key=='nav_text' else value
              if path.exists():np.testing.assert_array_equal(np.load(path,allow_pickle=False),value)
              else:np.save(path,value,allow_pickle=False)
            reference_folder=folder
          else:reference_folder=Path(row['sample'])
          record['available_only_reference_sample']=str(reference_folder)
          for path in reference_folder.glob('*.npy'):input_files[str(path)]=sha(path)
          if row['display_time_s'] in CASES:
            case_rows.append(dict(display_time_s=row['display_time_s'],query_ns=query,sample=str(reference_folder),saved_nearest_sample=row['sample'],
              metadata=metadata,expected_cpu_hashes=decoder.capture,exact_saved_nearest_input=record['full_arrays_exact_saved_nearest']))
          del arrays,expected,output
      except bridge.Unavailable as error:
        record.update(available_selection_completed=False,unavailable=str(error))
      records.append(record)
      print(json.dumps(dict(cpu_query=len(records),total=64,available=record['available_selection_completed'])),flush=True)
  assert len(records)==64
  assert originals==(bridge.Store.select,bridge.Preparer.prepare,live.Consumer._data,live.Consumer.predict,live.Consumer.synchronize)
  result=dict(cpu_only=True,gpu_executed=False,model_predictions_computed=0,policy=policy.POLICY,queries=64,controls=controls,
    records=records,original_methods_unchanged=True,source_snapshot_manifest_sha256=sha(BASE.parent/'manifest.json'),
    source_sha256={str(p):sha(p) for p in (HERE/'agent-nearest-live-policy.py',Path(__file__),EXP/'agent-live-bridge.py',EXP/'agent-live-consumer.py')},
    availability_limit='Original publication releases simulate admission. Actual subscriber receipt/IO timing is unlogged; parent paced Producer must enforce real ready times.',
    full64=args.full64,predeclared_parity_times=list(CASES),parent_gpu_cases=case_rows,unavailable_queries=sum(not r['available_selection_completed'] for r in records),
    exact_offline_nearest_frames=sum(r.get('available_selection_completed') and not r.get('offline_nearest_frame_differences') for r in records))
  if args.full64:
    assert not Device._opened_devices
    result.update(full_array_exact_count=sum(r.get('full_arrays_exact_saved_nearest',False) for r in records),
      consumer_cpu_hash_cases=sum('prepared_consumer_hashes' in r for r in records),input_sha256=input_files,raw_sha256=raw_files,
      CPU_fixture='Actual Consumer.predict/synchronize and real adapters, host storage double, synthetic shape-only runner. No inference or output quality claim.')
    consumer.close()
  save(target,result)
  print(json.dumps(dict(output=str(target),queries=64,unavailable=result['unavailable_queries'],gpu_executed=False),indent=2))


if __name__=='__main__':main()
