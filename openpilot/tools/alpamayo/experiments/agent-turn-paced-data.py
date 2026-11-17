"""One predeclared recorded turn interval. CPU cache preparation; no model or training."""
import argparse
from collections import Counter
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
import time
from types import SimpleNamespace

WORK=Path('/mnt/nvme-500/tmp/alpamayo-speed-20261116')
SOURCE=WORK/'agent-paced-dynamic-source-cpu02/alpamayo'
EXP=SOURCE/'experiments'
MANIFEST=Path('/mnt/nvme-256/footage/f9ad9a96/exports/0000008e--321b2ddc03/alpamayo-4hz-calibrated-two-six/manifest.json')
LOGS=Path('/mnt/nvme-256/footage/f9ad9a96/realdata')
OLD_RAW=Path('/mnt/nvme-500/tmp/alpamayo-route-4hz-20261116/native-frames')
CACHE=WORK/'agent-turn-paced-cache-111-119'
DECLARATION=WORK/'agent-turn-paced-interval.json'
ARTIFACTS=WORK/'agent-turn-paced-artifacts'
START=111.;SECONDS=8.
MANIFEST_SHA='5100176f7e48eaeeaf314e4295eccaabbde5dc1674028e8fcbe2faaf746c24af'
ANCHORS={16:'sample-3256552835',24:'sample-3260551549'}

def sha(path):
  with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def read(path):return json.loads(Path(path).read_text())
def write(path,value):
  with Path(path).open('x') as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n')
def load(path,name):
  spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec)
  sys.modules[name]=module;spec.loader.exec_module(module);return module

def configure():
  assert Path('/mnt/nvme-500').is_mount()
  for path in (ARTIFACTS,ARTIFACTS/'tmp',ARTIFACTS/'cache'):path.mkdir(exist_ok=True)
  os.environ.update(DEV='CPU',ALLOW_DEVICE_USAGE='0',PYTHONDONTWRITEBYTECODE='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
    CACHEDB=str(ARTIFACTS/'compiler-cache.db'),TMPDIR=str(ARTIFACTS/'tmp'),XDG_CACHE_HOME=str(ARTIFACTS/'cache'))
  sys.dont_write_bytecode=True
  sys.path[:0]=[str(EXP),str(SOURCE),'/tmp/opencode/openpilot-alpamayo','/home/john/asius/openpilot/opendbc_repo']

def declare():
  assert sha(MANIFEST)==MANIFEST_SHA
  manifest=read(MANIFEST);origin=manifest['capture_start_ns'];quality=read(WORK/'quality-samples.json')
  anchors=[]
  for index,name in ANCHORS.items():
    assert Path(quality[index]).name==name
    row=next(r for r in manifest['schedule'] if Path(r.get('sample','')).name==name)
    assert origin+round(START*1e9)<=row['query_ns']<origin+round((START+SECONDS)*1e9)
    anchors.append(dict(original_quality_index=index,sample=name,query_ns=row['query_ns'],route_offset_s=(row['query_ns']-origin)/1e9,
      navigation=row['navigation_text'],manifest_tick=row['tick']))
  value=dict(route='0000008e--321b2ddc03',evaluation_only=True,training_permitted=False,manifest=str(MANIFEST),manifest_sha256=MANIFEST_SHA,
    origin_ns=origin,start_seconds=START,seconds=SECONDS,window=[111,119],prime_window=[107.7,111],
    source_start_ns=origin+111_000_000_000,source_end_ns=origin+119_000_000_000,history_prime_start_ns=origin+107_700_000_000,
    anchors=anchors,selection='Original received native20Hz FLOOR; 10Hz ceiling; one active; zero prediction queue',
    navigation_source=manifest['navigation_source'],forced_reasoning=False,source_frame_exclusions=False,
    scene_limitation='Parent visually confirmed the intersection and reported the original model misses part of the recorded turn. This benchmark does not establish correct driving.')
  if DECLARATION.exists():assert read(DECLARATION)==value
  else:write(DECLARATION,value)
  return value

def trace():
  configure();declare();module=load(EXP/'agent-paced-replay.py','turn_paced_canonical')
  value=module.Trace(MANIFEST,LOGS,START,SECONDS)
  assert value.start==read(DECLARATION)['source_start_ns'] and value.prime_start==read(DECLARATION)['history_prime_start_ns']
  return module,value

def prepare():
  import numpy as np
  module,t=trace();assert not CACHE.exists()
  anticipated=len(t.frames)*(3*760*1344+256)+64*1024**2
  usage=shutil.disk_usage(WORK)
  assert anticipated<=1536*1024**2 and module.bridge.storage_ok(usage.total,usage.free,anticipated)
  old=read(OLD_RAW/'manifest.json');assert sha(old['source_manifest'])==old['source_manifest_sha256']
  videos={}
  for c in (1,6):
    camera=next(x for x in t.manifest['camera_manifest']['cameras'] if x['view']==c)
    path=Path(t.manifest['export'])/camera['video'];assert path.stat().st_size<=512*1024**2
    assert old['cameras'][str(c)]['video']==str(path) and old['cameras'][str(c)]['sha256']==sha(path)
    videos[str(c)]=dict(path=str(path),sha256=sha(path),bytes=path.stat().st_size)
  reused=[];missing=[]
  for r in t.frames:
    p=OLD_RAW/str(r['view'])/f'{r["video_index"]}.npy'
    (reused if p.is_file() else missing).append(r)
  # One existing frame per camera is decoded independently as an orientation/index control.
  controls=[next(r for r in reused if r['view']==c) for c in (1,6)]
  CACHE.mkdir();(CACHE/'source').mkdir()
  decode_frames=missing+controls
  write(CACHE/'source/manifest.json',dict(t.manifest,schedule=[dict(tick=0,frames=decode_frames)]))
  begin=time.monotonic_ns()
  from camera_profiles import extract
  # Canonical extraction, bounded to two decoder workers and finite requested indices.
  # The original AUTO frame threading, decoder, RGB conversion, and orientation remain.
  import av
  original_open=av.open
  def bounded_open(path,*a,**kw):
    assert str(path) in {v['path'] for v in videos.values()}
    container=original_open(path,*a,**kw);container.streams.video[0].thread_count=2;return container
  from unittest.mock import patch
  decoded=CACHE/'decoded'
  with patch.object(av,'open',bounded_open):extract(SimpleNamespace(source=CACHE/'source',output=decoded,recording_uuid=None,ticks=None))
  rgb=CACHE/'rgb';rgb.mkdir();[(rgb/str(c)).mkdir() for c in (1,6)]
  for r in controls:
    key=Path(str(r['view']))/f'{r["video_index"]}.npy'
    assert sha(decoded/key)==sha(OLD_RAW/key),'Existing canonical frame does not match a fresh canonical decode'
  files={}
  for r in t.frames:
    key=Path(str(r['view']))/f'{r["video_index"]}.npy';old_path=OLD_RAW/key
    source=old_path if old_path.is_file() else decoded/key
    pixels=np.load(source,allow_pickle=False);assert pixels.shape==(3,760,1344) and pixels.dtype==np.uint8
    del pixels
    os.link(source,rgb/key);files[str(key)]=sha(rgb/key)
  inventory=dict(label=module.LABEL,frame_metadata_sha256=t.frame_digest,frames=len(t.frames),files=files,trace=t.provenance,
    decode_and_save_ms=(time.monotonic_ns()-begin)/1e6,source_extract_sha256=sha(EXP/'camera_profiles.py'),
    reused_frames=len(reused),decoded_missing_frames=len(missing),independent_overlap_controls=[dict(view=r['view'],video_index=r['video_index']) for r in controls],
    original_raw_manifest_sha256=sha(OLD_RAW/'manifest.json'),videos=videos,decoder_workers=2,
    maximum_requested_video_index=max(r['video_index'] for r in t.frames),maximum_source_video_bytes=max(v['bytes'] for v in videos.values()),
    source_reads='Finite canonical native CPU extraction; existing files hard-linked after source-video identity and independent decode controls. No calibrated images or model features are prepared here.')
  write(CACHE/'paced-cache.json',inventory)
  records=[dict(ordinal=e.ordinal,release_ns=e.release_ns,published_ns=e.published_ns,kind=e.kind,stamp=e.stamp,valid=e.valid) for e in t.events]
  write(CACHE/'trace-metadata.json',dict(provenance=t.provenance,frames=t.frames,events=records,
    primed_event_count=sum(e.release_ns<=t.start for e in t.events),invalid_state_counts=dict(Counter(e.kind for e in t.events if not e.valid)),
    navigation=[dict(query_ns=q,text=text) for q,text in t.navigation if t.prime_start<=q<=t.end]))
  print(json.dumps({k:inventory[k] for k in ('frames','reused_frames','decoded_missing_frames','decode_and_save_ms','maximum_requested_video_index')}),flush=True)

def main():
  parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--prepare',action='store_true');args=parser.parse_args()
  configure()
  if args.prepare:prepare()
  else:
    module,t=trace();existing=sum((OLD_RAW/str(r['view'])/f'{r["video_index"]}.npy').is_file() for r in t.frames)
    print(json.dumps(dict(declaration=read(DECLARATION),trace=t.provenance,reusable_existing_frames=existing,missing_frames=len(t.frames)-existing)),flush=True)

if __name__=='__main__':main()
