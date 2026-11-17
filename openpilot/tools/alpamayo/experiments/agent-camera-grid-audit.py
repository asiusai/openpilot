"""CPU metadata audit of ONE declared model-only10Hz grid. No model or pixel work."""
from bisect import bisect_right
from collections import Counter, OrderedDict
import hashlib
import importlib.util
import json
from pathlib import Path
import statistics
import sys

WORK=Path('/mnt/nvme-500/tmp/alpamayo-speed-20261116')
OUT=WORK/'agent-camera-grid-data'
BASE=WORK/'agent-paced-dynamic-source-cpu02/alpamayo'
SOURCE=Path('/mnt/nvme-256/footage/f9ad9a96/exports/0000008e--321b2ddc03/alpamayo-4hz-calibrated-two-six/manifest.json')
OFFSETS=(-2_000_000_000,-1_000_000_000,-500_000_000,-100_000_000,0)
POLICY='native-first-shared-ID-phase10Hz-received-nearest'
# Declared timing assumptions. Neither is a measurement of grid performance.
ASSUMPTIONS=('constant_measured_median_total_service','measured_nonvision_plus_linear_newimage_vision')
sys.dont_write_bytecode=True


def read(path):return json.loads(Path(path).read_text())
def sha(path):
  with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
def save(path,value):
  with Path(path).open('x') as stream:stream.write(json.dumps(value,indent=2,allow_nan=False)+'\n')
def stats(values):
  values=sorted(values)
  if not values:return None
  i=(len(values)-1)*.95;j=int(i)
  return dict(count=len(values),mean=statistics.mean(values),median=statistics.median(values),
    p95=values[j]*(1-(i-j))+values[min(j+1,len(values)-1)]*(i-j),min=values[0],max=values[-1],sum=sum(values))
def module(path,name):
  s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m
def identity(frame):return frame['view'],frame['frame_id'],frame['capture_ns']
def eligible(frame,anchor):return (frame['frame_id']-anchor)%2==0
def nearest(frames,desired,query):
  candidates=[frame for frame in frames if frame['capture_ns']<=query]
  if not candidates or desired>query:raise ValueError('missing_or_future_history')
  return min(candidates,key=lambda frame:(abs(frame['capture_ns']-desired),frame['capture_ns']))


def select(available,now,mode,anchor,*,last_query=0,last_current=None):
  rings={c:[f for f in available[c] if mode=='floor20' or eligible(f,anchor)] for c in (1,6)}
  if any(not ring for ring in rings.values()):raise ValueError('missing_front_camera')
  query=max(f['capture_ns'] for f in rings[6])
  if query<=last_query:raise ValueError('no_new_query')
  if not 0<=now-query<=250_000_000:raise ValueError('stale_query')
  result=[]
  for camera in (1,6):
    frames=rings[camera]
    for offset in OFFSETS:
      desired=query+offset
      if mode=='floor20' or offset==0:
        candidates=[f for f in frames if f['capture_ns']<=desired]
        if not candidates:raise ValueError('missing_front_history')
        frame=max(candidates,key=lambda f:f['capture_ns'])
      else:frame=nearest(frames,desired,query)
      if abs(frame['capture_ns']-desired)>150_000_000:raise ValueError('stale_front_history')
      assert frame['capture_ns']<=query
      result.append(frame)
    if result[-1]['capture_ns']<=(last_current or {}).get(camera,0):raise ValueError('stale_current_camera')
    if len({f['capture_ns'] for f in result[-5:]})!=5:raise ValueError('repeated_history')
  assert len(result)==10 and len(set(map(identity,result)))==10
  return query,result


class Cache:
  def __init__(self):self.rows=OrderedDict();self.peak=0
  def call(self,keys):
    assert len(keys)==len(set(keys))==10
    missing=len(set(keys).difference(self.rows))
    for key in keys:self.rows[key]=None;self.rows.move_to_end(key)
    while len(self.rows)>48:self.rows.popitem(last=False)
    self.peak=max(self.peak,len(self.rows));return missing


def main():
  phase=read(OUT/'agent-camera-grid-phase.json');anchor=phase['anchor_frame_id']
  assert phase['outputs_or_predictions_read'] is False and phase['no_other_phase_evaluated']
  source=read(SOURCE);native_path=Path(phase['native_path']);assert sha(native_path)==phase['native_sha256']
  calibration_path=SOURCE.with_name('render-calibrations.json');calibrations=read(calibration_path);cal_times=[r['ns'] for r in calibrations]
  geometry=dict(source=source['source_camera_geometry'],target=source['target_camera_geometry']);del source
  observed=[];summary=read(WORK/'agent-startup-spec-paced/report.json')
  jsonl=WORK/'agent-startup-spec-paced/report.jsonl'
  for line in jsonl.open():
    value=json.loads(line)
    if value['kind']=='model_completion':
      observed.append(dict(input=value['input'],completed_ns=value['completed_ns'],timing=value['timing'],
        stages=value['prediction']['timings_ms'],stage_kind=value['prediction']['stage_timing_kind'],paced=value['paced_consumer']))
    del value
  assert len(observed)==20 and [row['input']['query_ns'] for row in observed]==[row['query_ns'] for row in summary['rows']]
  sys.path[:0]=[str(BASE),str(BASE/'experiments'),'/tmp/opencode/openpilot-alpamayo','/home/john/asius/openpilot/opendbc_repo']
  paced=module(BASE/'experiments/agent-paced-replay.py','agent_camera_grid_frozen_paced_metadata')
  trace=paced.Trace(SOURCE,paced.LOGS,35.,8.)
  frames=trace.frames;by_id={(f['view'],f['frame_id']):f for f in frames}
  known={}
  for row in observed:
    for f in row['input']['frames']:
      key=(f['view'],f['frame_id']);value={k:f[k] for k in ('received_ns','ready_ns')}
      if key in known:assert known[key]==value
      known[key]=value
  assert summary['source_frame_id_gaps']==0 and summary['raw_ingestion']['rgb_reads']==len(frames)==450
  def calibration(frame):
    i=bisect_right(cal_times,frame['capture_ns'])-1
    if i<0 or not 0<=frame['capture_ns']-cal_times[i]<=1_000_000_000:raise ValueError('Missing per-capture calibration')
    return cal_times[i]
  def keys(selected):return [(*identity(frame),calibration(frame)) for frame in selected]
  def ref(frame,query,slot):
    k=(frame['view'],frame['frame_id'])
    return dict(view=frame['view'],frame_id=frame['frame_id'],video_index=frame['video_index'],capture_ns=frame['capture_ns'],
      eof_ns=frame['eof_ns'],release_ns=frame['release_ns'],requested_ns=query+OFFSETS[slot%5],image_calibration_ns=calibration(frame),
      recorded_ready_ns=known[k]['ready_ns'] if k in known else None)
  caches={mode:Cache() for mode in ('floor20','grid10_nearest')};fixed=[]
  for index,row in enumerate(observed):
    meta=row['input'];now=meta['selected_ns'];caps={c:meta['frames'][slot]['capture_ns'] for c,slot in ((1,4),(6,9))}
    # Ordered, gap-free Producer + Store current records establish the available capture prefix.
    # A wide capture later than the original road query cannot become eligible after query reduction.
    available={c:[f for f in frames if f['view']==c and f['capture_ns']<=caps[c]] for c in (1,6)}
    for c in (1,6):
      assert all(known[(c,f['frame_id'])]['ready_ns']<=now for f in available[c] if (c,f['frame_id']) in known)
    q0,a=select(available,now,'floor20',anchor);q1,b=select(available,now,'grid10_nearest',anchor)
    assert q0==meta['query_ns']
    assert [(f['view'],f['frame_id'],f['capture_ns']) for f in a]==[(f['view'],f['frame_id'],f['capture_ns']) for f in meta['frames']]
    assert [calibration(f) for f in a]==[f['image_calibration_ns'] for f in meta['frames']]
    ma,mb=(caches[mode].call(keys(selected)) for mode,selected in (('floor20',a),('grid10_nearest',b)))
    assert ma==row['paced']['vision_misses']
    changes=[dict(slot=i,current=i%5==4,old=ref(f,q0,i),new=ref(g,q1,i)) for i,(f,g) in enumerate(zip(a,b,strict=True)) if identity(f)!=identity(g)]
    fixed.append(dict(index=index,dispatch_ns=now,old_query_ns=q0,new_query_ns=q1,query_changed=q0!=q1,
      all_history_motion_nav_must_be_recomputed=q0!=q1,nearest_grid_frames=[ref(f,q1,i) for i,f in enumerate(b)],changed_slots=changes,
      original_misses=ma,grid_metadata_misses=mb,old_current_age_ms=[(now-a[i]['capture_ns'])/1e6 for i in (4,9)],
      new_current_age_ms=[(now-b[i]['capture_ns'])/1e6 for i in (4,9)],
      added_current_age_ms=[(a[i]['capture_ns']-b[i]['capture_ns'])/1e6 for i in (4,9)],
      earliest_selected_age_ms=max((now-f['capture_ns'])/1e6 for f in b),navigation_recomputed=trace.nav(q1)))
  assert statistics.mean(r['original_misses'] for r in fixed)==8.4
  # Reconstruct a RECEIPT PROXY for counterfactual dispatch times only.
  # Actual ready anchors and original past-only priming are retained. Unknown
  # frames interpolate between predecessor-ready and successor-received anchors.
  ordered=sorted(frames,key=lambda f:f['ordinal'] if 'ordinal' in f else next(e.ordinal for e in trace.events if e.kind=='frame' and e.value is f))
  anchors={}
  for i,f in enumerate(ordered):
    k=(f['view'],f['frame_id'])
    if k in known:anchors[i]=dict(known[k],kind='actual_recorded')
    elif f['release_ns']<=trace.start:anchors[i]=dict(received_ns=trace.start,ready_ns=trace.start,kind='original_prime_protocol')
  close=trace.start+round(summary['observation_with_final_completion_s']*1e9)
  anchors[len(ordered)]=dict(received_ns=close,ready_ns=close,kind='known_all450_ingested_before_close')
  schedule={};previous=-1;left=trace.start
  for end,anchor_record in sorted(anchors.items()):
    right=anchor_record['received_ns'];assert right>=left
    for i in range(previous+1,end):
      fraction=(i-previous)/(end-previous)
      ready=max(ordered[i]['release_ns'],ordered[i]['eof_ns'],round(left+(right-left)*fraction))
      assert ready<=right
      schedule[(ordered[i]['view'],ordered[i]['frame_id'])]=(ready,'interpolated_unrecorded_proxy')
    if end<len(ordered):schedule[(ordered[end]['view'],ordered[end]['frame_id'])]=(anchor_record['ready_ns'],anchor_record['kind'])
    previous=end;left=anchor_record['ready_ns']
  assert len(schedule)==450
  service=[row['timing']['service_including_history_ms'] for row in observed]
  assert all(row['stage_kind']=='synchronized stages' for row in observed)
  nonvision=[row['timing']['service_including_history_ms']-row['stages']['vision'] for row in observed]
  vision_per_image=[row['stages']['vision']/row['paced']['vision_misses'] for row in observed]
  parameters=dict(constant_service_ms=statistics.median(service),nonvision_ms=statistics.median(nonvision),
    vision_per_new_image_ms=statistics.median(vision_per_image))
  assert min(parameters.values())>0
  def simulate(mode,assumption):
    t=trace.start;last=trace.start-1;current={};cache=Cache();rows=[];blocked=Counter();deadline=t
    ready_times=sorted({ready for ready,_ in schedule.values()})
    while t<trace.end:
      if t<deadline:t=deadline;continue
      available={c:[f for f in frames if f['view']==c and schedule[(c,f['frame_id'])][0]<=t] for c in (1,6)}
      try:q,selected=select(available,t,mode,anchor,last_query=last,last_current=current)
      except ValueError as error:
        blocked[str(error)]+=1;i=bisect_right(ready_times,t)
        if i==len(ready_times):break
        t=ready_times[i];continue
      misses=cache.call(keys(selected))
      duration=(parameters['constant_service_ms'] if assumption==ASSUMPTIONS[0] else
        parameters['nonvision_ms']+misses*parameters['vision_per_new_image_ms'])
      completion=t+round(duration*1e6)
      rows.append(dict(dispatch_ns=t,query_ns=q,frame_id=selected[-1]['frame_id'],completion_ns=completion,
        assumed_service_ms=duration,metadata_misses=misses,current_age_at_dispatch_ms=[(t-selected[i]['capture_ns'])/1e6 for i in (4,9)],
        assumed_current_age_at_completion_ms=[(completion-selected[i]['capture_ns'])/1e6 for i in (4,9)],
        captures=[ref(f,q,i) for i,f in enumerate(selected)],
        selected_unrecorded_ready_proxy_count=sum(schedule[(f['view'],f['frame_id'])][1]=='interpolated_unrecorded_proxy' for f in selected)))
      assert len(rows)<=160 and all(f['capture_ns']<=q and schedule[(f['view'],f['frame_id'])][0]<=t for f in selected)
      last=q;current={c:selected[i]['capture_ns'] for c,i in ((1,4),(6,9))};deadline=t+100_000_000;t=completion
    wall=(rows[-1]['completion_ns']-trace.start)/1e9 if rows else 8.
    return dict(policy=mode,assumption=assumption,ACTUAL_PERFORMANCE_MEASUREMENT=False,completed_simulated_inputs=len(rows),
      assumed_observation_with_final_completion_s=wall,assumed_completion_hz=len(rows)/wall,
      metadata_misses=stats(r['metadata_misses'] for r in rows),assumed_service_ms=stats(r['assumed_service_ms'] for r in rows),
      current_age_at_dispatch_ms={str(c):stats(r['current_age_at_dispatch_ms'][i] for r in rows) for i,c in enumerate((1,6))},
      blocked_attempts=dict(blocked),cache_capacity=48,peak_cache_entries=cache.peak,maximum_active=1,queued_queries=0,rows=rows)
  simulations=[simulate(mode,assumption) for assumption in ASSUMPTIONS for mode in ('floor20','grid10_nearest')]
  test=controls(anchor)
  fixed_summary=dict(queries=20,dispatch_times_unchanged=True,current_query_reselected_after_admission=True,
    original_each_observed_miss_exact=True,original_mean_misses=statistics.mean(r['original_misses'] for r in fixed),
    grid_mean_metadata_misses=statistics.mean(r['grid_metadata_misses'] for r in fixed),
    original_misses=sum(r['original_misses'] for r in fixed),grid_metadata_misses=sum(r['grid_metadata_misses'] for r in fixed),
    changed_query_count=sum(r['query_changed'] for r in fixed),changed_current_slots=sum(sum(c['current'] for c in r['changed_slots']) for r in fixed),
    changed_historical_slots=sum(sum(not c['current'] for c in r['changed_slots']) for r in fixed),
    added_current_age_ms={str(c):stats(r['added_current_age_ms'][i] for r in fixed) for i,c in enumerate((1,6))},
    new_current_age_ms={str(c):stats(r['new_current_age_ms'][i] for r in fixed) for i,c in enumerate((1,6))})
  report=dict(kind='CPU_CAMERA_GRID_METADATA_AUDIT',policy=POLICY,phase=phase,gpu_executed=False,no_pixels_read_or_decoded=True,
    no_model_code_or_source_edits=True,no_other_phase_evaluated=True,model_only_admission_keeps_original20Hz_capture_and_SmallModel=True,
    source_hashes={str(path):sha(path) for path in (Path(__file__),OUT/'agent-camera-grid-phase.json',SOURCE,native_path,calibration_path,
      WORK/'agent-startup-spec-paced/report.json',jsonl,WORK/'paced-cache-audit.json',BASE/'experiments/agent-live-bridge.py',BASE/'experiments/vision-cache.py')},
    availability=dict(native_frames=len(frames),recorded_ready_anchors=len(known),fixed_dispatch='Observed latest-current records establish gap-free capture-prefix availability; all candidate queries are <=original query. Exact recorded ready values are enforced where present.',
      simulation_ready_categories=dict(Counter(kind for _,kind in schedule.values())),
      simulator='Unknown ready times interpolate inside recorded predecessor-ready/successor-received bounds. This proxy is not logged receipt timing and does not predict changed ingestion timing.'),
    fixed_dispatch_summary=fixed_summary,fixed_dispatch_rows=fixed,measured_service_assumptions=parameters,
    assumption_limit='Counterfactual durations are imposed assumptions. Linear vision cost can be optimistic; fixed overhead/concurrency/JIT/content/state costs may change. Simulated Hz and latency are NOT measured speed.',
    simulations=simulations,controls=test,metadata_key_limit='Camera ID/frame ID/capture/calibration proxies for content hashes. Baseline observed counts match; different candidate pixels may deduplicate differently.',
    selection_limit='Camera metadata guards only in simulation; canonical received-state/motion/history/nav/SmallModel preparation must be rebuilt and qualified at every new query.',
    prior_nearest_only_audit='Existing same-query nearest mean8.6, not a new gain. This audit evaluates only one fixed-grid candidate.',
    under100ms_achieved=False,quality_validated=False,adopted=False)
  save(OUT/'agent-camera-grid-report.json',report)
  print(json.dumps(dict(fixed=fixed_summary,assumptions=parameters,simulations=[{k:v for k,v in r.items() if k!='rows'} for r in simulations]),indent=2))


def controls(anchor):
  # Independent brute-force nearest checks, including shuffled video-index labels.
  import random
  rng=random.Random(17112026)
  for _ in range(500):
    frames=[dict(view=6,frame_id=anchor+i,capture_ns=1_000_000_000+i*50_000_000,video_index=rng.randrange(100000)) for i in range(80)]
    q=frames[60]['capture_ns'];target=q-rng.randrange(2_000_000_001)
    admitted=[f for f in frames if eligible(f,anchor) and f['capture_ns']<=q]
    brute=min(admitted,key=lambda f:(abs(f['capture_ns']-target),f['capture_ns']))
    assert nearest(admitted,target,q) is brute
    changed=[dict(f,video_index=-f['video_index']-1) for f in admitted]
    assert nearest(changed,target,q)['frame_id']==brute['frame_id']
    future=[dict(f,capture_ns=f['capture_ns']+1_000_000_000_000) for f in frames if f['capture_ns']>q]
    assert nearest(admitted+future,target,q)['frame_id']==brute['frame_id']
  tie=[dict(view=6,frame_id=anchor,capture_ns=100),dict(view=6,frame_id=anchor+2,capture_ns=200)]
  assert nearest(tie,150,200)['capture_ns']==100
  try:nearest(tie,201,200)
  except ValueError:pass
  else:raise AssertionError('Future desired target accepted')
  available={c:[dict(view=c,frame_id=anchor+i,capture_ns=10_000_000_000+i*50_000_000) for i in range(45)] for c in (1,6)}
  q,frames=select(available,12_210_000_000,'grid10_nearest',anchor)
  assert q==12_200_000_000 and all(eligible(f,anchor) and f['capture_ns']<=q for f in frames)
  for now,last,expected in ((12_600_000_000,0,'stale_query'),(12_210_000_000,q,'no_new_query')):
    try:select(available,now,'grid10_nearest',anchor,last_query=last)
    except ValueError as error:assert str(error)==expected
    else:raise AssertionError('Camera freshness guard failed')
  return dict(independent_nearest_cases=500,video_index_does_not_select_phase=True,future_records_ignored=True,
    ties_choose_past=True,stale_query_rejected=True,repeated_query_rejected=True,one_phase_only=True)


if __name__=='__main__':main()
