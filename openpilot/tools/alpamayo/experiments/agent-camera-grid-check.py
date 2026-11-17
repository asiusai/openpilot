"""Independent CPU recomputation of phase, cache, changed slots, ages, and causal guards."""
from bisect import bisect_right
from collections import OrderedDict
import hashlib
import json
from pathlib import Path

WORK=Path('/mnt/nvme-500/tmp/alpamayo-speed-20261116')
DATA=WORK/'agent-camera-grid-data'
OFFSETS=(-2_000_000_000,-1_000_000_000,-500_000_000,-100_000_000,0)


def read(path):return json.loads(Path(path).read_text())
def sha(path):
  with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def main():
  report=read(DATA/'agent-camera-grid-report.json');phase=read(DATA/'agent-camera-grid-phase.json')
  for path,digest in report['source_hashes'].items():assert sha(path)==digest,path
  views=read(phase['native_path'])['views'];native={c:views[str(c)] for c in (1,6)}
  common=set(r['frame_id'] for r in native[1])&set(r['frame_id'] for r in native[6])
  assert phase['anchor_frame_id']==min(common)==4 and phase['retained_frame_id_mod2']==0
  assert phase['no_other_phase_evaluated'] and report['no_other_phase_evaluated']
  source=Path('/mnt/nvme-256/footage/f9ad9a96/exports/0000008e--321b2ddc03/alpamayo-4hz-calibrated-two-six')
  cal=read(source/'render-calibrations.json');cal_times=[r['ns'] for r in cal]
  observed=[]
  for line in (WORK/'agent-startup-spec-paced/report.jsonl').open():
    row=json.loads(line)
    if row['kind']=='model_completion':observed.append(row['input'])
  caches={mode:OrderedDict() for mode in ('original','grid')};totals={mode:0 for mode in caches}
  changed_current=changed_history=changed_query=0;last_query=0;last_current={}
  for old,row in zip(observed,report['fixed_dispatch_rows'],strict=True):
    now=old['selected_ns'];caps={1:old['frames'][4]['capture_ns'],6:old['frames'][9]['capture_ns']}
    admitted={c:[r for r in native[c] if r['frame_id']%2==0 and r['capture_ns']<=caps[c]] for c in (1,6)}
    q=max(r['capture_ns'] for r in admitted[6]);assert q==row['new_query_ns'] and last_query<q<=now
    assert now-q<=250_000_000
    expected=[]
    for c in (1,6):
      available=[f for f in admitted[c] if f['capture_ns']<=q]
      for off in OFFSETS:
        desired=q+off
        if off==0:frame=max(available,key=lambda f:f['capture_ns'])
        else:frame=sorted(available,key=lambda f:(abs(f['capture_ns']-desired),f['capture_ns']))[0]
        assert abs(frame['capture_ns']-desired)<=150_000_000
        expected.append((c,frame))
    current={c:expected[i][1]['capture_ns'] for c,i in ((1,4),(6,9))}
    assert all(current[c]>last_current.get(c,0) for c in (1,6));last_current=current;last_query=q
    for i,((camera,frame),got) in enumerate(zip(expected,row['nearest_grid_frames'],strict=True)):
      assert (camera,frame['frame_id'],frame['capture_ns'])==(got['view'],got['frame_id'],got['capture_ns'])
      assert got['requested_ns']==q+OFFSETS[i%5] and frame['capture_ns']<=q
      assert got['image_calibration_ns']==cal_times[bisect_right(cal_times,frame['capture_ns'])-1]
      assert got['recorded_ready_ns'] is None or got['recorded_ready_ns']<=now
      changed=(got['frame_id'],got['capture_ns'])!=(old['frames'][i]['frame_id'],old['frames'][i]['capture_ns'])
      changed_current+=changed and i%5==4;changed_history+=changed and i%5!=4
    changed_query+=q!=old['query_ns'];assert row['all_history_motion_nav_must_be_recomputed']==(q!=old['query_ns'])
    for i,index in enumerate((4,9)):
      assert row['added_current_age_ms'][i]==(old['frames'][index]['capture_ns']-row['nearest_grid_frames'][index]['capture_ns'])/1e6
    for mode,frames in (('original',old['frames']),('grid',row['nearest_grid_frames'])):
      keys=[(f['view'],f['frame_id'],f['capture_ns'],f['image_calibration_ns']) for f in frames]
      assert len(keys)==len(set(keys))==10
      missing=sum(key not in caches[mode] for key in keys);totals[mode]+=missing
      for key in keys:caches[mode][key]=True;caches[mode].move_to_end(key)
      while len(caches[mode])>48:caches[mode].popitem(last=False)
      assert missing==row['original_misses' if mode=='original' else 'grid_metadata_misses']
  assert totals==dict(original=168,grid=136) and (changed_current,changed_history,changed_query)==(22,80,11)
  for scenario in report['simulations']:
    assert scenario['ACTUAL_PERFORMANCE_MEASUREMENT'] is False and scenario['cache_capacity']==48
    assert scenario['maximum_active']==1 and scenario['queued_queries']==0
    previous=0
    for row in scenario['rows']:
      q=row['query_ns'];assert previous<q<=row['dispatch_ns']<row['completion_ns'];previous=q
      assert row['completion_ns']-row['dispatch_ns']==round(row['assumed_service_ms']*1e6)
      assert all(f['capture_ns']<=q and abs(f['capture_ns']-f['requested_ns'])<=150_000_000 for f in row['captures'])
      assert all(f['recorded_ready_ns'] is None or f['recorded_ready_ns']<=row['dispatch_ns'] for f in row['captures'])
      if scenario['policy']=='grid10_nearest':assert all(f['frame_id']%2==0 for f in row['captures'])
  protected=read(DATA/'agent-camera-grid-protected-before.json')
  for path,digest in protected.items():assert sha(path)==digest,path
  result=dict(cpu_only=True,gpu_executed=False,independent_fixed_queries=20,independent_image_slot_checks=200,
    exact_original_cache_misses=168,grid_metadata_cache_misses=136,cache_capacity=48,
    changed_current_slots=changed_current,changed_historical_slots=changed_history,changed_current_queries=changed_query,
    phase_independent_of_video_index=True,all_selected_captures_causal=True,known_ready_anchors_respected=True,
    new_query_requires_history_nav_motion_rebuild=True,counterfactual_timings_remain_assumptions=True,
    prior_frozen_files_unchanged=len(protected),audit_report_sha256=sha(DATA/'agent-camera-grid-report.json'),
    quality_or_performance_validated=False,propose_GPU_protocol=False)
  with (DATA/'agent-camera-grid-independent-check.json').open('x') as stream:stream.write(json.dumps(result,indent=2)+'\n')
  print(json.dumps(result,indent=2))


if __name__=='__main__':main()
