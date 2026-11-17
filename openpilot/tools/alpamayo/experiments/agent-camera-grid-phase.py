"""Metadata-only phase declaration. Reads no predictions or benchmark report."""
import hashlib
import json
from pathlib import Path
import statistics

WORK=Path('/mnt/nvme-500/tmp/alpamayo-speed-20261116')
OUT=WORK/'agent-camera-grid-data'
SOURCE=Path('/mnt/nvme-256/footage/f9ad9a96/exports/0000008e--321b2ddc03/alpamayo-4hz-calibrated-two-six/manifest.json')
RULE='Anchor both cameras to the smallest shared native frame_id in the original clip; retain (frame_id-anchor)%2==0. Never inspect the other phase.'


def sha(path):
  with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
def stats(values):
  values=list(values);return dict(min=min(values),median=statistics.median(values),max=max(values))


def main():
  source=json.loads(SOURCE.read_text());native_path=Path(source['export'])/'clips/native-timestamps.json'
  origin=source['capture_start_ns'];del source
  views=json.loads(native_path.read_text())['views'];rows={c:views[str(c)] for c in (1,6)}
  maps={c:{r['frame_id']:dict(r,video_index=i) for i,r in enumerate(rows[c])} for c in (1,6)}
  for camera in (1,6):
    assert len(maps[camera])==len(rows[camera])
    assert all(b['frame_id']==a['frame_id']+1 and b['capture_ns']>a['capture_ns'] for a,b in zip(rows[camera],rows[camera][1:]))
  common=sorted(set(maps[1])&set(maps[6]));assert set(maps[1])==set(maps[6])
  pairs=[dict(frame_id=i,wide_capture_ns=maps[1][i]['capture_ns'],road_capture_ns=maps[6][i]['capture_ns'],
    road_minus_wide_ns=maps[6][i]['capture_ns']-maps[1][i]['capture_ns']) for i in common]
  window=[r for r in pairs if origin+35_000_000_000<=r['road_capture_ns']<origin+43_000_000_000]
  # Alignment is a metadata prerequisite, not a phase/performance sweep.
  assert len(window)==160 and max(abs(r['road_minus_wide_ns']) for r in pairs)<10_000_000
  anchor=common[0]
  audit=dict(cpu_only=True,gpu_executed=False,outputs_or_predictions_read=False,predeclared_rule=RULE,
    anchor_frame_id=anchor,retained_frame_id_mod2=anchor%2,source_sha256=sha(SOURCE),native_sha256=sha(native_path),
    native_path=str(native_path),origin_ns=origin,paired_ids=len(pairs),unpaired_ids=0,no_frame_id_gaps_or_resets=True,
    full_clip_road_minus_wide_ns=stats(r['road_minus_wide_ns'] for r in pairs),
    window_road_minus_wide_ns=stats(r['road_minus_wide_ns'] for r in window),window_source_seconds=[35,43],
    per_camera={str(c):dict(frames=len(rows[c]),first_native=rows[c][0],
      native_period_ns=stats(b['capture_ns']-a['capture_ns'] for a,b in zip(rows[c],rows[c][1:])),
      verified_frame_id_minus_video_index=sorted(set(r['frame_id']-i for i,r in enumerate(rows[c]))),
      retained_native_ids=[r['frame_id'] for r in rows[c] if (r['frame_id']-anchor)%2==0][:5]) for c in (1,6)},
    implementation='Model eligibility only. Native20Hz capture/SmallModel remain unchanged. Use frame_id, never assume video_index identity.',
    no_other_phase_evaluated=True,phase_selection_uses_quality_or_cache_results=False)
  OUT.mkdir(exist_ok=True)
  with (OUT/'agent-camera-grid-phase.json').open('x') as stream:stream.write(json.dumps(audit,indent=2)+'\n')
  print(json.dumps(audit,indent=2))


if __name__=='__main__':main()
