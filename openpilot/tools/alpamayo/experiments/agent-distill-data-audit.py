"""Bounded, read-only owned-route metadata audit. Never opens the protected route."""
import argparse
from bisect import bisect_right
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

sys.dont_write_bytecode = True
EXP = Path(__file__).resolve().parent
WORK = Path('/mnt/nvme-500/tmp/alpamayo-speed-20261116')
ARCHIVE = Path('/mnt/nvme-256/footage/f9ad9a96')
PROTECTED = '0000008e--321b2ddc03'
# Fixed before the additional log audit. Dates, not outcomes, selected these routes.
DRIVES = {'train': '00000071--858232e52d', 'validation': '00000073--41474e402a',
          'sealed_test': '00000090--1cf3f7d926'}


def sha(path):
  h = hashlib.sha256()
  with Path(path).open('rb') as stream:
    for data in iter(lambda: stream.read(1024*1024), b''): h.update(data)
  return h.hexdigest()


def write(path, data):
  with path.open('x') as stream: stream.write(json.dumps(data, indent=2, allow_nan=False)+'\n')


def utc(ns): return datetime.fromtimestamp(ns/1e9, timezone.utc).isoformat()


def load_events(path, cap):
  assert PROTECTED not in str(path) and path.is_relative_to(ARCHIVE/'realdata')
  import zstandard
  from openpilot.cereal import log
  with path.open('rb') as stream, zstandard.ZstdDecompressor().stream_reader(stream) as decoder:
    raw = decoder.read(cap+1)
  if len(raw) > cap: raise ValueError(f'Decompression bound exceeded: {path}')
  status = dict(complete=True)
  def bounded_events():
    import capnp
    try:
      yield from log.Event.read_multiple_bytes(raw)
    except capnp.KjException as error:
      # Preserve complete messages from an explicitly incomplete final message.
      # Do not silently accept another parsing or integrity failure.
      if 'Message ends prematurely' not in str(error): raise
      status.update(complete=False, incomplete_capnp_tail=str(error).split('stack:')[0].strip())
  return bounded_events(), len(raw), status


def audit_qlog(path):
  events, size, parse_status = load_events(path, 32*1024**2)
  counts, valid, navigation = Counter(), Counter(), []
  initial, gps, ignition, started, clocks = [], [], [], [], []
  moving = 0
  for event in events:
    kind = event.which(); counts[kind] += 1
    if event.valid: valid[kind] += 1
    stamp = int(event.logMonoTime)
    if kind == 'initData':
      message = event.initData
      initial.append(dict(monotonic_ns=stamp, wall_ns=int(message.wallTimeNanos),
        wall_utc=utc(message.wallTimeNanos), bootlog_id=str(message.bootlogId),
        git_commit=str(message.gitCommit), git_source_commit=str(message.gitSrcCommit),
        device_type=str(message.deviceType), version=str(message.version)))
    elif kind in ('gpsLocationExternal', 'gpsLocation') and event.valid:
      message = getattr(event, kind)
      if message.hasFix and message.unixTimestampMillis:
        gps.append(dict(monotonic_ns=stamp, unix_ns=int(message.unixTimestampMillis)*1_000_000))
    elif kind == 'pandaStates' and event.valid:
      ignition.append(dict(monotonic_ns=stamp, on=any(s.ignitionLine or s.ignitionCan for s in event.pandaStates)))
    elif kind == 'deviceState' and event.valid:
      started.append(dict(monotonic_ns=stamp, started=bool(event.deviceState.started),
        started_mono_ns=int(event.deviceState.startedMonoTime)))
    elif kind == 'carState' and event.valid:
      moving += not event.carState.standstill
    elif kind in ('navInstruction', 'navRoute'):
      row = dict(kind=kind, monotonic_ns=stamp, valid=bool(event.valid))
      # No invalid/default navigation values, destinations, or future motion become inputs.
      if event.valid:
        row['recorded_value'] = getattr(event, kind).to_dict()
      navigation.append(row)
    elif kind == 'clocks' and event.valid:
      # The schema varies; use only recorded wall-clock fields that actually exist.
      m = event.clocks
      clocks.append(dict(monotonic_ns=stamp, wall_time_ns=int(m.wallTimeNanos)))
  result = dict(path=str(path), sha256=sha(path), compressed_bytes=path.stat().st_size,
    uncompressed_bytes=size, parse_status=parse_status, channels=dict(counts), valid_channels=dict(valid),
    initialization=initial, valid_navigation_records=navigation,
    navigation_counts=dict(Counter((r['kind']+(' valid' if r['valid'] else ' invalid')) for r in navigation)),
    valid_gps_fixes=len(gps), moving_car_state_count=moving,
    ignition_counts=dict(Counter(str(r['on']) for r in ignition)),
    started_counts=dict(Counter(str(r['started']) for r in started)))
  for key, rows in [('gps', gps), ('ignition', ignition), ('started', started), ('clocks', clocks)]:
    result[key+'_endpoints'] = [rows[0], rows[-1]] if rows else []
  if gps:
    result['gps_utc_range'] = [utc(gps[0]['unix_ns']), utc(gps[-1]['unix_ns'])]
    result['gps_boot_epoch_ns'] = gps[0]['unix_ns']-gps[0]['monotonic_ns']
  return result


def audit_anchor(path, split):
  """Check a metadata-selected causal anchor. Never decodes pixels or reads future targets."""
  import numpy as np
  events, size, parse_status = load_events(path, 128*1024**2)
  frames, calibration, motion, encodes = {1: [], 6: []}, {}, {}, {1: {}, 6: {}}
  motion_times, publication = [], {}
  camera_kind = {'wideRoadCameraState': 1, 'narrowRoadCameraState': 6}
  encode_kind = {'wideRoadEncodeIdx': 1, 'narrowRoadEncodeIdx': 6}
  counts = Counter()
  for event in events:
    kind = event.which()
    if not event.valid: continue
    counts[kind] += 1
    m = getattr(event, kind)
    if kind in camera_kind:
      frames[camera_kind[kind]].append(dict(capture_ns=int(m.timestampSof), eof_ns=int(m.timestampEof),
        published_ns=int(event.logMonoTime), frame_id=int(m.frameId), sensor=str(m.sensor)))
    elif kind in encode_kind:
      encodes[encode_kind[kind]][int(m.frameId)] = dict(segment=int(m.segmentNum), video_index=int(m.segmentId),
        capture_ns=int(m.timestampSof), eof_ns=int(m.timestampEof))
    elif kind == 'extrinsicsCalibration' and str(m.calStatus) == 'calibrated':
      calibration[int(event.logMonoTime)] = list(m.rpyCalib)
    elif kind == 'deviceMotion' and m.timestamp > 0 and m.inputsOK and m.orientationNED.valid and m.velocityDevice.valid:
      stamp = int(m.timestamp)
      motion_times.append(stamp); publication[stamp] = int(event.logMonoTime)
  for rows in frames.values(): rows.sort(key=lambda r: r['capture_ns'])
  ct = sorted(calibration); mt = sorted(set(motion_times))
  result = dict(path=str(path), sha256=sha(path), uncompressed_bytes=size, parse_status=parse_status,
    valid_channels=dict(counts), pixel_decode_executed=False, teacher_targets_read=False,
    motion_values_retained=split != 'sealed_test', future_motion_read_as_values=False)
  if not all(frames.values()) or not ct or not mt:
    return dict(result, status='missing required camera/calibration/motion events')
  # One fixed rule: first current narrow frame >=15 seconds after segment start.
  # This rule depends only on availability, not trajectories, teacher outputs, or fit loss.
  floor = max(frames[1][0]['capture_ns'], frames[6][0]['capture_ns'], mt[0], ct[0])+15_000_000_000
  query = next((r['capture_ns'] for r in frames[6] if r['capture_ns'] >= floor), None)
  if query is None: return dict(result, status='segment shorter than fixed anchor rule')
  chosen = []
  for view in (1, 6):
    times = [r['capture_ns'] for r in frames[view]]
    for offset in (-2_000_000_000, -1_000_000_000, -500_000_000, -100_000_000, 0):
      desired = query+offset; i = bisect_right(times, desired)-1
      assert i >= 0 and 0 <= desired-times[i] <= 150_000_000
      row = dict(frames[view][i], view=view, requested_ns=desired)
      encoding = encodes[view].get(row['frame_id'])
      assert encoding and encoding['capture_ns'] == row['capture_ns'] and encoding['eof_ns'] == row['eof_ns']
      row['encoding'] = encoding
      cal_index = bisect_right(ct, row['capture_ns'])-1
      assert cal_index >= 0 and 0 <= row['capture_ns']-ct[cal_index] <= 1_000_000_000
      row['prior_extrinsics_ns'] = ct[cal_index]
      chosen.append(row)
  for view in (1, 6):
    times = [r['capture_ns'] for r in chosen if r['view'] == view]
    assert all(a < b for a, b in zip(times, times[1:]))
  last = bisect_right(mt, query)-1; first = max(0, bisect_right(mt, query-1_900_000_000)-1)
  assert mt[first] <= query-1_900_000_000 and 0 <= query-mt[last] <= 100_000_000
  assert max(b-a for a, b in zip(mt[first:last], mt[first+1:last+1])) <= 150_000_000
  # Publication is causal at preparation time; measurements remain <=query.
  causal_times = [t for t in mt[:last+1]]
  result.update(status='history metadata complete; navigation and image preparation still blocked',
    query_ns=query, frames=chosen, last_motion_ns=mt[last], max_motion_history_gap_ns=max(np.diff(mt[first:last+1]).tolist()),
    motion_input_max_timestamp_ns=max(causal_times), latest_motion_publication_ns=max(publication[t] for t in causal_times),
    extrinsics_ns=ct[bisect_right(ct, query)-1],
    original_pose_constructor='replay-route.py:Motion.sample; causal records only',
    calibration_limit='Recorded extrinsics verified. Intrinsic/lens/device profile and exact native RGB preparation have not been checked.')
  if split != 'sealed_test':
    # The first pass establishes timestamps. Only causal motion values enter the
    # existing pose constructor; values after query are never extracted.
    events, _, second_status = load_events(path, 128*1024**2)
    for event in events:
      if event.which() != 'deviceMotion' or not event.valid: continue
      m = event.deviceMotion; stamp = int(m.timestamp)
      if stamp in publication and stamp <= query:
        motion[stamp] = ([getattr(m.orientationNED, a) for a in 'xyz'], [getattr(m.velocityDevice, a) for a in 'xyz'])
    assert second_status == parse_status
    spec = importlib.util.spec_from_file_location('agent_distill_data_existing_replay', EXP/'replay-route.py')
    replay = importlib.util.module_from_spec(spec); spec.loader.exec_module(replay)
    causal = {t: motion[t] for t in causal_times}
    xyz, rot, future, age = replay.Motion(causal).sample(query, calibration[result['extrinsics_ns']])
    assert future is None and xyz.shape == (1, 20, 3) and rot.shape == (1, 20, 3, 3)
    assert np.isfinite(xyz).all() and np.isfinite(rot).all()
    # Hash only; no incomplete input bundle or training example is exported.
    result['causal_pose_check'] = dict(shape_xyz=list(xyz.shape), shape_rot=list(rot.shape),
      dtype=str(xyz.dtype), xyz_sha256=hashlib.sha256(xyz.tobytes()).hexdigest(),
      rot_sha256=hashlib.sha256(rot.tobytes()).hexdigest(), extrapolation_ms=age,
      future_returned=False, source_sha256=sha(EXP/'replay-route.py'))
  return result


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--output', type=Path, required=True)
  args = parser.parse_args(); output = args.output.resolve()
  assert output.parent.is_relative_to(WORK) and output.name.startswith('agent-distill-data-')
  output.parent.mkdir(parents=True, exist_ok=True)
  assert subprocess.check_output(['findmnt', '-n', '-o', 'UUID', '-T', str(ARCHIVE)], text=True).strip() == 'aac28ecf-db48-4b4f-8a99-bde6a3726ab9'
  assert subprocess.check_output(['findmnt', '-n', '-o', 'TARGET', '-T', str(output.parent)], text=True).strip() == '/mnt/nvme-500'
  sys.path[:0] = ['/tmp/opencode/openpilot-alpamayo', '/home/john/asius/openpilot/opendbc_repo']
  routes, records, selected = [], [], []
  for summary in sorted((ARCHIVE/'realdata/.summaries').glob('*.json')):
    if summary.stem == PROTECTED: continue  # Do not even open its summary.
    info = json.loads(summary.read_text()); route = info['routeId']; assert route != PROTECTED
    segments = sorted((ARCHIVE/'realdata').glob(route+'--*'), key=lambda p: int(p.name.rsplit('--', 1)[1]))
    complete = [p for p in segments if all((p/f).is_file() for f in ('fcamera.mp4', 'ecamera.mp4', 'qlog.zst', 'rlog.zst'))]
    if not complete: continue
    paths = sorted({complete[0]/'qlog.zst', complete[len(complete)//2]/'qlog.zst', complete[-1]/'qlog.zst'})
    audited = [audit_qlog(p) for p in paths]; records.extend(audited)
    routes.append(dict(route=route, summary_path=str(summary), summary_sha256=sha(summary),
      available_complete_segments=[int(p.name.rsplit('--', 1)[1]) for p in complete],
      inspected_qlogs=[str(p) for p in paths], summary_time_estimated=info.get('timeEstimated')))
    if route in DRIVES.values():
      split = next(k for k, v in DRIVES.items() if v == route)
      segment = complete[len(complete)//2]
      anchor = audit_anchor(segment/'rlog.zst', split)
      selected.append(dict(split=split, route=route, status='provisional raw-drive evidence only; not an admissible training split',
        qlogs=audited, history=anchor))
    print(f'Audited {route}: {len(paths)} bounded qlogs', flush=True)
  totals = Counter()
  for row in records: totals.update(row['navigation_counts'])
  result = dict(schema=1, helper_sha256=sha(Path(__file__)), gpu_initialized=False, training_executed=False,
    protected_route_opened=False, protected_route=PROTECTED, video_decoded=False,
    policy='first/middle/last complete segment qlog per non-test route; one middle rlog per three fixed date-selected routes',
    qlog_decompression_cap_bytes=32*1024**2, rlog_decompression_cap_bytes=128*1024**2,
    routes=routes, qlogs=records, selected_drives=selected, navigation_totals=dict(totals),
    qlogs_inspected=len(records), qlog_decompressed_bytes=sum(r['uncompressed_bytes'] for r in records),
    claim_limit='Only inspected sections establish navigation coverage. No claim of absence in all route contents.',
    sources={str(p): sha(p) for p in [Path(__file__), EXP/'replay-route.py',
      Path('/tmp/opencode/openpilot-alpamayo/openpilot/cereal/log.capnp'), ARCHIVE/'README.md',
      ARCHIVE/'manifests/2026-11-16-office/dragon-status.json']})
  assert not any(name == 'torch' or name.startswith('tinygrad.runtime.ops_') for name in sys.modules)
  write(output, result)
  print(json.dumps(dict(output=str(output), routes=len(routes), qlogs=len(records), navigation_totals=dict(totals),
    selected=[dict(split=r['split'], route=r['route'], history=r['history']['status']) for r in selected]), indent=2))


if __name__ == '__main__': main()
