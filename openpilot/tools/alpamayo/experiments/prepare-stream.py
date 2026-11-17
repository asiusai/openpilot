"""Prepare causal front-camera streams for timing and paired accuracy experiments."""
import argparse
from collections import Counter, OrderedDict
import copy
import hashlib
from pathlib import Path

import av
import numpy as np

from calibration_trials import asof, calibrated_image, vehicle_history, vehicle_records
from camera_profiles import replay_module


def prepare(args):
  replay = replay_module()
  source = replay.read_json(args.source/'manifest.json')
  if source['camera_input_profile'] != 'calibrated_vehicle_history' or source['camera_subset'] != [1, 6]:
    raise ValueError('Use the calibrated two-camera reference')
  if args.output.exists():
    raise FileExistsError('Use a new output directory')
  args.output.mkdir(parents=True)
  export = Path(source['export'])
  _, camera_rows = replay.camera_records(export)
  times = {c: np.array([r['capture_ns'] for r in camera_rows[c]], dtype=np.int64) for c in (1, 6)}
  motions, calibrations, models, _, hashes = replay.read_logs(args.logs)
  records, vehicle_hashes = vehicle_records(args.logs)
  if hashes != source['rlog_sha256'] or hashes != vehicle_hashes:
    raise ValueError('Source recording hashes differ')
  motion = replay.Motion(motions)
  cal = replay.read_json(args.source/'render-calibrations.json')
  calibration = (np.array([r['ns'] for r in cal]), cal)
  model_calibration = (np.array(sorted(calibrations)), [calibrations[k] for k in sorted(calibrations)])
  start = source['capture_start_ns']
  seconds = (motion.ns-start)/1e9
  speed = np.linalg.norm(motion.v, axis=1)
  distance = np.r_[0., np.cumsum(np.diff(seconds)*(speed[1:]+speed[:-1])/2)]
  annotations = source['navigation_annotations']
  periods = 1e9/args.hz
  camera_hz=args.camera_hz or args.hz
  offsets=tuple(map(float,args.history_offsets.split(',')))
  if len(offsets)!=5 or tuple(sorted(set(offsets)))!=offsets or offsets[-1]!=0 or offsets[0]<-2:
    raise ValueError('Use five increasing history offsets from -2 to 0 seconds, ending at 0')
  # The ring simulates a camera published at the requested cadence. Its actual
  # capture timestamps remain in the prompt; it never selects a future frame.
  ring = {}
  for c in (1, 6):
    grid = start+np.rint(np.arange(-round(3*camera_hz), int(np.ceil(source['duration_s']*camera_hz))+1)*(1e9/camera_hz)).astype(np.int64)
    ring[c] = np.unique(np.searchsorted(times[c], grid, side='right')-1)
    ring[c] = ring[c][ring[c] >= 0]
  schedules = {policy: [] for policy in args.policies}
  needed = {1: set(), 6: set()}
  for begin, end in args.windows:
    if not 2 <= begin < end <= source['duration_s']:
      raise ValueError('Windows must lie inside the route with two seconds of history')
    for tick in range(round(begin*args.hz), round(end*args.hz)):
      requested = start+round(tick*periods)
      anchor = int(np.searchsorted(times[6], requested, side='right'))-1
      query = int(times[6][anchor])
      rpy, cal_ns = asof(model_calibration, query, 1_000_000_000)
      xyz, rot, actual, age = motion.sample(query, rpy)
      small = models[camera_rows[6][anchor]['frame_id']]
      if small['model']['timestampEof'] != camera_rows[6][anchor]['eof_ns']:
        raise ValueError('Small-model frame mismatch')
      row = dict(tick=tick, display_time_s=tick/args.hz, query_ns=query, skip_reasons=[], omitted_views={},
                 available_views=[1, 6], calibration_rpy=rpy, calibration_log_mono_time=cal_ns,
                 ego_history_xyz=xyz.tolist(), ego_history_rot=rot.tolist(),
                 recorded_motion_future_xyz=actual, motion_extrapolation_ms=age,
                 small_model=small, small_model_eof_difference_ns=0,
                 navigation_text=replay.navigation_text((query-start)/1e9, (seconds, distance),
                                                       annotations['turns'], annotations['arrival_s']))
      xyz, rot, audit = vehicle_history(row, records)
      row.update(ego_history_xyz=xyz.tolist(), ego_history_rot=rot.tolist(), vehicle_history_audit=audit,
                 speed_m_s=audit['speed_m_s'], camera_profile='calibrated_vehicle_history')
      for policy, rows in schedules.items():
        current = copy.deepcopy(row)
        frames = []
        for c in (1, 6):
          causal_ring = ring[c][times[c][ring[c]] <= query]
          for offset in offsets:
            desired = query+round(offset*1e9)
            index = (int(np.searchsorted(times[c], desired, side='right'))-1 if policy == 'floor' else
                     int(causal_ring[np.argmin(np.abs(times[c][causal_ring]-desired))]))
            stamp = int(times[c][index])
            if index < 0 or stamp > query or abs(desired-stamp) > 150_000_000 or (policy == 'floor' and stamp > desired):
              raise ValueError('Invalid or stale causal image')
            _, cal_ns = asof(calibration, stamp, 1_000_000_000)
            frames.append(dict(view=c, video_index=index, capture_ns=stamp, requested_ns=desired,
                               frame_id=camera_rows[c][index]['frame_id'], image_calibration_ns=cal_ns))
            needed[c].add(index)
        current['frames'] = frames
        rows.append(current)
  # Decode by frame index, as in the reference preparation. Container frame PTS
  # need not have a constant interval. Native timestamp metadata owns timing.
  for c, name in ((1, 'front-wide'), (6, 'front-telephoto')):
    folder = args.raw_cache/str(c)
    folder.mkdir(parents=True, exist_ok=True)
    missing = {i for i in needed[c] if not (folder/f'{i}.npy').exists()}
    if missing:
      with av.open(str(export/'clips'/f'{name}.mp4')) as container:
        stream = container.streams.video[0]
        stream.thread_type = 'AUTO'
        for index, frame in enumerate(container.decode(stream)):
          if index in missing:
            np.save(folder/f'{index}.npy', frame.to_ndarray(format='rgb24').transpose(2, 0, 1), allow_pickle=False)
          if index == max(missing):
            break
      if any(not (folder/f'{i}.npy').exists() for i in missing):
        raise ValueError('Video ended before a requested frame')
    print(f'{name}: {len(needed[c])} unique frames, {len(missing)} newly decoded', flush=True)
  cache = OrderedDict()
  overlap = {r['query_ns']: r for r in source['schedule']}
  for policy, rows in schedules.items():
    out = args.output/policy
    out.mkdir()
    exact_controls, misses, keys = 0, [], set()
    last_tick = None
    for row in rows:
      if last_tick is None or row['tick'] != last_tick+1:
        keys.clear()
      last_tick = row['tick']
      path = out/'samples'/f'sample-{row["query_ns"]//1000}'
      path.mkdir(parents=True)
      images, unique = [], set()
      for frame in row['frames']:
        c, index, stamp = frame['view'], frame['video_index'], frame['image_calibration_ns']
        key = (c, index, stamp)
        if key not in cache:
          raw = np.load(args.raw_cache/str(c)/f'{index}.npy')
          image_cal, _ = asof(calibration, frame['capture_ns'], 1_000_000_000)
          cache[key] = calibrated_image(raw, c, source['source_camera_geometry'],
                                       source['target_camera_geometry'], image_cal)
        images.append(cache[key][None])
        cache.move_to_end(key)
        if len(cache) > 160:
          cache.popitem(last=False)
        unique.add(key)
      misses.append(len(unique-keys))
      keys.update(unique)
      arrays = dict(image_frames=np.stack(images), camera_indices=np.array([f['view'] for f in row['frames']], dtype=np.int64),
                    absolute_timestamps=np.array([f['capture_ns']//1000 for f in row['frames']], dtype=np.int64),
                    ego_history_xyz=np.asarray(row['ego_history_xyz'], dtype=np.float32),
                    ego_history_rot=np.asarray(row['ego_history_rot'], dtype=np.float32), nav_text=np.array(row['navigation_text']))
      for name, value in arrays.items():
        np.save(path/f'{name}.npy', value, allow_pickle=False)
      row['sample'] = str(path.resolve())
      if row['query_ns'] in overlap:
        reference = overlap[row['query_ns']]
        for name, value in arrays.items():
          if name in ('image_frames','absolute_timestamps') and (policy!='floor' or offsets!=(-2.,-1.,-.5,-.1,0.)):
            continue
          if not np.array_equal(value, np.load(Path(reference['sample'])/f'{name}.npy')):
            raise ValueError(f'Existing baseline input differs: {name}, {row["query_ns"]}')
        exact_controls += int(policy=='floor' and offsets==(-2.,-1.,-.5,-.1,0.))
    manifest = copy.deepcopy(source)
    manifest.update(sample_hz=args.hz, schedule=rows, requested_samples=len(rows), valid_samples=len(rows),
                     streaming_fixture=dict(windows=args.windows, history_policy=policy,
                       camera_hz=camera_hz,history_offsets_s=offsets,
                      source_manifest_sha256=hashlib.sha256((args.source/'manifest.json').read_bytes()).hexdigest(),
                      exact_original_input_controls=exact_controls,
                      unique_image_misses_histogram=dict(Counter(misses)),
                      limit='Ring selection changes the history sampling policy; requires paired accuracy validation'))
    replay.write_json(out/'manifest.json', manifest)
    replay.write_json(out/'samples.json', [r['sample'] for r in rows])
    print(f'{policy}: {len(rows)} samples; {exact_controls} exact baseline controls; image misses {dict(Counter(misses))}', flush=True)


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description=__doc__)
  for name in ('source', 'logs', 'output', 'raw-cache'):
    parser.add_argument('--'+name, type=Path, required=True)
  parser.add_argument('--hz', type=float, default=10.)
  parser.add_argument('--camera-hz',type=float,help='Camera ring cadence; defaults to the prediction fixture cadence')
  parser.add_argument('--history-offsets',default='-2,-1,-0.5,-0.1,0',help='Explicit five-frame temporal sampling experiment')
  parser.add_argument('--window', dest='windows', type=float, nargs=2, action='append', required=True)
  parser.add_argument('--policies', nargs='+', choices=('floor', 'ring'), default=['floor', 'ring'])
  args = parser.parse_args()
  if not 0 < args.hz <= 20:
    parser.error('Use a positive camera cadence up to 20 Hz')
  if args.camera_hz is not None and not 0<args.camera_hz<=20:
    parser.error('Camera cadence must be positive and at most 20 Hz')
  prepare(args)
