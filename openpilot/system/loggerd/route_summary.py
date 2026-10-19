"""Local route summaries, independent of network access and upload settings."""

import fcntl
import json
import math
import os
import re
import subprocess
import threading
import tempfile
from contextlib import contextmanager
from pathlib import Path
from urllib.parse import urlsplit

from openpilot.common.hardware.hw import Paths
from openpilot.common.swaglog import cloudlog
from openpilot.tools.lib.logreader import LogReader

ROUTE = re.compile(r"^([0-9a-f]{8}--[0-9a-f]{10})--(\d+)$")


def summary_path(root, route):
  return Path(root) / '.summaries' / f'{route}.json'


def cloud_path(route):
  return f'summaries/{0xFFFFFFFF - int(route[:8], 16):08x}.json'


def repository_url(remote):
  if not remote or len(remote) > 1000:
    return None
  value = remote.strip()
  if '://' not in value:
    value = re.sub(r'^(?:[^/@]+@)?([^/:]+):', r'\1/', value)
    value = 'https://' + value
  try:
    url = urlsplit(value)
    host = url.hostname
    path = url.path.strip('/').removesuffix('.git')
    if url.scheme not in ('http', 'https', 'git', 'ssh') or not host or '.' not in host:
      return None
    if not re.fullmatch(r'[a-zA-Z0-9.-]+', host) or not re.fullmatch(r'[a-zA-Z0-9_.-]+(?:/[a-zA-Z0-9_.-]+)+', path):
      return None
    if any(part in ('.', '..') for part in path.split('/')):
      return None
    value = f'https://{host}/{path}'
    return value if len(value) <= 200 else None
  except ValueError:
    return None


def init_info(init):
  info = {}
  for key, value in (
    ('openpilotVersion', init.version), ('gitBranch', init.gitBranch),
    ('gitCommit', init.gitCommit), ('gitSrcCommit', init.gitSrcCommit),
    ('deviceType', str(init.deviceType)), ('gitRemote', repository_url(init.gitRemote)),
  ):
    if isinstance(value, str) and value.strip() and len(value) <= 200 and not (key == 'deviceType' and value == 'unknown'):
      info[key] = value.strip()
  return info


def save_json(path, value):
  path = Path(path)
  path.parent.mkdir(parents=True, exist_ok=True)
  temporary = None
  try:
    with tempfile.NamedTemporaryFile(mode='w', prefix=path.name + '.', dir=path.parent, delete=False) as stream:
      temporary = Path(stream.name)
      json.dump(value, stream, separators=(',', ':'), allow_nan=False)
      stream.flush()
      os.fsync(stream.fileno())
    temporary.replace(path)
  finally:
    if temporary is not None:
      temporary.unlink(missing_ok=True)


@contextmanager
def recording_lock(root, shared=False):
  Path(root).mkdir(parents=True, exist_ok=True)
  with (Path(root) / '.route-counter.lock').open('a') as stream:
    try:
      fcntl.flock(stream, (fcntl.LOCK_SH if shared else fcntl.LOCK_EX) | fcntl.LOCK_NB)
    except BlockingIOError:
      yield False
      return
    try:
      yield True
    finally:
      fcntl.flock(stream, fcntl.LOCK_UN)


def segment_summary(path):
  first = last = wall = origin = None
  speed = enabled = None
  speed_t = state_t = None
  meters = engaged = 0.0
  have_speed = have_state = False
  start_location = end_location = None
  ended = False
  estimated = True
  video_origin = None
  video_priority = 3
  coords = []
  intervals = []
  previous_state = None
  info = {}

  def detail(name, value):
    if isinstance(value, str) and value.strip() and len(value) <= 200:
      info.setdefault(name, value.strip())

  for event in LogReader(str(path)):
    timestamp = event.logMonoTime / 1e9
    kind = event.which()
    if kind != 'initData':
      first = timestamp if first is None else min(first, timestamp)
      last = timestamp if last is None else max(last, timestamp)
    if kind == 'initData':
      for key, value in init_info(event.initData).items():
        detail(key, value)
      value = event.initData.wallTimeNanos / 1e6
      if value >= 1577836800000:
        wall, origin = value, timestamp
    elif kind == 'carParams':
      detail('carFingerprint', event.carParams.carFingerprint)
      detail('carBrand', event.carParams.brand)
    elif kind in ('narrowRoadEncodeIdx', 'qNarrowRoadEncodeIdx', 'wideRoadEncodeIdx', 'cabinEncodeIdx'):
      priority = 0 if kind in ('narrowRoadEncodeIdx', 'qNarrowRoadEncodeIdx') else 1 if kind == 'wideRoadEncodeIdx' else 2
      if priority < video_priority:
        index = getattr(event, kind)
        video_origin = index.timestampEof / 1e9 - index.segmentId / 20
        video_priority = priority
    elif kind == 'carState':
      if speed_t is not None and math.isfinite(speed) and 0 <= timestamp - speed_t <= 2:
        meters += max(0, speed) * (timestamp - speed_t)
        have_speed = True
      speed, speed_t = event.carState.vEgo, timestamp
    elif kind == 'selfdriveState':
      if state_t is not None and 0 <= timestamp - state_t <= 2:
        engaged += bool(enabled) * (timestamp - state_t)
        have_state = True
      state = event.selfdriveState
      alert = '\n'.join(text for text in (state.alertText1, state.alertText2) if text) if str(state.alertSize) != 'none' else ''
      current = {'state': 'alert' if alert else 'engaged' if state.enabled and str(state.state) != 'overriding' else 'disengaged'}
      if alert:
        current.update(alert=alert, severity='critical' if str(state.alertStatus) == 'critical' else 'warning')
      if previous_state is not None and state_t is not None and 0 < timestamp - state_t <= 2:
        if intervals and intervals[-1]['end'] == state_t and all(intervals[-1].get(k) == previous_state.get(k) for k in ('state', 'alert', 'severity')):
          intervals[-1]['end'] = timestamp
        else:
          intervals.append({'start': state_t, 'end': timestamp, **previous_state})
      previous_state = current
      enabled, state_t = state.enabled, timestamp
    elif kind in ('gpsLocationExternal', 'gpsLocation'):
      gps = getattr(event, kind)
      if (gps.hasFix and math.isfinite(gps.latitude) and math.isfinite(gps.longitude)
          and abs(gps.latitude) <= 90 and abs(gps.longitude) <= 180 and 0 <= gps.horizontalAccuracy <= 100):
        location = {'latitude': gps.latitude, 'longitude': gps.longitude}
        start_location = start_location or location
        end_location = location
        if not coords or timestamp - coords[-1][0] >= 1:
          coords.append([timestamp, round(gps.latitude, 6), round(gps.longitude, 6)])
        if gps.unixTimestampMillis >= 1577836800000 and origin is not None:
          wall = gps.unixTimestampMillis - (timestamp - origin) * 1000
          estimated = False
    elif kind == 'sentinel' and str(event.sentinel.type) == 'endOfRoute':
      ended = True
  # Match the video clock. initData repeats the route start in every segment.
  base = video_origin if video_origin is not None else first
  if base is not None:
    coords = [[round(max(0, t - base), 3), lat, lon] for t, lat, lon in coords if t >= base]
    intervals = [{**item, 'start': round(max(0, item['start'] - base), 3), 'end': round(item['end'] - base, 3)} for item in intervals if item['end'] > base]
  return {
    'coords': coords,
    'events': intervals,
    'startTime': round(wall) if wall is not None else None,
    'durationSeconds': max(0, last - base) if base is not None and last is not None else None,
    'distanceMeters': meters if have_speed else None,
    'engagedSeconds': engaged if have_state else None,
    'startLocation': start_location,
    'endLocation': end_location,
    'ended': ended,
    'timeEstimated': estimated,
    'info': info,
  }


class SummaryCollector:
  def __init__(self, root):
    self.root = Path(root)
    self.cache = {}

  def update(self):
    routes = {}
    for folder in self.root.iterdir():
      match = ROUTE.fullmatch(folder.name)
      if not match or folder.is_symlink() or not folder.is_dir():
        continue
      routes.setdefault(match[1], []).append((int(match[2]), folder))
    for state in (self.root / '.summaries').glob('*.segments'):
      if ROUTE.fullmatch(state.stem + '--0'):
        routes.setdefault(state.stem, [])
    with recording_lock(self.root, shared=True) as available:
      if not available:
        return
      idle = subprocess.run(['pgrep', '-x', 'loggerd'], stdout=subprocess.DEVNULL, check=False).returncode == 1
      if idle:
        from openpilot.system.loggerd.data_upload_queue import clear_stale_locks

        clear_stale_locks(self.root)
      for route, folders in routes.items():
        if summary_path(self.root, route).with_suffix('.deleted').exists():
          continue
        state_path = summary_path(self.root, route).with_suffix('.segments')
        saved = json.loads(state_path.read_text()) if state_path.exists() else {}
        previous_saved = json.dumps(saved, sort_keys=True)
        parts = []
        active = False
        for segment, folder in sorted(folders):
          if any(folder.glob('*.lock')):
            active = True
            continue
          source = next((folder / name for name in ('qlog.zst', 'qlog', 'rlog.zst')
                         if (folder / name).is_file() and (folder / name).stat().st_size > 0), None)
          if source is None:
            continue
          stat = source.stat()
          fingerprint = (stat.st_size, stat.st_mtime_ns)
          if self.cache.get(source, (None,))[0] != fingerprint:
            try:
              self.cache[source] = (fingerprint, segment_summary(source))
            except Exception:
              cloudlog.exception('route summary log parse failed')
              continue
          part = self.cache[source][1]
          saved[str(segment)] = {**part, 'info': {**saved.get(str(segment), {}).get('info', {}), **part['info']}}
        if json.dumps(saved, sort_keys=True) != previous_saved:
          save_json(state_path, saved)
        parts = sorted((int(segment), value) for segment, value in saved.items())
        if not parts:
          continue
        values = [part for _, part in parts]
        timed = next((part for part in values if part['startTime'] is not None and not part.get('timeEstimated', True)), values[0])
        start = timed['startTime']
        # initData is a route-start timestamp repeated in every segment.
        summary = {
          'version': 1,
          'routeId': route,
          'status': 'complete' if not active and (idle or values[-1]['ended']) else 'unfinished',
          'startTime': start,
          'timeEstimated': timed.get('timeEstimated', True),
          'info': {key: value for part in reversed(values) for key, value in part.get('info', {}).items()},
        }
        for field in ('distanceMeters', 'engagedSeconds'):
          valid = [part[field] for part in values if part[field] is not None]
          summary[field] = round(sum(valid), 3) if valid else None
        ends = [segment * 60 + part['durationSeconds'] for segment, part in parts if part['durationSeconds'] is not None]
        summary['durationSeconds'] = round(max(ends), 3) if ends else None
        summary['coords'] = [[round(t + segment * 60, 3), lat, lon] for segment, part in parts for t, lat, lon in part.get('coords', [])]
        summary['events'] = []
        for segment, part in parts:
          for event in part.get('events', []):
            item = {**event, 'start': round(event['start'] + segment * 60, 3), 'end': round(event['end'] + segment * 60, 3)}
            previous = summary['events'][-1] if summary['events'] else None
            if previous and 0 <= item['start'] - previous['end'] < 0.05 and all(previous.get(k) == item.get(k) for k in ('state', 'alert', 'severity')):
              previous['end'] = item['end']
            else:
              summary['events'].append(item)
        summary['startLocation'] = next((p['startLocation'] for p in values if p['startLocation']), None)
        summary['endLocation'] = next((p['endLocation'] for p in reversed(values) if p['endLocation']), None)
        path = summary_path(self.root, route)
        old = json.loads(path.read_text()) if path.exists() else None
        if old != summary:
          save_json(path, summary)


def recover_counters(root, params, conflict_route, highest):
  with recording_lock(root) as idle:
    if not idle or not params.get_bool('IsOffroad'):
      return False
    directory = Path(root) / '.summaries'
    journal = directory / 'rename-plan.json'
    if journal.exists():
      plan = json.loads(journal.read_text())
    else:
      candidates = sorted(
        p.stem for p in directory.glob('*.json') if p.stem >= conflict_route and ROUTE.fullmatch(p.stem + '--0') and not p.with_suffix('.uploaded').exists()
      )
      next_count = max(highest + 1, int(params.get('RouteCount') or 0))
      if next_count + len(candidates) > 0xFFFFFFFF:
        raise RuntimeError('Route counter exhausted')
      plan = {route: f'{next_count + i:08x}--{route.split("--")[1]}' for i, route in enumerate(candidates)}
      save_json(journal, plan)
    for old, new in plan.items():
      for folder in Path(root).glob(old + '--*'):
        folder.rename(folder.with_name(new + folder.name[len(old) :]))
      source = summary_path(root, old)
      if source.exists():
        value = json.loads(source.read_text())
        value['routeId'] = new
        save_json(summary_path(root, new), value)
        source.unlink()
      state = source.with_suffix('.segments')
      if state.exists():
        state.rename(summary_path(root, new).with_suffix('.segments'))
    if plan:
      params.put('RouteCount', max(int(params.get('RouteCount') or 0), max(int(v[:8], 16) for v in plan.values()) + 1), block=True)
    journal.unlink(missing_ok=True)
    return True


def main():
  collector = SummaryCollector(Paths.log_root())
  stop = threading.Event()
  while not stop.is_set():
    try:
      collector.update()
    except Exception:
      cloudlog.exception('route summary failed')
    stop.wait(2)


if __name__ == '__main__':
  main()
