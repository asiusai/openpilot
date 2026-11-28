"""Local route summaries, independent of network access and upload settings."""

import fcntl
import json
import math
import os
import re
import stat
import subprocess
import sys
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path
from urllib.parse import urlsplit

from openpilot.common.hardware.hw import Paths
from openpilot.common.realtime import Ratekeeper, set_core_affinity
from openpilot.common.swaglog import cloudlog
from openpilot.system.loggerd.storage_upload_queue import ROUTE_SEGMENT_RE, clear_stale_locks
from openpilot.tools.lib.logreader import LogReader

ROUTE = re.compile(r"^([0-9a-f]{8}--[0-9a-f]{10})--(\d+)$")
# Increase this when segment_summary changes its result. All logs are then parsed again.
PARSER_VERSION = 1
# A directory change in the same clock tick as a listing leaves the directory time unchanged.
# Trust a listing only when the directory time stayed the same for this long.
SETTLE_SECONDS = 3
# Look at every directory again after this many passes. This finds a change in a closed segment,
# and a change with no new directory time.
RECHECK_PASSES = 30


def summary_path(root, route):
  return Path(root) / '.summaries' / f'{route}.json'


def storage_path(route):
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
      if speed is not None and speed_t is not None and math.isfinite(speed) and 0 <= timestamp - speed_t <= 2:
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


def segment_folders(root):
  """List the log root. Return the segments of each route, and the other folders that can hold recording locks."""
  routes: dict[str, list[tuple[int, str]]] = {}
  others = []
  with os.scandir(root) as entries:
    for entry in entries:
      match = ROUTE.fullmatch(entry.name)
      if not (match or ROUTE_SEGMENT_RE.fullmatch(entry.name)) or not entry.is_dir(follow_symlinks=False):
        continue
      if match:
        routes.setdefault(match[1], []).append((int(match[2]), entry.path))
      else:
        others.append(entry.path)
  return {route: sorted(segments) for route, segments in routes.items()}, others


def segment_state(folder):
  """Return if a segment is still being written, and the name, size and time of its finished log."""
  try:
    names = os.listdir(folder)
  except OSError:
    names = []
  if any(name.endswith('.lock') for name in names):
    return True, None
  for name in ('qlog.zst', 'qlog', 'rlog.zst'):
    if name in names:
      try:
        status = os.stat(os.path.join(folder, name))
      except OSError:
        continue
      if stat.S_ISREG(status.st_mode) and status.st_size > 0:
        return False, (name, status.st_size, status.st_mtime_ns)
  return False, None


def summary_files(directory):
  """Return the file name endings present for each route in the summary directory."""
  files: dict[str, set[str]] = {}
  try:
    names = os.listdir(directory)
  except OSError:
    names = []
  for name in names:
    route, _, ending = name.rpartition('.')
    files.setdefault(route, set()).add(ending)
  return files


def load_parsed(path):
  """Read which logs are already parsed into the saved segments. A missing or bad file means none."""
  try:
    value = json.loads(path.read_text())
    return value['segments'] if value['version'] == PARSER_VERSION and isinstance(value['segments'], dict) else {}
  except Exception:
    return {}


def process_name(pid):
  try:
    with open(f'/proc/{pid}/comm', 'rb') as stream:
      return stream.read()
  except OSError:
    return None


class SummaryCollector:
  def __init__(self, root):
    self.root = Path(root)
    self.passes = 0
    self.logger_pid = None
    # Directory path -> (directory time, first time seen with it, value read from the directory or None).
    self.directories = {}
    # Route -> what the last full pass saw, if its status ignores the logger, and the logger state then.
    self.routes = {}
    # Route -> logs parsed in this run and not saved yet. A failed save must not parse them again.
    self.unsaved = {}

  def cached(self, path, read, recheck=False):
    """Return read(path). Read again only when the directory time changed, or for a recheck."""
    try:
      changed = os.stat(path).st_mtime_ns
    except OSError:
      changed = None
    now = time.monotonic()
    known = self.directories.get(path)
    if known is None or known[0] != changed:
      known = self.directories[path] = (changed, now, None)
    elif known[2] is not None and not recheck:
      return known[2]
    value = read(path)
    self.directories[path] = (changed, known[1], value if now - known[1] >= SETTLE_SECONDS else None)
    return value

  def folder_state(self, folder, recheck=False):
    """Return segment_state(folder). A closed segment with a finished log is looked at again only for a recheck."""
    known = self.directories.get(folder)
    if known is not None and known[2] is not None and known[2][1] is not None and not recheck:
      return known[2]
    return self.cached(folder, segment_state, recheck)

  def logger_running(self, search):
    """Tell if a process named loggerd exists. A search gives the answer of `pgrep -x loggerd` without a new process."""
    if sys.platform != 'linux':
      return search and subprocess.run(['pgrep', '-x', 'loggerd'], stdout=subprocess.DEVNULL, check=False).returncode != 1
    # Look at the last known process first. Search all processes only when it is gone.
    if self.logger_pid is None or process_name(self.logger_pid) != b'loggerd\n':
      names = os.listdir('/proc') if search else []
      self.logger_pid = next((pid for pid in names if pid.isdigit() and process_name(pid) == b'loggerd\n'), None)
    return self.logger_pid is not None

  def update(self):
    recheck = self.passes % RECHECK_PASSES == 0
    self.passes += 1
    root, summaries = str(self.root), str(self.root / '.summaries')
    routes, others = self.cached(root, segment_folders, recheck)
    files = self.cached(summaries, summary_files, recheck)
    folders = [folder for segments in routes.values() for _, folder in segments] + others
    # Routes with no folder left still have a summary to keep.
    routes = {**routes, **{route: [] for route, endings in files.items()
                           if 'segments' in endings and route not in routes and ROUTE.fullmatch(route + '--0')}}
    with recording_lock(self.root, shared=True) as available:
      if not available:
        return
      states = {folder: self.folder_state(folder, recheck) for folder in folders}
      locked = any(active for active, _ in states.values())
      # A running logger keeps a lock file in its segment. Search for the logger only then, and in each recheck.
      idle = not self.logger_running(locked or recheck)
      if idle and locked:
        clear_stale_locks(self.root)
        states = {folder: self.folder_state(folder) for folder in folders}
      for route in self.routes.keys() - routes.keys():
        del self.routes[route]
      if recheck:
        for path in self.directories.keys() - states.keys() - {root, summaries}:
          del self.directories[path]
      for route, segments in routes.items():
        endings = files.get(route, set())
        if 'deleted' in endings:
          continue
        # Do the work below only when the route changed, or when the logger state changes its status.
        seen = (segments, [states[folder] for _, folder in segments], endings)
        last = self.routes.pop(route, None)
        if last is not None and last[0] == seen and (last[1] or last[2] == idle):
          self.routes[route] = last
          continue
        state_path = summary_path(self.root, route).with_suffix('.segments')
        saved = json.loads(state_path.read_text()) if state_path.exists() else {}
        previous_saved = None
        cache_path = state_path.with_suffix('.parsed')
        parsed = load_parsed(cache_path)
        previous_parsed = dict(parsed)
        unsaved = self.unsaved.setdefault(route, {})
        parts = []
        active = failed = False
        for segment, folder in segments:
          writing, fingerprint = states[folder]
          if writing:
            active = True
            continue
          if fingerprint is None:
            continue
          key = str(segment)
          if key in saved and parsed.get(key) == list(fingerprint):
            continue
          if unsaved.get(key, (None,))[0] != fingerprint:
            try:
              unsaved[key] = (fingerprint, segment_summary(os.path.join(folder, fingerprint[0])))
            except Exception:
              cloudlog.exception('route summary log parse failed')
              failed = True
              continue
          part = unsaved[key][1]
          previous_saved = previous_saved or json.dumps(saved, sort_keys=True)
          saved[key] = {**part, 'info': {**saved.get(key, {}).get('info', {}), **part['info']}}
          parsed[key] = list(fingerprint)
        if previous_saved is not None and json.dumps(saved, sort_keys=True) != previous_saved:
          save_json(state_path, saved)
        if parsed != previous_parsed:
          save_json(cache_path, {'version': PARSER_VERSION, 'segments': parsed})
        del self.unsaved[route]
        parts = sorted((int(segment), value) for segment, value in saved.items())
        # A failed log is tried again in the next pass.
        done = None if failed else (seen, active or not parts or parts[-1][1]['ended'], idle)
        if not parts:
          if done is not None:
            self.routes[route] = done
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
        events: list[dict] = []
        summary['events'] = events
        for segment, part in parts:
          for event in part.get('events', []):
            item = {**event, 'start': round(event['start'] + segment * 60, 3), 'end': round(event['end'] + segment * 60, 3)}
            previous = events[-1] if events else None
            if previous and 0 <= item['start'] - previous['end'] < 0.05 and all(previous.get(k) == item.get(k) for k in ('state', 'alert', 'severity')):
              previous['end'] = item['end']
            else:
              events.append(item)
        summary['startLocation'] = next((p['startLocation'] for p in values if p['startLocation']), None)
        summary['endLocation'] = next((p['endLocation'] for p in reversed(values) if p['endLocation']), None)
        path = summary_path(self.root, route)
        old = json.loads(path.read_text()) if path.exists() else None
        if old != summary:
          save_json(path, summary)
        if done is not None:
          self.routes[route] = done


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
      for suffix in ('.segments', '.parsed'):
        state = source.with_suffix(suffix)
        if state.exists():
          state.rename(summary_path(root, new).with_suffix(suffix))
    if plan:
      params.put('RouteCount', max(int(params.get('RouteCount') or 0), max(int(v[:8], 16) for v in plan.values()) + 1), block=True)
    journal.unlink(missing_ok=True)
    return True


def main():
  # Summaries can wait. Do not take processor time from driving processes.
  os.nice(19)
  try:
    set_core_affinity([0, 1, 2, 3])
  except Exception:
    cloudlog.exception('failed to set core affinity')

  collector = SummaryCollector(Paths.log_root())
  rk = Ratekeeper(0.5, print_delay_threshold=None)
  while True:
    try:
      collector.update()
    except Exception:
      cloudlog.exception('route summary failed')
    rk.keep_time()


if __name__ == '__main__':
  main()
