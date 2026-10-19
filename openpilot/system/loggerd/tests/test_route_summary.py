import fcntl
import json
from types import SimpleNamespace

from openpilot.cereal import log
from openpilot.system.loggerd.route_summary import SummaryCollector, cloud_path, init_info, recover_counters, repository_url, save_json, summary_path

ROUTE = '00000001--1234567890'


def write_segment(root, segment, start, end=False):
  folder = root / f'{ROUTE}--{segment}'
  folder.mkdir(parents=True)
  events = []

  def event(kind, t):
    e = log.Event.new_message(logMonoTime=int(t * 1e9))
    e.init(kind)
    events.append(e)
    return getattr(e, kind)

  init = event('initData', 1)
  init.wallTimeNanos = 1700000000000000000
  init.version, init.gitBranch, init.gitCommit, init.deviceType = '0.10.3', 'master', 'abc123456789', 'v0'
  init.gitRemote = 'git@github.com:asiusai/openpilot.git'
  car = event('carParams', start)
  car.carFingerprint, car.brand, car.carVin = 'TOYOTA_PRIUS', 'toyota', 'private-vin'
  for t in range(start, start + 61):
    event('carState', t).vEgo = 10
    event('selfdriveState', t).enabled = True
  if end:
    event('sentinel', start + 60).type = 'endOfRoute'
  (folder / 'qlog').write_bytes(b''.join(e.to_bytes() for e in events))
  return folder


def test_summary_uses_segment_times_and_retains_totals_after_cleanup(tmp_path, monkeypatch):
  monkeypatch.setattr('openpilot.system.loggerd.route_summary.subprocess.run', lambda *a, **k: SimpleNamespace(returncode=0))
  first = write_segment(tmp_path, 0, 1)
  collector = SummaryCollector(tmp_path)
  collector.update()
  initial = json.loads(summary_path(tmp_path, ROUTE).read_text())
  assert initial['status'] == 'unfinished'
  assert initial['durationSeconds'] == 60
  assert initial['distanceMeters'] == 600
  assert initial['startLocation'] is None
  (first / 'qlog').unlink()
  first.rmdir()
  write_segment(tmp_path, 1, 61, True)
  collector = SummaryCollector(tmp_path)
  collector.update()
  summary = json.loads(summary_path(tmp_path, ROUTE).read_text())
  assert summary['status'] == 'complete'
  assert summary['durationSeconds'] == 120
  assert summary['distanceMeters'] == 1200
  assert summary['engagedSeconds'] == 120
  assert summary['startTime'] == 1700000000000
  assert summary['info'] == {
    'openpilotVersion': '0.10.3', 'gitBranch': 'master', 'gitCommit': 'abc123456789',
    'deviceType': 'v0', 'carFingerprint': 'TOYOTA_PRIUS', 'carBrand': 'toyota',
    'gitRemote': 'https://github.com/asiusai/openpilot',
  }
  assert 'private-vin' not in json.dumps(summary)
  last = tmp_path / f'{ROUTE}--1'
  (last / 'qlog').unlink()
  last.rmdir()
  SummaryCollector(tmp_path).update()
  assert json.loads(summary_path(tmp_path, ROUTE).read_text()) == summary


def test_summary_backfills_existing_cache_and_keeps_recorded_info(tmp_path, monkeypatch):
  monkeypatch.setattr('openpilot.system.loggerd.route_summary.subprocess.run', lambda *a, **k: SimpleNamespace(returncode=1))
  folder = write_segment(tmp_path, 0, 1, True)
  SummaryCollector(tmp_path).update()
  path = summary_path(tmp_path, ROUTE)
  original = json.loads(path.read_text())
  parts = json.loads(path.with_suffix('.segments').read_text())
  parts['0'].pop('info')
  save_json(path.with_suffix('.segments'), parts)
  save_json(path, {k: v for k, v in original.items() if k != 'info'})
  SummaryCollector(tmp_path).update()
  assert json.loads(path.read_text()) == original
  # A later partial source must not erase previously recorded information.
  e = log.Event.new_message(logMonoTime=1_000_000_000)
  e.init('carState').vEgo = 0
  (folder / 'qlog').write_bytes(e.to_bytes())
  SummaryCollector(tmp_path).update()
  assert json.loads(path.read_text())['info'] == original['info']
  # Deleted routes must not be regenerated or backfilled.
  path.with_suffix('.deleted').touch()
  path.unlink()
  SummaryCollector(tmp_path).update()
  assert not path.exists()


def test_recorded_repository_excludes_credentials_and_retains_source_commit():
  repository = 'https://github.com/asiusai/openpilot'
  for remote in ('git@github.com:asiusai/openpilot.git', 'ssh://git@github.com/asiusai/openpilot.git',
                 'https://user:secret@github.com/asiusai/openpilot.git?token=secret', 'github.com/asiusai/openpilot'):
    assert repository_url(remote) == repository
  for remote in ('javascript:alert(1)', 'file:///data/openpilot', 'https://github.com/owner/../repo', ''):
    assert repository_url(remote) is None
  event = log.Event.new_message()
  init = event.init('initData')
  init.gitRemote, init.gitSrcCommit = repository + '.git', '123456789abcdef'
  assert init_info(init) == {'gitRemote': repository, 'gitSrcCommit': '123456789abcdef'}


def test_counter_recovery_waits_for_logger_and_preserves_route_order(tmp_path):
  write_segment(tmp_path, 0, 1)
  save_json(summary_path(tmp_path, ROUTE), {'routeId': ROUTE})

  class Params:
    count = 2

    def get_bool(self, key):
      return True

    def get(self, key):
      return self.count

    def put(self, key, value, **kw):
      self.count = value

  params = Params()
  with (tmp_path / '.route-counter.lock').open('a') as lock:
    fcntl.flock(lock, fcntl.LOCK_SH)
    assert not recover_counters(tmp_path, params, ROUTE, 10)
  assert recover_counters(tmp_path, params, ROUTE, 10)
  renamed = '0000000b--1234567890'
  assert (tmp_path / f'{renamed}--0/qlog').exists()
  assert json.loads(summary_path(tmp_path, renamed).read_text())['routeId'] == renamed
  assert params.count == 12
  assert cloud_path(renamed) == 'summaries/fffffff4.json'


def test_interrupted_empty_qlog_uses_available_rlog(tmp_path, monkeypatch):
  import zstandard
  monkeypatch.setattr('openpilot.system.loggerd.route_summary.subprocess.run', lambda *a, **k: SimpleNamespace(returncode=1))
  folder = write_segment(tmp_path, 0, 1, True)
  (folder / 'rlog.zst').write_bytes(zstandard.compress((folder / 'qlog').read_bytes()))
  (folder / 'qlog').unlink()
  (folder / 'qlog.zst').touch()
  SummaryCollector(tmp_path).update()
  summary = json.loads(summary_path(tmp_path, ROUTE).read_text())
  assert summary['durationSeconds'] == 60
  assert summary['distanceMeters'] == 600
  assert summary['status'] == 'complete'


def test_summary_coordinates_events_use_video_clock_and_keep_alert_details(tmp_path, monkeypatch):
  monkeypatch.setattr('openpilot.system.loggerd.route_summary.subprocess.run', lambda *a, **k: SimpleNamespace(returncode=1))
  folder = tmp_path / f'{ROUTE}--2'
  folder.mkdir()
  events = []

  def event(kind, t):
    e = log.Event.new_message(logMonoTime=int(t * 1e9), valid=True)
    e.init(kind)
    events.append(e)
    return getattr(e, kind)

  event('initData', 1).wallTimeNanos = 1700000000000000000
  index = event('qNarrowRoadEncodeIdx', 501)
  index.timestampEof, index.segmentId = int(501e9), 20
  for t in range(500, 505):
    gps = event('gpsLocation', t)
    gps.hasFix, gps.latitude, gps.longitude, gps.horizontalAccuracy = True, 58.3 + (t-500)*0.0001, 26.7, 2
    state = event('selfdriveState', t)
    state.enabled = True
    if t >= 502:
      state.alertText1, state.alertText2 = 'Take control', 'System unavailable'
      state.alertSize, state.alertStatus = 'full', 'critical'
  # A telemetry gap must not create an engagement span.
  event('selfdriveState', 510).enabled = False
  event('selfdriveState', 511).enabled = False
  (folder / 'qlog').write_bytes(b''.join(e.to_bytes() for e in events))
  SummaryCollector(tmp_path).update()
  summary = json.loads(summary_path(tmp_path, ROUTE).read_text())
  assert summary['coords'][0] == [120, 58.3, 26.7]
  assert summary['coords'][-1][0] == 124
  assert summary['durationSeconds'] == 131
  assert summary['events'] == [
    {'start': 120, 'end': 122, 'state': 'engaged'},
    {'start': 122, 'end': 124, 'state': 'alert', 'alert': 'Take control\nSystem unavailable', 'severity': 'critical'},
    {'start': 130, 'end': 131, 'state': 'disengaged'},
  ]
  (folder / 'qlog').unlink()
  # Retain measured paths and events when recording cleanup removes a log.
  SummaryCollector(tmp_path).update()
  assert json.loads(summary_path(tmp_path, ROUTE).read_text()) == summary
