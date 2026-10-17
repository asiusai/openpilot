import fcntl
import json
from types import SimpleNamespace

from openpilot.cereal import log
from openpilot.system.loggerd.route_summary import SummaryCollector, cloud_path, recover_counters, save_json, summary_path

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

  event('initData', 1).wallTimeNanos = 1700000000000000000
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
