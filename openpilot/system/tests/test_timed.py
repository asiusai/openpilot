import datetime
from unittest.mock import Mock

import pytest

from openpilot.system import timed


@pytest.mark.parametrize('offset', [-30, 0.1, 30])
def test_asius_gps_uses_clock_helper_for_all_corrections(monkeypatch, offset):
  monkeypatch.setattr(timed, 'ASIUS_HARDWARE', True)
  run = Mock()
  monkeypatch.setattr(timed.subprocess, 'run', run)
  sample = datetime.datetime.now(datetime.UTC).replace(tzinfo=None) + datetime.timedelta(seconds=offset)
  timed.set_time(sample, 100.5)
  command = run.call_args.args[0]
  assert command[:6] == ['sudo', '-n', '/usr/bin/vamos-clock', 'sync', '--source', 'gps']
  assert float(command[7]) == sample.replace(tzinfo=datetime.UTC).timestamp() * 1000
  assert command[8:] == ['--sample-mono', '100.5']
  assert run.call_args.kwargs['timeout'] == 5


def test_clock_helper_failure_does_not_kill_timed(monkeypatch):
  monkeypatch.setattr(timed, 'ASIUS_HARDWARE', True)
  monkeypatch.setattr(timed.subprocess, 'run', Mock(side_effect=OSError('helper unavailable')))
  timed.set_time(datetime.datetime.now(datetime.UTC).replace(tzinfo=None), 100)
