from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from openpilot.common import gps as selection
from openpilot.system.asius import methods


@pytest.mark.parametrize(('asius', 'ublox', 'expected'), [
  (True, False, ['gpsLocationExternal']),
  (True, True, ['gpsLocationExternal']),
  (False, True, ['gpsLocationExternal']),
  (False, False, ['gpsLocation']),
])
def test_device_gps_sources(monkeypatch, asius, ublox, expected):
  params = Mock()
  params.get_bool.return_value = ublox
  monkeypatch.setattr(selection, 'ASIUS_HARDWARE', asius)
  assert selection.get_gps_location_services(params) == expected


@pytest.mark.parametrize('alive', [False, True])
def test_live_snapshot_reports_receiver_connection_without_a_fix(monkeypatch, alive):
  params = Mock()
  params.get.return_value = None
  sm = Mock()
  sm.alive = dict.fromkeys(methods.LIVE_STATE_SERVICES, False)
  sm.valid = dict.fromkeys(methods.LIVE_STATE_SERVICES, True)
  for service in ('gpsLocation', 'gpsLocationExternal'):
    sm.alive[service] = alive
    sm.valid[service] = False
  sm.logMonoTime = dict.fromkeys(methods.LIVE_STATE_SERVICES, 20_000_000_000)
  sm.recv_frame = dict.fromkeys(methods.LIVE_STATE_SERVICES, 0)
  sm.seen = dict.fromkeys(methods.LIVE_STATE_SERVICES, False)
  build = SimpleNamespace(channel='master', openpilot=SimpleNamespace(version='test', git_normalized_origin='', git_commit=''))
  monkeypatch.setattr(methods, 'get_build_metadata', lambda: build)
  monkeypatch.setattr(methods, 'getDeviceName', lambda: 'Asius v0')
  monkeypatch.setattr(methods, 'load_authorized_peers', dict)
  monkeypatch.setattr(methods, '_software_update_state', lambda _: {})
  monkeypatch.setattr(methods, 'ignition_state', lambda: None)
  services = methods._live_state_snapshot(sm, params)['services']
  expected = {'hasFix': False, 'connected': alive}
  if alive:
    expected['sampleMonoTime'] = 20.0
  assert services['gpsLocation'] == services['gpsLocationExternal'] == expected
