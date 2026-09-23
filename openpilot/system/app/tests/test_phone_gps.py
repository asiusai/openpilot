import asyncio
from unittest.mock import AsyncMock

import pytest

from openpilot.cereal import log
from openpilot.cereal.services import SERVICE_LIST
from openpilot.system.app import bluetoothd, methods
from openpilot.system.app.phone_gps import PhoneGps, gps_message


def fix(**changes):
  return {
    'latitude': 59.437, 'longitude': 24.7536, 'accuracy': 8., 'altitude': 40., 'altitudeAccuracy': 12.,
    'speed': 15., 'heading': 90., 'timestamp': 1800000000000, 'ageMs': 0., 'source': 'android', **changes,
  }


@pytest.fixture
def gps():
  now = [100.]
  sent = []
  gps = PhoneGps(sent.append, lambda: now[0])
  return gps, now, sent


def test_standard_gps_message_roundtrips_in_route_log_format():
  message = gps_message(fix())
  with log.Event.from_bytes(message.to_bytes()) as event:
    assert event.which() == 'gpsLocation'
    assert event.valid
    location = event.gpsLocation
    assert location.hasFix
    assert location.latitude == pytest.approx(59.437)
    assert location.longitude == pytest.approx(24.7536)
    assert location.altitude == 40
    assert location.source == 'android'
    assert list(location.vNED) == pytest.approx([0., 15., 0.], abs=1e-6)
    assert location.unixTimestampMillis == 1800000000000
  service = SERVICE_LIST['gpsLocation']
  assert service.should_log
  assert service.frequency == 1
  assert service.decimation == 1  # every phone fix also reaches qlog


def test_missing_measurements_are_marked_uncertain_not_fabricated():
  location = gps_message(fix(altitude=None, altitudeAccuracy=None, speed=None, heading=None, source='iOS')).gpsLocation
  assert location.hasFix
  assert location.source == 'iOS'
  assert location.verticalAccuracy == 10000
  assert location.speedAccuracy == 1000
  assert location.bearingAccuracyDeg == 180
  assert list(location.vNED) == [0., 0., 0.]
  assert location.satelliteCount == 0


@pytest.mark.parametrize(('key', 'value'), [
  ('latitude', 91), ('latitude', float('nan')), ('longitude', -181), ('longitude', True),
  ('accuracy', -1), ('accuracy', float('inf')), ('altitude', 100001), ('altitudeAccuracy', -1),
  ('speed', -1), ('speed', 201), ('heading', -1), ('heading', 361), ('timestamp', 'yesterday'),
  ('timestamp', 253402300799001), ('ageMs', 3001), ('ageMs', -1), ('source', 'ublox'),
])
def test_invalid_fixes_do_not_publish_or_advance_sequence(gps, key, value):
  service, _, sent = gps
  session = service.start('app')['session']
  with pytest.raises(ValueError):
    service.update('app', session, 0, fix(**{key: value}))
  assert sent == []
  assert service.sequence == -1


def test_low_accuracy_and_no_fix_are_not_valid_route_points(gps):
  service, _, sent = gps
  session = service.start('app')['session']
  assert not service.update('app', session, 0, fix(accuracy=101))['hasFix']
  assert not sent[-1].gpsLocation.hasFix
  service.update('app', session, 1, None)
  assert not sent[-1].valid
  assert not sent[-1].gpsLocation.hasFix


def test_sessions_are_peer_bound_and_replay_protected(gps):
  service, _, sent = gps
  session = service.start('app')['session']
  service.update('app', session, 0, fix())
  for peer, token, sequence in [('other', session, 1), ('app', 'wrong', 1), ('app', session, 0), ('app', session, -1), ('app', session, True)]:
    with pytest.raises(ValueError):
      service.update(peer, token, sequence, fix(timestamp=1800000001000))
  with pytest.raises(ValueError, match='newer'):
    service.update('app', session, 1, fix())
  assert len(sent) == 1
  assert not service.stop('other', session)['stopped']
  assert not service.stop('app', 'wrong')['stopped']
  with pytest.raises(RuntimeError, match='Another phone'):
    service.start('other')


def test_fixes_expire_without_republishing_cached_coordinates(gps):
  service, now, sent = gps
  session = service.start('app')['session']
  service.update('app', session, 0, fix(ageMs=2000))
  now[0] += 1.01
  service.tick()
  assert not sent[-1].valid
  assert not sent[-1].gpsLocation.hasFix
  assert len(sent) == 2
  service.tick()
  assert len(sent) == 2
  now[0] += 5
  service.tick()
  with pytest.raises(ValueError, match='expired'):
    service.update('app', session, 1, fix(timestamp=1800000001000))
  assert service.owner is None


def test_new_session_invalidates_old_and_old_stop_cannot_stop_new(gps, monkeypatch):
  service, _, sent = gps
  monkeypatch.setattr('time.time', lambda: 0.)
  first = service.start('app')['session']
  service.update('app', first, 0, fix())
  second = service.start('app')['session']
  assert first != second
  assert not sent[-1].valid
  assert not service.stop('app', first)['stopped']
  service.update('app', second, 0, fix(timestamp=1800000001000))
  assert service.stop('app', second)['stopped']
  assert not sent[-1].valid


def test_revoking_authorization_removes_fix(gps):
  service, _, sent = gps
  session = service.start('app')['session']
  service.update('app', session, 0, fix())
  service.tick({'other': {}})
  assert service.owner is None
  assert not sent[-1].valid


def test_gps_methods_only_exist_on_bluetooth_dispatcher(gps, monkeypatch):
  service, _, sent = gps
  engine = bluetoothd.BlePeerEngine.__new__(bluetoothd.BlePeerEngine)
  engine.phone_gps = service
  engine.dongle_id = 'device'
  engine.active_peers = {}
  engine.send_body = AsyncMock()
  engine.unpack_body = lambda _: ('app', {'id': 'rpc', 'method': 'startPhoneGps'}, False)
  monkeypatch.setattr(bluetoothd, 'load_authorized_peers', dict)
  with pytest.raises(PermissionError):
    asyncio.run(engine.handle_encrypted(b''))
  assert service.owner is None
  monkeypatch.setattr(bluetoothd, 'load_authorized_peers', lambda: {'app': {}})
  asyncio.run(engine.handle_encrypted(b''))
  response = engine.send_body.call_args.args[1]
  assert response['result']['session'] == service.session
  for method in ('startPhoneGps', 'updatePhoneGps', 'stopPhoneGps'):
    assert method not in methods.dispatcher_for_peer('app')
  service.update('app', service.session, 0, fix())
  engine.clear_active_peers()
  assert not sent[-1].valid
  assert service.owner is None


def test_asius_uses_phone_gps_even_with_stale_ublox_parameter(monkeypatch):
  from openpilot.common import gps as selection
  from unittest.mock import Mock
  params = Mock()
  params.get_bool.return_value = True
  monkeypatch.setattr(selection, 'ASIUS_HARDWARE', True)
  assert selection.get_gps_location_service(params) == 'gpsLocation'
  monkeypatch.setattr(selection, 'ASIUS_HARDWARE', False)
  assert selection.get_gps_location_service(params) == 'gpsLocationExternal'


def test_live_snapshot_marks_stale_gps_unavailable(monkeypatch):
  from types import SimpleNamespace
  from unittest.mock import Mock
  params = Mock()
  params.get.return_value = None
  sm = Mock()
  sm.alive = {'gpsLocation': False}
  sm.valid = {'gpsLocation': True}
  sm.recv_frame = dict.fromkeys(methods.LIVE_STATE_SERVICES, 0)
  build = SimpleNamespace(channel='master', openpilot=SimpleNamespace(version='test', git_normalized_origin='', git_commit=''))
  monkeypatch.setattr(methods, 'get_build_metadata', lambda: build)
  monkeypatch.setattr(methods, 'get_device_name', lambda: 'Asius v0')
  monkeypatch.setattr(methods, 'load_authorized_peers', dict)
  monkeypatch.setattr(methods, '_software_update_state', lambda _: {})
  assert methods._live_state_snapshot(sm, params)['services']['gpsLocation'] == {'hasFix': False}
