import base64
import gzip
import hashlib
import json
import random
import time
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from openpilot.common.params import Params
from openpilot.selfdrive.navigation.navd import RouteEngine
from openpilot.selfdrive.navigation import navd
from openpilot.system.asius.navigation_route import RouteStore, CHUNK_BYTES


@pytest.mark.parametrize("position", [None, {}, [24], [24, 91], [181, 59], [True, 59], ["24", 59], [float("nan"), 59]])
def test_invalid_saved_position_is_ignored(tmp_path, position):
  store = RouteStore(Params(str(tmp_path)))
  store.root.mkdir(parents=True, exist_ok=True)
  (store.root / "position.json").write_text(json.dumps(position))
  assert store.position() is None
  (store.root / "state.json").write_text("[]")
  assert store.read() == {}


def test_route_chunks_are_shared_bounded_and_revision_checked(tmp_path):
  params = Params(str(tmp_path))
  place = {"latitude": 59., "longitude": 24., "requestId": "first"}
  params.put_bool("NavigationEnabled", True, block=True)
  params.put("NavDestination", place, block=True)
  store = RouteStore(params)
  rng = random.Random(42)
  route = {"geometry": {"coordinates": [[24 + rng.random(), 59 + rng.random()] for _ in range(4000)]}}
  assert store.publish(place, route=route)
  revision = store.state()["revision"]
  data = b""
  while True:
    part = store.chunk(revision, len(data))
    assert len(part["chunk"]) <= 4096
    data += base64.b64decode(part["chunk"])
    if part["done"]:
      break
  assert hashlib.sha256(data).hexdigest() == revision
  assert json.loads(gzip.decompress(data)) == route
  assert RouteStore(params).chunk(revision) == store.chunk(revision)
  for invalid in (-1, 1, True, len(data) + CHUNK_BYTES):
    with pytest.raises(ValueError):
      store.chunk(revision, invalid)
  with pytest.raises(ValueError):
    store.chunk("../../state.json")
  params.put("NavDestination", {**place, "requestId": "new"}, block=True)
  assert not store.publish(place, route=route)
  assert store.state()["status"] == "loading"
  with pytest.raises(ValueError, match="Route changed"):
    store.chunk(revision)
  params.remove("NavDestination")
  assert store.state() == {"status": "idle"}


def test_late_mapbox_reply_cannot_replace_new_destination(tmp_path):
  params = Params(str(tmp_path))
  place = {"latitude": 59., "longitude": 24., "requestId": "first"}
  params.put_bool("NavigationEnabled", True, block=True)
  params.put("NavDestination", place, block=True)
  store = RouteStore(params)
  engine = RouteEngine(MagicMock(), MagicMock(), params, store=store)
  engine.last_position = (24., 59.)
  def response(*args, **kwargs):
    params.put("NavDestination", {**place, "longitude": 25., "requestId": "second"}, block=True)
    result = MagicMock(status_code=200)
    result.json.return_value = {"routes": [{"legs": [{"steps": []}]}]}
    return result
  with patch("requests.get", side_effect=response), patch("openpilot.system.asius.navigation_proxy.NavigationApi"):
    engine.calculate_route((24., 59.))
  assert engine.route is None
  assert store.state()["destination"]["requestId"] == "second"
  assert "revision" not in store.state()


def test_parked_preview_uses_device_saved_origin_without_live_guidance(tmp_path):
  params = Params(str(tmp_path))
  place = {"latitude": 59., "longitude": 24.001, "requestId": "first"}
  params.put_bool("NavigationEnabled", True, block=True)
  params.put("NavDestination", place, block=True)
  store = RouteStore(params)
  store.save_position([24., 59.])
  publisher = MagicMock()
  engine = RouteEngine(MagicMock(), publisher, params, store=store)
  result = MagicMock(status_code=200)
  result.json.return_value = {"routes": [{"distance": 100, "duration": 15, "geometry": {"coordinates": [[24., 59.], [24.001, 59.]]},
    "legs": [{"steps": [{"distance": 100, "duration": 15, "geometry": {"coordinates": [[24., 59.], [24.001, 59.]]}, "bannerInstructions": []}]}]}]}
  with patch("requests.get", return_value=result), patch("openpilot.system.asius.navigation_proxy.NavigationApi"):
    engine.recompute_route(gps_ok=False)
  assert store.state()["status"] == "ready"
  assert store.state()["lastKnown"] is True
  engine.send_instruction()
  assert publisher.send.call_args.args[1].valid is False


@pytest.fixture
def saved_origin_route(tmp_path, monkeypatch):
  params = Params(str(tmp_path))
  place = {"latitude": 0., "longitude": .002, "requestId": "first"}
  params.put_bool("NavigationEnabled", True, block=True)
  params.put("NavDestination", place, block=True)
  store = RouteStore(params)
  store.save_position([0., 0.])
  sm, pm = MagicMock(), MagicMock()
  gps = SimpleNamespace(hasFix=False, longitude=0., latitude=0., speed=5., bearingDeg=90., bearingAccuracyDeg=5.)
  device = SimpleNamespace(started=False)
  sm.valid = dict.fromkeys(('gpsLocation', 'gpsLocationExternal', 'deviceState'), True)
  sm.alive = sm.valid.copy()
  sm.recv_time = dict.fromkeys(sm.valid, time.monotonic())
  sm.logMonoTime = dict.fromkeys(sm.valid, time.monotonic_ns())
  sm.frame = 1
  sm.__getitem__.side_effect = lambda service: device if service == 'deviceState' else gps
  route = {"distance": 222.39, "duration": 20, "geometry": {"coordinates": [[0., 0.], [.001, 0.], [.002, 0.]]},
           "legs": [{"steps": [
             {"distance": 111.195, "duration": 10, "geometry": {"coordinates": [[0., 0.], [.001, 0.]]},
              "bannerInstructions": [{"distanceAlongGeometry": 120, "primary": {"text": "Main Street", "type": "turn", "modifier": "right"}}]},
             {"distance": 111.195, "duration": 10, "geometry": {"coordinates": [[.001, 0.], [.002, 0.]]}, "bannerInstructions": []},
           ]}]}
  provider = MagicMock(return_value={"routes": [route]})
  monkeypatch.delenv('MAPBOX_TOKEN', raising=False)
  monkeypatch.setattr(navd, 'request_json', provider)
  monkeypatch.setattr(navd.requests, 'get', MagicMock(side_effect=AssertionError('External requests are forbidden')))
  engine = RouteEngine(sm, pm, params, store)
  engine.update()
  assert store.state()['lastKnown'] is True
  assert store.state()['status'] == 'ready'
  return SimpleNamespace(engine=engine, params=params, store=store, gps=gps, device=device, sm=sm, pm=pm,
                         route=route, provider=provider, place=place)


def latest_instruction(fixture):
  return next(call.args[1] for call in reversed(fixture.pm.send.call_args_list) if call.args[0] == 'navInstruction')


@pytest.mark.parametrize('started', [False, True])
def test_fresh_gps_revalidates_saved_origin_without_replacing_selected_geometry(saved_origin_route, started):
  f = saved_origin_route
  before = f.store.state()
  f.gps.hasFix, f.gps.longitude, f.device.started = True, .0005, started
  f.engine.update_position()
  f.engine.send_instruction()
  assert not latest_instruction(f).valid  # Fresh GPS alone cannot bypass route revalidation.
  f.engine.update()
  state = f.store.state()
  assert state['status'] == 'ready' and state['lastKnown'] is False
  assert state['destination'] == f.place
  assert state['origin'] == {'longitude': .0005, 'latitude': 0.}
  assert state['revision'] == before['revision']
  assert state['publishedAt'] > before['publishedAt']
  f.provider.assert_called_once()
  assert latest_instruction(f).valid is started
  f.device.started = True
  f.engine.update()
  assert latest_instruction(f).valid
  assert latest_instruction(f).logMonoTime / 1e9 >= state['publishedAt']
  assert 50 < latest_instruction(f).navInstruction.maneuverDistance < 60
  assert f.store.state()['publishedAt'] == state['publishedAt']  # Do not invalidate consumers on each GPS sample.
  f.provider.assert_called_once()


@pytest.mark.parametrize('wrong_way', [False, True])
def test_fresh_gps_off_route_or_wrong_way_reroutes_from_current_origin(saved_origin_route, wrong_way):
  f = saved_origin_route
  f.gps.hasFix, f.gps.longitude = True, .0005
  f.gps.latitude = 0. if wrong_way else .001
  f.gps.bearingDeg = 270. if wrong_way else 90.
  route = deepcopy(f.route)
  route['geometry']['coordinates'][0] = [f.gps.longitude, f.gps.latitude]
  route['legs'][0]['steps'][0]['geometry']['coordinates'][0] = [f.gps.longitude, f.gps.latitude]

  def response(path, query, **kwargs):
    assert f'/0.0005,{f.gps.latitude};0.002,0.0' in path
    assert kwargs['destination'] == f.place
    assert f.store.state()['status'] == 'loading'
    assert not latest_instruction(f).valid
    return {'routes': [route]}

  f.provider.side_effect = response
  f.engine.update()  # Revalidate even while parked and before the normal reroute hold expires.
  assert f.provider.call_count == 2
  assert f.store.state()['status'] == 'ready'
  assert f.store.state()['lastKnown'] is False
  assert f.store.state()['origin'] == {'longitude': f.gps.longitude, 'latitude': f.gps.latitude}
  assert not latest_instruction(f).valid  # Parked remains a preview.


@pytest.mark.parametrize('outcome', ['cancel', 'replace', 'error'])
def test_fresh_gps_reroute_cannot_restore_cancelled_replaced_or_failed_route(saved_origin_route, outcome):
  f = saved_origin_route
  f.gps.hasFix, f.gps.latitude, f.device.started = True, .001, True

  def response(*args, **kwargs):
    if outcome == 'cancel':
      f.params.remove('NavDestination')
    elif outcome == 'replace':
      f.params.put('NavDestination', {**f.place, 'requestId': 'replacement'}, block=True)
    else:
      raise ValueError('Mock provider failure')
    return {'routes': [f.route]}

  f.provider.side_effect = response
  f.engine.update()
  assert not latest_instruction(f).valid
  assert f.engine.route is None and f.engine.route_data is None
  state = f.store.state()
  assert 'revision' not in state
  assert state['status'] == {'cancel': 'idle', 'replace': 'loading', 'error': 'error'}[outcome]
  if outcome == 'replace':
    assert state['destination']['requestId'] == 'replacement'
  if outcome == 'error':
    f.engine.update()  # Another fresh fix cannot republish the saved route over the error.
    assert f.store.state()['status'] == 'error'
    assert not latest_instruction(f).valid
    assert f.provider.call_count == 2


def test_stale_gps_does_not_revalidate_saved_origin(saved_origin_route):
  f = saved_origin_route
  f.gps.hasFix, f.device.started = True, True
  f.sm.recv_time = dict.fromkeys(f.sm.valid, time.monotonic() - 3)
  f.engine.update()
  assert f.store.state()['lastKnown'] is True
  assert not latest_instruction(f).valid
  f.provider.assert_called_once()
