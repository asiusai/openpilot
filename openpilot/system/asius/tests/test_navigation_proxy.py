import asyncio
import base64
import gzip
import hashlib
import json
import secrets
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import AsyncMock, MagicMock

import pytest
import requests

from openpilot.common.params import Params
from openpilot.selfdrive.navigation.navd import REROUTE_COUNTER_MIN, RouteEngine
from openpilot.system.asius import navigation_proxy as proxy
from openpilot.system.asius.navigation_route import RouteStore


@pytest.fixture
def store(tmp_path, monkeypatch):
  monkeypatch.setattr(proxy.NavigationApi, "get_token", lambda self, path: "device-proof:" + path)
  monkeypatch.setattr(proxy.Paths, "shm_path", lambda: str(tmp_path))
  return proxy.NavigationRequests(Params(str(tmp_path / "params")))


def pending(store):
  deadline = time.monotonic() + 3
  while time.monotonic() < deadline:
    if jobs := store.pending():
      return jobs[0]
    time.sleep(0.01)
  raise AssertionError("Device did not request app forwarding")


def upload(store, request_id, claim, data, peer="phone", status=200):
  packed = gzip.compress(json.dumps(data).encode())
  return upload_bytes(store, request_id, claim, packed, peer, status)


def upload_bytes(store, request_id, claim, packed, peer="phone", status=200):
  for offset in range(0, len(packed), proxy.CHUNK_BYTES):
    part = packed[offset:offset + proxy.CHUNK_BYTES]
    result = store.write(peer, request_id, claim["lease"], status, len(packed), hashlib.sha256(packed).hexdigest(), offset,
                         base64.b64encode(part).decode())
    assert result == {"nextOffset": offset + len(part), "done": offset + len(part) == len(packed)}


def test_direct_http_and_provider_errors_never_queue(store, monkeypatch):
  response = MagicMock(content=b'{}')
  response.json.return_value = {"routes": []}
  get = MagicMock(return_value=response)
  monkeypatch.setattr(proxy.requests, "get", get)
  assert proxy.request_json("/search/searchbox/v1/suggest", {"q": "Tartu"}, params=store.params) == {"routes": []}
  assert store.pending() == []
  response.raise_for_status.side_effect = requests.HTTPError("429")
  with pytest.raises(requests.HTTPError):
    proxy.request_json("/search/searchbox/v1/suggest", {}, params=store.params)
  assert store.pending() == []
  with pytest.raises(ValueError, match="Unsupported"):
    proxy.request_json("/maps/v4/mapbox.mapbox-streets-v8/1/1/1.vector.pbf", {})


def test_offline_device_uses_app_response_with_request_bound_device_proof(store, monkeypatch):
  monkeypatch.setattr(proxy.requests, "get", MagicMock(side_effect=requests.ConnectionError()))
  data = {"suggestions": [], "padding": secrets.token_hex(12000)}
  with ThreadPoolExecutor() as pool:
    result = pool.submit(proxy.request_json, "/search/searchbox/v1/suggest", {"q": "Tartu & Tallinn", "session_token": "a" * 32},
                         params=store.params)
    request_id = pending(store)
    claim = store.claim("phone", request_id)
    assert "q=Tartu+%26+Tallinn" in claim["path"]
    assert claim["authorization"] == "Bearer device-proof:" + claim["path"]
    assert store.claim("other-phone", request_id) == {"accepted": False}
    assert store.pending() == []
    upload(store, request_id, claim, data)
    assert result.result(timeout=2) == data
    assert store.pending() == []


def test_device_keeps_route_ownership_with_offline_http(store, monkeypatch):
  destination = {"latitude": 59.5, "longitude": 24.8, "requestId": "first"}
  store.params.put("NavDestination", destination, block=True)
  store.params.put_bool("NavigationEnabled", True, block=True)
  monkeypatch.setattr(proxy.requests, "get", MagicMock(side_effect=requests.ConnectionError()))
  monkeypatch.delenv("MAPBOX_TOKEN", raising=False)
  engine = RouteEngine(MagicMock(), MagicMock(), store.params, RouteStore(store.params))
  engine.last_position = (24.7, 59.4)
  route = {"geometry": {"coordinates": [[24.7, 59.4], [24.75, 59.45], [24.8, 59.5]]}, "distance": 100, "duration": 10,
           "legs": [{"steps": [{"geometry": {"coordinates": [[24.7, 59.4], [24.75, 59.45]]}, "distance": 50, "duration": 5},
                               {"geometry": {"coordinates": [[24.75, 59.45], [24.8, 59.5]]}, "distance": 50, "duration": 5}]}]}
  with ThreadPoolExecutor() as pool:
    result = pool.submit(engine.calculate_route, (24.8, 59.5))
    request_id = pending(store)
    claim = store.claim("phone", request_id)
    assert claim["path"].startswith("/v1/nav/directions/v5/mapbox/driving-traffic/24.7,59.4;24.8,59.5?")
    assert "language=" not in claim["path"]  # requests omits unset optional parameters too.
    upload(store, request_id, claim, {"routes": [route]})
    result.result(timeout=2)
  assert engine.route == route["legs"][0]["steps"]
  assert engine.store.state()["status"] == "ready"
  assert engine.store.state()["destination"] == destination
  first_revision = engine.store.state()["revision"]

  # The device detects an off-route position and initiates another forwarded request.
  engine.started = True
  engine.last_position = (24.7, 59.5)
  engine.recompute_route(gps_ok=False)
  assert store.pending() == []
  for _ in range(REROUTE_COUNTER_MIN):
    engine.recompute_route(gps_ok=True)
    assert store.pending() == []
  with ThreadPoolExecutor() as pool:
    result = pool.submit(engine.recompute_route, gps_ok=True)
    request_id = pending(store)
    claim = store.claim("phone", request_id)
    assert "/24.7,59.5;24.8,59.5?" in claim["path"]
    route["geometry"]["coordinates"][0] = [24.7, 59.5]
    route["legs"][0]["steps"][0]["geometry"]["coordinates"][0] = [24.7, 59.5]
    upload(store, request_id, claim, {"routes": [route]})
    result.result(timeout=2)
  assert engine.store.state()["destination"] == destination
  assert engine.store.state()["revision"] != first_revision


def test_expired_claim_and_foreign_peer_cannot_replace_response(store):
  with ThreadPoolExecutor() as pool:
    result = pool.submit(store.request, "/v1/nav/search/searchbox/v1/suggest?q=Tartu", timeout=2)
    request_id = pending(store)
    old = store.claim("phone", request_id)
    with pytest.raises(ValueError, match="expired"):
      upload(store, request_id, old, {}, peer="foreign")
    with store.database() as db:
      db.execute("UPDATE requests SET lease_until = 0 WHERE id = ?", (request_id,))
    new = store.claim("phone", request_id)
    with pytest.raises(ValueError, match="expired"):
      upload(store, request_id, old, {})
    upload(store, request_id, new, {"accepted": 1})
    assert result.result(timeout=2) == {"accepted": 1}


@pytest.mark.parametrize("replacement", [None, {"requestId": "second", "latitude": 59, "longitude": 24}])
def test_cancelled_or_replaced_destination_rejects_late_response(store, replacement):
  destination = {"requestId": "first", "latitude": 59, "longitude": 24}
  store.params.put("NavDestination", destination, block=True)
  with ThreadPoolExecutor() as pool:
    result = pool.submit(store.request, "/v1/nav/directions/v5/mapbox/driving-traffic/24,59;24,59?steps=true", destination)
    request_id = pending(store)
    claim = store.claim("phone", request_id)
    if replacement is None:
      store.params.remove("NavDestination")
    else:
      store.params.put("NavDestination", replacement, block=True)
    with pytest.raises(ValueError, match="expired"):
      upload(store, request_id, claim, {})
    with pytest.raises(ValueError, match="changed"):
      result.result(timeout=2)


@pytest.mark.parametrize("data", [b"not gzip", gzip.compress(b"[1,2]"), gzip.compress(b"x" * (proxy.MAX_ROUTE_BYTES + 1))])
def test_malformed_and_oversized_responses_are_rejected(store, data):
  with ThreadPoolExecutor() as pool:
    result = pool.submit(store.request, "/v1/nav/search/searchbox/v1/suggest?q=Tartu")
    request_id = pending(store)
    upload_bytes(store, request_id, store.claim("phone", request_id), data)
    with pytest.raises(ValueError):
      result.result(timeout=2)


def test_expiry_and_release(store):
  with ThreadPoolExecutor() as pool:
    result = pool.submit(store.request, "/v1/nav/search/searchbox/v1/suggest?q=Tartu", timeout=0.2)
    request_id = pending(store)
    claim = store.claim("phone", request_id)
    store.release("phone", request_id, claim["lease"])
    assert store.claim("second", request_id)["accepted"]
    with pytest.raises(ValueError, match="Connect an app"):
      result.result(timeout=2)
    assert store.claim("phone", request_id) == {"accepted": False}


def test_bridge_replies_do_not_wait_for_busy_search_slots(monkeypatch):
  from openpilot.system.asius import methods
  from openpilot.system.asius.bluetoothd import BlePeerEngine
  monkeypatch.setattr(methods, "dispatcher_for_peer", lambda peer: {"claimNavigationRequest": lambda requestId: {"accepted": False}})
  engine = BlePeerEngine.__new__(BlePeerEngine)
  engine.rpc_active = 4
  engine.send_body = AsyncMock()
  engine.peer_devices = {'phone': '/test'}
  engine.sessions = {'/test': MagicMock(session_id=lambda _peer: 'test-session')}
  asyncio.run(engine.handle_rpc("phone", {"id": 1, "method": "claimNavigationRequest", "params": {"requestId": "a" * 32}}))
  assert engine.send_body.call_args.args[1]["result"] == {"accepted": False}


def test_relay_search_keeps_receiver_free(monkeypatch):
  from openpilot.system.asius import methods
  waiting, complete = threading.Event(), threading.Event()

  def search():
    waiting.set()
    assert complete.wait(2)
    return {}

  monkeypatch.setattr(methods, "dispatcher_for_peer", lambda peer: {"searchNavigation": search, "writeNavigationResponse": lambda: complete.set()})
  room = MagicMock()
  monkeypatch.setattr(methods, "relay_room", room)
  methods.handle_rpc("phone", {"id": 1, "method": "searchNavigation"}, "connection", "session")
  assert waiting.wait(1)
  methods.handle_rpc("phone", {"id": 2, "method": "writeNavigationResponse"}, "connection", "session")
  assert complete.is_set()
