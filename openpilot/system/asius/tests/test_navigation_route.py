import base64
import gzip
import hashlib
import json
import random
from unittest.mock import MagicMock, patch

import pytest

from openpilot.common.params import Params
from openpilot.selfdrive.navigation.navd import RouteEngine
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
  engine.api = MagicMock()
  def response(*args, **kwargs):
    params.put("NavDestination", {**place, "longitude": 25., "requestId": "second"}, block=True)
    result = MagicMock(status_code=200)
    result.json.return_value = {"routes": [{"legs": [{"steps": []}]}]}
    return result
  with patch("requests.get", side_effect=response):
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
  engine.api = MagicMock()
  result = MagicMock(status_code=200)
  result.json.return_value = {"routes": [{"distance": 100, "duration": 15, "geometry": {"coordinates": [[24., 59.], [24.001, 59.]]},
    "legs": [{"steps": [{"distance": 100, "duration": 15, "geometry": {"coordinates": [[24., 59.], [24.001, 59.]]}, "bannerInstructions": []}]}]}]}
  with patch("requests.get", return_value=result):
    engine.recompute_route(gps_ok=False)
  assert store.state()["status"] == "ready"
  assert store.state()["lastKnown"] is True
  engine.send_instruction()
  assert publisher.send.call_args.args[1].valid is False
