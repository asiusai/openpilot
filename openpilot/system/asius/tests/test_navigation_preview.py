import base64
import gzip
import hashlib
import json
from unittest.mock import MagicMock, patch

import pytest

from openpilot.common.params import Params
from openpilot.selfdrive.navigation.navd import RouteEngine
from openpilot.system.asius import navigation, navigation_preview
from openpilot.system.asius.navigation_preview import PreviewStore, previewNavigation
from openpilot.system.asius.navigation_route import RouteStore

PLACE = {"latitude": 59.001, "longitude": 24.002, "name": "Home", "address": "Tallinn"}


def route(index):
  coordinates = [[24., 59.], [24.001, 59. + index * .001], [24.002, 59.001]]
  return {"distance": 200 + index * 10, "duration": 30 + index * 10, "geometry": {"coordinates": coordinates},
          "legs": [{"summary": f"Road {index}", "steps": [{"distance": 200, "duration": 30,
            "geometry": {"coordinates": coordinates}, "bannerInstructions": [], "maneuver": {"instruction": "Head east", "type": "depart"}}]}]}


@pytest.fixture
def params(tmp_path, monkeypatch):
  params = Params(str(tmp_path))
  monkeypatch.setattr(navigation, "Params", lambda: params)
  monkeypatch.setattr(navigation_preview, "Params", lambda: params)
  RouteStore(params).save_position([24., 59.])
  return params


def test_preview_leaves_active_route_and_history_untouched(params):
  navigation.setNavDestination(58, 25, "Existing", "")
  store = RouteStore(params)
  store.publish(params.get("NavDestination"), route=route(0))
  before = [params.get("NavDestination"), params.get("NavRecentDestinations"), store.state()]
  with patch.object(navigation, "navigation_request", return_value={"routes": [route(0), route(1)]}) as request:
    result = previewNavigation(**PLACE)
  assert request.call_args.args[1]["alternatives"] == "true"
  assert "24.0,59.0;24.002,59.001" in request.call_args.args[0]
  assert before == [params.get("NavDestination"), params.get("NavRecentDestinations"), store.state()]
  data = b""
  while True:
    part = navigation_preview.getNavigationPreview(result["revision"], len(data))
    data += base64.b64decode(part["chunk"])
    if part["done"]:
      break
  assert hashlib.sha256(data).hexdigest() == result["revision"]
  assert json.loads(gzip.decompress(data))["routes"] == [route(0), route(1)]


def test_go_uses_exact_alternative_without_another_provider_request(params):
  store = PreviewStore(params)
  preview = store.save(PLACE, [24., 59.], [route(0), route(1)])
  assert navigation.setNavDestination(**PLACE, previewRevision=preview["revision"], routeIndex=1) == {"success": 1}
  active_revision = RouteStore(params).state()["revision"]
  engine = RouteEngine(MagicMock(), MagicMock(), params)
  engine.api = MagicMock()
  with patch("openpilot.selfdrive.navigation.navd.request_json", side_effect=AssertionError("Must reuse selected route")):
    engine.recompute_route(gps_ok=False)
  assert engine.route == route(1)["legs"][0]["steps"]
  assert RouteStore(params).state()["revision"] == active_revision
  assert params.get("NavRecentDestinations") == [PLACE]
  # A later deviation calculates from current GPS, rather than replaying the choice.
  with patch("openpilot.selfdrive.navigation.navd.request_json", return_value={"routes": [route(2)]}) as request:
    engine.calculate_route((PLACE["longitude"], PLACE["latitude"]))
  request.assert_called_once()
  assert engine.route == route(2)["legs"][0]["steps"]


def test_movement_along_preview_starts_at_nearest_step(params):
  planned = route(0)
  paths = [[[24., 59.], [24.002, 59.]], [[24.002, 59.], [24.01, 59.]]]
  planned["geometry"]["coordinates"] = [paths[0][0], *paths[1]]
  planned["legs"][0]["steps"] = [{**planned["legs"][0]["steps"][0], "geometry": {"coordinates": path}} for path in paths]
  preview = PreviewStore(params).save(PLACE, [24., 59.], [planned])
  RouteStore(params).save_position([24.005, 59.])
  assert navigation.setNavDestination(**PLACE, previewRevision=preview["revision"], routeIndex=0)["success"] == 1
  engine = RouteEngine(MagicMock(), MagicMock(), params)
  with patch("openpilot.selfdrive.navigation.navd.request_json", side_effect=AssertionError("Must reuse selected route")):
    engine.recompute_route(gps_ok=True)
  assert engine.step_idx == 1


@pytest.mark.parametrize("index", [-1, 2, 3, True, 1.2, None])
def test_invalid_selection_does_not_start_navigation(params, index):
  preview = PreviewStore(params).save(PLACE, [24., 59.], [route(0), route(1)])
  assert navigation.setNavDestination(**PLACE, previewRevision=preview["revision"], routeIndex=index)["success"] == 0
  assert params.get("NavDestination") is None
  assert not params.get("NavRecentDestinations")


def test_expiry_reboot_movement_and_destination_mismatch_require_new_preview(params):
  store = PreviewStore(params)
  preview = store.save(PLACE, [24., 59.], [route(0)])
  revision = preview["revision"]
  with patch.object(navigation_preview.time, "monotonic", return_value=store.read(revision)["created"] + 301):
    assert navigation.setNavDestination(**PLACE, previewRevision=revision, routeIndex=0)["success"] == 0
    with pytest.raises(ValueError, match="expired"):
      store.chunk(revision, 0)
  store.boot = "another boot"
  with pytest.raises(ValueError, match="expired"):
    store.chunk(revision, 0)
  assert navigation.setNavDestination(**{**PLACE, "longitude": 25}, previewRevision=revision, routeIndex=0)["success"] == 0
  RouteStore(params).save_position([25., 59.])
  assert navigation.setNavDestination(**PLACE, previewRevision=revision, routeIndex=0)["success"] == 0
  assert params.get("NavDestination") is None


def test_preview_requires_origin_and_bounds_retained_data(params):
  store = PreviewStore(params)
  revisions = [store.save(PLACE, [24., 59.], [route(i)])["revision"] for i in range(6)]
  assert len(list(store.root.glob("*.gz"))) == 4
  with pytest.raises(ValueError, match="expired"):
    store.chunk(revisions[0], 0)
  for revision, offset in [("../bad", 0), (revisions[-1], -1), (revisions[-1], 1)]:
    with pytest.raises(ValueError):
      store.chunk(revision, offset)
  (store.routes.root / "position.json").unlink()
  with patch.object(navigation, "navigation_request", side_effect=AssertionError("No origin")):
    with pytest.raises(ValueError, match="GPS"):
      previewNavigation(**PLACE)
