"""Device-calculated route choices. Browsing never replaces active guidance."""
import fcntl
import gzip
import hashlib
import json
import math
import time
from contextlib import contextmanager
from pathlib import Path

from openpilot.common.params import Params
from openpilot.system.asius.navigation_route import MAX_ROUTE_BYTES, RouteStore, atomic_write, route_chunk, validate_chunk_request

PREVIEW_SECONDS = 300
MAX_PREVIEWS = 4


class PreviewStore:
  def __init__(self, params):
    self.routes = RouteStore(params)
    self.root = self.routes.root / "previews"
    self.boot = Path("/proc/sys/kernel/random/boot_id").read_text().strip()

  @contextmanager
  def lock(self):
    self.root.mkdir(parents=True, exist_ok=True)
    with (self.root / ".lock").open("a") as lock:
      fcntl.flock(lock, fcntl.LOCK_EX)
      yield

  def read(self, revision):
    validate_chunk_request(revision, 0)
    try:
      meta = json.loads((self.root / f"{revision}.json").read_bytes())
      if meta["boot"] == self.boot and 0 <= time.monotonic() - meta["created"] < PREVIEW_SECONDS:
        return meta
    except (OSError, ValueError, KeyError):
      pass
    raise ValueError("Route preview expired. Refresh the routes.")

  def save(self, place, origin, routes):
    if not routes:
      raise ValueError("No driving route found for this destination.")
    # The provider can return fewer alternatives. Never fabricate another route.
    routes = routes[:3]
    for route in routes:
      if not route.get("legs") or not route["legs"][0].get("steps") or len(route["geometry"]["coordinates"]) < 2:
        raise ValueError("Invalid route response")
    data = json.dumps({"routes": routes}, separators=(",", ":"), allow_nan=False).encode()
    packed = gzip.compress(data, mtime=0)
    if max(len(data), len(packed)) > MAX_ROUTE_BYTES:
      raise ValueError("Route preview is too large")
    revision = hashlib.sha256(packed).hexdigest()
    meta = {"destination": place, "origin": origin, "created": time.monotonic(), "boot": self.boot}
    with self.lock():
      atomic_write(self.root / f"{revision}.gz", packed)
      atomic_write(self.root / f"{revision}.json", json.dumps(meta).encode())
      others = sorted((p for p in self.root.glob("*.gz") if p.stem != revision), key=lambda p: p.stat().st_mtime, reverse=True)
      for path in others[MAX_PREVIEWS - 1:]:
        path.unlink(missing_ok=True)
        path.with_suffix(".json").unlink(missing_ok=True)
    return {"revision": revision, "origin": {"longitude": origin[0], "latitude": origin[1]}, "lastKnown": True}

  def chunk(self, revision, offset):
    validate_chunk_request(revision, offset)
    with self.lock():
      self.read(revision)
      return route_chunk(revision, offset, (self.root / f"{revision}.gz").read_bytes())

  def select(self, revision, index, request):
    import numpy as np
    from openpilot.selfdrive.navigation.route import along_path
    if type(index) is not int or not 0 <= index < 3:
      raise ValueError("Invalid route selection")
    with self.lock():
      meta = self.read(revision)
      if any(request.get(key) != value for key, value in meta["destination"].items()):
        raise ValueError("Route preview destination changed")
      current = self.routes.position()
      origin = meta["origin"]
      routes = json.loads(gzip.decompress((self.root / f"{revision}.gz").read_bytes()))["routes"]
      if index >= len(routes):
        raise ValueError("Invalid route selection")
      route = routes[index]
      moved = current is None or math.hypot((current[0] - origin[0]) * math.cos(math.radians(origin[1])), current[1] - origin[1]) * 111320 > 100
      if current is None or (moved and along_path(np.asarray(route["geometry"]["coordinates"]), current)[1] > 25):
        raise ValueError("Device position changed. Refresh the routes.")
      atomic_write(self.routes.root / "selected.json", json.dumps({"request": request, "route": route}).encode())
      return route, {"longitude": origin[0], "latitude": origin[1]}

  def take_selected(self, request):
    # The initial route must use the exact choice. Later reroutes use current GPS.
    with self.lock():
      path = self.routes.root / "selected.json"
      try:
        selected = json.loads(path.read_bytes())
      except (OSError, ValueError):
        return None
      if selected.get("request") != request:
        return None
      path.unlink(missing_ok=True)
      return selected["route"]


def previewNavigation(latitude, longitude, name, address):
  from openpilot.system.asius.navigation import destination, navigation_request
  place = destination(latitude, longitude, name, address)
  store = PreviewStore(Params())
  origin = store.routes.position()
  if origin is None:
    raise ValueError("Waiting for device GPS. Try again when a position is available.")
  language = (store.routes.params.get("LanguageSetting") or "en").replace("main_", "")
  coordinates = f'{origin[0]},{origin[1]};{place["longitude"]},{place["latitude"]}'
  result = navigation_request(f"directions/v5/mapbox/driving-traffic/{coordinates}", {
    "geometries": "geojson", "overview": "full", "steps": "true", "banner_instructions": "true", "alternatives": "true", "language": language,
  }, max_bytes=MAX_ROUTE_BYTES)
  return store.save(place, origin, result.get("routes", []))


def getNavigationPreview(revision, offset=0):
  return PreviewStore(Params()).chunk(revision, offset)
