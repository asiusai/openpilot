"""One device-owned route, transferred in bounded authenticated RPC replies."""
import base64
import gzip
import hashlib
import json
import math
import os
import tempfile
import time
from pathlib import Path

from openpilot.common.params import Params

CHUNK_BYTES = 3072
MAX_ROUTE_BYTES = 8 * 1024 * 1024


def atomic_write(path, data):
  path.parent.mkdir(parents=True, exist_ok=True)
  with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as f:
    try:
      f.write(data)
      f.close()
      os.replace(f.name, path)
    finally:
      if os.path.exists(f.name):
        os.unlink(f.name)


class RouteStore:
  def __init__(self, params=None):
    self.params = params if params is not None else Params()
    self.root = Path(self.params.get_param_path()).parent / "navigation"

  def read(self):
    try:
      state = json.loads((self.root / "state.json").read_bytes())
      return state if isinstance(state, dict) else {}
    except (OSError, ValueError):
      return {}

  def state(self):
    destination = self.params.get("NavDestination")
    if not destination:
      return {"status": "idle"}
    state = self.read()
    if state.get("destination") != destination:
      return {"status": "loading", "destination": destination}
    return state

  def publish(self, destination, *, route=None, status="ready", origin=None, last_known=False, error=None):
    # A slow HTTP reply must never replace a newer destination or cancellation.
    if self.params.get("NavDestination") != destination:
      return False
    state = {"destination": destination, "status": status, "origin": origin, "lastKnown": last_known, "publishedAt": time.monotonic()}
    if error:
      state["error"] = error
    old = self.read()
    if route is None and {k: v for k, v in old.items() if k != "publishedAt"} == {k: v for k, v in state.items() if k != "publishedAt"}:
      return True
    if route is not None:
      data = json.dumps(route, separators=(",", ":"), allow_nan=False).encode()
      if len(data) > MAX_ROUTE_BYTES:
        raise ValueError("Route is too large")
      packed = gzip.compress(data, mtime=0)
      if len(packed) > MAX_ROUTE_BYTES:
        raise ValueError("Route is too large")
      revision = hashlib.sha256(packed).hexdigest()
      atomic_write(self.root / f"{revision}.gz", packed)
      state.update(revision=revision, bytes=len(packed))
    atomic_write(self.root / "state.json", json.dumps(state, separators=(",", ":")).encode())
    # Readers request a revision on every chunk and retry after a route change.
    for path in self.root.glob("*.gz"):
      if path.stem != state.get("revision"):
        path.unlink(missing_ok=True)
    return True

  def chunk(self, revision, offset=0):
    if not isinstance(revision, str) or len(revision) != 64 or any(c not in "0123456789abcdef" for c in revision):
      raise ValueError("Invalid route revision")
    if type(offset) is not int or offset < 0 or offset % CHUNK_BYTES:
      raise ValueError("Invalid route offset")
    state = self.state()
    if state.get("revision") != revision:
      raise ValueError("Route changed")
    data = (self.root / f"{revision}.gz").read_bytes()
    if offset >= len(data):
      raise ValueError("Invalid route offset")
    part = data[offset:offset + CHUNK_BYTES]
    return {"revision": revision, "offset": offset, "total": len(data), "chunk": base64.b64encode(part).decode(),
            "nextOffset": offset + len(part), "done": offset + len(part) == len(data)}

  def save_position(self, position):
    atomic_write(self.root / "position.json", json.dumps(position).encode())

  def position(self):
    try:
      position = json.loads((self.root / "position.json").read_bytes())
      if isinstance(position, list) and len(position) == 2:
        if all(type(value) in (int, float) and math.isfinite(value) for value in position):
          if abs(position[0]) <= 180 and abs(position[1]) <= 90:
            return position
    except (OSError, ValueError):
      pass
    return None


def getNavigationRoute(revision, offset=0):
  return RouteStore().chunk(revision, offset)


def getNavigationState():
  return RouteStore().state()
