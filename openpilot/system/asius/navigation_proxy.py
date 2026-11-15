"""Device-initiated navigation HTTP fallback through an authorized app.

Only search and directions use this queue. Map tiles keep their direct HTTP path.
The RAM-backed database joins navd, relayd and bluetoothd without a new daemon.
"""
import base64
import hashlib
import json
import re
import secrets
import sqlite3
import time
import zlib
from contextlib import contextmanager
from pathlib import Path
from urllib.parse import urlsplit

import requests

from openpilot.common.hardware.hw import Paths
from openpilot.common.params import Params
from openpilot.system.asius.mapbox import NAVIGATION_API, NavigationApi
from openpilot.system.asius.navigation_route import CHUNK_BYTES, MAX_ROUTE_BYTES

BRIDGE_METHODS = {"claimNavigationRequest", "writeNavigationResponse", "releaseNavigationRequest"}
SEARCH_METHODS = {"searchNavigation", "retrieveNavigationLocation", "previewNavigation"}
REQUEST_SECONDS = 90
LEASE_SECONDS = 25
MAX_PENDING = 4
REQUEST_PATH = re.compile(r"/(?:search/searchbox/v1/(?:suggest|retrieve/[a-zA-Z0-9._%=-]{1,512})|directions/v5/mapbox/driving-traffic/[-0-9.,;]+)")


class NavigationRequests:
  def __init__(self, params=None, root=None):
    self.params = params if params is not None else Params()
    namespace = hashlib.sha256(self.params.get_param_path().encode()).hexdigest()[:16]
    self.path = Path(root or Paths.shm_path()) / f"asius-navigation-{namespace}.sqlite3"

  @contextmanager
  def database(self):
    self.path.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(self.path, timeout=2)
    db.row_factory = sqlite3.Row
    try:
      db.execute("CREATE TABLE IF NOT EXISTS requests (id TEXT PRIMARY KEY, path TEXT, destination TEXT, expires REAL, " +
                 "owner TEXT, lease TEXT, lease_until REAL, status INTEGER, total INTEGER, digest TEXT, data BLOB, done INTEGER)")
      db.execute("BEGIN IMMEDIATE")
      db.execute("DELETE FROM requests WHERE expires <= ?", (time.monotonic(),))
      yield db
      db.commit()
    except Exception:
      db.rollback()
      raise
    finally:
      db.close()

  def current(self, row):
    return row is not None and (row["destination"] is None or json.loads(row["destination"]) == self.params.get("NavDestination"))

  def pending(self):
    if not self.path.exists():
      return []
    with self.database() as db:
      return [row["id"] for row in db.execute("SELECT * FROM requests WHERE done = 0 AND lease_until < ?", (time.monotonic(),))
              if self.current(row)]

  def request(self, path, destination=None, timeout=REQUEST_SECONDS):
    request_id = secrets.token_hex(16)
    with self.database() as db:
      if db.execute("SELECT COUNT(*) FROM requests").fetchone()[0] >= MAX_PENDING:
        raise ValueError("Too many navigation requests. Try again.")
      db.execute("INSERT INTO requests VALUES (?, ?, ?, ?, NULL, NULL, 0, 0, 0, '', ?, 0)",
                 (request_id, path, json.dumps(destination) if destination is not None else None,
                  time.monotonic() + timeout, b""))
    try:
      while True:
        with self.database() as db:
          row = db.execute("SELECT * FROM requests WHERE id = ?", (request_id,)).fetchone()
        if row is None:
          raise ValueError("Cannot reach navigation. Connect an app with internet access.")
        if not self.current(row):
          raise ValueError("Navigation destination changed")
        if row["done"]:
          if row["status"] != 200:
            raise ValueError("Navigation is busy. Try again shortly." if row["status"] == 429 else "Navigation service unavailable")
          decoder = zlib.decompressobj(16 + zlib.MAX_WBITS)
          try:
            data = decoder.decompress(row["data"], MAX_ROUTE_BYTES + 1)
          except zlib.error:
            raise ValueError("Invalid compressed navigation response") from None
          if len(data) > MAX_ROUTE_BYTES or not decoder.eof or decoder.unused_data:
            raise ValueError("Invalid navigation response")
          result = json.loads(data)
          if not isinstance(result, dict):
            raise ValueError("Invalid navigation response")
          return result
        time.sleep(0.05)
    finally:
      with self.database() as db:
        db.execute("DELETE FROM requests WHERE id = ?", (request_id,))

  def claim(self, peer, requestId):
    with self.database() as db:
      row = db.execute("SELECT * FROM requests WHERE id = ?", (requestId,)).fetchone()
      if not self.current(row) or row["done"] or row["lease_until"] >= time.monotonic():
        return {"accepted": False}
      lease = secrets.token_hex(16)
      db.execute("UPDATE requests SET owner = ?, lease = ?, lease_until = ?, status = 0, total = 0, digest = '', data = ? WHERE id = ?",
                 (peer, lease, time.monotonic() + LEASE_SECONDS, b"", requestId))
      # Sign at claim time so direct HTTP timeouts cannot expire the app's proof.
      return {"accepted": True, "lease": lease, "path": row["path"],
              "authorization": f"Bearer {NavigationApi().get_token(row['path'])}"}

  def write(self, peer, requestId, lease, status, total, digest, offset, chunk):
    if (type(status) is not int or status not in range(200, 600) or type(total) is not int or not 1 <= total <= MAX_ROUTE_BYTES or
        type(offset) is not int or offset < 0 or offset % CHUNK_BYTES or not isinstance(chunk, str) or len(chunk) > 4096 or
        not isinstance(digest, str) or re.fullmatch(r"[a-f0-9]{64}", digest) is None):
      raise ValueError("Invalid navigation response chunk")
    part = base64.b64decode(chunk, validate=True)
    if not 1 <= len(part) <= CHUNK_BYTES or offset + len(part) > total or (offset + len(part) < total and len(part) != CHUNK_BYTES):
      raise ValueError("Invalid navigation response size")
    with self.database() as db:
      row = db.execute("SELECT * FROM requests WHERE id = ?", (requestId,)).fetchone()
      if (not self.current(row) or row["done"] or row["owner"] != peer or row["lease"] != lease or row["lease_until"] < time.monotonic()):
        raise ValueError("Navigation request expired")
      if offset != len(row["data"]) or (offset and (row["status"], row["total"], row["digest"]) != (status, total, digest)):
        raise ValueError("Navigation response changed")
      data = row["data"] + part
      done = len(data) == total
      if done and hashlib.sha256(data).hexdigest() != digest:
        raise ValueError("Navigation response integrity check failed")
      db.execute("UPDATE requests SET status = ?, total = ?, digest = ?, data = ?, done = ?, lease_until = ? WHERE id = ?",
                 (status, total, digest, data, done, time.monotonic() + LEASE_SECONDS, requestId))
      return {"nextOffset": len(data), "done": done}

  def release(self, peer, requestId, lease):
    with self.database() as db:
      db.execute("UPDATE requests SET owner = NULL, lease = NULL, lease_until = 0 WHERE id = ? AND owner = ? AND lease = ? AND done = 0",
                 (requestId, peer, lease))
    return {"released": True}

  def methods(self, peer):
    return {"claimNavigationRequest": lambda **kwargs: self.claim(peer, **kwargs),
            "writeNavigationResponse": lambda **kwargs: self.write(peer, **kwargs),
            "releaseNavigationRequest": lambda **kwargs: self.release(peer, **kwargs)}


def request_json(path, query, *, params=None, destination=None, max_bytes=MAX_ROUTE_BYTES):
  if REQUEST_PATH.fullmatch(path) is None:
    raise ValueError("Unsupported navigation request")
  url = requests.Request('GET', NAVIGATION_API + path, params=query).prepare().url
  parsed = urlsplit(url)
  forwarded = parsed.path + ('?' + parsed.query if parsed.query else '')
  if len(forwarded) > 4096:
    raise ValueError("Navigation request is too large")
  try:
    response = requests.get(url, headers={"Authorization": f"Bearer {NavigationApi().get_token(forwarded)}"}, timeout=(2, 8))
    response.raise_for_status()
    if len(response.content) > max_bytes:
      raise ValueError("Navigation response is too large") from None
    return response.json()
  except (requests.ConnectionError, requests.Timeout):
    # An HTTP error proves the server is reachable. Do not retry provider errors through every app.
    result = NavigationRequests(params).request(forwarded, destination, timeout=90 if path.startswith('/directions/') else 30)
    if len(json.dumps(result).encode()) > max_bytes:
      raise ValueError("Navigation response is too large") from None
    return result
