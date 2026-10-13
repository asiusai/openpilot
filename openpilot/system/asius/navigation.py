"""Device-owned navigation destinations, shared by authorized app transports."""

import fcntl
import math
from contextlib import contextmanager
from pathlib import Path

from openpilot.common.params import Params

RECENT_DESTINATIONS = "NavRecentDestinations"
MAX_RECENT_DESTINATIONS = 20


def destination(latitude, longitude, name, address):
  if any(type(value) not in (int, float) or not math.isfinite(value) for value in (latitude, longitude)):
    raise ValueError("Invalid destination coordinates")
  if not (-90 <= latitude <= 90 and -180 <= longitude <= 180):
    raise ValueError("Invalid destination coordinates")
  if not isinstance(name, str) or not 1 <= len(name.strip()) <= 256:
    raise ValueError("Invalid destination name")
  if not isinstance(address, str) or len(address.strip()) > 1024:
    raise ValueError("Invalid destination address")
  return {"latitude": latitude, "longitude": longitude, "name": name.strip(), "address": address.strip()}


def read_recent(params):
  try:
    saved = params.get(RECENT_DESTINATIONS)
  except (ValueError, UnicodeError):
    return []
  recent = []
  for place in saved if isinstance(saved, list) else []:
    try:
      place = destination(**place)
      if not any((item["latitude"], item["longitude"]) == (place["latitude"], place["longitude"]) for item in recent):
        recent.append(place)
    except (TypeError, ValueError):
      continue
    if len(recent) == MAX_RECENT_DESTINATIONS:
      break
  return recent


@contextmanager
def history_lock(params):
  # Relay and Bluetooth run in separate processes. Keep their updates serialized.
  with (Path(params.get_param_path()).parent / ".nav-history.lock").open("a") as lock:
    fcntl.flock(lock, fcntl.LOCK_EX)
    try:
      yield
    finally:
      fcntl.flock(lock, fcntl.LOCK_UN)


def getRecentNavDestinations():
  return read_recent(Params())


def setNavDestination(latitude, longitude, name, address):
  try:
    place = destination(latitude, longitude, name, address)
  except ValueError as error:
    return {"success": 0, "error": str(error)}
  params = Params()
  with history_lock(params):
    recent = [item for item in read_recent(params) if (item["latitude"], item["longitude"]) != (latitude, longitude)]
    params.put(RECENT_DESTINATIONS, [place, *recent][:MAX_RECENT_DESTINATIONS], block=True)
    params.put("NavDestination", place, block=True)
  return {"success": 1}
