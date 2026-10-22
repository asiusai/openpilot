"""Device-owned navigation destinations, shared by authorized app transports."""

import fcntl
import math
import secrets
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


def setNavDestination(latitude, longitude, name, address, previewRevision=None, routeIndex=None):
  try:
    place = destination(latitude, longitude, name, address)
  except ValueError as error:
    return {"success": 0, "error": str(error)}
  params = Params()
  with history_lock(params):
    request = {**place, "requestId": secrets.token_hex(16)}
    selected = None
    if previewRevision is not None or routeIndex is not None:
      from openpilot.system.asius.navigation_preview import PreviewStore
      try:
        selected = PreviewStore(params).select(previewRevision, routeIndex, request)
      except ValueError as error:
        return {"success": 0, "error": str(error)}
    recent = [item for item in read_recent(params) if (item["latitude"], item["longitude"]) != (latitude, longitude)]
    params.put(RECENT_DESTINATIONS, [place, *recent][:MAX_RECENT_DESTINATIONS], block=True)
    params.put_bool("NavigationEnabled", True, block=True)
    params.put("NavDestination", request, block=True)
    if selected is not None:
      from openpilot.system.asius.navigation_route import RouteStore
      route, origin = selected
      RouteStore(params).publish(request, route=route, origin=origin, last_known=True)
  return {"success": 1}


def navigation_request(path, params, max_bytes=256 * 1024):
  import requests
  from openpilot.system.asius.mapbox import NavigationApi
  from openpilot.system.asius.navigation_proxy import request_json
  try:
    return request_json('/' + path, params, {"Authorization": f"Bearer {NavigationApi().get_token()}"}, max_bytes=max_bytes)
  except requests.RequestException:
    raise ValueError("Navigation service unavailable. Try again shortly.") from None


def searchNavigation(query, session, latitude=None, longitude=None, language="en"):
  if not isinstance(query, str) or not 3 <= len(query.strip()) <= 256:
    raise ValueError("Invalid search")
  if not isinstance(session, str) or not 1 <= len(session) <= 128:
    raise ValueError("Invalid search session")
  if not isinstance(language, str) or not 2 <= len(language) <= 8 or not language.replace("-", "").isalpha():
    raise ValueError("Invalid language")
  params = {"q": query, "session_token": session, "limit": "6", "language": language}
  if latitude is not None or longitude is not None:
    point = destination(latitude, longitude, "Search", "")
    params["proximity"] = f'{point["longitude"]},{point["latitude"]}'
  result = navigation_request("search/searchbox/v1/suggest", params)
  return {"suggestions": [{key: item[key] for key in ("mapbox_id", "name", "place_formatted", "full_address") if key in item}
                          for item in result.get("suggestions", [])[:6]]}


def retrieveNavigationLocation(mapboxId, session):
  from urllib.parse import quote
  if not isinstance(mapboxId, str) or not 1 <= len(mapboxId) <= 2048:
    raise ValueError("Invalid location")
  if not isinstance(session, str) or not 1 <= len(session) <= 128:
    raise ValueError("Invalid search session")
  result = navigation_request(f"search/searchbox/v1/retrieve/{quote(mapboxId, safe='')}", {"session_token": session})
  longitude, latitude = result["features"][0]["geometry"]["coordinates"]
  point = destination(latitude, longitude, "Location", "")
  return {"latitude": point["latitude"], "longitude": point["longitude"]}
