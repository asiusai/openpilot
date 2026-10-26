#!/usr/bin/env python3
import os
import time

import numpy as np
import requests

import openpilot.cereal.messaging as messaging
from openpilot.system.asius.mapbox import NAVIGATION_API
from openpilot.system.asius.navigation_route import RouteStore
from openpilot.system.asius.navigation_proxy import request_json
from openpilot.common.params import Params
from openpilot.common.realtime import Ratekeeper
from openpilot.common.swaglog import cloudlog
from openpilot.selfdrive.navigation.helpers import latest_gps_source
from openpilot.selfdrive.navigation.route import along_path, local_meters, nearest_segment

REROUTE_DISTANCE = 25
MANEUVER_TRANSITION_THRESHOLD = 10
NAVIGATION_HZ = 5
REROUTE_COUNTER_MIN = 3 * NAVIGATION_HZ
ROUTE_RESEND_PERIOD = 10 * NAVIGATION_HZ  # updates; a restarted mapsd gets the route within this
RECOMPUTE_HOLD = 10 * NAVIGATION_HZ  # ten seconds between route requests


def set_banner(instruction, banners, distance_to_maneuver):
  """Fill the banner fields from the banner to show now: the last one still ahead of the ego, else the first."""
  if not banners:
    return
  banner = next((b for b in reversed(banners) if distance_to_maneuver < b['distanceAlongGeometry']), banners[0])
  instruction.showFull = distance_to_maneuver < banner['distanceAlongGeometry']
  primary = banner['primary']
  instruction.maneuverPrimaryText = primary.get('text') or ''
  instruction.maneuverType = primary.get('type') or ''
  instruction.maneuverModifier = primary.get('modifier') or ''
  if banner.get('secondary') is not None:
    instruction.maneuverSecondaryText = banner['secondary']['text']


class RouteEngine:
  def __init__(self, sm, pm, params=None, store=None):
    self.sm = sm
    self.pm = pm
    self.params = params if params is not None else Params()

    self.store = store if store is not None else RouteStore(self.params)
    saved_position = self.store.position()
    self.last_position = tuple(saved_position) if saved_position else None  # (longitude, latitude)
    self.last_position_save = 0.
    self.last_known = True
    self.destination_request = None
    self.started = False
    self.last_speed = 0.
    self.last_bearing = None

    self.nav_destination = None
    self.step_idx = None
    self.route = None
    self.route_geometry = None  # an (n, 2) array of (longitude, latitude) per step

    self.recompute_hold = 0
    self.reroute_counter = 0

    self.mapbox_token = os.environ.get("MAPBOX_TOKEN")
    self.mapbox_host = "https://api.mapbox.com" if self.mapbox_token is not None else NAVIGATION_API

  def update_position(self):
    self.started = bool(self.sm.valid["deviceState"] and self.sm.alive["deviceState"] and self.sm["deviceState"].started)
    source = latest_gps_source(self.sm)
    if source is not None:
      gps = self.sm[source]
      self.last_position = (gps.longitude, gps.latitude)
      self.last_speed = getattr(gps, "speed", 0.)
      self.last_bearing = gps.bearingDeg if self.last_speed >= 1 and getattr(gps, "bearingAccuracyDeg", 0.) <= 45 else None
      if time.monotonic() - self.last_position_save >= 5:
        self.last_position_save = time.monotonic()
        try:
          self.store.save_position(self.last_position)
        except OSError:
          cloudlog.exception("navd.failed_to_save_position")
    self.last_known = source is None

  def update(self):
    self.sm.update(0)
    self.update_position()
    route = self.route
    self.did_calculate = False
    try:
      self.recompute_route(gps_ok=not self.last_known)
      if self.did_calculate:
        # HTTP can take seconds. Never publish guidance from the pre-request fix.
        self.sm.update(0)
        self.update_position()
      self.send_instruction()
    except Exception:
      cloudlog.exception("navd.failed_to_compute")
    # mapsd dedupes repeats, so a periodic resend is how a restarted mapsd gets the route
    if self.route is not route or self.sm.frame % ROUTE_RESEND_PERIOD == 0:
      self.send_route()

  def recompute_route(self, gps_ok: bool):
    destination = self.params.get("NavDestination") if self.params.get_bool("NavigationEnabled") else None
    if destination is None:
      self.clear_route()
      self.recompute_hold = 0
      return
    if self.last_position is None:
      self.store.publish(destination, status="waitingForGps")
      return
    new_destination = (destination['longitude'], destination['latitude'])
    changed = destination != self.destination_request

    # Don't recompute when GPS drifts in tunnels
    if not changed and self.step_idx is not None and (not gps_ok or not self.started):
      return

    self.recompute_hold -= 1
    if changed or (self.should_recompute() and self.recompute_hold <= 0):
      self.recompute_hold = RECOMPUTE_HOLD
      self.calculate_route(new_destination)
      self.reroute_counter = 0

  def calculate_route(self, destination):
    self.did_calculate = True
    assert self.last_position is not None
    cloudlog.warning(f"Calculating route {self.last_position} -> {destination}")
    request = self.params.get("NavDestination")
    from openpilot.system.asius.navigation_preview import PreviewStore
    changed = isinstance(request, dict) and request.get("requestId") and request != self.destination_request
    selected = PreviewStore(self.params).take_selected(request) if changed else None
    self.destination_request = request
    self.clear_route()
    self.nav_destination = destination
    if selected is not None:
      try:
        self.accept_route(request, selected, resume=True)
      except (ValueError, KeyError, IndexError):
        self.store.publish(request, status="error", error="Invalid selected route. Refresh the routes.")
        self.clear_route()
      return
    self.store.publish(request, status="loading")

    lang = self.params.get('LanguageSetting')
    if lang is not None:
      lang = lang.replace('main_', '')

    params = {
      'access_token': self.mapbox_token,
      'geometries': 'geojson',
      'overview': 'full',
      'steps': 'true',
      'banner_instructions': 'true',
      'alternatives': 'false',
      'annotations': 'maxspeed',
      'language': lang,
    }
    if self.last_bearing is not None:
      params['bearings'] = f"{(self.last_bearing + 360) % 360:.0f},90;"

    coords_str = ';'.join(f'{lon},{lat}' for lon, lat in (self.last_position, destination))
    url = self.mapbox_host + '/directions/v5/mapbox/driving-traffic/' + coords_str
    headers = {}
    if self.mapbox_token is None:
      params.pop('access_token')
      # The Asius proxy adds maxspeed annotations and rejects unknown parameters.
      params.pop('annotations')
    try:
      if self.mapbox_token is None:
        r = request_json(url[len(NAVIGATION_API):], params, params=self.params, destination=request)
      else:
        resp = requests.get(url, params=params, headers=headers, timeout=10)
        resp.raise_for_status()
        r = resp.json()

      if self.params.get("NavDestination") != request:
        return

      if len(r['routes']):
        self.accept_route(request, r['routes'][0])
      else:
        self.store.publish(request, status="error", error="No driving route found")
        cloudlog.warning("Got empty route response")
        self.clear_route()

    except (requests.exceptions.RequestException, ValueError, KeyError, IndexError):
      self.store.publish(request, status="error", error="Cannot load route. Connect the device or an app to the internet.")
      cloudlog.exception("failed to get route")
      self.clear_route()

  def accept_route(self, request, route, resume=False):
    steps = route['legs'][0]['steps']
    geometry = [np.array(step['geometry']['coordinates'], dtype=np.float64).reshape((-1, 2)) for step in steps]
    if not steps or any(len(path) < 2 or not np.isfinite(path).all() for path in geometry):
      raise ValueError("Invalid route geometry")
    origin = {'longitude': self.last_position[0], 'latitude': self.last_position[1]}
    if self.store.publish(request, route=route, origin=origin, last_known=self.last_known):
      self.route = steps
      self.route_geometry = geometry
      self.step_idx = min(range(len(geometry)), key=lambda i: along_path(geometry[i], self.last_position)[1]) if resume else 0

  def send_instruction(self):
    msg = messaging.new_message('navInstruction', valid=True)

    if self.step_idx is None or not self.started or self.last_known:
      msg.valid = False
      self.pm.send('navInstruction', msg)
      return

    assert self.route is not None and self.route_geometry is not None and self.last_position is not None
    step = self.route[self.step_idx]
    along_geometry, _ = along_path(self.route_geometry[self.step_idx], self.last_position)
    distance_to_maneuver_along_geometry = step['distance'] - along_geometry

    # Banner instructions are for the following maneuver step, don't use empty last step
    banner_step = step
    if not len(banner_step['bannerInstructions']) and self.step_idx == len(self.route) - 1:
      banner_step = self.route[max(self.step_idx - 1, 0)]

    # Current instruction
    msg.navInstruction.maneuverDistance = distance_to_maneuver_along_geometry
    set_banner(msg.navInstruction, banner_step['bannerInstructions'], distance_to_maneuver_along_geometry)

    # Total remaining distance and time: the rest of this step, then every step after it
    remaining = max(0., min(1., 1.0 - along_geometry / max(step['distance'], 1)))
    future = self.route[self.step_idx + 1:]
    msg.navInstruction.distanceRemaining = step['distance'] * remaining + sum(s['distance'] for s in future)
    msg.navInstruction.timeRemaining = step['duration'] * remaining + sum(s['duration'] for s in future)

    self.pm.send('navInstruction', msg)

    # Transition to next route segment
    if distance_to_maneuver_along_geometry < -MANEUVER_TRANSITION_THRESHOLD:
      if self.step_idx + 1 < len(self.route):
        self.step_idx += 1
        self.recompute_hold = 0
      else:
        cloudlog.warning("Destination reached")

        # Clear route if driving away from destination
        if np.linalg.norm(local_meters(self.last_position, self.nav_destination)) > REROUTE_DISTANCE:
          self.params.remove("NavDestination")
          self.clear_route()

  def send_route(self):
    coords = np.concatenate(self.route_geometry) if self.route_geometry is not None else np.empty((0, 2))
    msg = messaging.new_message('navRoute', valid=True)
    msg.navRoute.coordinates = [{'longitude': lon, 'latitude': lat} for lon, lat in coords.tolist()]
    self.pm.send('navRoute', msg)

  def clear_route(self):
    self.route = None
    self.route_geometry = None
    self.step_idx = None
    self.nav_destination = None

  def should_recompute(self):
    if self.step_idx is None:
      return True

    assert self.route is not None and self.route_geometry is not None and self.last_position is not None
    # Don't recompute in last segment, assume destination is reached
    if self.step_idx == len(self.route) - 1:
      return False

    path = self.route_geometry[self.step_idx]
    wrong_way = False
    if self.last_bearing is not None and self.last_speed >= 3 and len(path) > 1:
      local = local_meters(path, self.last_position)
      index, _, _ = nearest_segment(local[:-1], local[1:], np.zeros(2))
      direction = local[index + 1] - local[index]
      heading = np.degrees(np.arctan2(direction[0], direction[1]))
      wrong_way = abs((self.last_bearing - heading + 180) % 360 - 180) > 100
    if wrong_way or along_path(path, self.last_position)[1] > REROUTE_DISTANCE:
      self.reroute_counter += 1
    else:
      self.reroute_counter = 0
    return self.reroute_counter > REROUTE_COUNTER_MIN


def main():
  pm = messaging.PubMaster(['navInstruction', 'navRoute'])
  sm = messaging.SubMaster(['gpsLocation', 'gpsLocationExternal', 'deviceState'])

  rk = Ratekeeper(NAVIGATION_HZ)
  route_engine = RouteEngine(sm, pm)
  while True:
    route_engine.update()
    rk.keep_time()


if __name__ == "__main__":
  main()
