"""Mapbox vector tiles for the upstream navigation renderer, without comma account authentication."""
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor

import requests

from openpilot.system.asius.identity import dongle_id_from_public_key, get_device_public_key
from openpilot.system.asius.relayd import sign_jwt


NAVIGATION_API = 'https://api.asius.ai/v1/nav'


class NavigationApi:
  def get_token(self):
    return sign_jwt({'type': 'navigation', 'aud': 'asius-navigation', 'identity': dongle_id_from_public_key(get_device_public_key())}, expiry_seconds=60)


class MapboxTiles:
  min_zoom = 0
  max_zoom = 16

  def __init__(self):
    self.api = NavigationApi()
    self.cache = OrderedDict()
    self.pending = {}
    self.pool = ThreadPoolExecutor(max_workers=3, thread_name_prefix='mapbox')

  def _fetch(self, key):
    z, x, y = key
    try:
      response = requests.get(f'{NAVIGATION_API}/maps/v4/mapbox.mapbox-streets-v8/{z}/{x}/{y}.vector.pbf',
                              headers={'Authorization': f'Bearer {self.api.get_token()}'}, timeout=10)
      response.raise_for_status()
      if len(response.content) > 4 * 1024 * 1024:
        raise OSError('Oversized map tile')
      return response.content
    except requests.RequestException:
      # A requests exception can contain the token-bearing URL.
      raise OSError('Unable to fetch Mapbox tiles') from None

  def get(self, z: int, x: int, y: int) -> bytes:
    key = (z, x, y)
    if key in self.cache:
      self.cache.move_to_end(key)
      return self.cache[key]
    future = self.pending.pop(key, None)
    data = future.result() if future else self._fetch(key)
    self.cache[key] = data
    while len(self.cache) > 128:
      self.cache.popitem(last=False)
    return data

  def prefetch(self, tiles):
    # Bound lookahead instead of downloading every tile of a long route.
    # The renderer loads visible tiles on demand. No persistent Mapbox cache.
    return

  def prefetch_view(self, zoom: int, tiles):
    for x, y in tiles:
      key = (zoom, x, y)
      if key not in self.cache and key not in self.pending and len(self.pending) < 16:
        self.pending[key] = self.pool.submit(self._fetch, key)

  def close(self):
    self.pool.shutdown(wait=False, cancel_futures=True)
    self.cache.clear()
    self.pending.clear()


def navigation_archive():
  return MapboxTiles()
