import time

from openpilot.common.params import Params
from openpilot.common.hardware import ASIUS_HARDWARE


def get_gps_location_services(params: Params) -> list[str]:
  if ASIUS_HARDWARE:
    return ['gpsLocation', 'gpsLocationExternal']
  return ['gpsLocationExternal' if params.get_bool('UbloxAvailable') else 'gpsLocation']


def latest_gps_source(sm, services=('gpsLocation', 'gpsLocationExternal')) -> str | None:
  """The service with the freshest usable GPS fix, or None."""
  now = time.monotonic()
  sources = [s for s in services if sm.valid[s] and sm[s].hasFix and 0 <= now - sm.recv_time[s] < 2]
  return max(sources, key=lambda s: sm.logMonoTime[s]) if sources else None
