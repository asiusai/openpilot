from openpilot.common.params import Params
from openpilot.common.hardware import ASIUS_HARDWARE


def get_gps_location_service(params: Params) -> str:
  if not ASIUS_HARDWARE and params.get_bool("UbloxAvailable"):
    return "gpsLocationExternal"
  else:
    return "gpsLocation"
