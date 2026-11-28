from openpilot.common.params import Params
from openpilot.common.hardware import ASIUS_HARDWARE


def get_gps_location_service(params: Params) -> str:
  if ASIUS_HARDWARE or params.get_bool("UbloxAvailable"):
    return "gpsLocationExternal"
  else:
    return "gpsLocation"
