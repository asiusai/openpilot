"""Live stream requests for the Asius app."""
from opendbc.car.structs import car

from openpilot.cereal.services import SERVICE_LIST
from openpilot.common.params import Params


def start_stream(sdp: str, cameras: list[str], bridge_services_out: list[str], enabled: bool = True, in_car: bool = False) -> dict:
  from openpilot.system.webrtc.helpers import StreamRequestBody, post_stream_request, wait_for_webrtcd
  if not isinstance(cameras, list) or any(c not in ("road", "wideRoad", "driver") for c in cameras) or len(set(cameras)) != len(cameras):
    raise ValueError("cameras must contain unique road, wideRoad or driver names")
  if not isinstance(bridge_services_out, list) or any(not isinstance(s, str) or s not in SERVICE_LIST for s in bridge_services_out):
    raise ValueError("bridge_services_out must contain valid cereal service names")
  bridge_services_in = []

  cp_bytes = Params().get("CarParamsPersistent")
  if cp_bytes is not None:
    with car.CarParams.from_bytes(cp_bytes) as CP:
      if CP.notCar and not in_car and cameras:
        bridge_services_in.append("testJoystick")

  # webrtcd owns the streaming lifetime, including ignition transitions.
  wait_for_webrtcd()
  return post_stream_request(StreamRequestBody(sdp, cameras, enabled, bridge_services_in, bridge_services_out, in_car=in_car))
