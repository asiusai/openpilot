import os
import operator
import platform

from opendbc.car.structs import car
from openpilot.common.params import Params
from openpilot.common.hardware import ASIUS_HARDWARE, BLUETOOTH_AVAILABLE, COMMA_HARDWARE, PC
from openpilot.system.manager.process import PythonProcess, NativeProcess, DaemonProcess

WEBCAM = os.getenv("USE_WEBCAM") is not None

def offroad_camera_ready() -> bool:
  # hardwared restores these CPUs asynchronously when a headless preview starts.
  return not ASIUS_HARDWARE or {6, 7}.issubset(os.sched_getaffinity(0))

def driverview(started: bool, params: Params, CP: car.CarParams) -> bool:
  return started or (params.get_bool("IsDriverViewEnabled") and offroad_camera_ready())

def sound(started: bool, params: Params, CP: car.CarParams) -> bool:
  from openpilot.system.asius.pairing_confirmation import read_announcement
  from openpilot.system.asius.sound import preview_request
  return (driverview(started, params, CP) or params.get_bool("VoiceAlertsEnabled") or
          bool(params.get("BluetoothPairingUntil")) or read_announcement() is not None or
          (ASIUS_HARDWARE and preview_request(params) is not None))

def notcar(started: bool, params: Params, CP: car.CarParams) -> bool:
  return started and CP.notCar

def iscar(started: bool, params: Params, CP: car.CarParams) -> bool:
  return started and not CP.notCar

def logging(started: bool, params: Params, CP: car.CarParams) -> bool:
  run = (not CP.notCar) or not params.get_bool("DisableLogging")
  return started and run

def ublox_available() -> bool:
  return os.path.exists('/dev/ttyHS0') and not os.path.exists('/persist/comma/use-quectel-gps')

def ublox(started: bool, params: Params, CP: car.CarParams) -> bool:
  use_ublox = ublox_available()
  if use_ublox != params.get_bool("UbloxAvailable"):
    params.put_bool("UbloxAvailable", use_ublox, block=True)
  return started and use_ublox

def joystick(started: bool, params: Params, CP: car.CarParams) -> bool:
  return started and params.get_bool("JoystickDebugMode")

def not_joystick(started: bool, params: Params, CP: car.CarParams) -> bool:
  return started and not params.get_bool("JoystickDebugMode")

def long_maneuver(started: bool, params: Params, CP: car.CarParams) -> bool:
  return started and params.get_bool("LongitudinalManeuverMode")

def lat_maneuver(started: bool, params: Params, CP: car.CarParams) -> bool:
  return started and params.get_bool("LateralManeuverMode")

def not_long_maneuver(started: bool, params: Params, CP: car.CarParams) -> bool:
  return started and not params.get_bool("LongitudinalManeuverMode")

def qcomgps(started: bool, params: Params, CP: car.CarParams) -> bool:
  return started and not ublox_available()

def always_run(started: bool, params: Params, CP: car.CarParams) -> bool:
  return True

def only_onroad(started: bool, params: Params, CP: car.CarParams) -> bool:
  return started

def alpamayo_shadow(started: bool, params: Params, CP: car.CarParams) -> bool:
  # ALPAMAYO_KEEP_WARM=1 loads and compiles offroad and keeps the worker across ignition cycles.
  # Warmup then never overlaps the onroad camera and Adreno load (Karel dropped USB under that combined load).
  keep_warm = os.environ.get('ALPAMAYO_KEEP_WARM') == '1'
  if (not started and not keep_warm) or os.environ.get('ALPAMAYO_SHADOW') == '1':
    return started or keep_warm
  from openpilot.selfdrive.alpamayo.shadowd import planner_mode
  if planner_mode() and os.path.isfile(os.environ['ALPAMAYO_CONFIG']):
    from openpilot.selfdrive.modeld.helpers import chestnut_present
    return chestnut_present()
  return False

def navigation(started: bool, params: Params, CP: car.CarParams) -> bool:
  return started and params.get_bool("NavigationEnabled")

def only_offroad(started: bool, params: Params, CP: car.CarParams) -> bool:
  return not started

def livestream(started: bool, params: Params, CP: car.CarParams) -> bool:
  return params.get_bool("IsLiveStreaming") and (started or offroad_camera_ready())

def sidecam(started: bool, params: Params, CP: car.CarParams) -> bool:
  # Camera devices always look for their host; the host runs its hotspot while onroad.
  from openpilot.system.sidecam.config import current
  config = current()
  return config is not None and (config['role'] == 'camera' or started or config.get('always', False))

def sidecam_streaming(started: bool, params: Params, CP: car.CarParams) -> bool:
  from openpilot.system.sidecam.config import streaming
  return streaming() and offroad_camera_ready()

def side_camerad_args() -> list[str]:
  from openpilot.system.sidecam.config import role
  return ['--no-cabin'] if role() == 'camera' else []

def side_views() -> list[str]:
  from openpilot.system.sidecam.config import current
  config = current()
  return [config['wide'], config['road']] if config and config['role'] == 'camera' else ['lf', 'lr']

def asius_upload(started: bool, params: Params, CP: car.CarParams) -> bool:
  return not PC and params.get_bool("CloudUploadEnabled")

def or_(*fns):
  return lambda *args: operator.or_(*(fn(*args) for fn in fns))

def and_(*fns):
  return lambda *args: operator.and_(*(fn(*args) for fn in fns))

procs = [
  DaemonProcess("manage_athenad", "openpilot.system.athena.manage_athenad", "AthenadPid", enabled=False),
  DaemonProcess("manage_relayd", "openpilot.system.asius.manage_relayd", "RelayPid"),
  DaemonProcess("manage_bluetoothd", "openpilot.system.asius.manage_bluetoothd", "BluetoothdPid", enabled=BLUETOOTH_AVAILABLE),

  NativeProcess("loggerd", "openpilot/system/loggerd", ["./loggerd"], logging),
  NativeProcess("encoderd", "openpilot/system/loggerd", ["./encoderd"], logging, restart_delay=5.0 if ASIUS_HARDWARE else 0.0),
  NativeProcess("stream_encoderd", "openpilot/system/loggerd", ["./encoderd", "--stream"], or_(livestream, notcar),
                restart_delay=5.0 if ASIUS_HARDWARE else 0.0),
  PythonProcess("logmessaged", "openpilot.system.logmessaged", always_run),

  PythonProcess("sidecamd", "openpilot.system.sidecam.sidecamd", sidecam, enabled=ASIUS_HARDWARE),
  NativeProcess("side_encoderd", "openpilot/system/loggerd", ["./encoderd", "--side", *side_views()], sidecam_streaming,
                enabled=ASIUS_HARDWARE, restart_delay=1.0),
  NativeProcess("camerad", "openpilot/system/camerad", ["./camerad", *side_camerad_args()],
                or_(or_(driverview, livestream), sidecam_streaming), enabled=not WEBCAM,
                restart_delay=1.0 if ASIUS_HARDWARE else 0.0),
  PythonProcess("webcamerad", "openpilot.system.camerad.webcam.camerad", driverview, enabled=WEBCAM),
  PythonProcess("proclogd", "openpilot.system.proclogd", only_onroad, enabled=platform.system() != "Darwin"),
  PythonProcess("journald", "openpilot.system.journald", only_onroad, platform.system() != "Darwin"),
  PythonProcess("micd", "openpilot.system.micd", iscar),
  PythonProcess("timed", "openpilot.system.timed", always_run, enabled=not PC),

  PythonProcess("modeld", "openpilot.selfdrive.modeld.modeld", only_onroad),
  PythonProcess("alpamayod", "openpilot.selfdrive.alpamayo.shadowd", alpamayo_shadow),
  PythonProcess("navd", "openpilot.selfdrive.navigation.navd", lambda started, params, CP: started or params.get_bool("NavigationEnabled")),
  PythonProcess("mapsd", "openpilot.selfdrive.navigation.mapsd", navigation),

  PythonProcess("dmonitoringmodeld", "openpilot.selfdrive.modeld.dmonitoringmodeld", driverview,
                enabled=(WEBCAM or not PC)),
  PythonProcess("sensord", "openpilot.system.sensord.sensord", only_onroad, enabled=not PC),
  PythonProcess("ui", "openpilot.selfdrive.ui.ui", always_run, enabled=not ASIUS_HARDWARE),
  PythonProcess("soundd", "openpilot.selfdrive.ui.soundd", sound),
  PythonProcess("ledd", "openpilot.selfdrive.v0.ledd", always_run, enabled=ASIUS_HARDWARE),
  PythonProcess("buttond", "openpilot.selfdrive.v0.buttond", always_run, enabled=ASIUS_HARDWARE),
  PythonProcess("locationd", "openpilot.selfdrive.locationd.locationd", only_onroad),
  NativeProcess("_pandad", "openpilot/selfdrive/pandad", ["./pandad"], always_run, enabled=False),
  PythonProcess("calibrationd", "openpilot.selfdrive.locationd.calibrationd", only_onroad),
  PythonProcess("torqued", "openpilot.selfdrive.locationd.torqued", only_onroad),
  PythonProcess("controlsd", "openpilot.selfdrive.controls.controlsd", and_(not_joystick, iscar)),
  PythonProcess("joystickd", "openpilot.tools.joystick.joystickd", or_(joystick, notcar)),
  PythonProcess("selfdrived", "openpilot.selfdrive.selfdrived.selfdrived", only_onroad),
  PythonProcess("card", "openpilot.selfdrive.car.card", only_onroad),
  PythonProcess("deleter", "openpilot.system.loggerd.deleter", always_run),
  PythonProcess("dmonitoringd", "openpilot.selfdrive.monitoring.dmonitoringd", driverview, enabled=(WEBCAM or not PC)),
  PythonProcess("qcomgpsd", "openpilot.system.qcomgpsd.qcomgpsd", qcomgps, enabled=COMMA_HARDWARE),
  PythonProcess("pandad", "openpilot.selfdrive.pandad.pandad", always_run),
  PythonProcess("paramsd", "openpilot.selfdrive.locationd.paramsd", only_onroad),
  PythonProcess("lagd", "openpilot.selfdrive.locationd.lagd", only_onroad),
  PythonProcess("ubloxd", "openpilot.system.ubloxd.ubloxd", ublox, enabled=COMMA_HARDWARE),
  PythonProcess("pigeond", "openpilot.system.ubloxd.pigeond", ublox, enabled=COMMA_HARDWARE),
  PythonProcess("m10d", "openpilot.system.ubloxd.m10d", always_run, enabled=ASIUS_HARDWARE),
  PythonProcess("plannerd", "openpilot.selfdrive.controls.plannerd", not_long_maneuver),
  PythonProcess("maneuversd", "openpilot.tools.longitudinal_maneuvers.maneuversd", long_maneuver),
  PythonProcess("lateral_maneuversd", "openpilot.tools.lateral_maneuvers.lateral_maneuversd", lat_maneuver),
  PythonProcess("radard", "openpilot.selfdrive.controls.radard", only_onroad),
  PythonProcess("hardwared", "openpilot.system.hardware.hardwared", always_run),
  PythonProcess("tombstoned", "openpilot.system.tombstoned", always_run, enabled=not PC),
  PythonProcess("updated", "openpilot.system.updated.updated", only_offroad, enabled=not PC),
  PythonProcess("uploader", "openpilot.system.loggerd.uploader", always_run, enabled=False),
  PythonProcess("route_summaryd", "openpilot.system.loggerd.route_summary", always_run),
  PythonProcess("asius_uploader", "openpilot.system.loggerd.asius_uploader", asius_upload),
  # debug procs
  PythonProcess("webrtcd", "openpilot.system.webrtc.webrtcd", always_run),
  PythonProcess("joystick", "openpilot.tools.joystick.joystick_control", and_(joystick, iscar)),
]

managed_processes = {p.name: p for p in procs}
