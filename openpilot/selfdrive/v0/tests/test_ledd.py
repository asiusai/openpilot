import time
from collections import defaultdict
from types import SimpleNamespace

from openpilot.cereal import log
from openpilot.selfdrive.v0 import ledd


class FakeSubMaster:
  def __init__(self):
    self.seen = defaultdict(bool)
    self.alive = defaultdict(bool)
    self.valid = defaultdict(bool)
    self.messages = {}

  def set(self, service, message, *, alive=True, valid=True):
    self.seen[service] = True
    self.alive[service] = alive
    self.valid[service] = valid
    self.messages[service] = message

  def __getitem__(self, service):
    return self.messages[service]


def healthy_sm(*, started=True):
  sm = FakeSubMaster()
  sm.set('deviceState', SimpleNamespace(started=started))
  sm.set('extrinsicsCalibration', SimpleNamespace(calStatus=log.ExtrinsicsCalibration.Status.calibrated, calPerc=100))
  sm.set('managerState', SimpleNamespace(processes=[]))
  sm.set('pandaStates', [SimpleNamespace(
    pandaType=log.PandaState.PandaType.tres,
    faultStatus=None,
    faults=[],
    heartbeatLost=False,
  )])
  sm.set('driverMonitoringState', SimpleNamespace(alertLevel=log.DriverMonitoringState.AlertLevel.none, lockout=False, alwaysOnLockout=False))
  sm.set('selfdriveState', SimpleNamespace(
    active=False,
    engageable=True,
    state=log.SelfdriveState.OpenpilotState.disabled,
    alertSound=SimpleNamespace(raw='none'),
    alertType='',
  ))
  return sm


def setup_module():
  ledd.log = log


def test_white_when_ready():
  assert ledd.led_state(healthy_sm()) == ledd.WHITE


def test_yellow_when_not_engageable_or_waiting_for_brake_release():
  sm = healthy_sm()
  sm['selfdriveState'].engageable = False
  assert ledd.led_state(sm) == ledd.YELLOW

  sm['selfdriveState'].engageable = True
  sm['selfdriveState'].state = log.SelfdriveState.OpenpilotState.preEnabled
  assert ledd.led_state(sm) == ledd.YELLOW


def test_red_immediately_when_started_without_selfdrive_state():
  sm = healthy_sm()
  sm.seen['selfdriveState'] = False
  sm.alive['selfdriveState'] = False
  assert ledd.led_state(sm) == ledd.RED


def test_missing_selfdrive_state_is_ignored_when_offroad():
  sm = healthy_sm()
  sm.seen['selfdriveState'] = False
  sm.alive['selfdriveState'] = False
  sm['deviceState'].started = False
  assert ledd.led_state(sm) == ledd.WHITE


def test_orange_calibration_overrides_non_engageable():
  sm = healthy_sm()
  sm['extrinsicsCalibration'].calStatus = log.ExtrinsicsCalibration.Status.uncalibrated
  sm['selfdriveState'].engageable = False
  assert ledd.led_state(sm) == ledd.ORANGE


def test_green_when_engaged():
  sm = healthy_sm()
  sm['selfdriveState'].active = True
  sm['selfdriveState'].state = log.SelfdriveState.OpenpilotState.enabled
  assert ledd.led_state(sm) == ledd.GREEN


def test_warning_sound_blinks_red_only_while_engaged():
  sm = healthy_sm()
  sm['selfdriveState'].alertSound.raw = 'promptRepeat'
  assert ledd.led_state(sm, now=0.) == ledd.WHITE

  sm['selfdriveState'].active = True
  sm['selfdriveState'].state = log.SelfdriveState.OpenpilotState.enabled
  assert ledd.led_state(sm, now=0.) == ledd.RED
  assert ledd.led_state(sm, now=0.5) == ledd.OFF


def test_driver_monitoring_blinks_magenta_while_engaged():
  sm = healthy_sm()
  sm['selfdriveState'].active = True
  sm['selfdriveState'].state = log.SelfdriveState.OpenpilotState.enabled
  sm['driverMonitoringState'].alertLevel = log.DriverMonitoringState.AlertLevel.one
  assert ledd.led_state(sm, now=0.) == ledd.DM_WARNING
  assert ledd.led_state(sm, now=0.5) == ledd.OFF


def test_driver_monitoring_warning_is_ignored_when_disengaged():
  sm = healthy_sm()
  sm['driverMonitoringState'].alertLevel = log.DriverMonitoringState.AlertLevel.one
  sm['selfdriveState'].alertType = 'driverDistracted1/warning'
  sm['selfdriveState'].alertSound.raw = 'promptDistracted'
  assert ledd.led_state(sm, now=0.) == ledd.WHITE


def test_soft_disabling_blinks_red_over_driver_monitoring_warning():
  sm = healthy_sm()
  sm['selfdriveState'].active = True
  sm['selfdriveState'].state = log.SelfdriveState.OpenpilotState.softDisabling
  sm['selfdriveState'].alertType = 'driverDistracted3/warning'
  assert ledd.led_state(sm, now=0.) == ledd.RED
  assert ledd.led_state(sm, now=0.5) == ledd.OFF


def test_persistent_process_failure_is_solid_red(monkeypatch):
  sm = healthy_sm()
  sm['managerState'].processes = [SimpleNamespace(name='camerad', shouldBeRunning=True, running=False)]
  monkeypatch.setattr(ledd, 'STARTED_AT', time.monotonic() - ledd.STARTUP_GRACE - 1.)
  assert ledd.led_state(sm, now=0.) == ledd.RED
  assert ledd.led_state(sm, now=0.5) == ledd.RED


def test_offroad_stays_white_when_processes_are_not_running(monkeypatch):
  sm = healthy_sm(started=False)
  sm['managerState'].processes = [SimpleNamespace(name='camerad', shouldBeRunning=True, running=False)]
  monkeypatch.setattr(ledd, 'STARTED_AT', time.monotonic() - ledd.STARTUP_GRACE - 1.)
  assert ledd.led_state(sm) == ledd.WHITE


def test_pairing_blinks_green_without_driver_camera(monkeypatch):
  monkeypatch.setattr(ledd, 'pairing_mode_active', lambda: True)
  monkeypatch.setattr(ledd.time, 'monotonic', lambda: 0.)
  channels = ledd.pairing_led_channels(26)
  assert channels == {
    1: [0] * 9,
    2: [0, 26, 0, 0, 3, 0, 0, 26, 0],
    3: [0, 26, 0, 0, 3, 0, 0, 26, 0],
  }

  monkeypatch.setattr(ledd.time, 'monotonic', lambda: 0.5)
  assert ledd.pairing_led_channels(26) == {1: [0] * 9, 2: [0] * 9, 3: [0] * 9}


def test_camera_brightness_uses_openpilot_wide_road_exposure_curve():
  sm = FakeSubMaster()
  assert ledd.camera_led_brightness(sm) == 26

  sm.set('narrowRoadCameraState', SimpleNamespace(exposureValPercent=100.))
  assert ledd.camera_led_brightness(sm) == 26

  sm.set('wideRoadCameraState', SimpleNamespace(exposureValPercent=100.))
  assert ledd.camera_led_brightness(sm) == 13

  sm['wideRoadCameraState'].exposureValPercent = 0.
  assert ledd.camera_led_brightness(sm) == 26


def test_runtime_brightness_reaches_requested_peak():
  assert ledd.max_brightness(ledd.WHITE, 26) == ledd.LedState("white", 26, 20, 12)
  assert ledd.max_brightness(ledd.YELLOW, 26) == ledd.LedState("yellow", 26, 19, 0)


def test_status_colors_dim_leds_two_and_five_to_ten_percent(monkeypatch):
  for camera_num in ledd.CAM_LED_STATUS_CAMERAS:
    board = ledd.CameraLedBoard("test", camera_num=camera_num, bus_num=0)
    sent = []
    monkeypatch.setattr(board, "set_channels", sent.append)
    for state in (ledd.WHITE, ledd.GREEN, ledd.YELLOW, ledd.ORANGE, ledd.RED, ledd.DM_WARNING, ledd.OFF):
      board.set(state)
      rgb = [state.red, state.green, state.blue]
      assert sent[-1] == rgb + [round(channel * 0.1) for channel in rgb] + rgb
