import time
import pytest
from collections import defaultdict
from types import SimpleNamespace

from openpilot.cereal import log
from openpilot.selfdrive.selfdrived.events import EVENTS, ET
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
  sm.set('deviceState', SimpleNamespace(started=started, thermalStatus=log.DeviceState.ThermalStatus.ok))
  sm.set('extrinsicsCalibration', SimpleNamespace(calStatus=log.ExtrinsicsCalibration.Status.calibrated, calPerc=100))
  sm.set('managerState', SimpleNamespace(processes=[]))
  sm.set('onroadEvents', [])
  sm.set('pandaStates', [SimpleNamespace(
    pandaType=log.PandaState.PandaType.tres,
    faultStatus=None,
    faults=[],
    heartbeatLost=False,
  )])
  sm.set('driverMonitoringState', SimpleNamespace(alertLevel=log.DriverMonitoringState.AlertLevel.none, lockout=False, alwaysOnLockout=False,
                                                visionPolicyState=SimpleNamespace(faceDetected=True)))
  sm.set('selfdriveState', log.SelfdriveState.new_message(
    active=False,
    engageable=True,
    state=log.SelfdriveState.OpenpilotState.disabled,
    alertSound='none',
    alertType='',
  ))
  return sm


def setup_module():
  ledd.log = log


def test_blue_when_ready():
  assert ledd.led_state(healthy_sm()) == ledd.BLUE


def test_blue_when_not_engageable_or_waiting_for_brake_release():
  sm = healthy_sm()
  sm['selfdriveState'].engageable = False
  assert ledd.led_state(sm) == ledd.BLUE

  sm['selfdriveState'].engageable = True
  sm['selfdriveState'].state = log.SelfdriveState.OpenpilotState.preEnabled
  assert ledd.led_state(sm) == ledd.BLUE


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
  assert ledd.led_state(sm) == ledd.BLUE


@pytest.mark.parametrize('event', sorted(ledd.SYSTEM_FAULT_EVENTS))
@pytest.mark.parametrize('active', [False, True])
def test_system_fault_is_red_even_with_another_screen_alert(event, active):
  sm = healthy_sm()
  sm['selfdriveState'].active = active
  sm['selfdriveState'].alertType = 'wrongGear/noEntry'
  sm.set('onroadEvents', [log.OnroadEvent.new_message(name=event)])
  assert ledd.led_state(sm, now=0.) == ledd.RED
  assert ledd.led_state(sm, now=0.5) == (ledd.OFF if active else ledd.RED)

  sm.set('onroadEvents', [])
  assert ledd.led_state(sm, now=0.) == (ledd.GREEN if active else ledd.BLUE)


@pytest.mark.parametrize('service', ['onroadEvents', 'deviceState'])
@pytest.mark.parametrize('field', ['seen', 'alive', 'valid'])
def test_camera_fault_requires_current_valid_state(service, field):
  sm = healthy_sm()
  sm.set('onroadEvents', [log.OnroadEvent.new_message(name='cameraMalfunction')])
  getattr(sm, field)[service] = False
  assert ledd.led_state(sm) == ledd.BLUE


def test_camera_fault_is_ignored_offroad():
  sm = healthy_sm(started=False)
  sm.set('onroadEvents', [log.OnroadEvent.new_message(name='cameraMalfunction')])
  assert ledd.led_state(sm) == ledd.BLUE


@pytest.mark.parametrize('started', [False, True])
@pytest.mark.parametrize('thermal_status', ['overheated', 'critical'])
def test_overheating_is_red_including_startup_cooling(started, thermal_status):
  sm = healthy_sm(started=started)
  sm['deviceState'].thermalStatus = getattr(log.DeviceState.ThermalStatus, thermal_status)
  assert ledd.led_state(sm) == ledd.RED
  sm['deviceState'].thermalStatus = log.DeviceState.ThermalStatus.ok
  assert ledd.led_state(sm) == ledd.BLUE


@pytest.mark.parametrize('field', ['seen', 'alive', 'valid'])
def test_overheating_requires_current_valid_state(field):
  sm = healthy_sm()
  sm['deviceState'].thermalStatus = log.DeviceState.ThermalStatus.critical
  getattr(sm, field)['deviceState'] = False
  assert ledd.led_state(sm) == ledd.BLUE


@pytest.mark.parametrize('event', ['wrongGear', 'wrongCarMode', 'pcmDisable', 'pedalPressed', 'reverseGear',
                                 'doorOpen', 'seatbeltNotLatched', 'driverUnresponsive1', 'startup', 'userBookmark'])
def test_normal_events_do_not_show_a_system_fault(event):
  sm = healthy_sm()
  sm.set('onroadEvents', [log.OnroadEvent.new_message(name=event)])
  assert ledd.led_state(sm) == ledd.BLUE


@pytest.mark.parametrize('status', ['uncalibrated', 'recalibrating'])
def test_orange_calibration_overrides_non_engageable(status):
  sm = healthy_sm()
  sm['extrinsicsCalibration'].calStatus = getattr(log.ExtrinsicsCalibration.Status, status)
  sm['selfdriveState'].engageable = False
  assert ledd.led_state(sm) == ledd.ORANGE


@pytest.mark.parametrize('active', [False, True])
def test_invalid_calibration_is_red_over_driver_monitoring_and_setup(active):
  sm = healthy_sm()
  sm['extrinsicsCalibration'].calStatus = log.ExtrinsicsCalibration.Status.invalid
  sm['selfdriveState'].active = active
  sm['driverMonitoringState'].alertLevel = log.DriverMonitoringState.AlertLevel.one
  assert ledd.led_state(sm, now=0., setup_complete=False) == ledd.RED
  assert ledd.led_state(sm, now=0.5, setup_complete=False) == (ledd.OFF if active else ledd.RED)
  assert ledd.automatic_led_channels(sm, brightness=127, now=0.) == ledd.camera_channels([[127, 0, 0]] * 6)


@pytest.mark.parametrize('service', ['extrinsicsCalibration', 'deviceState'])
@pytest.mark.parametrize('field', ['seen', 'alive', 'valid'])
def test_invalid_calibration_requires_current_valid_state(service, field):
  sm = healthy_sm()
  sm['extrinsicsCalibration'].calStatus = log.ExtrinsicsCalibration.Status.invalid
  getattr(sm, field)[service] = False
  assert ledd.led_state(sm) == ledd.BLUE


def test_invalid_calibration_is_ignored_offroad():
  sm = healthy_sm(started=False)
  sm['extrinsicsCalibration'].calStatus = log.ExtrinsicsCalibration.Status.invalid
  assert ledd.led_state(sm) == ledd.BLUE


def test_green_when_engaged():
  sm = healthy_sm()
  sm['selfdriveState'].active = True
  sm['selfdriveState'].state = log.SelfdriveState.OpenpilotState.enabled
  assert ledd.led_state(sm) == ledd.GREEN


def test_warning_sound_blinks_red_only_while_engaged():
  sm = healthy_sm()
  sm['selfdriveState'].alertSound = 'promptRepeat'
  assert ledd.led_state(sm, now=0.) == ledd.BLUE

  sm['selfdriveState'].active = True
  sm['selfdriveState'].state = log.SelfdriveState.OpenpilotState.enabled
  assert ledd.led_state(sm, now=0.) == ledd.RED
  assert ledd.led_state(sm, now=0.5) == ledd.OFF


@pytest.mark.parametrize('event', ['driverUnresponsive', 'driverDistracted'])
@pytest.mark.parametrize('stage', [1, 2, 3])
def test_driver_monitoring_blinks_orange_then_red_with_upstream_alerts(event, stage):
  sm = healthy_sm()
  alert = EVENTS[getattr(log.OnroadEvent.EventName, f'{event}{stage}')][ET.PERMANENT]
  sm['selfdriveState'].active = True
  sm['selfdriveState'].state = log.SelfdriveState.OpenpilotState.enabled
  sm['selfdriveState'].alertType = f'{event}{stage}/permanent'
  sm['selfdriveState'].alertSound = alert.audible_alert
  sm['selfdriveState'].alertStatus = alert.alert_status
  assert ledd.led_state(sm, now=0.) == (ledd.RED if stage == 3 else ledd.DM_WARNING)
  assert ledd.led_state(sm, now=0.5) == ledd.OFF
  green = 0 if stage == 3 else 3
  assert ledd.automatic_led_channels(sm, brightness=26, now=0.) == {
    1: [0] * 9,
    2: [26, green, 0, 0, 0, 0, 26, green, 0],
    3: [26, green, 0, 0, 0, 0, 26, green, 0],
  }
  sm['selfdriveState'].alertType = ''
  sm['selfdriveState'].alertSound = 'none'
  assert ledd.led_state(sm, now=0.) == ledd.GREEN


@pytest.mark.parametrize('stage,sound', [(1, 'none'), (2, 'promptDistracted')])
def test_driver_monitoring_warning_is_ignored_when_disengaged(stage, sound):
  sm = healthy_sm()
  sm['driverMonitoringState'].alertLevel = log.DriverMonitoringState.AlertLevel.one
  sm['selfdriveState'].alertType = f'driverUnresponsive{stage}/permanent'
  sm['selfdriveState'].alertSound = sound
  assert ledd.led_state(sm, now=0.) == ledd.BLUE
  assert ledd.led_state(sm, now=0.5) == ledd.BLUE


def test_soft_disabling_blinks_red_over_driver_monitoring_warning():
  sm = healthy_sm()
  sm['selfdriveState'].active = True
  sm['selfdriveState'].state = log.SelfdriveState.OpenpilotState.softDisabling
  sm['selfdriveState'].alertType = 'driverDistracted3/warning'
  assert ledd.led_state(sm, now=0.) == ledd.RED
  assert ledd.led_state(sm, now=0.5) == ledd.OFF


@pytest.mark.parametrize('status', ['uncalibrated', 'invalid'])
def test_calibration_priority_over_no_face_and_driver_warning(status):
  sm = healthy_sm()
  sm['extrinsicsCalibration'].calStatus = getattr(log.ExtrinsicsCalibration.Status, status)
  calibration_color = ledd.RED if status == 'invalid' else ledd.ORANGE
  sm['selfdriveState'].alertType = 'driverUnresponsive1/permanent'
  for face_detected in [False, True, False]:
    sm['driverMonitoringState'].visionPolicyState.faceDetected = face_detected
    sm['driverMonitoringState'].alertLevel = log.DriverMonitoringState.AlertLevel.one
    for now in [0., 0.5]:
      assert ledd.led_state(sm, now=now) == calibration_color

  sm['selfdriveState'].active = True
  assert ledd.led_state(sm, now=0.) == (ledd.RED if status == 'invalid' else ledd.DM_WARNING)
  sm['selfdriveState'].alertType = 'driverUnresponsive2/permanent'
  sm['selfdriveState'].alertSound = 'promptDistracted'
  assert ledd.led_state(sm, now=0., setup_complete=False) == (ledd.RED if status == 'invalid' else ledd.DM_WARNING)
  # A higher-priority blink stays off between flashes instead of revealing orange or setup magenta.
  assert ledd.led_state(sm, now=0.5, setup_complete=False) == ledd.OFF
  assert ledd.automatic_led_channels(sm, brightness=127, now=0.5) == ledd.camera_channels([[0, 0, 0]] * 6)


def test_persistent_process_failure_is_solid_red(monkeypatch):
  sm = healthy_sm()
  sm['managerState'].processes = [SimpleNamespace(name='camerad', shouldBeRunning=True, running=False)]
  monkeypatch.setattr(ledd, 'STARTED_AT', time.monotonic() - ledd.STARTUP_GRACE - 1.)
  assert ledd.led_state(sm, now=0.) == ledd.RED
  assert ledd.led_state(sm, now=0.5) == ledd.RED


def test_offroad_stays_blue_when_processes_are_intentionally_stopped(monkeypatch):
  sm = healthy_sm(started=False)
  sm['managerState'].processes = [SimpleNamespace(name='camerad', shouldBeRunning=False, running=False)]
  monkeypatch.setattr(ledd, 'STARTED_AT', time.monotonic() - ledd.STARTUP_GRACE - 1.)
  assert ledd.led_state(sm) == ledd.BLUE


def test_pairing_blinks_blue_without_driver_camera(monkeypatch):
  monkeypatch.setattr(ledd, 'read_confirmation', lambda **kwargs: None)
  monkeypatch.setattr(ledd, 'pairing_mode_active', lambda: True)
  monkeypatch.setattr(ledd.time, 'monotonic', lambda: 0.)
  channels = ledd.pairing_led_channels(26)
  assert channels == {
    1: [0] * 9,
    2: [0, 0, 26, 0, 0, 0, 0, 0, 26],
    3: [0, 0, 26, 0, 0, 0, 0, 0, 26],
  }

  monkeypatch.setattr(ledd.time, 'monotonic', lambda: 0.5)
  assert ledd.pairing_led_channels(26) == {1: [0] * 9, 2: [0] * 9, 3: [0] * 9}


def test_pairing_confirmation_uses_synchronized_amber_double_pulse(monkeypatch):
  monkeypatch.setattr(ledd, 'pairing_mode_active', lambda: True)
  monkeypatch.setattr(ledd, 'read_confirmation', lambda **kwargs: {'created': 10.})
  for now, lit in [(10., True), (10.3, False), (10.5, True), (10.9, False), (12., True)]:
    monkeypatch.setattr(ledd.time, 'monotonic', lambda now=now: now)
    channels = ledd.pairing_led_channels(26)
    assert channels[1] == [0] * 9
    assert channels[2] == channels[3]
    assert channels[2][3:6] == [0] * 3
    assert channels[2][:3] == ([26, 12, 0] if lit else [0, 0, 0])
    assert channels[2][6:] == channels[2][:3]
  monkeypatch.setattr(ledd, 'pairing_mode_active', lambda: False)
  assert ledd.pairing_led_channels(26) is None


def test_camera_brightness_uses_wide_road_exposure_stops():
  sm = FakeSubMaster()
  assert ledd.camera_led_brightness(sm) == 6

  sm.set('narrowRoadCameraState', SimpleNamespace(exposureValPercent=100.))
  assert ledd.camera_led_brightness(sm) == 6

  sm.set('wideRoadCameraState', SimpleNamespace(exposureValPercent=100.))
  assert ledd.camera_led_brightness(sm) == 6

  sm['wideRoadCameraState'].exposureValPercent = 0.
  assert ledd.camera_led_brightness(sm) == 127

  # The reported dark scene was 0.96749%, previously mapped to 124/255.
  for exposure in (.5, .96749, 1., 100.):
    sm['wideRoadCameraState'].exposureValPercent = exposure
    assert ledd.camera_led_brightness(sm) == 6
  sm['wideRoadCameraState'].exposureValPercent = .1
  assert 6 < ledd.camera_led_brightness(sm) < 127
  sm['wideRoadCameraState'].exposureValPercent = -.1
  assert ledd.camera_led_brightness(sm) == 6

  values = []
  for exposure in [i / 1000 for i in range(501)]:
    sm['wideRoadCameraState'].exposureValPercent = exposure
    values.append(ledd.camera_led_brightness(sm))
  assert values == sorted(values, reverse=True)
  assert all(6 <= value <= 127 for value in values)
  sm['wideRoadCameraState'].exposureValPercent = float('nan')
  assert ledd.camera_led_brightness(sm) == 6
  sm.valid['wideRoadCameraState'] = False
  assert ledd.camera_led_brightness(sm) == 6


def test_runtime_brightness_reaches_requested_peak():
  assert ledd.max_brightness(ledd.BLUE, 26) == ledd.LedState("blue", 0, 0, 26)
  assert ledd.max_brightness(ledd.ORANGE, 26) == ledd.LedState("orange", 26, 3, 0)


def test_status_colors_leave_leds_two_and_five_off(monkeypatch):
  for camera_num in ledd.CAM_LED_STATUS_CAMERAS:
    board = ledd.CameraLedBoard("test", camera_num=camera_num, bus_num=0)
    sent = []
    monkeypatch.setattr(board, "set_channels", sent.append)
    for state in (ledd.BLUE, ledd.GREEN, ledd.ORANGE, ledd.RED, ledd.DM_WARNING, ledd.OFF):
      board.set(state)
      rgb = [state.red, state.green, state.blue]
      assert sent[-1] == rgb + [0, 0, 0] + rgb


@pytest.mark.parametrize('kernel_leds', [False, True])
def test_direct_channel_writes_cannot_enable_middle_led(monkeypatch, tmp_path, kernel_leds):
  monkeypatch.setattr(ledd, 'CAM_LED_SYSFS_ROOT', tmp_path)
  board = ledd.CameraLedBoard("test", camera_num=2, bus_num=0, initialized=True, kernel_leds=kernel_leds)
  writes = {}
  monkeypatch.setattr(board, 'write', lambda register, value: writes.update({register: value}))
  if kernel_leds:
    for path in board.channel_paths:
      path.mkdir()
      (path / 'brightness').write_text('99')
  board.set_channels([255] * 9)
  actual = ([int((path / 'brightness').read_text()) for path in board.channel_paths] if kernel_leds else
            [writes[ledd.AW9523_DIM_BASE + i] for i in range(6)])
  assert actual == [255] * 6


def test_offroad_fault_is_red_and_engaged_fault_blinks(monkeypatch):
  monkeypatch.setattr(ledd, 'STARTED_AT', time.monotonic() - ledd.STARTUP_GRACE - 1.)
  sm = healthy_sm(started=False)
  sm['managerState'].processes = [SimpleNamespace(name='bluetoothd', shouldBeRunning=True, running=False)]
  assert ledd.led_state(sm, now=0.) == ledd.RED
  assert ledd.led_state(sm, now=0.5) == ledd.RED
  sm['deviceState'].started = True
  sm['selfdriveState'].active = True
  assert ledd.led_state(sm, now=0.) == ledd.RED
  assert ledd.led_state(sm, now=0.5) == ledd.OFF


@pytest.mark.parametrize('engaged', [False, True])
def test_missing_driver_without_audible_alert_does_not_change_led(engaged):
  sm = healthy_sm()
  sm['selfdriveState'].active = engaged
  sm['driverMonitoringState'].visionPolicyState.faceDetected = False
  assert ledd.led_state(sm, now=0.) == (ledd.GREEN if engaged else ledd.BLUE)
  assert ledd.led_state(sm, now=0.5) == (ledd.GREEN if engaged else ledd.BLUE)
  sm['managerState'].processes = [SimpleNamespace(name='dmonitoringmodeld', shouldBeRunning=False, running=False)]
  assert ledd.led_state(sm, now=0.) == (ledd.GREEN if engaged else ledd.BLUE)
  sm['deviceState'].started = False
  assert ledd.led_state(sm, now=0.) == ledd.BLUE


def test_stale_driver_state_is_not_a_driver_warning():
  sm = healthy_sm()
  sm['driverMonitoringState'].visionPolicyState.faceDetected = False
  sm.valid['driverMonitoringState'] = False
  assert ledd.led_state(sm) == ledd.BLUE


@pytest.mark.parametrize('state', [ledd.BLUE, ledd.GREEN, ledd.ORANGE, ledd.RED, ledd.DM_WARNING])
def test_automatic_states_cannot_exceed_brightness_cap(state):
  limited = ledd.max_brightness(state, 255)
  assert max(limited.red, limited.green, limited.blue) == 127


def test_pending_setup_is_solid_magenta_with_fault_priority(monkeypatch):
  sm = healthy_sm(started=False)
  assert ledd.led_state(sm, now=0., setup_complete=False) == ledd.SETUP_PENDING
  assert ledd.led_state(sm, now=0.5, setup_complete=False) == ledd.SETUP_PENDING
  assert ledd.led_state(sm, setup_complete=True) == ledd.BLUE
  monkeypatch.setattr(ledd, 'persistent_error', lambda sm: True)
  assert ledd.led_state(sm, setup_complete=False) == ledd.RED


def test_aw9523_initialization_keeps_ir_off(monkeypatch):
  board = ledd.CameraLedBoard("test", camera_num=2, bus_num=0)
  writes = []
  bus = SimpleNamespace(read_byte_data=lambda addr, register: 0x23)
  def connect():
    board.bus = bus
    return bus
  monkeypatch.setattr(board, 'connect', connect)
  monkeypatch.setattr(board, 'write', lambda register, value: writes.append((register, value)))
  board.set(ledd.BLUE)
  assert writes[:2] == [(4, 255), (5, 255)]
  assert writes[2:18] == [(register, 0) for register in range(0x20, 0x30)]
  assert writes[18:21] == [(0x11, 2), (0x12, 0), (0x13, 0)]
  assert writes[21:] == [(0x20 + i, v) for i, v in enumerate([0, 0, 180, 0, 0, 180])]
  assert not any(value for register, value in writes if 0x26 <= register <= 0x2f)


def test_aw9523_rejects_wrong_chip_before_writes(monkeypatch):
  board = ledd.CameraLedBoard("test", camera_num=2, bus_num=0)
  board.bus = SimpleNamespace(read_byte_data=lambda addr, register: 0)
  monkeypatch.setattr(board, 'connect', lambda: board.bus)
  writes = []
  monkeypatch.setattr(board, 'write', lambda *args: writes.append(args))
  with pytest.raises(OSError, match='AW9523'):
    board.init()
  assert not writes
