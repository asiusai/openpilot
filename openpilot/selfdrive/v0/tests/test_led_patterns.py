import pytest

from openpilot.selfdrive.v0 import ledd
from openpilot.selfdrive.v0.led_patterns import calibration_channels, startup_channels, startup_levels
from openpilot.selfdrive.v0.tests.test_ledd import healthy_sm
from openpilot.cereal import log


@pytest.mark.parametrize(('percent', 'road', 'wide'), [
  (0, [0] * 9, [0, 0, 0, 0, 0, 0, 255, 32, 0]),
  (25, [0] * 9, [0, 0, 0, 13, 2, 0, 255, 32, 0]),
  (50, [0] * 9, [255, 32, 0, 26, 3, 0, 255, 32, 0]),
  (75, [0, 0, 0, 13, 2, 0, 255, 32, 0], [255, 32, 0, 26, 3, 0, 255, 32, 0]),
  (100, [255, 32, 0, 26, 3, 0, 255, 32, 0], [255, 32, 0, 26, 3, 0, 255, 32, 0]),
])
def test_calibration_fills_in_physical_order(percent, road, wide):
  assert calibration_channels(percent, 255) == {1: [0] * 9, 2: road, 3: wide}


def test_calibration_partial_progress_and_zero_brightness():
  assert calibration_channels(12.5, 255)[3][-3:] == [255, 32, 0]
  assert calibration_channels(37.5, 255)[3][:3] == [128, 16, 0]
  assert calibration_channels(0, 13)[3][-3:] == [13, 2, 0]
  assert calibration_channels(0, 0) == {1: [0] * 9, 2: [0] * 9, 3: [0] * 9}
  assert calibration_channels(100, 0) == {1: [0] * 9, 2: [0] * 9, 3: [0] * 9}


def test_startup_fades_together_smoothly_with_dim_centers():
  assert startup_levels(0.) == [0.] * 6
  assert startup_levels(1.5) == [1.] * 6
  rising = [startup_levels(frame / 30.)[0] for frame in range(46)]
  falling = [startup_levels(frame / 30.)[0] for frame in range(45, 91)]
  assert rising == sorted(rising)
  assert falling == sorted(falling, reverse=True)
  assert max(b - a for a, b in zip(rising[:-1], rising[1:], strict=True)) < 0.04
  assert startup_channels(1.5)[3][3:6] == [26, 20, 12]
  for frame in range(90):
    elapsed = frame / 30.
    channels = startup_channels(elapsed)
    assert len(set(startup_levels(elapsed))) == 1
    assert channels == startup_channels(elapsed + 3.)
    assert channels[1] == [0] * 9
    assert channels[2] == channels[3]
    assert channels[2][:3] == channels[2][6:]
    assert all(value <= 26 for camera in (2, 3) for value in channels[camera][3:6])


@pytest.mark.parametrize('percent', [0, 25, 100])
def test_live_calibration_uses_percentage_and_returns_to_normal_without_success_flash(monkeypatch, percent):
  ledd.log = log
  monkeypatch.setattr(ledd, 'STARTED_AT', 0.)
  sm = healthy_sm()
  sm['extrinsicsCalibration'].calStatus = log.ExtrinsicsCalibration.Status.uncalibrated
  sm['extrinsicsCalibration'].calPerc = percent
  assert ledd.automatic_led_channels(sm, 255, 100.) == calibration_channels(percent, 255)
  sm['extrinsicsCalibration'].calStatus = log.ExtrinsicsCalibration.Status.calibrated
  assert ledd.automatic_led_channels(sm, 255, 101.) is None
  assert ledd.led_state(sm, 101.) == ledd.WHITE


def test_stale_calibration_and_safety_alerts_do_not_render_progress(monkeypatch):
  ledd.log = log
  monkeypatch.setattr(ledd, 'STARTED_AT', 0.)
  sm = healthy_sm()
  sm['extrinsicsCalibration'].calStatus = log.ExtrinsicsCalibration.Status.uncalibrated
  sm.valid['extrinsicsCalibration'] = False
  assert ledd.automatic_led_channels(sm, 255, 100.) is None
  sm.valid['extrinsicsCalibration'] = True
  sm['selfdriveState'].active = True
  sm['selfdriveState'].alertSound.raw = 'warningImmediate'
  assert ledd.automatic_led_channels(sm, 255, 100.) == {
    1: [0] * 9, 2: [255, 0, 0, 26, 0, 0, 255, 0, 0], 3: [255, 0, 0, 26, 0, 0, 255, 0, 0],
  }
  assert ledd.automatic_led_channels(sm, 255, 100.5) == {1: [0] * 9, 2: [0] * 9, 3: [0] * 9}
  assert ledd.led_state(sm, 100.) == ledd.RED


def test_startup_fade_only_while_parked(monkeypatch):
  ledd.log = log
  monkeypatch.setattr(ledd, 'STARTED_AT', 100.)
  sm = healthy_sm(started=False)
  assert ledd.automatic_led_channels(sm, 26, 101.) == startup_channels(101.)
  assert ledd.automatic_led_channels(sm, 26, 104.) is None
  sm['deviceState'].started = True
  assert ledd.automatic_led_channels(sm, 26, 101.) is None
