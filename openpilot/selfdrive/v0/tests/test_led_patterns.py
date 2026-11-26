import pytest

from openpilot.selfdrive.v0 import ledd
from openpilot.selfdrive.v0.led_patterns import STARTUP_PERIOD, calibration_channels, startup_channels, startup_levels
from openpilot.selfdrive.v0.tests.test_ledd import healthy_sm
from openpilot.cereal import log


@pytest.mark.parametrize(('percent', 'road', 'wide'), [
  (0, [0] * 12, [0] * 9 + [127, 127 * 32 / 255, 0]),
  (25, [0] * 12, [0] * 6 + [127, 127 * 32 / 255, 0] * 2),
  (50, [0] * 12, [127, 127 * 32 / 255, 0] * 4),
  (75, [0] * 6 + [127, 127 * 32 / 255, 0] * 2, [127, 127 * 32 / 255, 0] * 4),
  (100, [127, 127 * 32 / 255, 0] * 4, [127, 127 * 32 / 255, 0] * 4),
])
def test_calibration_fills_in_physical_order(percent, road, wide):
  assert calibration_channels(percent, 255) == {1: [0] * 12, 2: road, 3: wide}


def test_calibration_partial_progress_and_zero_brightness():
  assert calibration_channels(12.5, 255)[3][-3:] == pytest.approx([127, 127 * 32 / 255, 0])
  assert calibration_channels(43.75, 255)[3][:3] == pytest.approx([63.5, 63.5 * 32 / 255, 0])
  assert calibration_channels(0, 26)[3][-3:] == pytest.approx([26, 26 * 32 / 255, 0])
  assert calibration_channels(0, 0) == {1: [0] * 12, 2: [0] * 12, 3: [0] * 12}
  assert calibration_channels(100, 0) == {1: [0] * 12, 2: [0] * 12, 3: [0] * 12}


def test_startup_comet_crosses_both_windows_and_returns_without_a_flash():
  frames = [startup_levels(frame / 60.) for frame in range(265)]
  assert max(range(8), key=frames[0].__getitem__) == 0
  assert max(range(8), key=frames[132].__getitem__) == 7
  assert frames[-1] == frames[0]
  assert {max(range(8), key=row.__getitem__) for row in frames} == set(range(8))
  assert max(abs(b - a) for before, after in zip(frames[:-1], frames[1:], strict=True)
             for a, b in zip(before, after, strict=True)) < .1
  for frame in range(264):
    elapsed = frame / 60.
    channels = startup_channels(elapsed)
    for camera, values in channels.items():
      assert values == pytest.approx(startup_channels(elapsed + STARTUP_PERIOD)[camera])
    assert channels[1] == [0] * 12
    for camera in (2, 3):
      assert all(0 <= value <= 5 for value in channels[camera][2::3])
      assert channels[camera][0::3] == channels[camera][1::3] == [0] * 4
  assert startup_channels(0.)[3][-1] == 5
  assert startup_channels(STARTUP_PERIOD / 2)[2][2] == 5
  assert startup_channels(0.)[2] == [0] * 12
  assert startup_channels(1.2, 0) == {1: [0] * 12, 2: [0] * 12, 3: [0] * 12}


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
  assert ledd.led_state(sm, 101.) == ledd.BLUE


def test_stale_calibration_and_safety_alerts_do_not_render_progress(monkeypatch):
  ledd.log = log
  monkeypatch.setattr(ledd, 'STARTED_AT', 0.)
  sm = healthy_sm()
  sm['extrinsicsCalibration'].calStatus = log.ExtrinsicsCalibration.Status.uncalibrated
  sm.valid['extrinsicsCalibration'] = False
  assert ledd.automatic_led_channels(sm, 255, 100.) is None
  sm.valid['extrinsicsCalibration'] = True
  sm['selfdriveState'].active = True
  sm['selfdriveState'].alertSound = 'warningImmediateSystem'
  assert ledd.automatic_led_channels(sm, 255, 100.) == {
    1: [0] * 12, 2: [127, 0, 0] * 4, 3: [127, 0, 0] * 4,
  }
  assert ledd.automatic_led_channels(sm, 255, 100.5) == {1: [0] * 12, 2: [0] * 12, 3: [0] * 12}
  assert ledd.led_state(sm, 100.) == ledd.RED


def test_startup_comet_keeps_boot_phase_only_while_parked(monkeypatch):
  ledd.log = log
  monkeypatch.setattr(ledd, 'STARTED_AT', 100.)
  sm = healthy_sm(started=False)
  assert ledd.automatic_led_channels(sm, 127, 101.) == startup_channels(101.)
  assert ledd.automatic_led_channels(sm, 26, 104.3) == startup_channels(104.3)
  assert ledd.automatic_led_channels(sm, 26, 104.5) is None
  sm['deviceState'].started = True
  assert ledd.automatic_led_channels(sm, 26, 101.) is None
