import threading
import numpy as np

from openpilot.common.test import OpenpilotTestCase
from openpilot.cereal import log, messaging
from openpilot.cereal.messaging import SubMaster, PubMaster
from openpilot.selfdrive.ui.soundd import SELFDRIVE_STATE_TIMEOUT, check_selfdrive_timeout_alert
from openpilot.selfdrive.ui import soundd

AudibleAlert = log.SelfdriveState.AudibleAlert


class TestSoundd(OpenpilotTestCase):
  def test_check_selfdrive_timeout_alert(self, mocker):
    sm = SubMaster(['selfdriveState'])
    pm = PubMaster(['selfdriveState'])

    cs = messaging.new_message('selfdriveState')
    cs.selfdriveState.enabled = True
    threading.Timer(0.01, pm.send, args=("selfdriveState", cs)).start()
    sm.update(100)
    assert sm.updated['selfdriveState']

    sm.recv_time['selfdriveState'] = 0
    clock = mocker.patch("openpilot.selfdrive.ui.soundd.time.monotonic", return_value=SELFDRIVE_STATE_TIMEOUT)
    assert not check_selfdrive_timeout_alert(sm)

    clock.return_value = SELFDRIVE_STATE_TIMEOUT + 0.1
    assert check_selfdrive_timeout_alert(sm)

    clock.return_value = SELFDRIVE_STATE_TIMEOUT + 10
    assert not check_selfdrive_timeout_alert(sm)

  # TODO: add test with micd for checking that soundd actually outputs sounds

  def test_v0_volume_curve(self, mocker):
    device = mocker.patch.object(soundd.HARDWARE, 'get_device_type', return_value='mici')
    comma = soundd.Soundd()
    levels = [0, 26, 35, 41, 50, 56, 100]
    original = [comma.calculate_volume(db) for db in levels]
    assert comma.current_volume == soundd.MIN_VOLUME
    np.testing.assert_allclose(original, [20 ** -.9, 20 ** -.78, 20 ** -.51, 20 ** -.33, 20 ** -.06, 1., 1.])

    device.return_value = 'v0'
    asius = soundd.Soundd()
    louder = [asius.calculate_volume(db) for db in levels]
    np.testing.assert_allclose(louder, np.minimum(2 * np.array(original), 1.))
    assert asius.current_volume == louder[0]
    assert all(a <= b for a, b in zip(louder, louder[1:], strict=False))

  def test_v0_sound_headroom(self, mocker):
    device = mocker.patch.object(soundd.HARDWARE, 'get_device_type', return_value='mici')
    comma = soundd.Soundd()
    device.return_value = 'v0'
    asius = soundd.Soundd()
    for alert, samples in asius.loaded_sounds.items():
      assert np.max(np.abs(samples)) <= 1.
      assert samples.shape == comma.loaded_sounds[alert].shape
      assert samples.dtype == np.float32
      assert np.linalg.norm(samples) >= np.linalg.norm(comma.loaded_sounds[alert])
      assert np.all(samples[comma.loaded_sounds[alert] == 0] == 0)
    for alert in (AudibleAlert.engage, AudibleAlert.disengage, AudibleAlert.prompt):
      assert np.linalg.norm(asius.loaded_sounds[alert]) > 2.5 * np.linalg.norm(comma.loaded_sounds[alert])
    for alert in (AudibleAlert.warningSoft, AudibleAlert.warningImmediate, soundd.CRITICAL_MAX):
      np.testing.assert_array_equal(asius.loaded_sounds[alert], comma.loaded_sounds[alert])

    device.return_value = 'mici'
    for alert, samples in soundd.Soundd().loaded_sounds.items():
      np.testing.assert_array_equal(samples, comma.loaded_sounds[alert])
