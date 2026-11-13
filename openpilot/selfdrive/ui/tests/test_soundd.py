import threading
from types import SimpleNamespace
import numpy as np

from openpilot.common.test import OpenpilotTestCase
from openpilot.cereal import log, messaging
from openpilot.cereal.messaging import SubMaster, PubMaster
from openpilot.selfdrive.ui.soundd import SELFDRIVE_STATE_TIMEOUT, check_selfdrive_timeout_alert
from openpilot.selfdrive.ui import soundd
from openpilot.common.params import Params


class PreviewState(dict):
  def __init__(self):
    super().__init__(deviceState=SimpleNamespace(started=False), pandaStates=[SimpleNamespace(
      pandaType=log.PandaState.PandaType.cuatro, ignitionLine=False, ignitionCan=False, controlsAllowed=False)])
    self.logMonoTime = {'deviceState': 100 * 10**9, 'pandaStates': 100 * 10**9}

  def all_checks(self, services):
    return True

AudibleAlert = log.SelfdriveState.AudibleAlert


class TestSoundd(OpenpilotTestCase):
  def test_saved_volume_and_preview_once(self):
    from unittest.mock import patch
    device = patch.object(soundd.HARDWARE, 'get_device_type', return_value='v0')
    device.start()
    self.addCleanup(device.stop)
    for volume in (10, 37, 80, 100):
      with self.subTest(volume=volume):
        params = Params()
        params.put('SoundVolume', volume, block=True)
        params.put('SoundVolumePreview', {'id': 'preview', 'expires': 105}, block=True)
        daemon = soundd.Soundd()
        assert daemon.current_volume == volume / 100
        assert daemon.calculate_volume(90) == volume / 100
        sm = PreviewState()
        daemon.update_preview(sm, 100)
        samples = daemon.loaded_sounds[AudibleAlert.engage]
        np.testing.assert_array_equal(daemon.get_sound_data(len(samples)), samples * (volume / 100))
        daemon.update_preview(sm, 100.1)
        assert not np.any(daemon.get_sound_data(len(samples)))

  def test_preview_cancels_on_real_activity(self):
    from unittest.mock import patch
    device = patch.object(soundd.HARDWARE, 'get_device_type', return_value='v0')
    device.start()
    self.addCleanup(device.stop)
    for reason in ('ignition', 'controls', 'started', 'stale', 'expired', 'critical', 'pairing'):
      with self.subTest(reason=reason):
        daemon = soundd.Soundd()
        daemon.params.put('SoundVolumePreview', {'id': 'preview', 'expires': 105}, block=True)
        sm = PreviewState()
        daemon.update_preview(sm, 100)
        assert daemon.preview_frame == 0
        if reason == 'ignition': sm['pandaStates'][0].ignitionLine = True
        if reason == 'controls': sm['pandaStates'][0].controlsAllowed = True
        if reason == 'started': sm['deviceState'].started = True
        if reason == 'stale': sm.logMonoTime['pandaStates'] = 0
        if reason == 'expired': daemon.params.remove('SoundVolumePreview')
        if reason == 'critical': daemon.update_alert(AudibleAlert.warningImmediate)
        if reason == 'pairing':
          daemon.pairing_speech.clip = np.full(10, .5, dtype=np.float32)
          daemon.pairing_speech.offset = 0
        daemon.update_preview(sm, 100.1)
        data = daemon.get_sound_data(10)
        assert daemon.preview_frame is None
        if reason == 'pairing': np.testing.assert_allclose(data, .4)
        elif reason == 'critical': np.testing.assert_array_equal(data, daemon.loaded_sounds[AudibleAlert.warningImmediate][:10] * .8)
        else: assert not np.any(data)

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
    np.testing.assert_array_equal([asius.calculate_volume(db) for db in levels], [.8] * len(levels))
    assert asius.current_volume == .8

  def test_v0_all_alerts_use_selected_gain(self, mocker):
    mocker.patch.object(soundd.HARDWARE, 'get_device_type', return_value='v0')
    for alert in soundd.sound_list:
      if alert == soundd.CRITICAL_MAX:
        continue  # Selected by the existing critical escalation, not an alert event.
      daemon = soundd.Soundd()
      daemon.update_alert(alert)
      samples = daemon.loaded_sounds[alert]
      np.testing.assert_array_equal(daemon.get_sound_data(len(samples)), samples * .8)

  def test_v0_pairing_uses_selected_gain(self, mocker):
    mocker.patch.object(soundd.HARDWARE, 'get_device_type', return_value='v0')
    daemon = soundd.Soundd()
    daemon.pairing_speech.clip = np.full(10, .5, dtype=np.float32)
    daemon.pairing_speech.offset = 0
    np.testing.assert_allclose(daemon.get_sound_data(10), .4)

  def test_v0_sound_headroom(self, mocker):
    device = mocker.patch.object(soundd.HARDWARE, 'get_device_type', return_value='mici')
    comma = soundd.Soundd()
    device.return_value = 'v0'
    asius = soundd.Soundd()
    for alert, samples in asius.loaded_sounds.items():
      assert np.max(np.abs(samples)) <= 1.
      assert samples.shape == comma.loaded_sounds[alert].shape
      assert samples.dtype == np.float32
      np.testing.assert_array_equal(samples, comma.loaded_sounds[alert])
      assert np.all(samples[comma.loaded_sounds[alert] == 0] == 0)
    for alert in (AudibleAlert.warningSoft, AudibleAlert.warningImmediate, soundd.CRITICAL_MAX):
      np.testing.assert_array_equal(asius.loaded_sounds[alert], comma.loaded_sounds[alert])

    device.return_value = 'mici'
    for alert, samples in soundd.Soundd().loaded_sounds.items():
      np.testing.assert_array_equal(samples, comma.loaded_sounds[alert])
