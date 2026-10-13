import json
from unittest.mock import Mock

import numpy as np
import pytest

from openpilot.cereal import log
from openpilot.common.params import Params
from openpilot.selfdrive.selfdrived.events import EVENTS, ET, EventName, Events
from openpilot.selfdrive.selfdrived.state import StateMachine
from openpilot.selfdrive.ui import soundd, voice_alerts
from openpilot.selfdrive.ui.voice_alerts import VoiceAlerts, alert_voice
from openpilot.system.manager.process_config import sound


def state(event="driverDistracted1", size="small", sound="none", status="normal", text="", event_type="permanent"):
  return log.SelfdriveState.new_message(alertType=f"{event}/{event_type}", alertSize=size, alertSound=sound, alertStatus=status, alertText2=text)


@pytest.fixture
def speech(tmp_path):
  params = Params(str(tmp_path))
  params.put_bool("VoiceAlertsEnabled", True, block=True)
  return VoiceAlerts(params)


def finish(speech):
  if speech.playing:
    speech.mix(np.zeros(len(speech.current), dtype=np.float32), 1., False)


def test_manifest_bindings_and_bundled_audio(speech):
  speech.load()
  assert speech.loaded.keys() == voice_alerts.MESSAGES.keys()
  assert len(voice_alerts.EVENT_VOICES) == sum(len(v["events"]) for v in voice_alerts.MESSAGES.values())
  assert voice_alerts.EVENT_VOICES.keys() <= EventName.schema.enumerants.keys()
  assert len(voice_alerts.NO_ENTRY_VOICES) == sum(len(v["no_entry"]) for v in voice_alerts.MESSAGES.values())
  assert not voice_alerts.EVENT_VOICES.keys() & voice_alerts.NO_ENTRY_VOICES.keys()
  for event in voice_alerts.NO_ENTRY_VOICES:
    assert ET.NO_ENTRY in EVENTS[getattr(EventName, event)]
  alerts = json.loads((voice_alerts.VOICE_DIR.parents[2] / "selfdrived/alerts_offroad.json").read_text())
  assert voice_alerts.OFFROAD_VOICES.keys() <= alerts.keys()
  for param in voice_alerts.OFFROAD_VOICES:
    speech.params.check_key(param)
  for samples in speech.loaded.values():
    assert samples.dtype == np.float32
    assert np.isfinite(samples).all()
    assert 0.1 < np.max(np.abs(samples)) <= .851


def test_toggle_defaults_off(tmp_path):
  params = Params(str(tmp_path))
  assert not params.get_bool("VoiceAlertsEnabled")
  assert params.get_default_value("VoiceAlertsEnabled") is False
  speech = VoiceAlerts(params)
  speech.update(state(), True, False, 100.)
  assert not speech.playing and not speech.load_attempted


def test_once_per_active_message_and_retrigger_delay(speech):
  speech.update(state(), True, False, 100.)
  assert speech.playing
  finish(speech)
  speech.update(state("driverDistracted2"), True, False, 200.)
  assert not speech.playing  # The next attention stage uses the same phrase.
  speech.update(None, True, False, 201.)
  speech.update(state(), True, False, 202.)
  assert speech.playing
  finish(speech)
  speech.update(None, True, False, 203.)
  speech.update(state(), True, False, 204.)
  assert not speech.playing


def test_short_alert_finishes_but_cleared_pending_alert_does_not_play(speech):
  speech.update(state(), True, False, 100.)
  speech.update(state("cameraMalfunction"), True, False, 100.1)
  assert speech.playing
  speech.update(None, True, False, 100.2)
  finish(speech)
  speech.update(None, True, False, 110.)
  assert not speech.playing
  assert "camera_error" not in speech.last_played


def test_disable_and_critical_stop_speech(speech):
  speech.update(state(), True, False, 100.)
  speech.update(state(), True, True, 100.1)
  assert not speech.playing
  speech.update(state("cameraMalfunction"), True, False, 110.)
  assert speech.playing
  speech.params.put_bool("VoiceAlertsEnabled", False, block=True)
  speech.update(state("cameraMalfunction"), True, False, 111.)
  assert not speech.playing


def test_offroad_alerts_and_no_stale_queue(speech):
  speech.params.put("Offroad_ChestnutNotDetected", {"text": "missing"}, block=True)
  speech.update(None, False, False, 100.)
  assert "chestnut_error" in speech.last_played
  speech.params.put("Offroad_TemperatureTooHigh", {"text": "hot"}, block=True)
  speech.update(None, False, False, 101.)
  speech.params.remove("Offroad_TemperatureTooHigh")
  finish(speech)
  speech.update(None, False, False, 110.)
  assert not speech.playing
  assert "overheat" not in speech.last_played
  speech.update(None, True, False, 140.)
  assert not speech.playing


def test_missing_clip_does_not_stop_standard_audio(speech, tmp_path, monkeypatch):
  monkeypatch.setattr(voice_alerts, "VOICE_DIR", tmp_path)
  monkeypatch.setattr(voice_alerts.cloudlog, "exception", Mock())
  speech.update(state(), True, False, 100.)
  assert not speech.playing
  alert = np.linspace(-1, 1, 100, dtype=np.float32)
  np.testing.assert_array_equal(speech.mix(alert, 1., False), alert)


@pytest.mark.parametrize("event,size,event_type,expected", [
  ("wrongCarMode", "none", "userDisable", None),
  ("wrongCarMode", "mid", "noEntry", "cannot_engage"),
  ("wrongCarMode", "mid", "permanent", None),
  ("pedalPressed", "none", "userDisable", None),
  ("pedalPressed", "mid", "noEntry", "pedal_pressed"),
  ("pcmDisable", "none", "userDisable", None),
  ("steerDisengage", "none", "userDisable", None),
  ("fanMalfunction", "mid", "userDisable", None),
  ("fanMalfunction", "mid", "permanent", "system_error"),
  ("commIssue", "mid", "noEntry", "system_error"),
  ("driverUnresponsive1", "small", "permanent", "touch_wheel"),
  ("driverUnresponsive2", "mid", "permanent", "touch_wheel"),
  ("preEnableStandstill", "small", "preEnable", None),
  ("resumeRequired", "small", "warning", None),
  ("brakeHold", "small", "warning", None),
  ("preLaneChangeLeft", "mid", "warning", None),
  ("tooDistracted", "mid", "noEntry", "cannot_engage"),
  ("notAnEvent", "small", "permanent", None),
])
def test_screen_alert_selection(event, size, event_type, expected):
  assert alert_voice(state(event, size, event_type=event_type)) == expected


@pytest.mark.parametrize("initial,attempt,expected", [
  ("disabled", False, set()),  # Holding a pedal without trying to engage.
  ("disabled", True, {"pedal_pressed"}),  # Refused engagement attempt.
  ("enabled", False, set()),  # Normal brake disengagement.
])
def test_pedal_speech_follows_actual_engagement_state(initial, attempt, expected):
  machine = StateMachine()
  machine.state = getattr(log.SelfdriveState.OpenpilotState, initial)
  events = Events()
  events.add(EventName.pedalPressed)
  if attempt:
    events.add(EventName.buttonEnable)
  machine.update(events)
  alerts = events.create_alerts(machine.current_alert_types)
  voices = {alert_voice(log.SelfdriveState.new_message(alertType=a.alert_type, alertSize=a.alert_size)) for a in alerts}
  assert voices - {None} == expected


def test_readiness_messages_require_no_entry():
  for event, key in voice_alerts.NO_ENTRY_VOICES.items():
    assert alert_voice(state(event, event_type=ET.NO_ENTRY)) == key
    for event_type in (ET.PERMANENT, ET.WARNING, ET.USER_DISABLE, ET.SOFT_DISABLE, ET.PRE_ENABLE):
      assert alert_voice(state(event, event_type=event_type)) is None


def test_update_and_internet_errors_do_not_trigger_speech(speech):
  for key in ('Offroad_ConnectivityNeeded', 'Offroad_ConnectivityNeededPrompt', 'Offroad_UpdateFailed'):
    speech.params.put(key, {'text': 'Cannot reach update server'}, block=True)
  speech.params.put('LastUpdateException', 'Could not resolve host: github.com', block=True)
  speech.params.put('UpdateFailedCount', 5, block=True)
  speech.update(None, False, False, 100.)
  assert not speech.playing
  assert not speech.last_played


def test_grouped_chestnut_faults_do_not_repeat(speech):
  speech.params.put("Offroad_ChestnutNotDetected", {"text": "missing"}, block=True)
  speech.update(None, False, False, 100.)
  finish(speech)
  speech.params.put("Offroad_ChestnutUsbSlow", {"text": "slow"}, block=True)
  speech.update(None, False, False, 140.)
  assert not speech.playing
  assert speech.last_played == {"chestnut_error": 100.}


@pytest.mark.parametrize("text,expected", [("modeld", "model_error"), ("camerad, modeld", "camera_error"),
                                         ("dmonitoringmodeld", "system_error"), ("mysteryd", "system_error")])
def test_fixed_process_messages(text, expected):
  assert alert_voice(state("processNotRunning", text=text)) == expected


def test_mixing_headroom_and_siren_priority(speech):
  speech.update(state(), True, False, 100.)
  speech.current = np.full(10, .851, dtype=np.float32)
  mixed = speech.mix(np.ones(10, dtype=np.float32), 1., False)
  assert np.max(mixed) <= 1.
  speech.update(state("cameraMalfunction"), True, False, 110.)
  original = np.linspace(-1., 1., 4096, dtype=np.float32)
  np.testing.assert_array_equal(speech.mix(original, 1., True), original)
  assert not speech.playing


@pytest.mark.parametrize("device,expected", [("v0", .5), ("mici", .05)])
def test_voice_volume_preserves_comma_volume(device, expected, tmp_path, monkeypatch):
  params = Params(str(tmp_path))
  monkeypatch.setattr(soundd, "Params", lambda: params)
  monkeypatch.setattr(soundd.HARDWARE, "get_device_type", lambda: device)
  daemon = soundd.Soundd()
  daemon.current_volume = .1
  daemon.voice.enabled = True
  daemon.voice.current = np.full(10, .5, dtype=np.float32)
  np.testing.assert_allclose(daemon.get_sound_data(10), expected)
  daemon.voice.current = np.full(10, .5, dtype=np.float32)
  daemon.voice.frame = 0
  daemon.update_alert(soundd.AudibleAlert.warningImmediate)
  expected_siren = daemon.loaded_sounds[soundd.AudibleAlert.warningImmediate][:10] * .1
  np.testing.assert_array_equal(daemon.get_sound_data(10), expected_siren)
  assert not daemon.voice.playing
  # Completing the last siren loop must not mix speech into its final buffer.
  samples = daemon.loaded_sounds[soundd.AudibleAlert.warningImmediate]
  daemon.current_sound_frame = len(samples) - 10
  daemon.pending_stop = True
  daemon.voice.current = np.full(10, .5, dtype=np.float32)
  daemon.voice.frame = 0
  np.testing.assert_array_equal(daemon.get_sound_data(10), samples[-10:] * .1)
  assert not daemon.voice.playing


def test_offroad_process_only_runs_for_enabled_speech(speech):
  assert sound(False, speech.params, None)
  speech.params.put_bool("VoiceAlertsEnabled", False, block=True)
  assert not sound(False, speech.params, None)
  assert sound(True, speech.params, None)
