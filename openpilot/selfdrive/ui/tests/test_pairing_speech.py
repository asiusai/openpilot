import json
from pathlib import Path

import numpy as np
import pytest

from openpilot.selfdrive.ui import pairing_speech
from openpilot.system.asius.pairing_confirmation import CONFIRM_SECONDS


@pytest.mark.parametrize('button_confirmation', [True, False])
def test_all_digits_are_bundled_and_spoken_in_order_including_zeros(monkeypatch, button_confirmation):
  monkeypatch.setattr(pairing_speech, 'read_announcement', lambda **kwargs: None)
  speaker = pairing_speech.PairingSpeech(button_confirmation=button_confirmation)
  assert len(speaker.digits) == 10
  assert all(len(samples) > 1000 and 0 < np.max(np.abs(samples)) <= .86 for samples in speaker.digits.values())
  status = {'id': 'first', 'digits': '000042'}
  monkeypatch.setattr(pairing_speech, 'read_confirmation', lambda **kwargs: status)
  speaker.update(False, False, 10.)
  confirm = 'confirm_button' if button_confirmation else 'confirm_screen'
  expected = np.concatenate([speaker.instructions['compare'], *[speaker.digits[digit] for digit in status['digits']],
                             speaker.instructions[confirm]])
  output = speaker.mix(np.zeros(len(expected) + 10, dtype=np.float32), .5, False)
  np.testing.assert_allclose(output[:len(expected)], expected * .5)
  assert not speaker.playing
  speaker.update(False, False, 11.)
  assert not speaker.playing  # A status refresh must not repeat the number.
  status['id'] = 'second'
  speaker.update(False, False, 12.)
  assert speaker.playing
  speaker.mix(np.zeros(100, dtype=np.float32), 1., True)
  assert not speaker.playing


def test_cancel_or_ignition_stops_comparison_speech(monkeypatch):
  monkeypatch.setattr(pairing_speech, 'read_announcement', lambda **kwargs: None)
  status = {'id': 'first', 'digits': '123456'}
  monkeypatch.setattr(pairing_speech, 'read_confirmation', lambda **kwargs: status)
  speaker = pairing_speech.PairingSpeech()
  speaker.update(False, False, 10.)
  assert speaker.playing
  speaker.update(True, False, 11.)
  assert not speaker.playing
  status = {'id': 'second', 'digits': '654321'}
  speaker.update(False, False, 12.)
  assert speaker.playing
  status = None
  speaker.update(False, False, 13.)
  assert not speaker.playing


def test_pairing_outcomes_play_once_after_comparison_and_stop_on_ignition(monkeypatch):
  status = None
  announcement = {'id': 'start', 'outcome': 'started'}
  monkeypatch.setattr(pairing_speech, 'read_confirmation', lambda **kwargs: status)
  monkeypatch.setattr(pairing_speech, 'read_announcement', lambda **kwargs: announcement)
  speaker = pairing_speech.PairingSpeech()
  assert set(speaker.outcomes) == {'started', 'success', 'failed', 'ignition_on'}
  speaker.update(False, False, 1.)
  assert speaker.playing
  status = {'id': 'comparison', 'digits': '000042'}
  speaker.update(False, False, 2.)
  assert speaker.clip_kind == 'digits'
  status = None
  speaker.update(False, False, 3.)
  assert not speaker.playing  # Do not repeat the older start announcement.
  for outcome in ('success', 'failed'):
    announcement = {'id': outcome, 'outcome': outcome}
    speaker.update(False, False, 4.)
    assert speaker.playing
    speaker.mix(np.zeros(8 * 48000, dtype=np.float32), .5, False)
    speaker.update(False, False, 5.)
    assert not speaker.playing
  announcement = {'id': 'another', 'outcome': 'started'}
  speaker.update(False, False, 6.)
  speaker.update(True, False, 7.)
  assert not speaker.playing


def test_runtime_loads_every_catalog_clip_and_instructions_leave_time_to_confirm():
  speaker = pairing_speech.PairingSpeech()
  catalog = Path(pairing_speech.BASEDIR) / 'openpilot/selfdrive/assets/sounds/voice/messages.json'
  pairing_keys = {key.removeprefix('pairing_') for key in json.loads(catalog.read_text()) if key.startswith('pairing_')}
  assert set(speaker.digits) | set(speaker.outcomes) | set(speaker.instructions) == pairing_keys
  longest_code = 6 * max(map(len, speaker.digits.values()))
  for confirm in ('confirm_button', 'confirm_screen'):
    duration = (len(speaker.instructions['compare']) + longest_code + len(speaker.instructions[confirm])) / pairing_speech.SAMPLE_RATE
    assert duration < CONFIRM_SECONDS - 5  # Keep time for the three-second hold and user response.


def test_ignition_refusal_speaks_onroad_once_but_critical_alerts_stop_it(monkeypatch):
  monkeypatch.setattr(pairing_speech, 'read_confirmation', lambda **kwargs: None)
  announcement = {'id': 'refused', 'outcome': 'ignition_on'}
  monkeypatch.setattr(pairing_speech, 'read_announcement', lambda **kwargs: announcement)
  speaker = pairing_speech.PairingSpeech()
  speaker.update(True, False, 1.)
  expected = speaker.outcomes['ignition_on']
  output = speaker.mix(np.zeros(len(expected), dtype=np.float32), .5, False)
  np.testing.assert_allclose(output, expected * .5)
  speaker.update(True, False, 2.)
  assert not speaker.playing
  announcement = {'id': 'another-refusal', 'outcome': 'ignition_on'}
  speaker.update(True, False, 3.)
  assert speaker.playing
  alert = np.full(100, .5, dtype=np.float32)
  np.testing.assert_array_equal(speaker.mix(alert, 1., True), alert)
  assert not speaker.playing
  speaker.update(True, True, 4.)
  speaker.update(True, False, 5.)
  assert not speaker.playing


@pytest.mark.parametrize('outcome', ['started', 'success', 'failed'])
def test_only_the_ignition_refusal_can_speak_onroad(monkeypatch, outcome):
  monkeypatch.setattr(pairing_speech, 'read_confirmation', lambda **kwargs: {'id': 'code', 'digits': '123456'})
  monkeypatch.setattr(pairing_speech, 'read_announcement', lambda **kwargs: {'id': 'announcement', 'outcome': outcome})
  speaker = pairing_speech.PairingSpeech()
  speaker.update(True, False, 1.)
  assert not speaker.playing
