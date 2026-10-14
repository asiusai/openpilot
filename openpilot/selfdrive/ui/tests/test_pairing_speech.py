import numpy as np

from openpilot.selfdrive.ui import pairing_speech


def test_all_digits_are_bundled_and_spoken_in_order_including_zeros(monkeypatch):
  monkeypatch.setattr(pairing_speech, 'read_announcement', lambda **kwargs: None)
  speaker = pairing_speech.PairingSpeech()
  assert len(speaker.digits) == 10
  assert all(len(samples) > 1000 and 0 < np.max(np.abs(samples)) <= .86 for samples in speaker.digits.values())
  status = {'id': 'first', 'digits': '000042'}
  monkeypatch.setattr(pairing_speech, 'read_confirmation', lambda **kwargs: status)
  speaker.update(False, False, 10.)
  expected = np.concatenate([speaker.digits[digit] for digit in status['digits']])
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
  assert set(speaker.outcomes) == {'started', 'success', 'failed'}
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
