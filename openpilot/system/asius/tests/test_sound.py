import base64
from unittest.mock import Mock

import pytest

from openpilot.common.params import Params
from openpilot.system.asius import methods, sound
from openpilot.system.asius.param_editor import ParameterEditor


@pytest.mark.parametrize('value', [None, -1, 0, 9, 101, True, 50.5, '80'])
def test_invalid_gain_uses_default(value):
  assert sound.volume_gain(Mock(get=lambda _: value)) == .8


def test_saved_volume_validated_through_both_editors(tmp_path, monkeypatch):
  params = Params(str(tmp_path))
  monkeypatch.setattr(methods, 'Params', lambda: params)
  for value in (10, 37, 80, 100):
    assert methods.saveParams({'SoundVolume': value}) == {'SoundVolume': 'ok'}
    assert sound.volume_gain(params) == value / 100
  for value in (9, 101, True, 20.5, '80'):
    assert methods.saveParams({'SoundVolume': value})['SoundVolume'].startswith('error:')
    assert params.get('SoundVolume') == 100
  editor = ParameterEditor(methods.SAVE_PARAMS_BLOCKED_KEYS, lambda: params)
  revision = editor._value(params, 'SoundVolume')[0]['revision']
  with pytest.raises(ValueError, match='10 to 100'):
    editor.write('test', 'SoundVolume', revision, 'upload', 0, base64.b64encode(b'101').decode(), True)
  assert methods.saveParams({'SoundVolumePreview': {'id': 'bad'}}) == {'SoundVolumePreview': 'error: blocked'}
  assert methods.saveParams({'SoundVolume': None}) == {'SoundVolume': 'ok: removed'}
  assert sound.volume_gain(params) == .8


@pytest.mark.parametrize('expires', [99, 100, 106, float('inf'), True, '105'])
def test_preview_rejects_expired_or_invalid_request(expires):
  assert sound.preview_request(Mock(get=lambda _: {'id': 'test', 'expires': expires}), 100) is None


def test_preview_is_short_lived_and_starts_offroad_soundd(tmp_path, monkeypatch):
  from openpilot.system.manager import process_config
  params = Params(str(tmp_path))
  monkeypatch.setattr(sound.time, 'monotonic', lambda: 100)
  monkeypatch.setattr(process_config, 'ASIUS_HARDWARE', True)
  assert not process_config.sound(False, params, Mock())
  sound.request_preview(params)
  request = sound.preview_request(params)
  assert request is not None and request['expires'] == 105
  assert process_config.sound(False, params, Mock())
  monkeypatch.setattr(sound.time, 'monotonic', lambda: 105)
  assert not process_config.sound(False, params, Mock())
