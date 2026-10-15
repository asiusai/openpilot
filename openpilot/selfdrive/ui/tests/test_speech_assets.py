import json
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from openpilot.selfdrive.assets.sounds.voice import generate

SOUNDS = generate.ROOT.parent


def test_all_catalog_phrases_have_current_bundled_audio():
  assert list(SOUNDS.rglob('messages.json')) == [generate.ROOT / 'messages.json']
  assert list(SOUNDS.rglob('generated.json')) == [generate.ROOT / 'generated.json']
  messages = json.loads((generate.ROOT / 'messages.json').read_text())
  generate.check(messages)


@pytest.mark.parametrize('key', ['pay_attention', 'pairing_started', 'pairing_0'])
@pytest.mark.parametrize('damage', ['missing', 'corrupt', 'changed_text'])
def test_incomplete_or_stale_speech_is_rejected(key, damage, tmp_path, monkeypatch):
  directory = tmp_path / 'voice'
  shutil.copytree(generate.ROOT, directory)
  monkeypatch.setattr(generate, 'ROOT', directory)
  messages = json.loads((directory / 'messages.json').read_text())
  if damage == 'missing':
    (directory / f'{key}.wav').unlink()
  elif damage == 'corrupt':
    (directory / f'{key}.wav').write_bytes(b'broken audio')
  else:
    messages[key]['text'] = 'Changed phrase.'
  with pytest.raises(ValueError):
    generate.check(messages)


@pytest.mark.parametrize('extra_audio', [False, True])
def test_new_phrases_require_generated_audio_and_extra_clips_are_rejected(extra_audio, tmp_path, monkeypatch):
  directory = tmp_path / 'voice'
  shutil.copytree(generate.ROOT, directory)
  monkeypatch.setattr(generate, 'ROOT', directory)
  messages = json.loads((directory / 'messages.json').read_text())
  if extra_audio:
    shutil.copyfile(directory / 'pay_attention.wav', directory / 'extra.wav')
  else:
    messages['new_phrase'] = {'text': 'New phrase.'}
  with pytest.raises(ValueError):
    generate.check(messages)


def test_default_build_checks_all_speech_without_tts(tmp_path):
  # Exercise the SCons action without uv, ffmpeg, edge-tts or site-packages.
  shutil.copytree(generate.ROOT, tmp_path / 'voice')
  shutil.copyfile(SOUNDS / 'SConscript', tmp_path / 'SConscript')
  (tmp_path / 'SConstruct').write_text("env = Environment(tools=[], ENV={'PATH': ''})\nExport('env')\nSConscript('SConscript')\n")
  command = [str(Path(sys.executable).with_name('scons')), '-Q']
  result = subprocess.run(command, cwd=tmp_path, capture_output=True, text=True)
  assert result.returncode == 0, result.stdout + result.stderr
  verified = [line for line in result.stdout.splitlines() if line.startswith('Verified ')]
  assert verified == [f"Verified {len(json.loads((tmp_path / 'voice/messages.json').read_text()))} offline speech clips."]
  # Every build rechecks files, including damage after a successful build.
  (tmp_path / 'voice/pairing_0.wav').unlink()
  result = subprocess.run(command, cwd=tmp_path, capture_output=True, text=True)
  assert result.returncode != 0
  assert 'Speech files do not match the catalog' in result.stderr
