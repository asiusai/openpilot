import json
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from openpilot.selfdrive.assets.sounds.voice import generate

SOUNDS = generate.ROOT.parent
CATALOGS = ('voice', 'pairing')


@pytest.mark.parametrize('catalog', CATALOGS)
def test_all_catalog_phrases_have_current_bundled_audio(catalog, monkeypatch):
  directory = SOUNDS / catalog
  monkeypatch.setattr(generate, 'ROOT', directory)
  messages = json.loads((directory / 'messages.json').read_text())
  generate.check(messages)


@pytest.mark.parametrize('catalog', CATALOGS)
@pytest.mark.parametrize('damage', ['missing', 'corrupt', 'new_phrase', 'changed_text', 'extra'])
def test_incomplete_or_stale_speech_is_rejected(catalog, damage, tmp_path, monkeypatch):
  directory = tmp_path / catalog
  shutil.copytree(SOUNDS / catalog, directory)
  monkeypatch.setattr(generate, 'ROOT', directory)
  messages = json.loads((directory / 'messages.json').read_text())
  key = next(iter(messages))
  if damage == 'missing':
    (directory / f'{key}.wav').unlink()
  elif damage == 'corrupt':
    (directory / f'{key}.wav').write_bytes(b'broken audio')
  elif damage == 'new_phrase':
    messages['new_phrase'] = {'text': 'New phrase.'}
  elif damage == 'changed_text':
    messages[key]['text'] = 'Changed phrase.'
  else:
    shutil.copyfile(directory / f'{key}.wav', directory / 'extra.wav')
  with pytest.raises(ValueError):
    generate.check(messages)


def test_default_build_checks_both_catalogs_without_tts(tmp_path):
  # Exercise the SCons action without uv, ffmpeg, edge-tts or site-packages.
  for catalog in CATALOGS:
    shutil.copytree(SOUNDS / catalog, tmp_path / catalog)
  shutil.copyfile(SOUNDS / 'SConscript', tmp_path / 'SConscript')
  (tmp_path / 'SConstruct').write_text("env = Environment(tools=[], ENV={'PATH': ''})\nExport('env')\nSConscript('SConscript')\n")
  command = [str(Path(sys.executable).with_name('scons')), '-Q']
  result = subprocess.run(command, cwd=tmp_path, capture_output=True, text=True)
  assert result.returncode == 0, result.stdout + result.stderr
  verified = [line for line in result.stdout.splitlines() if line.startswith('Verified ')]
  assert verified == [f"Verified {len(json.loads((tmp_path / catalog / 'messages.json').read_text()))} offline speech clips."
                      for catalog in CATALOGS]
  # Every build rechecks files, including damage after a successful build.
  (tmp_path / 'pairing/0.wav').unlink()
  result = subprocess.run(command, cwd=tmp_path, capture_output=True, text=True)
  assert result.returncode != 0
  assert 'Speech files do not match the catalog' in result.stderr
