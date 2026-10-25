from unittest.mock import patch

import pytest
from openpilot.system.asius.pqc import RootIdentity

from openpilot.system.asius import identity
from openpilot.system.asius.recording_keys import get_recording_key, public_text
from openpilot.system.loggerd.data_api import access_document, initial_state, recover_state


@pytest.fixture
def key_file(tmp_path, monkeypatch):
  path = tmp_path / 'identity.seed'
  monkeypatch.setattr(identity, 'PRIVATE_KEY_PATH', path)
  return path


def test_missing_identity_is_never_created(key_file):
  with pytest.raises(FileNotFoundError):
    identity.get_device_identity()
  assert not key_file.exists()
  assert identity.get_device_public_key() is None


def test_identity_can_be_loaded_without_writes(key_file):
  key = RootIdentity.generate()
  key_file.write_bytes(key.file_bytes())
  key_file.chmod(0o400)
  with patch('os.open', side_effect=AssertionError('unexpected write or lock')):
    assert identity.get_device_identity() == identity.dongle_id_from_public_key(identity.get_device_public_key())
  assert key_file.stat().st_mode & 0o777 == 0o400


def test_corrupt_key_is_not_replaced(key_file):
  key_file.write_text('corrupt')
  with pytest.raises(ValueError):
    identity.get_device_identity()
  assert key_file.read_text() == 'corrupt'


def test_recording_key_is_deterministic_and_identity_bound():
  seed = bytes(range(64))
  first = RootIdentity(seed)
  reloaded = RootIdentity(seed)
  second = RootIdentity(bytes(reversed(seed)))
  assert len(public_text(get_recording_key(first))) == 1579
  assert public_text(get_recording_key(first)) == public_text(get_recording_key(reloaded))
  assert public_text(get_recording_key(first)) != public_text(get_recording_key(second))
  assert first.derive(b'identity:ML-DSA-65', 32) != first.derive(b'recording:ML-KEM-768', 64)[:32]


def test_identity_alone_recovers_recording_grants():
  first = RootIdentity(bytes(range(64)))
  state = initial_state([])
  document = access_document(first, state, [])
  # No recording.pem or local key state is needed after a factory reflash.
  restored = RootIdentity(bytes(range(64)))
  assert recover_state(restored, document, [])['keys'] == state['keys']
