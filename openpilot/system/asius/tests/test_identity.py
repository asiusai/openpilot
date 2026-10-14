import concurrent.futures
import os
from unittest.mock import patch

import pytest

from openpilot.system.asius import identity


@pytest.fixture
def keys(tmp_path, monkeypatch):
  monkeypatch.setattr(identity, 'PRIVATE_KEY_PATH', tmp_path / 'id_ed25519')
  monkeypatch.setattr(identity, 'PUBLIC_KEY_PATH', tmp_path / 'id_ed25519.pub')
  return identity.PRIVATE_KEY_PATH, identity.PUBLIC_KEY_PATH


def test_concurrent_generation_and_public_recovery(keys):
  with concurrent.futures.ThreadPoolExecutor(8) as pool:
    identities = list(pool.map(lambda _: identity.get_or_create_device_identity(), range(16)))
  assert len(set(identities)) == 1
  private, public = keys
  original = private.read_bytes()
  assert private.stat().st_mode & 0o777 == 0o600
  public.unlink()
  assert identity.get_or_create_device_identity() == identities[0]
  assert private.read_bytes() == original


def test_entropy_failure_creates_no_identity(keys):
  with patch.object(identity, 'secure_seed', side_effect=OSError('entropy unavailable')):
    with pytest.raises(OSError):
      identity.get_or_create_device_identity()
  assert not any(path.exists() for path in keys)


def test_corrupt_or_mismatched_identity_is_not_replaced(keys):
  identity.get_or_create_device_identity()
  private, public = keys
  original = private.read_bytes()
  public.write_text('wrong public key')
  with pytest.raises(ValueError):
    identity.get_or_create_device_identity()
  assert private.read_bytes() == original
  private.write_text('corrupted')
  with pytest.raises(ValueError):
    identity.get_or_create_device_identity()
  assert private.read_text() == 'corrupted'
