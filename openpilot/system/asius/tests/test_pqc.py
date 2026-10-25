import pytest

from cryptography.exceptions import InvalidSignature

from openpilot.system.asius.pqc import ROOT_FILE_BYTES, RootIdentity, identity_from_public, verify_signature


def test_root_restores_both_permanent_keys():
  first = RootIdentity.generate()
  restored = RootIdentity.from_file_bytes(first.file_bytes())
  assert restored.identity() == first.identity()
  assert restored.signing_key().public_key().public_bytes_raw() == first.signing_key().public_key().public_bytes_raw()
  assert restored.recording_key().public_key().public_bytes_raw() == first.recording_key().public_key().public_bytes_raw()
  assert len(first.file_bytes()) == ROOT_FILE_BYTES == 104
  assert len(first.identity()) == 44
  assert first.seed.hex() not in repr(first)


def test_root_corruption_never_silently_changes_identity():
  encoded = RootIdentity.generate().file_bytes()
  for offset in range(len(encoded)):
    damaged = bytearray(encoded)
    damaged[offset] ^= 1
    with pytest.raises(ValueError):
      RootIdentity.from_file_bytes(bytes(damaged))
  for invalid in [b"", encoded[:-1], encoded + b"\x00"]:
    with pytest.raises(ValueError):
      RootIdentity.from_file_bytes(invalid)


def test_signature_binds_public_key_to_expected_identity():
  owner, other = RootIdentity.generate(), RootIdentity.generate()
  key = owner.signing_key()
  public = key.public_key().public_bytes_raw()
  message = b"Asius PQ identity test"
  signature = key.sign(message)
  verify_signature(owner.identity(), public, signature, message)
  for identity, candidate_public, candidate_signature, candidate_message in [
    (other.identity(), public, signature, message),
    (owner.identity(), other.signing_key().public_key().public_bytes_raw(), signature, message),
    (owner.identity(), public, signature, message + b"!"),
    (owner.identity(), public, signature[:-1], message),
  ]:
    with pytest.raises(InvalidSignature):
      verify_signature(identity, candidate_public, candidate_signature, candidate_message)
  with pytest.raises(ValueError):
    identity_from_public(public[:-1])


def test_recording_recovery_and_domain_separation():
  root = RootIdentity.generate()
  key = root.recording_key()
  shared, ciphertext = key.public_key().encapsulate()
  restored = RootIdentity.from_file_bytes(root.file_bytes())
  assert restored.recording_key().decapsulate(ciphertext) == shared
  assert root.derive(b"identity:ML-DSA-65", 32) != root.derive(b"recording:ML-KEM-768", 32)
  assert RootIdentity.generate().recording_key().decapsulate(ciphertext) != shared
