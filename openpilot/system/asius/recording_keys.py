"""X-Wing recording access derived from the device root secret."""
from openpilot.system.asius import xwing
from openpilot.system.asius.pqc import RootIdentity
from openpilot.system.asius.session import b64, unb64


def decode_public(value: str) -> xwing.PublicKey:
  if not isinstance(value, str) or len(value) != 1622:
    raise ValueError('Invalid recording public key')
  raw = unb64(value)
  if len(raw) != xwing.PUBLIC_BYTES:
    raise ValueError('Invalid recording public key')
  return xwing.PublicKey(raw)


def public_text(key: xwing.PrivateKey) -> str:
  return b64(key.public_key().public_bytes_raw())


def get_recording_key(identity: RootIdentity) -> xwing.PrivateKey:
  return identity.recording_key()
