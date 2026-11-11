"""X-Wing recording access derived from the device root secret."""
import base64
from openpilot.system.asius import xwing
from openpilot.system.asius.pqc import RootIdentity


def decode_public(value: str) -> xwing.PublicKey:
  if not isinstance(value, str) or len(value) != 1622:
    raise ValueError('Invalid recording public key')
  raw = base64.b64decode(value + '==', altchars=b'-_', validate=True)
  if len(raw) != xwing.PUBLIC_BYTES:
    raise ValueError('Invalid recording public key')
  return xwing.PublicKey(raw)


def public_text(key: xwing.PrivateKey) -> str:
  return base64.urlsafe_b64encode(key.public_key().public_bytes_raw()).decode().rstrip('=')


def get_recording_key(identity: RootIdentity) -> xwing.PrivateKey:
  return identity.recording_key()
