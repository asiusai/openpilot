"""ML-KEM recording access derived from the device root secret."""
import base64
from cryptography.hazmat.primitives.asymmetric import mlkem
from openpilot.system.asius.pqc import RootIdentity


def decode_public(value: str) -> mlkem.MLKEM768PublicKey:
  if not isinstance(value, str) or len(value) != 1579:
    raise ValueError('Invalid recording public key')
  raw = base64.b64decode(value + '=', altchars=b'-_', validate=True)
  if len(raw) != 1184:
    raise ValueError('Invalid recording public key')
  return mlkem.MLKEM768PublicKey.from_public_bytes(raw)


def public_text(key: mlkem.MLKEM768PrivateKey) -> str:
  return base64.urlsafe_b64encode(key.public_key().public_bytes_raw()).decode().rstrip('=')


def get_recording_key(identity: RootIdentity) -> mlkem.MLKEM768PrivateKey:
  return identity.recording_key()
