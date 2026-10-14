"""Separate P-256 recording keys. App enrollment binds each public key."""
import base64
import fcntl
import os
from pathlib import Path

from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives.serialization import Encoding, NoEncryption, PrivateFormat, PublicFormat, load_pem_private_key

from openpilot.system.asius.identity import _atomic_key, secure_seed

KEY_PATH = Path('/data/persist/comma/recording_p256.pem')


def decode_public(value: str) -> ec.EllipticCurvePublicKey:
  if not isinstance(value, str) or len(value) != 87:
    raise ValueError('Invalid recording public key')
  raw = base64.b64decode(value + '=', altchars=b'-_', validate=True)
  if len(raw) != 65 or raw[0] != 4:
    raise ValueError('Invalid recording public key')
  return ec.EllipticCurvePublicKey.from_encoded_point(ec.SECP256R1(), raw)


def public_text(key: ec.EllipticCurvePrivateKey) -> str:
  return base64.urlsafe_b64encode(key.public_key().public_bytes(Encoding.X962, PublicFormat.UncompressedPoint)).decode().rstrip('=')


def get_recording_key() -> ec.EllipticCurvePrivateKey:
  KEY_PATH.parent.mkdir(parents=True, exist_ok=True)
  fd = os.open(KEY_PATH.with_suffix('.lock'), os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
  with os.fdopen(fd, 'w') as lock:
    fcntl.flock(lock, fcntl.LOCK_EX)
    if KEY_PATH.exists():
      key = load_pem_private_key(KEY_PATH.read_bytes(), password=None)
      if not isinstance(key, ec.EllipticCurvePrivateKey) or not isinstance(key.curve, ec.SECP256R1):
        raise ValueError('Invalid stored recording key')
      KEY_PATH.chmod(0o600)
      return key
    secure_seed()  # Wait for the OS CSPRNG before the library generates a key.
    key = ec.generate_private_key(ec.SECP256R1())
    _atomic_key(KEY_PATH, key.private_bytes(Encoding.PEM, PrivateFormat.PKCS8, NoEncryption()), 0o600)
    return key
