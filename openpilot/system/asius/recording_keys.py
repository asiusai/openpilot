"""Recording access keys derived from the device signing identity."""
import base64

from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import ec, ed25519
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat

# Order of NIST P-256. Use 512 derived bits before reduction to avoid bias.
P256_ORDER = 0xFFFFFFFF00000000FFFFFFFFFFFFFFFFBCE6FAADA7179E84F3B9CAC2FC632551


def decode_public(value: str) -> ec.EllipticCurvePublicKey:
  if not isinstance(value, str) or len(value) != 87:
    raise ValueError('Invalid recording public key')
  raw = base64.b64decode(value + '=', altchars=b'-_', validate=True)
  if len(raw) != 65 or raw[0] != 4:
    raise ValueError('Invalid recording public key')
  return ec.EllipticCurvePublicKey.from_encoded_point(ec.SECP256R1(), raw)


def public_text(key: ec.EllipticCurvePrivateKey) -> str:
  return base64.urlsafe_b64encode(key.public_key().public_bytes(Encoding.X962, PublicFormat.UncompressedPoint)).decode().rstrip('=')


def get_recording_key(identity: ed25519.Ed25519PrivateKey) -> ec.EllipticCurvePrivateKey:
  material = HKDF(algorithm=hashes.SHA256(), length=64, salt=b"asius-device-keys-v1",
                  info=b"asius-device-recording-p256-v1").derive(identity.private_bytes_raw())
  scalar = int.from_bytes(material, 'big') % (P256_ORDER - 1) + 1
  return ec.derive_private_key(scalar, ec.SECP256R1())
