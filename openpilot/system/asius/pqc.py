"""Asius post-quantum identities and persistent root-secret format.

The root recovers permanent signing and recording-access keys. Ephemeral
session keys and content keys always use independent operating-system randomness.
"""

import hashlib
import hmac
import os
from dataclasses import dataclass, field
from functools import cached_property

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import mldsa, mlkem
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

ROOT_BYTES = 64
ROOT_MAGIC = b"ASIUSID\x01"
ROOT_FILE_BYTES = len(ROOT_MAGIC) + ROOT_BYTES + 32
SIGNING_PUBLIC_BYTES = 1952
SIGNATURE_BYTES = 3309
KEM_PUBLIC_BYTES = 1184
KEM_CIPHERTEXT_BYTES = 1088
IDENTITY_CONTEXT = b"asius-identity-v1\x00ML-DSA-65\x00"
ROOT_SALT = b"asius-root-keys-v1"
SIGNING_CONTEXT = b"identity:ML-DSA-65"
RECORDING_CONTEXT = b"recording:ML-KEM-768"
BASE58 = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"


def identity_from_public(public: bytes) -> str:
  if len(public) != SIGNING_PUBLIC_BYTES:
    raise ValueError("Invalid ML-DSA-65 public key length")
  # Parse through the maintained provider before accepting an identity key.
  mldsa.MLDSA65PublicKey.from_public_bytes(public)
  value = int.from_bytes(hashlib.sha256(IDENTITY_CONTEXT + public).digest(), "big")
  result = ""
  while value:
    value, digit = divmod(value, 58)
    result = BASE58[digit] + result
  return result.rjust(44, BASE58[0])


def verify_signature(identity: str, public: bytes, signature: bytes, message: bytes) -> None:
  if len(signature) != SIGNATURE_BYTES or not hmac.compare_digest(identity_from_public(public), identity):
    raise InvalidSignature("Signature identity mismatch")
  mldsa.MLDSA65PublicKey.from_public_bytes(public).verify(signature, message)


@dataclass(frozen=True)
class RootIdentity:
  seed: bytes = field(repr=False)

  def __post_init__(self):
    if not isinstance(self.seed, bytes) or len(self.seed) != ROOT_BYTES:
      raise ValueError("Identity requires a 64-byte root secret")

  @classmethod
  def generate(cls) -> "RootIdentity":
    return cls(os.urandom(ROOT_BYTES))

  @classmethod
  def from_file_bytes(cls, data: bytes) -> "RootIdentity":
    if len(data) != ROOT_FILE_BYTES or not data.startswith(ROOT_MAGIC):
      raise ValueError("Invalid identity file format")
    body, checksum = data[:-32], data[-32:]
    if not hmac.compare_digest(hashlib.sha256(body).digest(), checksum):
      raise ValueError("Identity file checksum mismatch")
    return cls(body[len(ROOT_MAGIC):])

  def file_bytes(self) -> bytes:
    body = ROOT_MAGIC + self.seed
    # This checksum detects storage damage. It is not access protection.
    return body + hashlib.sha256(body).digest()

  def derive(self, context: bytes, length: int) -> bytes:
    return HKDF(hashes.SHA384(), length, ROOT_SALT, context).derive(self.seed)

  def signing_key(self) -> mldsa.MLDSA65PrivateKey:
    return self._signing_key

  @cached_property
  def _signing_key(self) -> mldsa.MLDSA65PrivateKey:
    return mldsa.MLDSA65PrivateKey.from_seed_bytes(self.derive(SIGNING_CONTEXT, 32))

  def recording_key(self) -> mlkem.MLKEM768PrivateKey:
    return self._recording_key

  @cached_property
  def _recording_key(self) -> mlkem.MLKEM768PrivateKey:
    return mlkem.MLKEM768PrivateKey.from_seed_bytes(self.derive(RECORDING_CONTEXT, 64))

  def sign(self, message: bytes) -> bytes:
    return self.signing_key().sign(message)

  def public_key(self) -> mldsa.MLDSA65PublicKey:
    return self.signing_key().public_key()

  def identity(self) -> str:
    return identity_from_public(self.signing_key().public_key().public_bytes_raw())
