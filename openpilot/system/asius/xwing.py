"""X-Wing (draft-connolly-cfrg-xwing-kem-06 and later) provider adapter.

The construction and encodings are fixed by the X-Wing specification.
OpenSSL implements ML-KEM, X25519, SHAKE256 and SHA3-256. No primitive is
implemented here. Known-answer tests use the specification's public vectors.
"""
import hashlib
import os

from cryptography.hazmat.primitives.asymmetric import mlkem, x25519

PUBLIC_BYTES = 1216
CIPHERTEXT_BYTES = 1120
LABEL = bytes.fromhex("5c2e2f2f5e5c")


def _combine(pq: bytes, classical: bytes, ciphertext: bytes, public: bytes) -> bytes:
  return hashlib.sha3_256(pq + classical + ciphertext + public + LABEL).digest()


class PublicKey:
  def __init__(self, raw: bytes):
    if len(raw) != PUBLIC_BYTES:
      raise ValueError("Invalid X-Wing public key length")
    self._pq = mlkem.MLKEM768PublicKey.from_public_bytes(raw[:1184])
    self._classical = x25519.X25519PublicKey.from_public_bytes(raw[1184:])
    self._raw = raw

  def public_bytes_raw(self) -> bytes:
    return self._raw

  def encapsulate(self) -> tuple[bytes, bytes]:
    ephemeral = x25519.X25519PrivateKey.generate()
    ct_x = ephemeral.public_key().public_bytes_raw()
    ss_x = ephemeral.exchange(self._classical)
    ss_m, ct_m = self._pq.encapsulate()
    return _combine(ss_m, ss_x, ct_x, self._raw[1184:]), ct_m + ct_x


class PrivateKey:
  def __init__(self, seed: bytes):
    if len(seed) != 32:
      raise ValueError("Invalid X-Wing seed length")
    expanded = hashlib.shake_256(seed).digest(96)
    self._pq = mlkem.MLKEM768PrivateKey.from_seed_bytes(expanded[:64])
    self._classical = x25519.X25519PrivateKey.from_private_bytes(expanded[64:])

  @classmethod
  def generate(cls) -> "PrivateKey":
    return cls(os.urandom(32))

  def public_key(self) -> PublicKey:
    return PublicKey(self._pq.public_key().public_bytes_raw() + self._classical.public_key().public_bytes_raw())

  def decapsulate(self, ciphertext: bytes) -> bytes:
    if len(ciphertext) != CIPHERTEXT_BYTES:
      raise ValueError("Invalid X-Wing ciphertext length")
    ct_m, ct_x = ciphertext[:1088], ciphertext[1088:]
    ss_x = self._classical.exchange(x25519.X25519PublicKey.from_public_bytes(ct_x))
    ss_m = self._pq.decapsulate(ct_m)
    return _combine(ss_m, ss_x, ct_x, self._classical.public_key().public_bytes_raw())
