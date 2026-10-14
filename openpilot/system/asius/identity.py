from pathlib import Path
import fcntl
import os
import tempfile
import ctypes

from cryptography.hazmat.primitives.asymmetric import ed25519
from cryptography.hazmat.primitives.serialization import Encoding, NoEncryption, PrivateFormat, PublicFormat, load_pem_private_key, load_pem_public_key

BASE58 = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"
ED25519_KEY_BYTES = 32
DONGLE_ID_LEN = 44
PRIVATE_KEY_PATH = Path("/data/persist/comma/id_ed25519")
PUBLIC_KEY_PATH = Path("/data/persist/comma/id_ed25519.pub")


def base58_encode(value: int) -> str:
  if value == 0:
    return BASE58[0]

  out = []
  while value:
    value, rem = divmod(value, 58)
    out.append(BASE58[rem])
  return "".join(reversed(out))


def base58_decode(value: str) -> int:
  decoded = 0
  for char in value:
    decoded *= 58
    decoded += BASE58.index(char)
  return decoded


def identity_to_bytes(value: str) -> bytes:
  if len(value) != DONGLE_ID_LEN or not all(char in BASE58 for char in value):
    raise ValueError("invalid identity")

  decoded = base58_decode(value)
  if decoded >= 1 << (ED25519_KEY_BYTES * 8):
    raise ValueError("invalid identity")
  return decoded.to_bytes(ED25519_KEY_BYTES, "big")


def bytes_to_identity(value: bytes) -> str:
  if len(value) != ED25519_KEY_BYTES:
    raise ValueError("identity requires a 32-byte Ed25519 public key")
  return base58_encode(int.from_bytes(value, "big")).rjust(DONGLE_ID_LEN, BASE58[0])


def is_dongle_id(dongle_id: str | None) -> bool:
  if dongle_id is None:
    return False
  try:
    identity_to_bytes(dongle_id)
    return True
  except Exception:
    return False


def dongle_id_from_public_key(public_key: str) -> str:
  key = load_pem_public_key(public_key.encode())
  if not isinstance(key, ed25519.Ed25519PublicKey):
    raise ValueError("identity requires an Ed25519 public key")

  public_bytes = key.public_bytes(Encoding.Raw, PublicFormat.Raw)
  return bytes_to_identity(public_bytes)


def get_device_private_key() -> ed25519.Ed25519PrivateKey:
  key = load_pem_private_key(PRIVATE_KEY_PATH.read_bytes(), password=None)
  if not isinstance(key, ed25519.Ed25519PrivateKey):
    raise ValueError("device identity key is not Ed25519")
  return key


def get_device_public_key() -> str | None:
  try:
    return PUBLIC_KEY_PATH.read_text()
  except OSError:
    return None


def get_or_create_device_identity() -> str:
  PRIVATE_KEY_PATH.parent.mkdir(parents=True, exist_ok=True)
  lock_fd = os.open(PRIVATE_KEY_PATH.with_suffix('.lock'), os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
  with os.fdopen(lock_fd, 'w') as lock:
    fcntl.flock(lock, fcntl.LOCK_EX)
    if PRIVATE_KEY_PATH.exists():
      private_key = get_device_private_key()  # Corruption must not silently replace an identity.
      PRIVATE_KEY_PATH.chmod(0o600)
    else:
      if PUBLIC_KEY_PATH.exists():
        raise ValueError('Device private key is missing; explicit identity reset is required')
      # getrandom blocks until the kernel CSPRNG is initialized, independent of wall time.
      seed = secure_seed()
      private_key = ed25519.Ed25519PrivateKey.from_private_bytes(seed)
      _atomic_key(PRIVATE_KEY_PATH, private_key.private_bytes(Encoding.PEM, PrivateFormat.PKCS8, NoEncryption()), 0o600)
    public_key = private_key.public_key().public_bytes(Encoding.PEM, PublicFormat.SubjectPublicKeyInfo)
    if PUBLIC_KEY_PATH.exists() and PUBLIC_KEY_PATH.read_bytes() != public_key:
      raise ValueError('Device identity public and private keys disagree')
    if not PUBLIC_KEY_PATH.exists():
      _atomic_key(PUBLIC_KEY_PATH, public_key, 0o644)
    return dongle_id_from_public_key(public_key.decode())


def secure_seed() -> bytes:
  # Some bundled Python builds do not expose os.getrandom. Use the same Linux
  # API directly, with blocking initialization and no weaker random fallback.
  libc = ctypes.CDLL(None, use_errno=True)
  libc.getrandom.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_uint]
  libc.getrandom.restype = ctypes.c_ssize_t
  output = ctypes.create_string_buffer(32)
  if libc.getrandom(output, 32, 0) != 32:
    raise OSError(ctypes.get_errno(), 'Secure random generation failed')
  return output.raw


def _atomic_key(path: Path, data: bytes, mode: int) -> None:
  fd, name = tempfile.mkstemp(prefix=f'.{path.name}.', dir=path.parent)
  try:
    with os.fdopen(fd, 'wb') as output:
      os.fchmod(output.fileno(), mode)
      output.write(data)
      output.flush()
      os.fsync(output.fileno())
    os.replace(name, path)
    directory = os.open(path.parent, os.O_DIRECTORY)
    try:
      os.fsync(directory)
    finally:
      os.close(directory)
  finally:
    if os.path.exists(name):
      os.unlink(name)
