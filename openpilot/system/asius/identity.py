import ctypes
import os
import tempfile
from pathlib import Path

from openpilot.system.asius.pqc import BASE58, ROOT_BYTES, ROOT_FILE_BYTES, RootIdentity

IDENTITY_HASH_BYTES = 32
DONGLE_ID_LEN = 44
PRIVATE_KEY_PATH = Path("/persist/asius/identity.seed")


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
  if decoded >= 1 << (IDENTITY_HASH_BYTES * 8):
    raise ValueError("invalid identity")
  return decoded.to_bytes(IDENTITY_HASH_BYTES, "big")


def is_dongle_id(dongle_id: str | None) -> bool:
  if dongle_id is None:
    return False
  try:
    identity_to_bytes(dongle_id)
    return True
  except Exception:
    return False


def get_device_private_key() -> RootIdentity:
  with PRIVATE_KEY_PATH.open('rb') as source:
    return RootIdentity.from_file_bytes(source.read(ROOT_FILE_BYTES + 1))


def _secure_random(length: int) -> bytes:
  # getrandom blocks until the kernel random generator is initialized. Not every Python build has os.getrandom.
  libc = ctypes.CDLL(None, use_errno=True)
  libc.getrandom.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_uint]
  libc.getrandom.restype = ctypes.c_ssize_t
  buffer = ctypes.create_string_buffer(length)
  if libc.getrandom(buffer, length, 0) != length:
    raise OSError(ctypes.get_errno(), "Secure randomness unavailable")
  return buffer.raw


def provision_agnos_identity() -> bool:
  """Create the root secret once on a comma device. AGNOS has no identity service.

  Returns True when a new identity was written. Never replaces an existing file,
  and never writes outside a mounted /persist partition.
  """
  directory = PRIVATE_KEY_PATH.parent
  if PRIVATE_KEY_PATH.exists() or PRIVATE_KEY_PATH.is_symlink():
    return False
  if not os.path.ismount("/persist") or directory.is_symlink() or not directory.is_dir():
    raise RuntimeError("/persist/asius is not available")
  data = RootIdentity(_secure_random(ROOT_BYTES)).file_bytes()
  fd, temporary = tempfile.mkstemp(prefix=".identity.", dir=directory)
  try:
    with os.fdopen(fd, "wb") as output:
      os.fchmod(output.fileno(), 0o600)
      output.write(data)
      output.flush()
      os.fsync(output.fileno())
    try:
      os.link(temporary, PRIVATE_KEY_PATH)  # fails instead of replacing an identity
    except FileExistsError:
      return False
    directory_fd = os.open(directory, os.O_DIRECTORY)
    try:
      os.fsync(directory_fd)
    finally:
      os.close(directory_fd)
  finally:
    Path(temporary).unlink(missing_ok=True)
  return True


def get_device_identity() -> str:
  # VamOS provisions the identity before workloads start. Never create a key in
  # the root overlay when /persist is missing, read-only, or damaged.
  return get_device_private_key().identity()
