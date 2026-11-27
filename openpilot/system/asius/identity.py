from pathlib import Path

from openpilot.system.asius.pqc import BASE58, ROOT_FILE_BYTES, RootIdentity

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


def get_device_identity() -> str:
  # VamOS provisions the identity before workloads start. Never create a key in
  # the root overlay when /persist is missing, read-only, or damaged.
  return get_device_private_key().identity()
