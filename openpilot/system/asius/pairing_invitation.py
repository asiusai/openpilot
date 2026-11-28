"""Single-use invitations shared by relay and Bluetooth processes through tmpfs."""
from contextlib import contextmanager
import fcntl
import hashlib
import hmac
import json
import os
from pathlib import Path
import re
import secrets
import time

ROOT = Path('/dev/shm/asius-invitations')


@contextmanager
def locked(root: Path):
  root.mkdir(mode=0o700, parents=True, exist_ok=True)
  os.chmod(root, 0o700)
  fd = os.open(root / '.lock', os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
  with os.fdopen(fd, 'r+') as lock:
    fcntl.flock(lock, fcntl.LOCK_EX)
    yield


def invalidate(root: Path = ROOT) -> None:
  with locked(root):
    for path in root.glob('*.json'):
      path.unlink(missing_ok=True)


def issue(recipient: str, lifetime: int, root: Path = ROOT, now: float | None = None) -> str:
  if not 1 <= lifetime <= 300:
    raise ValueError('Invalid invitation lifetime')
  with locked(root):
    now = time.monotonic() if now is None else now
    for path in root.glob('*.json'):
      try:
        if json.loads(path.read_text())['expires'] <= now:
          path.unlink(missing_ok=True)
      except (OSError, ValueError, KeyError, TypeError):
        continue
    if len(list(root.glob('*.json'))) >= 32:
      raise ValueError('Too many pending invitations')
    token_id = secrets.token_hex(32)
    fd = os.open(root / f'{token_id}.json', os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, 'w') as f:
      json.dump({'recipient': recipient, 'created': now, 'expires': now + lifetime}, f)
    return token_id


def consume(token_id: str, recipient: str, root: Path = ROOT, now: float | None = None) -> bool:
  if not isinstance(token_id, str) or not re.fullmatch('[0-9a-f]{64}', token_id):
    return False
  now = time.monotonic() if now is None else now
  path = root / f'{token_id}.json'
  try:
    fd = os.open(path, os.O_RDWR | os.O_NOFOLLOW)
    with os.fdopen(fd, 'r+') as f:
      fcntl.flock(f, fcntl.LOCK_EX)
      value = json.load(f)
      # Clearing the locked inode also defeats concurrent readers opened before unlink.
      f.seek(0)
      f.truncate()
      path.unlink(missing_ok=True)
    return value['recipient'] == recipient and value['created'] <= now < value['expires'] <= value['created'] + 300
  except (OSError, ValueError, KeyError, TypeError):
    return False


def permits_admission(admission: str, recipient: str, root: Path = ROOT, now: float | None = None) -> bool:
  """A hash permits a pairing handshake. Only the encrypted full invitation grants access."""
  if not isinstance(admission, str) or not re.fullmatch('[0-9a-f]{64}', admission):
    return False
  now = time.monotonic() if now is None else now
  with locked(root):
    for path in root.glob('*.json'):
      if not hmac.compare_digest(hashlib.sha256(path.stem.encode()).hexdigest(), admission):
        continue
      try:
        value = json.loads(path.read_text())
        return value['recipient'] == recipient and value['created'] <= now < value['expires'] <= value['created'] + 300
      except (OSError, ValueError, KeyError, TypeError):
        return False
  return False
