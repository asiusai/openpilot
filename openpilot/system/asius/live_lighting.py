"""Short, authenticated lighting leases for the Live viewer."""
from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import re
import tempfile
import time

from openpilot.system.asius.access_policy import ignition_state, require_ignition_off

STATE = Path('/dev/shm/asius-live-lighting.json')
LEDS = Path('/sys/class/leds')
LEASE_SECONDS = 5.
MAX_LED = 127
MAX_IR_MA = 30


def read_state() -> dict:
  try:
    state = json.loads(STATE.read_text())
    if isinstance(state, dict) and 0 < state.get('expires', 0) - time.monotonic() <= LEASE_SECONDS:
      return state
  except (OSError, ValueError, TypeError):
    pass
  return {}


@contextmanager
def locked():
  with os.fdopen(os.open(str(STATE) + '.lock', os.O_CREAT | os.O_RDWR, 0o600), 'w') as lock:
    fcntl.flock(lock, fcntl.LOCK_EX)
    yield


def capabilities() -> dict:
  return {
    'leds': all((LEDS / f'asius:cam{cam}:{color}:{package}/brightness').is_file()
                for cam in (2, 3) for color in ('red', 'green', 'blue') for package in (0, 2)),
    'ir': all((LEDS / f'asius:cam{cam}:ir/brightness').is_file() for cam in (2, 3)),
    'maxLedBrightness': MAX_LED, 'maxIrMa': MAX_IR_MA,
  }


def get_state() -> dict:
  return {'capabilities': capabilities(), 'canEdit': ignition_state() is False}


def set_state(owner: str, lease: str, led: dict | None = None, irMa: int | None = None) -> dict:
  if not isinstance(lease, str) or re.fullmatch(r'[a-f0-9]{32}', lease) is None:
    raise ValueError('Invalid lighting session')
  if led is not None:
    if (not isinstance(led, dict) or set(led) != {'color', 'brightness'} or not isinstance(led['color'], str)
        or re.fullmatch(r'#[0-9a-fA-F]{6}', led['color']) is None
        or type(led['brightness']) is not int or not 0 <= led['brightness'] <= MAX_LED):
      raise ValueError('Invalid LED color or brightness')
  if irMa is not None and (type(irMa) is not int or not 0 <= irMa <= MAX_IR_MA):
    raise ValueError(f'IR must be between 0 and {MAX_IR_MA} mA')
  with locked():
    previous = read_state()
    if previous and (previous.get('owner'), previous.get('lease')) != (owner, lease):
      raise PermissionError('Another Live viewer is controlling the lights')
    # Restoring automatic control must also work after ignition turns on.
    if led is None and irMa is None:
      STATE.unlink(missing_ok=True)
      return {'expiresIn': 0}
    require_ignition_off()
    supported = capabilities()
    if (led is not None and not supported['leds']) or (irMa is not None and not supported['ir']):
      raise ValueError('These camera lights are not available on this device')
    state = {'owner': owner, 'lease': lease, 'expires': time.monotonic() + LEASE_SECONDS, 'led': led, 'irMa': irMa}
    fd, path = tempfile.mkstemp(dir=STATE.parent, prefix='.live-lighting-')
    try:
      with os.fdopen(fd, 'w') as out:
        json.dump(state, out)
      os.replace(path, STATE)
    finally:
      Path(path).unlink(missing_ok=True)
  return {'expiresIn': LEASE_SECONDS}


def methods(owner: str) -> dict:
  return {'getLiveLighting': get_state, 'setLiveLighting': lambda **kwargs: set_state(owner, **kwargs)}
