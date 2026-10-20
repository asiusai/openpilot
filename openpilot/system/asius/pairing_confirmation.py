"""Physical Bluetooth comparison approval. All state lives in tmpfs, never Params."""
from __future__ import annotations

import asyncio
import json
import math
import os
from pathlib import Path
import secrets
import socket
import time
from collections.abc import Callable
from dataclasses import dataclass

from openpilot.common.swaglog import cloudlog

RUNTIME = Path('/dev/shm/asius-pairing')
CONFIRM_SECONDS = 30.
ENROLL_SECONDS = 30.
ANNOUNCEMENTS = ('started', 'success', 'failed', 'ignition_on')


def announce(outcome: str, root: Path = RUNTIME, now: float | None = None) -> None:
  if outcome not in ANNOUNCEMENTS:
    raise ValueError('Invalid pairing announcement')
  now = time.monotonic() if now is None else now
  root.mkdir(mode=0o700, parents=True, exist_ok=True)
  path = root / 'speech.tmp'
  fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_NOFOLLOW, 0o600)
  with os.fdopen(fd, 'w') as output:
    json.dump({'id': secrets.token_hex(16), 'outcome': outcome, 'created': now, 'expires': now + 8.}, output)
  os.replace(path, root / 'speech.json')


def read_announcement(root: Path = RUNTIME, now: float | None = None) -> dict | None:
  try:
    value = json.loads((root / 'speech.json').read_text())
    now = time.monotonic() if now is None else now
    if value['outcome'] in ANNOUNCEMENTS and value['created'] <= now < value['expires'] <= value['created'] + 8.:
      return value
  except (OSError, ValueError, KeyError, TypeError):
    pass
  return None


def read_confirmation(root: Path = RUNTIME, now: float | None = None) -> dict | None:
  try:
    value = json.loads((root / 'status.json').read_text())
    now = time.monotonic() if now is None else now
    if (value['phase'] == 'confirm' and math.isfinite(value['expires']) and
        value['created'] <= now < value['expires'] <= value['created'] + CONFIRM_SECONDS):
      return value
  except (OSError, ValueError, KeyError, TypeError):
    pass
  return None


def send_button(action: str, started: float, root: Path = RUNTIME) -> bool:
  """Return false when bluetoothd is unavailable. Never approve through a fallback."""
  try:
    with socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM) as sock:
      sock.settimeout(.1)
      sock.sendto(json.dumps({'action': action, 'started': started}).encode(), str(root / 'button.sock'))
    return True
  except OSError:
    return False


def send_ui_response(request_id: str, accepted: bool, root: Path = RUNTIME) -> None:
  with socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM) as sock:
    sock.settimeout(.1)
    sock.sendto(json.dumps({'action': 'screen', 'id': request_id, 'accepted': accepted}).encode(), str(root / 'button.sock'))


@dataclass
class Pending:
  device: str
  request_id: str
  created: float
  expires: float
  result: asyncio.Future


class Confirmation:
  def __init__(self, enabled: Callable[[], bool], root: Path = RUNTIME, clock: Callable[[], float] = time.monotonic, allow_ui: bool = False,
               on_reject: Callable[[str], None] | None = None):
    self.enabled, self.root, self.clock = enabled, root, clock
    self.allow_ui = allow_ui
    self.on_reject = on_reject
    self.pending: Pending | None = None
    self.approved: tuple[str, float] | None = None
    root.mkdir(mode=0o700, parents=True, exist_ok=True)
    os.chmod(root, 0o700)
    self.publish(None)

  def publish(self, value: dict | None) -> None:
    path = self.root / 'status.json'
    if value is None:
      path.unlink(missing_ok=True)
      return
    temporary = self.root / 'status.tmp'
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, 'w') as f:
      json.dump(value, f, separators=(',', ':'))
    os.replace(temporary, path)

  async def request(self, device: str, passkey: int) -> None:
    self.expire()
    if not self.enabled() or self.pending or self.approved or not 0 <= passkey <= 999999:
      raise PermissionError('Pairing confirmation is unavailable')
    now = self.clock()
    pending = Pending(device, secrets.token_hex(16), now, now + CONFIRM_SECONDS, asyncio.get_running_loop().create_future())
    self.pending = pending
    cloudlog.event('asius.pairing.confirmation_requested', device=device)
    # The code stays on the device. Only soundd and the local device UI read it.
    self.publish({'phase': 'confirm', 'id': pending.request_id, 'digits': f'{passkey:06d}', 'created': now, 'expires': pending.expires})
    try:
      accepted = await asyncio.wait_for(asyncio.shield(pending.result), CONFIRM_SECONDS)
      if not accepted or self.pending is not pending or not self.enabled() or self.clock() >= pending.expires:
        raise PermissionError('Pairing confirmation was cancelled or expired')
      self.approved = (device, self.clock() + ENROLL_SECONDS)
      cloudlog.event('asius.pairing.confirmation_accepted', device=device)
    except BaseException:
      if self.pending is pending:
        self.cancel(device)
      raise
    finally:
      if self.pending is pending:
        self.pending = None
        self.publish(None)
      if not pending.result.done():
        pending.result.cancel()

  def button(self, action: str, started: float) -> bool:
    """Consume physical input only for this request, never the hold that opened pairing."""
    self.expire()
    pending = self.pending
    if pending is None:
      if action == 'press' and self.approved:
        self.cancel(self.approved[0])
        return True
      return False
    now = self.clock()
    cloudlog.event('asius.pairing.button', action=action, elapsed=now - started)
    if not math.isfinite(started) or not pending.created <= started <= now:
      return True
    if action == 'press':
      self.cancel(pending.device)
    elif action == 'hold' and now - started >= 3. and not pending.result.done():
      pending.result.set_result(True)
    return True

  def cancel(self, device: str | None = None) -> None:
    rejected = None
    if self.pending and (device is None or self.pending.device == device):
      rejected = self.pending.device
      if not self.pending.result.done():
        self.pending.result.set_result(False)
      self.pending = None
      self.publish(None)
    if self.approved and (device is None or self.approved[0] == device):
      rejected = self.approved[0]
      self.approved = None
    if rejected:
      cloudlog.event('asius.pairing.cancelled', device=rejected)
      self.failed(rejected)

  def failed(self, device: str) -> None:
    announce('failed', self.root, self.clock())
    if self.on_reject:
      self.on_reject(device)

  def screen(self, request_id: str, accepted: bool) -> None:
    self.expire()
    if not self.allow_ui or not self.pending or self.pending.request_id != request_id:
      return
    if not accepted:
      self.cancel()
    elif not self.pending.result.done():
      self.pending.result.set_result(True)

  def expire(self) -> None:
    if not self.enabled():
      self.cancel()
    if self.pending and self.clock() >= self.pending.expires:
      self.cancel(self.pending.device)
    if self.approved and self.clock() >= self.approved[1]:
      self.cancel(self.approved[0])

  def consume(self, device: str) -> bool:
    self.expire()
    if not self.approved or self.approved[0] != device:
      return False
    self.approved = None
    cloudlog.event('asius.pairing.enrollment_approved', device=device)
    return True

  async def buttons(self, open_window: Callable[[], object]) -> None:
    path = self.root / 'button.sock'
    path.unlink(missing_ok=True)
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
    sock.setblocking(False)
    sock.bind(str(path))
    os.chmod(path, 0o600)
    try:
      while True:
        self.expire()
        try:
          payload = await asyncio.wait_for(asyncio.get_running_loop().sock_recv(sock, 512), .1)
          value = json.loads(payload)
          if value.get('action') == 'screen' and isinstance(value.get('accepted'), bool):
            self.screen(value.get('id'), value['accepted'])
            continue
          action, started = value['action'], float(value['started'])
          if action not in ('press', 'hold') or not math.isfinite(started):
            continue
          handled = self.button(action, started)
          if not handled and action == 'hold' and 3. <= self.clock() - started <= 5. and not self.approved:
            open_window()
        except (TimeoutError, ValueError, KeyError, TypeError, PermissionError):
          pass
    finally:
      self.cancel()
      sock.close()
      path.unlink(missing_ok=True)


def main() -> None:
  """Exercise the physical input path from a local maintenance shell."""
  import argparse

  parser = argparse.ArgumentParser(description='Simulate a pairing button event through the private local socket.')
  parser.add_argument('action', choices=('press', 'hold', 'status'))
  action = parser.parse_args().action
  if action == 'status':
    pending = read_confirmation()
    print(json.dumps({'pending': pending is not None,
                      'secondsRemaining': max(0, math.ceil(pending['expires'] - time.monotonic())) if pending else 0}))
    return
  started = time.monotonic()
  if action == 'hold':
    time.sleep(3.1)
  if not send_button(action, started):
    raise SystemExit('Bluetooth button service is unavailable')
  print('Button event delivered')


if __name__ == '__main__':
  main()
