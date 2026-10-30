#!/usr/bin/env python3
import argparse
import glob
import fcntl
import os
import signal
import stat
import struct
import time
from dataclasses import dataclass, field
from enum import Enum, auto

EV_KEY = 0x01
KEY_PROG1 = 148
KEY_RELEASED = 0
KEY_PRESSED = 1
PAIR_HOLD_SECONDS = 3.0
RECONNECT_SECONDS = 5.0
RUNTIME_HZ = 20.0
INPUT_EVENT = struct.Struct("@llHHI")


class ButtonAction(Enum):
  PAIR = auto()
  CANCEL = auto()


@dataclass(frozen=True)
class ButtonEvent:
  action: ButtonAction
  started: float


def find_button_keys() -> list[str]:
  paths = []
  for name_path in sorted(glob.glob("/sys/class/input/event*/device/name")):
    try:
      with open(name_path, encoding="utf-8") as f:
        if f.read().strip() in ("gpio-keys", "pmic_pwrkey"):
          event_name = name_path.split("/")[-3]
          paths.append(f"/dev/input/{event_name}")
    except OSError:
      continue
  return paths


@dataclass
class InputButton:
  event_device: str
  hold_seconds: float = PAIR_HOLD_SECONDS
  fd: int | None = None
  pressed_at: float | None = None
  pair_sent: bool = False
  next_retry: float = 0.0
  pending: bytes = b""

  def connect(self, now: float) -> bool:
    if self.fd is not None:
      return True
    if now < self.next_retry:
      return False

    try:
      self.fd = os.open(self.event_device, os.O_RDONLY | os.O_NONBLOCK)
      if stat.S_ISCHR(os.fstat(self.fd).st_mode):
        # EVIOCSCLOCKID: use event timestamps, including releases queued during a stall.
        fcntl.ioctl(self.fd, 0x400445a0, struct.pack('i', time.CLOCK_MONOTONIC))
      self.pending = b""
      return True
    except OSError:
      self.close()
      self.next_retry = now + RECONNECT_SECONDS
      return False

  def close(self) -> None:
    if self.fd is not None:
      os.close(self.fd)
      self.fd = None
    self.pressed_at = None
    self.pair_sent = False
    self.pending = b""

  def _long_hold(self, now: float) -> list[ButtonEvent]:
    if self.pressed_at is not None and not self.pair_sent and now - self.pressed_at >= self.hold_seconds:
      self.pair_sent = True
      return [ButtonEvent(ButtonAction.PAIR, self.pressed_at)]
    return []

  def poll(self, now: float | None = None) -> list[ButtonEvent]:
    now = time.monotonic() if now is None else now
    actions: list[ButtonEvent] = []
    if not self.connect(now):
      return actions
    fd = self.fd
    if fd is None:
      return actions

    try:
      while True:
        chunk = os.read(fd, INPUT_EVENT.size * 16)
        if not chunk:
          self.close()
          break
        self.pending += chunk
    except BlockingIOError:
      pass
    except OSError:
      self.close()
      return actions

    complete = len(self.pending) - (len(self.pending) % INPUT_EVENT.size)
    for offset in range(0, complete, INPUT_EVENT.size):
      seconds, micros, event_type, code, value = INPUT_EVENT.unpack_from(self.pending, offset)
      event_time = seconds + micros / 1e6 if seconds or micros else now
      if event_type != EV_KEY or code != KEY_PROG1:
        continue

      if value == KEY_PRESSED and self.pressed_at is None:
        self.pressed_at = event_time
        self.pair_sent = False
      elif value == KEY_RELEASED and self.pressed_at is not None:
        actions.extend(self._long_hold(event_time))
        if not self.pair_sent:
          actions.append(ButtonEvent(ButtonAction.CANCEL, self.pressed_at))
        self.pressed_at = None
        self.pair_sent = False

    self.pending = self.pending[complete:]
    actions.extend(self._long_hold(now))
    return actions


@dataclass
class DeviceButtons:
  event_device: str | None = None
  buttons: dict[str, InputButton] = field(default_factory=dict)
  next_scan: float = 0.0

  def poll(self, now: float | None = None) -> list[ButtonEvent]:
    now = time.monotonic() if now is None else now
    if now >= self.next_scan:
      paths = [self.event_device] if self.event_device else find_button_keys()
      for path in self.buttons.keys() - set(paths):
        self.buttons.pop(path).close()
      for path in paths:
        if path not in self.buttons:
          self.buttons[path] = InputButton(path)
      self.next_scan = now + RECONNECT_SECONDS
    # Each physical button keeps its own hold timer and release state.
    return sorted((event for button in self.buttons.values() for event in button.poll(now)), key=lambda event: event.started)

  def close(self) -> None:
    for button in self.buttons.values():
      button.close()
    self.buttons.clear()
    self.next_scan = 0.0


def main() -> None:
  from openpilot.common.realtime import Ratekeeper
  from openpilot.common.swaglog import cloudlog
  from openpilot.system.asius.pairing_confirmation import send_button

  parser = argparse.ArgumentParser()
  parser.add_argument("--event-device")
  args = parser.parse_args()

  done = False

  def sigterm_handler(signum, frame) -> None:
    nonlocal done
    done = True

  signal.signal(signal.SIGINT, sigterm_handler)
  signal.signal(signal.SIGTERM, sigterm_handler)

  button = DeviceButtons(event_device=args.event_device)
  rk = Ratekeeper(RUNTIME_HZ)

  while not done:
    for event in button.poll():
      if not send_button('hold' if event.action == ButtonAction.PAIR else 'press', event.started):
        cloudlog.warning("asius.button.bluetooth_unavailable")
    rk.keep_time()

  button.close()


if __name__ == "__main__":
  main()
