import os
from unittest.mock import patch

from openpilot.selfdrive.v0.buttond import (
  EV_KEY,
  INPUT_EVENT,
  KEY_PRESSED,
  KEY_PROG1,
  KEY_RELEASED,
  ButtonAction,
  PandaButton,
  find_panda_key,
)


def input_event(value: int) -> bytes:
  return INPUT_EVENT.pack(0, 0, EV_KEY, KEY_PROG1, value)


def test_pair_after_long_hold():
  read_fd, write_fd = os.pipe()
  try:
    button = PandaButton(event_device=f"/proc/self/fd/{read_fd}")
    os.write(write_fd, input_event(KEY_PRESSED))

    assert button.poll(now=1.0) == []
    assert button.poll(now=3.9) == []
    assert button.poll(now=4.0) == [ButtonAction.PAIR]
    assert button.poll(now=5.0) == []

    os.write(write_fd, input_event(KEY_RELEASED))
    assert button.poll(now=5.1) == []
  finally:
    button.close()
    os.close(read_fd)
    os.close(write_fd)


def test_short_press_does_not_pair():
  read_fd, write_fd = os.pipe()
  try:
    button = PandaButton(event_device=f"/proc/self/fd/{read_fd}")
    os.write(write_fd, input_event(KEY_PRESSED))
    assert button.poll(now=1.0) == []

    os.write(write_fd, input_event(KEY_RELEASED))
    assert button.poll(now=2.0) == []
    assert button.poll(now=5.0) == []
  finally:
    button.close()
    os.close(read_fd)
    os.close(write_fd)


def test_find_panda_button_ignores_power_key(tmp_path):
  names = []
  for index, name in enumerate(("pmic_pwrkey", "gpio-keys")):
    path = tmp_path / f"event{index}" / "device/name"
    path.parent.mkdir(parents=True)
    path.write_text(name + "\n")
    names.append(str(path))
  with patch("openpilot.selfdrive.v0.buttond.glob.glob", return_value=names):
    assert find_panda_key() == "/dev/input/event1"
  with patch("openpilot.selfdrive.v0.buttond.glob.glob", return_value=names[:1]):
    assert find_panda_key() is None


def test_other_keys_do_not_pair():
  read_fd, write_fd = os.pipe()
  button = PandaButton(event_device=f"/proc/self/fd/{read_fd}")
  try:
    for code in (116, 115):  # power, volume up
      os.write(write_fd, INPUT_EVENT.pack(0, 0, EV_KEY, code, KEY_PRESSED))
    assert button.poll(now=1.) == []
    assert button.poll(now=5.) == []
  finally:
    button.close()
    os.close(read_fd)
    os.close(write_fd)
