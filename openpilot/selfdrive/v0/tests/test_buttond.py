import os
from unittest.mock import patch

from openpilot.selfdrive.v0.buttond import (
  EV_KEY,
  INPUT_EVENT,
  KEY_PRESSED,
  KEY_PROG1,
  KEY_RELEASED,
  ButtonAction,
  ButtonEvent,
  DeviceButtons,
  find_button_keys,
)


def input_event(value: int) -> bytes:
  return INPUT_EVENT.pack(0, 0, EV_KEY, KEY_PROG1, value)


def test_pair_after_long_hold():
  read_fd, write_fd = os.pipe()
  try:
    button = DeviceButtons(event_device=f"/proc/self/fd/{read_fd}")
    os.write(write_fd, input_event(KEY_PRESSED))

    assert button.poll(now=1.0) == []
    assert button.poll(now=3.9) == []
    assert button.poll(now=4.0) == [ButtonEvent(ButtonAction.PAIR, 1.)]
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
    button = DeviceButtons(event_device=f"/proc/self/fd/{read_fd}")
    os.write(write_fd, input_event(KEY_PRESSED))
    assert button.poll(now=1.0) == []

    os.write(write_fd, input_event(KEY_RELEASED))
    assert button.poll(now=2.0) == [ButtonEvent(ButtonAction.CANCEL, 1.)]
    assert button.poll(now=5.0) == []
  finally:
    button.close()
    os.close(read_fd)
    os.close(write_fd)


def test_find_both_buttons_ignores_headset(tmp_path):
  names = []
  for index, name in enumerate(("pmic_pwrkey", "gpio-keys", "QCS6490-Radxa-Dragon-Q6A Headset Jack")):
    path = tmp_path / f"event{index}" / "device/name"
    path.parent.mkdir(parents=True)
    path.write_text(name + "\n")
    names.append(str(path))
  with patch("openpilot.selfdrive.v0.buttond.glob.glob", return_value=names):
    assert find_button_keys() == ["/dev/input/event0", "/dev/input/event1"]
  with patch("openpilot.selfdrive.v0.buttond.glob.glob", return_value=names[:1]):
    assert find_button_keys() == ["/dev/input/event0"]


def test_other_keys_do_not_pair():
  read_fd, write_fd = os.pipe()
  button = DeviceButtons(event_device=f"/proc/self/fd/{read_fd}")
  try:
    for code in (116, 115):  # power, volume up
      os.write(write_fd, INPUT_EVENT.pack(0, 0, EV_KEY, code, KEY_PRESSED))
    assert button.poll(now=1.) == []
    assert button.poll(now=5.) == []
  finally:
    button.close()
    os.close(read_fd)
    os.close(write_fd)


def test_queued_short_release_does_not_become_confirmation_after_stall():
  read_fd, write_fd = os.pipe()
  button = DeviceButtons(event_device=f"/proc/self/fd/{read_fd}")
  try:
    os.write(write_fd, INPUT_EVENT.pack(1, 0, EV_KEY, KEY_PROG1, KEY_PRESSED))
    assert button.poll(now=1.) == []
    os.write(write_fd, INPUT_EVENT.pack(1, 100000, EV_KEY, KEY_PROG1, KEY_RELEASED))
    assert button.poll(now=5.) == [ButtonEvent(ButtonAction.CANCEL, 1.)]
  finally:
    button.close()
    os.close(read_fd)
    os.close(write_fd)


def test_both_buttons_keep_independent_holds_and_releases():
  pipes = [os.pipe(), os.pipe()]
  paths = [f"/proc/self/fd/{r}" for r, _ in pipes]
  button = DeviceButtons()
  try:
    with patch("openpilot.selfdrive.v0.buttond.find_button_keys", return_value=paths):
      os.write(pipes[0][1], input_event(KEY_PRESSED))
      assert button.poll(1.) == []
      os.write(pipes[1][1], input_event(KEY_PRESSED))
      assert button.poll(2.) == []
      os.write(pipes[1][1], input_event(KEY_RELEASED))
      assert button.poll(2.2) == [ButtonEvent(ButtonAction.CANCEL, 2.)]
      assert button.poll(4.) == [ButtonEvent(ButtonAction.PAIR, 1.)]
      os.write(pipes[0][1], input_event(KEY_RELEASED))
      assert button.poll(4.1) == []
      os.write(pipes[1][1], input_event(KEY_PRESSED))
      assert button.poll(5.) == []
      assert button.poll(8.) == [ButtonEvent(ButtonAction.PAIR, 5.)]
      os.write(pipes[1][1], input_event(KEY_RELEASED))
      assert button.poll(8.1) == []
  finally:
    button.close()
    for r, w in pipes:
      os.close(r)
      os.close(w)


def test_disconnected_button_cannot_complete_hold_and_replacement_works():
  r, w = os.pipe()
  path = f"/proc/self/fd/{r}"
  button = DeviceButtons()
  try:
    with patch("openpilot.selfdrive.v0.buttond.find_button_keys", return_value=[path]):
      os.write(w, input_event(KEY_PRESSED))
      assert button.poll(1.) == []
    with patch("openpilot.selfdrive.v0.buttond.find_button_keys", return_value=[]):
      assert button.poll(6.) == []
      assert not button.buttons
    with patch("openpilot.selfdrive.v0.buttond.find_button_keys", return_value=[path]):
      os.write(w, input_event(KEY_RELEASED))
      assert button.poll(11.) == []
      os.write(w, input_event(KEY_PRESSED))
      assert button.poll(12.) == []
      os.write(w, input_event(KEY_RELEASED))
      assert button.poll(12.1) == [ButtonEvent(ButtonAction.CANCEL, 12.)]
  finally:
    button.close()
    os.close(r)
    os.close(w)
