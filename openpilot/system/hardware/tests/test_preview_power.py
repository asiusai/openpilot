from itertools import product
from unittest.mock import Mock, patch

import pytest

from openpilot.system.hardware.hardwared import should_power_save
from openpilot.system.manager import process_config


@pytest.mark.parametrize("ignition,driver_view,live_stream", list(product((False, True), repeat=3)))
def test_headless_camera_activity_keeps_cpus_online(ignition, driver_view, live_stream):
  assert should_power_save(ignition, 0, driver_view, live_stream) == (not any((ignition, driver_view, live_stream)))


def test_visible_screen_prevents_power_save():
  assert not should_power_save(False, 1, False, False)


@pytest.mark.parametrize("predicate", (process_config.driverview, process_config.livestream))
def test_preview_waits_for_both_camera_cpus(predicate):
  params = Mock()
  params.get_bool.return_value = True
  with patch.object(process_config, "ASIUS_HARDWARE", True), patch.object(process_config.os, "sched_getaffinity") as affinity:
    for online in (set(range(4)), set(range(7)), {0, 1, 2, 3, 7}):
      affinity.return_value = online
      assert not predicate(False, params, None)
    affinity.return_value = set(range(8))
    assert predicate(False, params, None)
    params.get_bool.return_value = False
    assert not predicate(False, params, None)


@pytest.mark.parametrize("predicate", (process_config.driverview, process_config.livestream))
def test_comma_preview_keeps_existing_process_policy(predicate):
  params = Mock()
  params.get_bool.return_value = True
  with patch.object(process_config, "ASIUS_HARDWARE", False), patch.object(process_config.os, "sched_getaffinity") as affinity:
    assert predicate(False, params, None)
    affinity.assert_not_called()


def test_onroad_process_policy_does_not_wait_for_preview():
  params = Mock()
  params.get_bool.return_value = True
  with patch.object(process_config, "offroad_camera_ready") as ready:
    assert process_config.driverview(True, params, None)
    assert process_config.livestream(True, params, None)
    ready.assert_not_called()
