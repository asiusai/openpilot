#!/usr/bin/env python3

import importlib
import os
import unittest

import openpilot.common.hardware as hardware
from openpilot.common.test import OpenpilotTestCase
from openpilot.common.params import Params
import openpilot.system.manager.manager as manager
import openpilot.system.manager.process_config as process_config
from openpilot.system.manager.process_config import managed_processes, procs
from openpilot.common.hardware import HARDWARE

os.environ['FAKEUPLOAD'] = "1"


class TestManager(OpenpilotTestCase):
  def setup_method(self):
    HARDWARE.set_power_save(False)

    # ensure clean CarParams
    params = Params()
    params.clear_all()

  def teardown_method(self):
    manager.manager_cleanup()

  def test_duplicate_procs(self):
    assert len(procs) == len(managed_processes), "Duplicate process names"

  def test_device_specific_vision_processes(self):
    original_asius = hardware.ASIUS
    try:
      for asius in (False, True):
        hardware.ASIUS = asius
        config = importlib.reload(process_config)
        processes = config.managed_processes

        if asius:
          assert {"camerad_v1", "encoderd_v1", "stream_encoderd_v1"} <= processes.keys()
          assert not {"camerad", "encoderd", "stream_encoderd"} & processes.keys()
          assert processes["camerad_v1"].cmdline == ["./camerad_v1"]
          assert processes["modeld"].module == "openpilot.selfdrive.modeld.modeld_v1"
          assert processes["dmonitoringmodeld"].module == "openpilot.selfdrive.modeld.dmonitoringmodeld_v1"
        else:
          assert {"camerad", "encoderd", "stream_encoderd"} <= processes.keys()
          assert not {"camerad_v1", "encoderd_v1", "stream_encoderd_v1"} & processes.keys()
          assert processes["camerad"].cmdline == ["./camerad"]
          assert processes["modeld"].module == "openpilot.selfdrive.modeld.modeld"
          assert processes["dmonitoringmodeld"].module == "openpilot.selfdrive.modeld.dmonitoringmodeld"
    finally:
      hardware.ASIUS = original_asius
      importlib.reload(process_config)

  def test_set_params_with_default_value(self):
    params = Params()
    params.clear_all()

    os.environ['PREPAREONLY'] = '1'
    manager.main()
    for k in params.all_keys():
      default_value = params.get_default_value(k)
      if default_value is not None:
        assert params.get(k) == default_value
    assert params.get("OpenpilotEnabledToggle")
    assert params.get("RouteCount") == 0


if __name__ == "__main__":
  unittest.main()
