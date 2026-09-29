from pathlib import Path
from unittest.mock import patch

from openpilot.common.hardware.asius.thermal import AsiusThermalConfig, HwmonThermalZone
from openpilot.common.hardware.base import ThermalConfig, ThermalZone


def write_value(root: Path, relative_path: str, value: str) -> None:
  path = root / relative_path
  path.parent.mkdir(parents=True, exist_ok=True)
  path.write_text(value)


def test_hwmon_temperature_and_generic_thermal_zones(tmp_path: Path):
  hwmon = tmp_path / "hwmon7"
  write_value(hwmon, "name", "ufs\n")
  write_value(hwmon, "temp1_input", "41875\n")

  ufs_case = HwmonThermalZone("ufsCase", "ufs", poll_interval=0., hwmon_root=str(tmp_path))
  ufs_board = ThermalZone("unused")
  ufs_board.read = lambda: 38.5  # type: ignore[method-assign]

  assert ufs_case.read() == 41.875
  assert AsiusThermalConfig(cpu=[ufs_board], thermal_zones={"ufsBoard": ufs_board, "ufsCase": ufs_case}).get_msg() == {
    "cpuTempC": [38.5],
    "thermalZones": [
      {"name": "ufsBoard", "temp": 38.5},
      {"name": "ufsCase", "temp": 41.875},
    ],
  }
  assert ThermalConfig(cpu=[ufs_board]).get_msg() == {"cpuTempC": [38.5]}


def test_hwmon_polling_and_device_replacement(tmp_path: Path):
  hwmon = tmp_path / "hwmon7"
  write_value(hwmon, "name", "ufs\n")
  write_value(hwmon, "temp1_input", "41875\n")
  sensor = HwmonThermalZone("ufsCase", "ufs", hwmon_root=str(tmp_path))
  with patch("openpilot.common.hardware.asius.thermal.time.monotonic", side_effect=[0., 1., 30., 60.]):
    assert sensor.read() == 41.875
    (hwmon / "temp1_input").unlink()
    assert sensor.read() == 41.875  # cached until the next poll
    assert sensor.read() == 41.875  # retain the last reading during reprobe
    (hwmon / "name").unlink()
    write_value(tmp_path / "hwmon8", "name", "ufs\n")
    write_value(tmp_path / "hwmon8", "temp1_input", "42000\n")
    assert sensor.read() == 42.
