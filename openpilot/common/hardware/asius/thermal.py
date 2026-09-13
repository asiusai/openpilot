import os
import time
from dataclasses import dataclass, field

from openpilot.common.hardware.base import ThermalConfig, ThermalZone


class HwmonThermalZone(ThermalZone):
  def __init__(self, name: str, hwmon_name: str, attribute: str = "temp1_input", scale: float = 1000.,
               poll_interval: float = 30., hwmon_root: str = "/sys/class/hwmon"):
    super().__init__(name, scale)
    self.hwmon_name = hwmon_name
    self.attribute = attribute
    self.poll_interval = poll_interval
    self.hwmon_root = hwmon_root
    self._path: str | None = None
    self._last_read: float | None = None
    self._temperature = 0.

  def _find_path(self) -> str | None:
    try:
      hwmon_devices = os.listdir(self.hwmon_root)
    except FileNotFoundError:
      return None

    for device in hwmon_devices:
      device_path = os.path.join(self.hwmon_root, device)
      try:
        with open(os.path.join(device_path, "name")) as f:
          if f.read().strip() == self.hwmon_name:
            return os.path.join(device_path, self.attribute)
      except OSError:
        continue
    return None

  def read(self) -> float:
    now = time.monotonic()
    if self._last_read is not None and now - self._last_read < self.poll_interval:
      return self._temperature
    self._last_read = now

    if self._path is None:
      self._path = self._find_path()
    if self._path is None:
      return self._temperature

    try:
      with open(self._path) as f:
        self._temperature = int(f.read()) / self.scale
    except FileNotFoundError:
      self._path = None
    except (OSError, ValueError):
      pass
    return self._temperature


@dataclass
class AsiusThermalConfig(ThermalConfig):
  thermal_zones: dict[str, ThermalZone] = field(default_factory=dict)

  def get_msg(self):
    ret = super().get_msg()
    zones = [(name, zone.read()) for name, zone in self.thermal_zones.items()]
    ret["thermalZones"] = [{"name": name, "temp": temp} for name, temp in zones if temp != 0]
    return ret
