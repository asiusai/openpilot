import subprocess
from functools import cached_property

from openpilot.common.hardware.base import ThermalZone
from openpilot.common.hardware.asius.thermal import AsiusThermalConfig, HwmonThermalZone
from openpilot.common.hardware.asius.ufs import UfsHealthReader
from openpilot.common.hardware.comma.hardware import HardwareComma


class HardwareAsius(HardwareComma):
  @cached_property
  def amplifier(self):
    return None

  def get_device_type(self):
    return "v0"

  def get_serial(self):
    with open("/sys/devices/soc0/serial_number") as serial_file:
      return serial_file.read().strip()

  @cached_property
  def ufs_health(self):
    return UfsHealthReader()

  def get_thermal_config(self):
    return AsiusThermalConfig(cpu=[ThermalZone(f"cpu{i}-thermal") for i in range(8)],
                              gpu=[ThermalZone("gpuss0-thermal"), ThermalZone("gpuss1-thermal")],
                              dsp=ThermalZone("nspss0-thermal"),
                              memory=ThermalZone("ddr-thermal"),
                              thermal_zones={"ufsBoard": ThermalZone("ufs-thermal"),
                                             "ufsCase": HwmonThermalZone("ufsCase", "ufs", poll_interval=30.)})

  def get_ufs_health(self) -> dict:
    return self.ufs_health.read()

  def set_power_save(self, powersave_enabled):
    subprocess.run(["sudo", "/usr/bin/vamos-hardware", "gpu-power-save", "on" if powersave_enabled else "off"], check=True)
    super().set_power_save(powersave_enabled)

  def initialize_hardware(self):
    subprocess.run(["sudo", "/usr/bin/vamos-hardware", "initialize"], check=True)
