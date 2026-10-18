import subprocess
from functools import cached_property

from openpilot.common.hardware.base import ThermalConfig, ThermalZone
from openpilot.common.hardware.comma.hardware import HardwareComma, affine_irq


class HardwareAsius(HardwareComma):
  @cached_property
  def amplifier(self):
    return None

  def get_device_type(self):
    return "v0"

  def get_serial(self):
    with open("/sys/devices/soc0/serial_number") as serial_file:
      return serial_file.read().strip()

  def set_ir_power(self, percent: int):
    with open("/sys/class/leds/asius:ir/brightness", "w") as brightness:
      brightness.write(str(max(0, min(100, percent))))

  def get_thermal_config(self):
    return ThermalConfig(cpu=[ThermalZone(f"cpu{i}-thermal") for i in range(8)],
                         gpu=[ThermalZone("gpuss0-thermal"), ThermalZone("gpuss1-thermal")],
                         dsp=ThermalZone("nspss0-thermal"),
                         memory=ThermalZone("ddr-thermal"))

  def set_power_save(self, powersave_enabled):
    subprocess.run(["sudo", "/usr/bin/vamos-hardware", "gpu-power-save", "on" if powersave_enabled else "off"], check=True)
    super().set_power_save(powersave_enabled)
    if not powersave_enabled:
      # Mainline CAMSS uses different IRQ names from AGNOS. Keep camera work
      # on camerad's core after the big CPUs come online.
      for action in ("ac4a000.cci", "ac4b000.cci"):
        affine_irq(6, action)
      for i in range(5):
        affine_irq(6, f"acb3000.isp_msm_csid{i}")
        affine_irq(6, f"acb3000.isp_msm_vfe{i}")

  def initialize_hardware(self):
    subprocess.run(["sudo", "/usr/bin/vamos-hardware", "initialize"], check=True)

  def reset_internal_panda(self):
    subprocess.run(["sudo", "/usr/bin/vamos-hardware", "panda-reset"], check=True)

  def recover_internal_panda(self):
    subprocess.run(["sudo", "/usr/bin/vamos-hardware", "panda-reset", "--bootloader"], check=True)
