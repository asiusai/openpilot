import subprocess
from functools import cached_property

from openpilot.common.hardware.base import ThermalConfig, ThermalZone
from openpilot.common.hardware.asius.gpu import GpuUsage
from openpilot.common.hardware.comma.hardware import HardwareComma, affine_irq
from openpilot.common.utils import sudo_write

# Karel f9ad9a96, 2026-11-19: with the small model on Adreno at 812 MHz, the internal USB hub, Chestnut and
# onboard PCIe dropped together within ~15 s (19-40 USB disconnects per run, two supply cables).
# At 700 and 608 MHz, 46 s runs logged none; the small model took 24.0 ms instead of 22.5 ms.
GPU_DEVFREQ = '/sys/class/devfreq/3d00000.gpu'
ONROAD_GPU_HZ = 700_000_000


class HardwareAsius(HardwareComma):
  @cached_property
  def gpu_usage(self):
    return GpuUsage()

  def get_gpu_usage_percent(self):
    return self.gpu_usage.read()

  @cached_property
  def amplifier(self):
    return None

  def get_device_type(self):
    return "v0"

  def get_serial(self):
    with open("/sys/devices/soc0/serial_number") as serial_file:
      return serial_file.read().strip()

  def set_ir_power(self, percent: int):
    # Automatic exposure uses at most 100 mA of the kernel's 150 mA full scale.
    level = max(0, min(100, percent)) * 100 // 150
    for camera in (2, 3):
      with open(f"/sys/class/leds/asius:cam{camera}:ir/brightness", "w") as brightness:
        brightness.write(str(level))

  def get_thermal_config(self):
    return ThermalConfig(cpu=[ThermalZone(f"cpu{i}-thermal") for i in range(8)],
                         gpu=[ThermalZone("gpuss0-thermal"), ThermalZone("gpuss1-thermal")],
                         dsp=ThermalZone("nspss0-thermal"),
                         memory=ThermalZone("ddr-thermal"))

  def set_power_save(self, powersave_enabled):
    subprocess.run(["sudo", "/usr/bin/vamos-hardware", "gpu-power-save", "on" if powersave_enabled else "off"], check=True)
    super().set_power_save(powersave_enabled)
    if not powersave_enabled:
      sudo_write(str(ONROAD_GPU_HZ), f'{GPU_DEVFREQ}/max_freq')
      sudo_write(str(ONROAD_GPU_HZ), f'{GPU_DEVFREQ}/userspace/set_freq')
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
