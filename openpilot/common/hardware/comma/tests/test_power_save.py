from pathlib import Path
from unittest.mock import call, patch

import pytest

from openpilot.common.hardware.comma import hardware


@pytest.mark.parametrize('domains', [(0, 4), (0, 4, 7)])
@pytest.mark.parametrize('powersave_enabled', [False, True])
def test_cpu_frequency_domains(domains, powersave_enabled):
  policies = [Path(f'/sys/devices/system/cpu/cpufreq/policy{n}') for n in domains]
  device = hardware.HardwareComma()
  device.amplifier = None
  with patch.object(Path, 'glob', return_value=policies), patch.object(hardware, 'sudo_write') as write, \
       patch.object(hardware, 'affine_irq'):
    device.set_power_save(powersave_enabled)

  frequency_writes = [c for c in write.call_args_list if '/cpufreq/' in c.args[1]]
  if powersave_enabled:
    assert frequency_writes == [call('ondemand', str(policies[0] / 'scaling_governor'))]
  else:
    assert frequency_writes == [c for p in policies for c in (
      call('performance', str(p / 'scaling_governor')), call('1689600', str(p / 'scaling_max_freq')))]
