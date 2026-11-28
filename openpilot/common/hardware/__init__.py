import os
from importlib.util import find_spec
from typing import cast

from openpilot.common.hardware.base import HardwareBase
from openpilot.common.hardware.asius.hardware import HardwareAsius
from openpilot.common.hardware.comma.hardware import HardwareComma
from openpilot.common.hardware.pc.hardware import HardwarePc

AGNOS = os.path.isfile('/AGNOS')
COMMA_HARDWARE = AGNOS
ASIUS_HARDWARE = os.path.isfile('/ASIUS')
PC = not (COMMA_HARDWARE or ASIUS_HARDWARE)
# AGNOS is not changed for Asius. Bluetooth needs a kernel adapter class and the D-Bus client package there.
BLUETOOTH_AVAILABLE = ASIUS_HARDWARE or (COMMA_HARDWARE and os.path.isdir('/sys/class/bluetooth') and find_spec('dbus_fast') is not None)

if ASIUS_HARDWARE:
  HARDWARE = cast(HardwareBase, HardwareAsius())
elif COMMA_HARDWARE:
  HARDWARE = cast(HardwareBase, HardwareComma())
else:
  HARDWARE = cast(HardwareBase, HardwarePc())
