from unittest.mock import mock_open, patch

import pytest

from openpilot.common.hardware.asius.hardware import HardwareAsius


@pytest.mark.parametrize(('requested', 'expected'), [(-1, '0'), (0, '0'), (50, '33'), (100, '66'), (200, '66')])
def test_ir_controls_both_road_boards(requested, expected):
  files = mock_open()
  with patch('builtins.open', files):
    HardwareAsius().set_ir_power(requested)
  assert [call.args[0] for call in files.call_args_list] == [
    '/sys/class/leds/asius:cam2:ir/brightness', '/sys/class/leds/asius:cam3:ir/brightness',
  ]
  assert [call.args[0] for call in files().write.call_args_list] == [expected, expected]
