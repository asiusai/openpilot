import io
from unittest.mock import MagicMock

import pytest

from openpilot.selfdrive.v0 import ledd
from openpilot.system.manager import build


@pytest.mark.parametrize('asius,results', [(True, [1, 1, 1]), (True, [1, 0]), (False, [1, 1, 1])])
def test_build_failure_leds(monkeypatch, tmp_path, asius, results):
  monkeypatch.setattr(build, 'ASIUS_HARDWARE', asius)
  monkeypatch.setattr(build, 'AGNOS', False)
  monkeypatch.setattr(build, 'Spinner', MagicMock())
  monkeypatch.setattr(build.HARDWARE, 'set_power_save', MagicMock())
  monkeypatch.setattr(build.os, 'sync', MagicMock())
  monkeypatch.setenv('CI', '1')
  monkeypatch.setattr(ledd, 'CAM_LED_SYSFS_ROOT', tmp_path)
  boards = ledd.CameraLeds().boards
  for board in boards:
    for path in board.channel_paths:
      path.mkdir()
      (path / 'brightness').write_text('17')
      (path / 'trigger').write_text('asius-startup')

  attempts = []
  for result in results:
    process = MagicMock(returncode=result, stderr=io.BytesIO(b'compiler error\n' if result else b''))
    process.poll.return_value = result
    process.__enter__.return_value = process
    attempts.append(process)
  popen = MagicMock(side_effect=attempts)
  monkeypatch.setattr(build.subprocess, 'Popen', popen)

  if results[-1]:
    with pytest.raises(SystemExit, match='1'):
      build.build()
  else:
    build.build()
  assert popen.call_count == len(results)

  for board in boards:
    for channel, path in enumerate(board.channel_paths):
      if asius and results[-1]:
        # No camera exposure exists yet. Middle LEDs and the driver board stay off.
        expected = ledd.BOOT_BRIGHTNESS if board.camera_num in (2, 3) and channel in (0, 6) else 0
        assert (path / 'trigger').read_text() == 'none'
      else:
        expected = 17
      assert int((path / 'brightness').read_text()) == expected
