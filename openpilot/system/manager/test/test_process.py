import multiprocessing as mp
import time

import pytest

from openpilot.system.manager.process import PythonProcess


@pytest.mark.parametrize("start_method", [None, "spawn"])
def test_python_process_restart_and_cleanup(tmp_path, monkeypatch, start_method):
  """Exercise the real launcher, SIGINT shutdown, restart, and spawned child exit handlers."""
  (tmp_path / "manager_process_fixture.py").write_text('''
import atexit
import multiprocessing as mp
import os
import time
from pathlib import Path

def main():
  output = Path(os.environ["MANAGER_PROCESS_TEST_OUTPUT"])
  atexit.register((output / "exited").write_text, "clean")
  (output / "started").write_text(mp.get_start_method())
  while True:
    time.sleep(.01)
''')
  monkeypatch.syspath_prepend(str(tmp_path))
  monkeypatch.setenv("MANAGER_PROCESS_TEST_OUTPUT", str(tmp_path))
  process = PythonProcess("process-test", "manager_process_fixture", lambda *args: True, start_method=start_method)
  for _ in range(2):
    (tmp_path / "started").unlink(missing_ok=True)
    (tmp_path / "exited").unlink(missing_ok=True)
    try:
      process.start()
      deadline = time.monotonic() + 10
      while not (tmp_path / "started").is_file():
        assert process.proc.exitcode is None, "Child exited before starting"
        assert time.monotonic() < deadline, "Child initialization timed out"
        time.sleep(.01)
      assert (tmp_path / "started").read_text() == (start_method or mp.get_start_method())
      assert process.stop() == 0
      assert process.proc is None and not process.shutting_down
      if start_method == "spawn":
        assert (tmp_path / "exited").read_text() == "clean"
    finally:
      process.stop()
