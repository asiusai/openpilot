from unittest.mock import patch

import pytest

from openpilot.common.hardware.asius.gpu import GpuUsage, gpu_clients


def client(proc, pid, fd, client_id, busy, driver="msm"):
  process = proc / str(pid)
  (process / "fd").mkdir(parents=True, exist_ok=True)
  (process / "fdinfo").mkdir(exist_ok=True)
  link = process / "fd" / str(fd)
  if not link.is_symlink():
    link.symlink_to("/dev/dri/renderD128")
  info = process / "fdinfo" / str(fd)
  info.write_text(f"drm-driver:\t{driver}\ndrm-client-id:\t{client_id}\ndrm-engine-gpu:\t{busy} ns\n")
  return info


def test_duplicate_descriptors_and_other_drivers(tmp_path):
  client(tmp_path, 1, 4, 7, 100)
  client(tmp_path, 1, 5, 7, 150)
  client(tmp_path, 2, 6, 7, 125)
  client(tmp_path, 2, 7, 8, 200)
  client(tmp_path, 3, 8, 9, 999, driver="amdgpu")
  (tmp_path / "self").mkdir()
  assert gpu_clients(tmp_path) == {7: 150, 8: 200}


def test_busy_time_and_client_lifecycle(tmp_path):
  usage = GpuUsage(tmp_path)
  info = client(tmp_path, 1, 4, 7, 2_000_000_000)
  with patch("time.monotonic_ns", side_effect=[0, 1_000_000_000, 2_000_000_000, 3_000_000_000]):
    assert usage.read() == -1  # Existing lifetime work is not interval usage.
    client(tmp_path, 1, 4, 7, 2_250_000_000)
    client(tmp_path, 1, 5, 8, 1_000_000_000)  # Establish a new client's baseline.
    assert usage.read() == -1
    client(tmp_path, 1, 4, 7, 2_500_000_000)
    client(tmp_path, 1, 5, 8, 1_500_000_000)
    assert usage.read() == 75
    info.unlink()  # Processes and descriptors can disappear during a scan.
    assert usage.read() == 0


def test_regressing_counters_retain_high_water_mark(tmp_path):
  usage = GpuUsage(tmp_path)
  with patch("time.monotonic_ns", side_effect=[0, 1_000_000_000, 2_000_000_000, 3_000_000_000]):
    client(tmp_path, 1, 4, 7, 1_000_000_000)
    assert usage.read() == -1
    client(tmp_path, 1, 4, 7, 900_000_000)
    assert usage.read() == 0
    client(tmp_path, 1, 4, 7, 1_100_000_000)
    assert usage.read() == 10
    client(tmp_path, 1, 4, 7, 4_000_000_000)
    assert usage.read() == 100


@pytest.mark.parametrize("text", ["drm-driver: msm\n", "drm-driver: msm\ndrm-client-id: 7\ndrm-engine-gpu: -1 ns\n"])
def test_unavailable_counters_are_not_reported_as_idle(tmp_path, text):
  usage = GpuUsage(tmp_path)
  client(tmp_path, 1, 4, 7, 0).write_text(text)
  assert usage.read() == -1
  client(tmp_path, 1, 4, 7, 100)
  assert usage.read() == -1


def test_idle_and_missing_proc(tmp_path):
  assert GpuUsage(tmp_path).read() == 0
  assert GpuUsage(tmp_path / "missing").read() == -1
