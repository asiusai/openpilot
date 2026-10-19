import os
import time
from pathlib import Path


def gpu_clients(proc: Path) -> dict[int, int]:
  """Read MSM busy time in nanoseconds, counting each DRM client once."""
  clients: dict[int, int] = {}
  for process in proc.iterdir():
    if not process.name.isdecimal():
      continue
    try:
      fds = list((process / "fd").iterdir())
    except (FileNotFoundError, ProcessLookupError, PermissionError):
      continue
    for fd in fds:
      try:
        if not os.readlink(fd).startswith("/dev/dri/"):
          continue
      except (FileNotFoundError, ProcessLookupError, PermissionError):
        continue
      try:
        fields = dict(line.split(":", 1) for line in (process / "fdinfo" / fd.name).read_text().splitlines() if ":" in line)
      except (FileNotFoundError, ProcessLookupError):
        continue
      if fields.get("drm-driver", "").strip() != "msm":
        continue
      client = int(fields["drm-client-id"])
      busy, unit = fields["drm-engine-gpu"].split()
      if unit != "ns" or int(busy) < 0:
        raise ValueError("Invalid MSM busy-time counter")
      clients[client] = max(clients.get(client, 0), int(busy))
  return clients


class GpuUsage:
  def __init__(self, proc: Path = Path("/proc")):
    self.proc = proc
    self.previous: dict[int, int] = {}
    self.time_ns: int | None = None

  def read(self) -> float:
    try:
      clients = gpu_clients(self.proc)
    except (OSError, ValueError, KeyError):
      self.previous = {}
      self.time_ns = None
      return -1
    now = time.monotonic_ns()
    elapsed = now - self.time_ns if self.time_ns is not None else 0
    new_clients = clients.keys() - self.previous.keys()
    # DRM counters can briefly move backward. Keep their high-water marks.
    clients = {client: max(busy, self.previous.get(client, busy)) for client, busy in clients.items()}
    busy = sum(value - self.previous.get(client, value) for client, value in clients.items())
    self.previous, self.time_ns = clients, now
    if not clients:
      return 0
    if elapsed <= 0 or new_clients:
      return -1
    return min(100., 100. * busy / elapsed)
