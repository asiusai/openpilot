"""Fresh, peer-bound phone time samples, independent of the device wall clock."""
import json
import math
import secrets
import subprocess
import threading
import time


class ClockChallenges:
  def __init__(self):
    self.pending: dict[str, tuple[str, float]] = {}
    self.lock = threading.Lock()

  def challenge(self, peer: str) -> dict:
    with self.lock:
      now = time.monotonic()
      self.pending = {key: value for key, value in self.pending.items() if now - value[1] < 5}
      token = secrets.token_hex(24)
      self.pending[peer] = (token, now)
      return {'challenge': token}

  def sync(self, peer: str, challenge: str, unixTimeMs: float) -> dict:
    if isinstance(unixTimeMs, bool) or not isinstance(unixTimeMs, (int, float)) or not math.isfinite(unixTimeMs):
      raise ValueError('Invalid time sample')
    with self.lock:
      pending = self.pending.get(peer)
      if pending is None or not isinstance(challenge, str) or not secrets.compare_digest(pending[0], challenge):
        raise ValueError('Invalid clock challenge')
      del self.pending[peer]
      if not 0 <= time.monotonic() - pending[1] <= 5:
        raise ValueError('Clock challenge expired')
    result = subprocess.run(['sudo', '-n', '/usr/bin/vamos-clock', 'sync', '--source', 'app', '--unix-ms', str(unixTimeMs)],
                            capture_output=True, text=True, timeout=10, check=True)
    return json.loads(result.stdout)
