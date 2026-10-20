"""Small, transient upload status shared with authenticated app sessions."""
import json
import os
import time
from pathlib import Path

STATUS_PATH = Path('/dev/shm/asius-upload-progress.json')

def publish_error(path: str, error: str) -> None:
  try:
    temporary = STATUS_PATH.with_suffix(f'.{os.getpid()}.tmp')
    temporary.write_text(json.dumps({'path': path, 'sent': 0, 'total': 0, 'error': error, 'updated': time.monotonic()}))
    temporary.replace(STATUS_PATH)
  except OSError:
    pass


def read_progress():
  try:
    value = json.loads(STATUS_PATH.read_text())
    return value if 0 <= time.monotonic() - value['updated'] < 180 else None
  except (OSError, ValueError, KeyError):
    return None


class ProgressReader:
  def __init__(self, stream, path, total):
    self.stream, self.path, self.total = stream, path, total
    self.last = -1.0
    self.publish(0, force=True)

  def __getattr__(self, name):
    return getattr(self.stream, name)

  def __len__(self):
    return self.total

  def publish(self, sent, force=False):
    now = time.monotonic()
    if not force and now - self.last < 0.25:
      return
    self.last = now
    try:
      temporary = STATUS_PATH.with_suffix(f'.{os.getpid()}.tmp')
      temporary.write_text(json.dumps({'path': self.path, 'sent': sent, 'total': self.total, 'updated': now}))
      temporary.replace(STATUS_PATH)
    except OSError:
      pass  # Progress reporting must not interrupt an upload.

  def read(self, size=-1):
    data = self.stream.read(size)
    # Reserve completion until the cloud has confirmed the uploaded object.
    self.publish(min(self.stream.tell(), max(0, self.total - 1)))
    return data

  def complete(self):
    self.publish(self.total, force=True)
