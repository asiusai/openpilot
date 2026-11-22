"""Private side-camera configuration in /data/sidecam/config.json (mode 0600)."""
import json
import os
import time
from pathlib import Path

from openpilot.system.asius.identity import identity_to_bytes

CONFIG = Path('/data/sidecam/config.json')
RUNTIME = Path('/dev/shm/sidecam')
CONNECTED = RUNTIME / 'connected'  # present while a camera device is connected to its host
ADDRESS, PORT = '192.168.43.1', 39041
HOTSPOT_MHZ = 5180  # the VamOS camera hotspot uses channel 36
VIEWS = ('lf', 'lr', 'rf', 'rr')  # left/right side, front/rear facing
HELPER = '/usr/bin/vamos-local-video'


def validate(value):
  if not isinstance(value, dict) or value.get('role') not in ('host', 'camera'):
    raise ValueError('Choose host or camera role')
  allowed = {'role', 'network', 'helper'} | ({'peers', 'always'} if value['role'] == 'host' else {'host', 'wide', 'road'})
  if set(value) - allowed:
    raise ValueError('Unknown side camera configuration field')
  network = value.get('network')
  if not isinstance(network, dict) or set(network) != {'ssid', 'password'}:
    raise ValueError('Network needs an SSID and password')
  if not 1 <= len(network['ssid'].encode()) <= 32 or not 8 <= len(network['password']) <= 63 or \
     any(not 32 <= ord(c) <= 126 for c in network['ssid'] + network['password']):
    raise ValueError('Invalid camera network')
  if not isinstance(value.get('helper', HELPER), str):
    raise ValueError('Invalid network helper')
  if value['role'] == 'host':
    if type(value.get('always', False)) is not bool:
      raise ValueError('always must be a boolean')
    peers = value.get('peers')
    if not isinstance(peers, dict):
      raise ValueError('A host needs authorized camera peers')
    assigned = []
    for peer, views in peers.items():
      identity_to_bytes(peer)
      if not isinstance(views, list) or not views or any(v not in VIEWS for v in views):
        raise ValueError('Each peer needs its views: lf, lr, rf or rr')
      assigned += views
    if len(set(assigned)) != len(assigned):
      raise ValueError('Each view belongs to one peer')
  else:
    identity_to_bytes(value.get('host', ''))
    if value.get('wide') not in VIEWS or value.get('road') not in VIEWS or value['wide'] == value['road']:
      raise ValueError('A camera needs distinct wide and road views')
  return value


def load(path=CONFIG):
  if path.stat().st_mode & 0o077:
    raise ValueError('Side camera configuration must be private (0600)')
  return validate(json.loads(path.read_text()))


def save(value, path=CONFIG):
  validate(value)
  path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
  temporary = path.with_suffix('.tmp')
  fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
  with os.fdopen(fd, 'w') as output:
    json.dump(value, output)
  os.replace(temporary, path)


def current():
  try:
    return load()
  except (OSError, ValueError):
    return None


def streaming():
  """A camera device connected to its host within the last seconds."""
  try:
    return time.monotonic() - float(CONNECTED.read_text()) < 5 and role() == 'camera'
  except (OSError, ValueError):
    return False


def role():
  config = current()
  return config['role'] if config else None
