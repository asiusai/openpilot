import socket
import threading

import pytest

from openpilot.system.asius.pqc import RootIdentity
from openpilot.system.sidecam import config as sidecam_config
from openpilot.system.sidecam.transport import FRAME, HELLO, Connection

NETWORK = {'ssid': 'Asius-Cameras-test', 'password': 'not-a-real-password'}


def test_config_roles(tmp_path):
  host, camera = RootIdentity.generate().identity(), RootIdentity.generate().identity()
  path = tmp_path / 'config.json'
  sidecam_config.save({'role': 'camera', 'network': NETWORK, 'host': host, 'wide': 'lf', 'road': 'lr'}, path)
  assert path.stat().st_mode & 0o777 == 0o600
  assert sidecam_config.load(path)['wide'] == 'lf'
  sidecam_config.validate({'role': 'host', 'network': NETWORK, 'peers': {camera: ['lf', 'lr']}})

  with pytest.raises(ValueError):
    sidecam_config.validate({'role': 'camera', 'network': NETWORK, 'host': host, 'wide': 'lf', 'road': 'lf'})
  with pytest.raises(ValueError):
    sidecam_config.validate({'role': 'host', 'network': NETWORK, 'peers': {camera: ['lf', 'xx']}})
  with pytest.raises(ValueError):
    other = RootIdentity.generate().identity()
    sidecam_config.validate({'role': 'host', 'network': NETWORK, 'peers': {camera: ['lf'], other: ['lf']}})
  path.chmod(0o644)
  with pytest.raises(ValueError):
    sidecam_config.load(path)


def connect(host_id, camera_id, allowed):
  with socket.create_server(('127.0.0.1', 0)) as server:
    b = socket.create_connection(server.getsockname())
    a, _ = server.accept()
  host, camera = Connection(a, host_id), Connection(b, camera_id)
  result = {}

  def accept():
    try:
      host.accept(allowed)
      result['ok'] = True
    except ValueError as error:
      result['error'] = error
      host.close()
  thread = threading.Thread(target=accept)
  thread.start()
  try:
    camera.initiate(host_id.identity())
  except (ValueError, EOFError, OSError):
    pass
  thread.join()
  return host, camera, result


def test_session_carries_binary_frames():
  host_id, camera_id = RootIdentity.generate(), RootIdentity.generate()
  host, camera, result = connect(host_id, camera_id, {camera_id.identity()})
  assert result == {'ok': True}
  frame = bytes(range(256)) * 64
  camera.send(HELLO, b'{}')
  camera.send(FRAME, frame)
  assert host.receive() == (HELLO, b'{}')
  assert host.receive() == (FRAME, frame)
  assert not camera.needs_renewal()
  host.close()
  camera.close()


def test_unauthorized_camera_is_rejected():
  host_id, camera_id = RootIdentity.generate(), RootIdentity.generate()
  host, camera, result = connect(host_id, camera_id, {RootIdentity.generate().identity()})
  assert 'error' in result
  host.close()
  camera.close()


def test_network_rejoins_only_when_host_stays_unreachable(monkeypatch):
  from openpilot.system.sidecam import sidecamd
  calls, active = [], {'uuid': sidecamd.Network.UUID}

  def run(cmd, **kwargs):
    calls.append(cmd)
    out = active['uuid'] if 'GENERAL.CON-UUID' in cmd else ''
    if cmd[:2] == ['sudo', 'iw']:
      out = 'BSS 00:11:22:33:44:55(on wlan0)\n\tSSID: Asius-Cameras-test\n'
    return type('Result', (), {'stdout': out, 'returncode': 0})()
  monkeypatch.setattr(sidecamd.subprocess, 'run', run)
  network = sidecamd.Network({'network': NETWORK}, threading.Event())

  def joins(down_s):
    calls.clear()
    network.next_scan = 0.
    network.join_if_visible(down_s)
    return any(cmd[:3] == ['sudo', 'nmcli', '--wait'] for cmd in calls)

  assert not joins(5.)  # joined and recently connected: keep the link
  assert joins(sidecamd.REJOIN_AFTER)  # joined but the host is unreachable: re-activate
  assert not joins(sidecamd.REJOIN_AFTER + 1)  # at most once per REJOIN_AFTER
  active['uuid'] = 'other-network'
  assert joins(0.)  # visible hotspot while on another network: join
