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
