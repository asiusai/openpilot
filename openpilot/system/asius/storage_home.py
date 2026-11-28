"""Publish the selected storage server so private and public route viewers can find it."""
import threading

import requests

from openpilot.common.params import Params
from openpilot.common.swaglog import cloudlog
from openpilot.system.asius.service_hosts import api_origin

_lock = threading.Lock()
_published: tuple[str, str] | None = None


def run(exit_event: threading.Event, params: Params) -> None:
  """Refresh changed selections and retry failed publications while the relay is connected."""
  while not exit_event.is_set():
    publish(params)
    exit_event.wait(30)


def publish(params=None) -> bool:
  """Send a device-signed storage pointer once per API and storage pair. Never raises."""
  global _published
  params = params if params is not None else Params()
  try:
    api, storage = api_origin(params), params.get('StorageHost', return_default=True)
    with _lock:
      if _published == (api, storage):
        return True
    from openpilot.system.asius.identity import get_device_identity
    from openpilot.system.asius.relayd import sign_jwt
    device = get_device_identity()
    token = sign_jwt({'type': 'storage-home', 'aud': 'asius-services', 'device': device, 'storage': storage}, expiry_seconds=300)
    response = requests.put(f'{api}/v1/devices/{device}/storage', headers={'Authorization': f'Bearer {token}'}, timeout=10, allow_redirects=False)
    if response.status_code != 200:
      cloudlog.event('storage_home.publish_failed', status=response.status_code)
      return False
    with _lock:
      _published = (api, storage)
    return True
  except RuntimeError:
    return False  # The API is disabled.
  except Exception:
    cloudlog.exception('storage_home.publish_exception')
    return False
