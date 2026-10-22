"""Device proofs and service selection for an already authorized app."""
import re
from urllib.parse import urlsplit

from openpilot.common.params import Params
from openpilot.common.version import terms_version

TERMS_VERSION = '2026-10-22'


def registration_proof(challenge: str, terms: str, params=None) -> str:
  if not isinstance(challenge, str) or not re.fullmatch('[a-f0-9]{64}', challenge):
    raise ValueError('Invalid registration challenge')
  if terms != TERMS_VERSION:
    raise ValueError('Service terms changed. Update the app and device.')
  params = params if params is not None else Params()
  if params.get('HasAcceptedTerms') != terms_version:
    raise PermissionError('Accept the current openpilot terms before service registration')
  from openpilot.system.asius.identity import get_device_identity
  from openpilot.system.asius.relayd import sign_jwt
  return sign_jwt({'type': 'service-registration', 'aud': 'asius-services', 'device': get_device_identity(),
                   'challenge': challenge, 'terms': terms, 'openpilotTerms': terms_version}, expiry_seconds=300)


def service_host(value: str, scheme: str) -> str:
  if not isinstance(value, str):
    raise ValueError('Invalid service URL')
  text = value.strip()
  if not text:
    return ''
  url = urlsplit(text)
  if (any(c.isspace() or c in '\\?#' for c in text) or url.scheme != scheme or not url.hostname or url.username is not None or url.password is not None
      or url.path not in ('', '/') or not (-1 <= (url.port or -1) <= 65535)):
    raise ValueError('Use a secure service URL without a path or credentials')
  return text.rstrip('/')


def set_service_hosts(relay: str, cloud: str, params=None) -> dict:
  params = params if params is not None else Params()
  if not params.get_bool('IsOffroad'):
    raise PermissionError('Park before changing services')
  relay, cloud = service_host(relay, 'wss'), service_host(cloud, 'https')
  # Empty URL parameters resolve to defaults. Disable services before changing their hosts.
  params.put_bool('RelayEnabled', False, block=True)
  params.put_bool('CloudUploadEnabled', False, block=True)
  params.put('RelayHost', relay, block=True)
  params.put('CloudHost', cloud, block=True)
  params.put_bool('RelayEnabled', bool(relay), block=True)
  params.put_bool('CloudUploadEnabled', bool(cloud), block=True)
  return {'relay': relay, 'cloud': cloud}
