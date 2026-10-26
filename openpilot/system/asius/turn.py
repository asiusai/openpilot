"""Prepare live video through a registered device, never an app-issued proof."""
import json
import math
import time

import requests

from openpilot.system.asius.identity import dongle_id_from_public_key, get_device_public_key
from openpilot.system.asius.local_video import HELPER
from openpilot.system.asius.relayd import sign_jwt

TURN_PATH = '/v1/turn/credentials'
TURN_URL = 'https://api.asius.ai' + TURN_PATH


def prepare_live(phone_online: bool = True, local_wifi_supported: bool = False) -> dict:
  if not isinstance(phone_online, bool) or not isinstance(local_wifi_supported, bool):
    raise ValueError('Video connectivity flags must be boolean')

  def offline(reason):
    if local_wifi_supported and HELPER.is_file():
      return {'mode': 'localWifi'}
    return {'mode': 'unavailable', 'error': reason}

  if not phone_online:
    return offline('Connect the phone to the internet or use device Wi-Fi.')

  token = sign_jwt({'type': 'turn', 'aud': 'asius-turn', 'identity': dongle_id_from_public_key(get_device_public_key()),
                    'method': 'POST', 'path': TURN_PATH}, expiry_seconds=60)
  try:
    with requests.post(TURN_URL, headers={'Authorization': f'Bearer {token}'}, timeout=(3, 6),
                       allow_redirects=False, stream=True) as response:
      if response.status_code != 200:
        error = {
          401: 'Device TURN authentication failed. Check the device clock and software.',
          403: 'Enable Asius services on the device to use internet video.',
          429: 'Video connection limit reached. Try again in one minute.',
        }.get(response.status_code, 'Video connection service is unavailable. Try again.')
        return {'mode': 'unavailable', 'error': error}
      body = bytearray()
      for chunk in response.iter_content(4096):
        body.extend(chunk)
        if len(body) > 32768:
          raise ValueError('Oversized TURN response')
      credentials = json.loads(body)
      servers, expires = credentials['iceServers'], credentials['expiresAt']
      if not isinstance(servers, list) or not 1 <= len(servers) <= 16:
        raise ValueError('Invalid ICE servers')
      now_ms = time.time() * 1000  # noqa: TID251 - The gateway returns an absolute Unix expiry.
      if not isinstance(expires, (int, float)) or not math.isfinite(expires) or expires <= now_ms:
        raise ValueError('Expired TURN credentials')
      for server in servers:
        urls = server['urls']
        urls = [urls] if isinstance(urls, str) else urls
        if (not isinstance(urls, list) or not urls or
            any(not isinstance(url, str) or not url.startswith(('stun:', 'stuns:', 'turn:', 'turns:')) for url in urls)):
          raise ValueError('Invalid ICE URLs')
        if any(url.startswith(('turn:', 'turns:')) for url in urls):
          if not all(isinstance(server.get(key), str) and server[key] for key in ('username', 'credential')):
            raise ValueError('Missing TURN credentials')
      return {'mode': 'internet', 'iceServers': servers, 'expiresAt': expires}
  except requests.exceptions.SSLError:
    return {'mode': 'unavailable', 'error': 'Cannot verify the video service. Check the device clock.'}
  except (requests.ConnectionError, requests.Timeout):
    return offline('Device cannot reach the video service. Connect device Wi-Fi and retry.')
  except (requests.RequestException, ValueError, KeyError, TypeError):
    return {'mode': 'unavailable', 'error': 'Invalid response from the video connection service. Try again.'}
