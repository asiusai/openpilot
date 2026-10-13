"""Authenticated local video setup; Wi-Fi policy stays in VamOS."""
import json
from pathlib import Path
import subprocess

HELPER = Path('/usr/bin/vamos-local-video')


def wifi(owner: str, action: str, lease: str = '') -> dict:
  if not HELPER.is_file():
    raise ValueError('Update VamOS to use local video Wi-Fi')
  response = subprocess.run(['sudo', '-n', str(HELPER)],
                            input=json.dumps({'owner': owner, 'action': action, 'lease': lease}),
                            capture_output=True, text=True, timeout=45)
  try:
    result = json.loads(response.stdout)
  except ValueError:
    raise RuntimeError('Local video Wi-Fi did not respond') from None
  if response.returncode or result.get('error'):
    raise RuntimeError(result.get('error', 'Local video Wi-Fi failed'))
  return result


def start_stream(owner: str, lease: str, sdp: str, camera: str) -> dict:
  from openpilot.system.webrtc.helpers import StreamRequestBody, post_stream_request, wait_for_webrtcd
  if camera not in ('road', 'wideRoad', 'driver'):
    raise ValueError('Unknown camera')
  if not isinstance(sdp, str) or len(sdp) > 65536:
    raise ValueError('Invalid local video offer')
  network = wifi(owner, 'check', lease)
  wait_for_webrtcd()
  # No incoming control services. Bluetooth carries setup only, never video.
  return post_stream_request(StreamRequestBody(sdp, [camera], True, in_car=True, bind_address=network['address']))


def methods(owner: str) -> dict:
  return {
    'startLocalVideo': lambda: wifi(owner, 'start'),
    'renewLocalVideo': lambda lease: wifi(owner, 'renew', lease),
    'stopLocalVideo': lambda lease: wifi(owner, 'stop', lease),
    'startLocalStream': lambda lease, sdp, camera: start_stream(owner, lease, sdp, camera),
  }
