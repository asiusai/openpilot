"""The selected Asius API host. It serves the relay, registration, navigation and TURN."""
from openpilot.common.params import Params

DISABLED = 'Asius API is disabled. Use Bluetooth or select an API server in the app.'


def api_origin(params=None) -> str:
  params = params if params is not None else Params()
  if not params.get('ApiEnabled', return_default=True):
    raise RuntimeError(DISABLED)
  return params.get('ApiHost', return_default=True)


def relay_url(origin: str) -> str:
  if not origin.startswith('https://'):
    raise ValueError('API URL must use https://')
  return 'wss://' + origin.removeprefix('https://')
