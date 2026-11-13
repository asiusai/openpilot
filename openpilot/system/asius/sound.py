"""Saved Asius playback gain and short-lived app preview requests."""
import math
import time
import uuid

DEFAULT_VOLUME = 80
MIN_VOLUME = 10
MAX_VOLUME = 100


def valid_volume(value):
  return type(value) is int and MIN_VOLUME <= value <= MAX_VOLUME


def volume_gain(params):
  try:
    value = params.get('SoundVolume')
  except (ValueError, TypeError):
    value = None
  return (value if valid_volume(value) else DEFAULT_VOLUME) / 100.


def preview_request(params, now=None):
  now = time.monotonic() if now is None else now
  try:
    value = params.get('SoundVolumePreview')
  except (ValueError, TypeError):
    return None
  if not isinstance(value, dict) or not isinstance(value.get('id'), str) or not value['id']:
    return None
  expires = value.get('expires')
  if type(expires) not in (int, float) or not math.isfinite(expires) or not 0 < expires - now <= 5:
    return None
  return value


def request_preview(params):
  params.put('SoundVolumePreview', {'id': uuid.uuid4().hex, 'expires': time.monotonic() + 5}, block=True)
