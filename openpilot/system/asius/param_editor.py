"""Fork-owned typed parameter catalog and bounded, atomic writes over any app transport."""
import base64
import datetime
import hashlib
import json
import math
import threading
import time
from bisect import bisect_right
from dataclasses import dataclass, field
from collections.abc import Callable

from openpilot.common.params import Params

CHUNK_BYTES = 8192
MAX_VALUE_BYTES = 16 * 1024 * 1024
WRITE_TTL = 180
PARAM_TYPES = {'bool', 'int', 'float', 'string', 'time', 'json', 'bytes'}


def encode_value(value, kind: str) -> str | None:
  if value is None:
    return None
  if kind == 'bytes':
    return base64.b64encode(value).decode('ascii')
  if kind == 'time':
    return value.isoformat()
  if kind in ('bool', 'json'):
    return json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(',', ':'))
  return str(value)


def decode_value(text: str, kind: str):
  try:
    if kind == 'bool':
      if text not in ('true', 'false'):
        raise ValueError
      return text == 'true'
    if kind == 'int':
      if not text or any(c not in '+-0123456789' for c in text):
        raise ValueError
      return int(text)
    if kind == 'float':
      value = float(text)
      if not math.isfinite(value):
        raise ValueError
      return value
    if kind == 'json':
      value = json.loads(text, parse_constant=lambda _: (_ for _ in ()).throw(ValueError()))
      json.dumps(value, allow_nan=False)
      if not isinstance(value, (dict, list)):
        raise ValueError
      return value
    if kind == 'time':
      return datetime.datetime.fromisoformat(text)
    if kind == 'bytes':
      return base64.b64decode(text, validate=True)
    if kind == 'string':
      return text
  except (ValueError, TypeError, OverflowError):
    raise ValueError(f'Invalid {kind} value') from None
  raise ValueError('Unsupported parameter type')


@dataclass
class PendingWrite:
  name: str
  revision: str
  touched: float
  data: bytearray = field(default_factory=bytearray)


class ParameterEditor:
  def __init__(self, blocked_keys: set[str], params: Callable = Params):
    self.blocked_keys = blocked_keys
    self.params = params
    self.pending: dict[tuple[str, str], PendingWrite] = {}
    self.lock = threading.RLock()

  def _value(self, params, name: str):
    params.check_key(name)
    kind = params.get_type(name).name.lower()
    reason = 'Managed by the device. Use its dedicated controls.' if name in self.blocked_keys else None
    if kind not in PARAM_TYPES:
      reason = 'This parameter type is not supported by this device editor.'
    if reason:
      return {'name': name, 'type': kind, 'value': None, 'defaultValue': None, 'isSet': None,
              'size': 0, 'revision': '', 'readOnlyReason': reason}, b''
    value = encode_value(params.get(name), kind)
    default = encode_value(params.get_default_value(name), kind)
    data = (value if value is not None else default or '').encode('utf-8')
    revision = hashlib.sha256(json.dumps([kind, value, default], ensure_ascii=False).encode()).hexdigest()
    return {'name': name, 'type': kind, 'value': value if len(data) <= 256 else None,
            'defaultValue': default if default is None or len(default.encode('utf-8')) <= 256 else None,
            'isSet': value is not None, 'size': len(data), 'revision': revision, 'readOnlyReason': None}, data

  def catalog(self, after: str = '', limit: int = 24):
    if not isinstance(after, str) or type(limit) is not int or not 1 <= limit <= 40:
      raise ValueError('Invalid parameter page')
    params = self.params()
    keys = sorted(key.decode('utf-8') if isinstance(key, bytes) else key for key in params.all_keys())
    start = bisect_right(keys, after) if after else 0
    names = keys[start:start + limit]
    entries = []
    for name in names:
      try:
        entries.append(self._value(params, name)[0])
      except (TypeError, ValueError, OverflowError):
        entries.append({'name': name, 'type': params.get_type(name).name.lower(), 'value': None, 'defaultValue': None,
                        'isSet': None, 'size': 0, 'revision': '', 'readOnlyReason': 'The device could not read this value.'})
    return {'params': entries, 'next': names[-1] if start + len(names) < len(keys) else None, 'total': len(keys),
            'canEdit': params.get_bool('IsOffroad')}

  def read(self, name: str, revision: str, offset: int = 0):
    entry, data = self._value(self.params(), name)
    if entry['readOnlyReason']:
      raise ValueError(entry['readOnlyReason'])
    if revision != entry['revision']:
      raise ValueError('This value changed on the device. Refresh it before editing.')
    if type(offset) is not int or not 0 <= offset <= len(data):
      raise ValueError('Invalid value offset')
    chunk = data[offset:offset + CHUNK_BYTES]
    return {'chunk': base64.b64encode(chunk).decode('ascii'), 'nextOffset': offset + len(chunk), 'done': offset + len(chunk) == len(data)}

  def write(self, peer: str, name: str, revision: str, uploadId: str, offset: int, chunk: str, final: bool, clear: bool = False):
    with self.lock:
      now = time.monotonic()
      self.pending = {key: value for key, value in self.pending.items() if now - value.touched < WRITE_TTL}
      params = self.params()
      entry, _ = self._value(params, name)
      if entry['readOnlyReason']:
        raise ValueError(entry['readOnlyReason'])
      if not params.get_bool('IsOffroad'):
        raise ValueError('Park the car before editing parameters.')
      if revision != entry['revision']:
        raise ValueError('This value changed on the device. Refresh it before saving.')
      if not isinstance(uploadId, str) or not 1 <= len(uploadId) <= 64 or type(offset) is not int or offset < 0:
        raise ValueError('Invalid parameter upload')
      if type(final) is not bool or type(clear) is not bool or not isinstance(chunk, str) or len(chunk) > 4 * ((CHUNK_BYTES + 2) // 3):
        raise ValueError('Invalid parameter chunk')
      try:
        data = base64.b64decode(chunk, validate=True)
      except ValueError:
        raise ValueError('Invalid parameter chunk') from None
      if len(data) > CHUNK_BYTES or (clear and (data or offset or not final)):
        raise ValueError('Invalid parameter chunk')
      key = (peer, uploadId)
      pending = self.pending.get(key)
      if pending is None:
        if offset != 0 or len(self.pending) >= 8:
          raise ValueError('Parameter upload expired or is busy. Try saving again.')
        pending = self.pending[key] = PendingWrite(name, revision, now)
      if pending.name != name or pending.revision != revision or offset != len(pending.data):
        raise ValueError('Parameter chunks are out of order. Try saving again.')
      if offset + len(data) > MAX_VALUE_BYTES:
        del self.pending[key]
        raise ValueError('Parameter value exceeds 16 MB.')
      pending.data.extend(data)
      pending.touched = now
      if not final:
        return {'nextOffset': len(pending.data), 'parameter': None}
      del self.pending[key]
      try:
        value = decode_value(pending.data.decode('utf-8'), entry['type']) if not clear else None
      except UnicodeDecodeError:
        raise ValueError('Value must be valid UTF-8 text.') from None
      # No partially received value reaches Params, even if Bluetooth disconnects.
      if clear:
        params.remove(name)
      else:
        params.put(name, value, block=True)
      return {'nextOffset': len(pending.data), 'parameter': self._value(params, name)[0]}
