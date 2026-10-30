"""Compact Bluetooth transport for unchanged, authenticated v6 session objects."""
import json
import struct

from openpilot.system.asius.identity import base58_encode, identity_to_bytes
from openpilot.system.asius.session import b64, unb64

MAGIC = 0xa6
MAX_BYTES = 256 * 1024
TYPES = ('hello', 'welcome', 'data')


def require(condition):
  if not condition:
    raise ValueError('Invalid Bluetooth packet')


def encode_message(text: str) -> bytes:
  outer = json.loads(text)
  if outer.get('type') == 'hello' and outer.get('v') == 1:
    packet = text.encode()
    require(len(packet) <= MAX_BYTES)
    return packet
  require(outer.get('type') == 'peer')
  value = json.loads(outer['payload'])
  kind = TYPES.index(value['type'])
  require(value['v'] == 6 and value['transport'] == 'bluetooth')
  require(value['from'] == outer['from'] and value['to'] == outer['to'])
  parts = [bytes((MAGIC, kind + 1)), identity_to_bytes(value['from']), identity_to_bytes(value['to'])]

  def add(field, size=None):
    data = unb64(value[field])
    require(len(data) >= 16 if size is None else len(data) == size)
    parts.append(data)

  if kind == 2:
    add('sid', 48)
    require(type(value['n']) is int and 1 <= value['n'] < 2**20)
    parts.append(struct.pack('>I', value['n']))
    add('ciphertext')
  else:
    add('nonce', 32)
    add('kem', 1184 if kind == 0 else 1088)
    add('key', 1952)
    if kind == 1:
      add('hello', 48)
    add('sig', 3309)
  packet = b''.join(parts)
  require(len(packet) <= MAX_BYTES)
  return packet


def decode_message(packet: bytes) -> str:
  require(0 < len(packet) <= MAX_BYTES)
  if packet[0] != MAGIC:
    text = packet.decode()
    value = json.loads(text)
    require(value.get('type') == 'hello' and value.get('v') == 1)
    return text
  require(len(packet) >= 2 and 1 <= packet[1] <= len(TYPES))
  kind = packet[1] - 1
  offset = 2

  def take(size):
    nonlocal offset
    require(0 <= size <= len(packet) - offset)
    data = packet[offset:offset + size]
    offset += size
    return data

  sender, recipient = (base58_encode(int.from_bytes(take(32), 'big')).rjust(44, '1') for _ in range(2))
  value = {'v': 6, 'type': TYPES[kind], 'from': sender, 'to': recipient, 'transport': 'bluetooth'}

  def read(field, size):
    value[field] = b64(take(size))

  if kind == 2:
    read('sid', 48)
    value['n'] = struct.unpack('>I', take(4))[0]
    require(1 <= value['n'] < 2**20 and len(packet) - offset >= 16)
    read('ciphertext', len(packet) - offset)
  else:
    read('nonce', 32)
    read('kem', 1184 if kind == 0 else 1088)
    read('key', 1952)
    if kind == 1:
      read('hello', 48)
    read('sig', 3309)
  require(offset == len(packet))
  return json.dumps({'type': 'peer', 'from': sender, 'to': recipient, 'payload': json.dumps(value, separators=(',', ':'))}, separators=(',', ':'))
