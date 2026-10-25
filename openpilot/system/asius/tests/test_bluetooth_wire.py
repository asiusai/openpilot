import json

import pytest
from cryptography.exceptions import InvalidTag

from openpilot.system.asius.bluetooth_wire import decode_message, encode_message, MAX_BYTES
from openpilot.system.asius.pqc import RootIdentity
from openpilot.system.asius.session import SecureSession, wrap_peer, unwrap_peer


def test_binary_handshake_and_authenticated_messages():
  a, b = (SecureSession(RootIdentity.generate(), 'bluetooth') for _ in range(2))

  def transmit(value):
    return unwrap_peer(decode_message(encode_message(wrap_peer(value))), value['to'])

  hello = a.begin(b.public_key)
  welcome = b.receive(transmit(hello))['reply']
  assert len(encode_message(wrap_peer(hello))) == 6543
  assert len(encode_message(wrap_peer(welcome))) == 6495
  a.receive(transmit(welcome))
  data = a.encrypt(b.public_key, 'status')
  assert len(encode_message(wrap_peer(data))) == 140
  assert b.receive(transmit(data))['text'] == 'status'
  with pytest.raises(ValueError):
    b.receive(transmit(data))
  next_data = a.encrypt(b.public_key, 'next')
  changed = bytearray(encode_message(wrap_peer(next_data)))
  changed[-1] ^= 1
  with pytest.raises(InvalidTag):
    b.receive(unwrap_peer(decode_message(bytes(changed)), b.public_key))
  assert b.receive(transmit(next_data))['text'] == 'next'

  for value in (hello, welcome, data):
    packet = encode_message(wrap_peer(value))
    assert encode_message(decode_message(packet)) == packet
    for length in range(134 if value['type'] == 'data' else len(packet)):
      with pytest.raises(ValueError):
        decode_message(packet[:length])
    if value['type'] != 'data':
      with pytest.raises(ValueError):
        decode_message(packet + b'\x00')
    with pytest.raises(ValueError):
      decode_message(wrap_peer(value).encode())
  with pytest.raises(ValueError):
    decode_message(bytes(MAX_BYTES + 1))


def test_only_setup_greeting_uses_json():
  greeting = json.dumps({'type': 'hello', 'v': 1, 'maxFrameBytes': 180})
  assert decode_message(encode_message(greeting)) == greeting
  for packet in (b'', b'\xa6', b'\xa6\x04', b'{"type":"hello","v":6}'):
    with pytest.raises(ValueError):
      decode_message(packet)
