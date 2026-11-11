"""Generate public interoperability fixtures with disposable, fixed test roots."""
import argparse
import json
from pathlib import Path

from openpilot.system.asius.bluetooth_wire import encode_message
from openpilot.system.asius.pqc import RootIdentity
from openpilot.system.asius.recording_keys import public_text
from openpilot.system.asius.session import SecureSession, wrap_peer
from openpilot.system.loggerd.data_api import access_document, b64url


def generate(output: Path):
  output.mkdir(parents=True, exist_ok=True)
  owner, recipient = RootIdentity(bytes(range(64))), RootIdentity(bytes(range(64, 128)))
  content_key = bytes(255 - i for i in range(32))
  reader = {'publicKey': recipient.identity(), 'storagePublicKey': public_text(recipient.recording_key())}
  state = {'version': 1, 'activeKey': 'interop-key', 'keys': [{'id': 'interop-key', 'createdAt': 1700000000000, 'key': b64url(content_key)}]}
  recording = {'root': b64url(recipient.seed), 'recipient': recipient.identity(), 'storagePublicKey': reader['storagePublicKey'],
               'contentKey': b64url(content_key), 'document': access_document(owner, state, [reader])}
  (output / 'recording-device.json').write_text(json.dumps(recording, indent=2) + '\n')
  a, b = SecureSession(owner, 'bluetooth'), SecureSession(recipient, 'bluetooth')
  hello = a.begin(b.public_key)
  welcome = b.receive(hello)['reply']
  a.receive(welcome)
  data = a.encrypt(b.public_key, 'status')
  greeting = json.dumps({'type': 'hello', 'v': 1, 'maxFrameBytes': 180})
  wires = [greeting, *(wrap_peer(wire) for wire in (hello, welcome, data))]
  (output / 'ble-wire.json').write_text(json.dumps([{'text': text, 'hex': encode_message(text).hex()} for text in wires], indent=2) + '\n')


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--output', type=Path, required=True)
  generate(parser.parse_args().output)
