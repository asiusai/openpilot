"""Authenticated ephemeral app sessions. No wall-clock or persistent replay state."""
import base64
import hashlib
import json
import secrets
import threading
import time
from dataclasses import dataclass

from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec, ed25519
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

from openpilot.system.asius.identity import bytes_to_identity, identity_to_bytes


def canonical(value):
  return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode()


def b64(value):
  return base64.urlsafe_b64encode(value).decode().rstrip('=')


def unb64(value):
  if not isinstance(value, str):
    raise ValueError('Invalid encoded bytes')
  return base64.b64decode(value + '=' * (-len(value) % 4), altchars=b'-_', validate=True)


def unsigned(value):
  return {key: item for key, item in value.items() if key != 'sig'}


def digest(value):
  return hashlib.sha256(canonical(value)).digest()


def message_nonce(counter):
  if type(counter) is not int or not 1 <= counter < 0xffffffff:
    raise ValueError('Invalid message counter')
  return counter.to_bytes(12, 'big')


def data_header(message):
  """The same authenticated envelope protects private calls and shared status."""
  return {key: message[key] for key in ('v', 'type', 'from', 'to', 'transport', 'sid', 'n')}


def encrypt_data(key, sender, recipient, transport, sid, counter, plaintext):
  header = {'v': 5, 'type': 'data', 'from': sender, 'to': recipient, 'transport': transport, 'sid': sid, 'n': counter}
  return {**header, 'ciphertext': b64(AESGCM(key).encrypt(message_nonce(counter), plaintext, canonical(header)))}


def decrypt_data(key, message):
  header = data_header(message)
  if header['v'] != 5 or header['type'] != 'data':
    raise ValueError('Invalid encrypted message')
  return AESGCM(key).decrypt(message_nonce(header['n']), unb64(message['ciphertext']), canonical(header))


@dataclass
class Channel:
  sid: str
  peer: str
  send: bytes
  receive: bytes
  created: float
  tx: int = 0
  rx: int = 0


class SessionChanged(ValueError):
  pass


class SecureSession:
  def __init__(self, identity: ed25519.Ed25519PrivateKey, transport: str, now=time.monotonic):
    if transport not in ('web', 'bluetooth'):
      raise ValueError('Invalid transport')
    self.identity = identity
    self.public_key = bytes_to_identity(identity.public_key().public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw))
    self.transport = transport
    self.now = now
    self.pending = {}
    self.active = {}
    self.candidates = {}
    self.lock = threading.RLock()

  def clear(self):
    with self.lock:
      self.pending.clear()
      self.active.clear()
      self.candidates.clear()

  def forget(self, peer):
    with self.lock:
      self.active.pop(peer, None)
      self.pending.pop(peer, None)
      self.candidates = {sid: item for sid, item in self.candidates.items() if item.peer != peer}

  def activate(self, channel):
    if channel.peer not in self.active and len(self.active) >= 128:
      self.active.pop(min(self.active, key=lambda peer: self.active[peer].created))
    channel.created = self.now()
    self.active[channel.peer] = channel

  def ready(self, peer):
    with self.lock:
      return peer in self.active

  def session_id(self, peer):
    with self.lock:
      channel = self.active.get(peer)
      return channel.sid if channel else None

  def prune(self):
    self.pending = {peer: item for peer, item in self.pending.items() if self.now() - item[2] <= 15}
    self.candidates = {sid: item for sid, item in self.candidates.items() if self.now() - item.created <= 15}

  def signed(self, value):
    return {**value, 'sig': b64(self.identity.sign(b'asius-session-v1\n' + canonical(value)))}

  @staticmethod
  def ephemeral():
    key = ec.generate_private_key(ec.SECP256R1())
    return key, b64(key.public_key().public_bytes(serialization.Encoding.X962, serialization.PublicFormat.UncompressedPoint))

  def begin(self, peer):
    with self.lock:
      identity_to_bytes(peer)
      self.prune()
      if len(self.pending) >= 32:
        raise ValueError('Too many pending sessions')
      key, eph = self.ephemeral()
      hello = self.signed({'v': 5, 'type': 'hello', 'from': self.public_key, 'to': peer,
                           'transport': self.transport, 'nonce': b64(secrets.token_bytes(32)), 'eph': eph})
      self.pending[peer] = (hello, key, self.now())
      return hello

  def derive(self, hello, welcome, key, initiator):
    salt = digest([unsigned(hello), unsigned(welcome)])
    peer = ec.EllipticCurvePublicKey.from_encoded_point(ec.SECP256R1(), unb64((welcome if initiator else hello)['eph']))
    shared = key.exchange(ec.ECDH(), peer)
    def derive(direction):
      return HKDF(algorithm=hashes.SHA256(), length=32, salt=salt, info=f'asius-session-v1:{direction}'.encode()).derive(shared)
    forward, reverse = derive('initiator'), derive('responder')
    return Channel(b64(salt), (welcome if initiator else hello)['from'], forward if initiator else reverse,
                   reverse if initiator else forward, self.now())

  def encrypt(self, peer, text, expected_sid=None):
    with self.lock:
      channel = self.active.get(peer)
      if channel is None:
        raise ValueError('Session is not ready')
      if expected_sid is not None and channel.sid != expected_sid:
        raise SessionChanged('Session changed before delivery')
      channel.created = self.now()
      channel.tx += 1
      return encrypt_data(channel.send, self.public_key, peer, self.transport, channel.sid, channel.tx, text.encode())

  def receive(self, message):
    with self.lock:
      self.prune()
      if not isinstance(message, dict) or message.get('v') != 5 or message.get('to') != self.public_key or message.get('transport') != self.transport:
        raise ValueError('Invalid session context')
      peer = message['from']
      peer_bytes = identity_to_bytes(peer)
      if peer == self.public_key:
        raise ValueError('Reflected session')
      if message.get('type') == 'data':
        channel = self.active.get(peer)
        if channel is None or channel.sid != message.get('sid'):
          channel = self.candidates.get(message.get('sid'))
        counter = message.get('n')
        message_nonce(counter)
        if channel is None or channel.peer != peer or counter <= channel.rx:
          raise ValueError('Replayed or unknown session')
        text = decrypt_data(channel.receive, message).decode()
        channel.rx = counter
        self.activate(channel)
        self.candidates = {sid: candidate for sid, candidate in self.candidates.items() if candidate.peer != peer}
        return {'peer': peer, 'text': text, 'ready': True}
      if message.get('type') not in ('hello', 'welcome') or len(unb64(message.get('nonce'))) != 32 or len(unb64(message.get('eph'))) != 65:
        raise ValueError('Invalid handshake')
      ed25519.Ed25519PublicKey.from_public_bytes(peer_bytes).verify(unb64(message['sig']), b'asius-session-v1\n' + canonical(unsigned(message)))
      if message['type'] == 'hello':
        if len(self.candidates) >= 32:
          raise ValueError('Too many pending sessions')
        key, eph = self.ephemeral()
        welcome = self.signed({'v': 5, 'type': 'welcome', 'from': self.public_key, 'to': peer, 'transport': self.transport,
                              'hello': b64(digest(unsigned(message))), 'nonce': b64(secrets.token_bytes(32)), 'eph': eph})
        channel = self.derive(message, welcome, key, False)
        self.candidates[channel.sid] = channel
        return {'peer': peer, 'reply': welcome}
      pending = self.pending.get(peer)
      if pending is None or message.get('hello') != b64(digest(unsigned(pending[0]))):
        raise ValueError('Unsolicited handshake response')
      channel = self.derive(pending[0], message, pending[1], True)
      del self.pending[peer]
      self.activate(channel)
      return {'peer': peer, 'ready': True}


def wrap_peer(message):
  return json.dumps({'type': 'peer', 'from': message['from'], 'to': message['to'], 'payload': json.dumps(message)})


def unwrap_peer(raw, recipient):
  outer = json.loads(raw)
  if not isinstance(outer, dict) or outer.get('type') != 'peer' or outer.get('to') != recipient:
    raise ValueError('Invalid peer envelope')
  message = json.loads(outer['payload'])
  if message.get('from') != outer.get('from') or message.get('to') != recipient:
    raise ValueError('Peer envelope identity mismatch')
  return message
