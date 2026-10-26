"""One status ciphertext with a separate authentication key for each connection."""
import hashlib
import hmac
import secrets
import time

from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from openpilot.system.asius.live_state import StateEncoder
from openpilot.system.asius.session import b64, canonical, message_nonce


def status_tag(key, packet):
  return b64(hmac.digest(key, b'asius-status-v1\n' + canonical(packet), hashlib.sha384)[:32])


class StatusStream:
  def __init__(self, owner, now=time.monotonic):
    self.owner, self.now = owner, now
    self.key = secrets.token_bytes(32)
    self.epoch = b64(secrets.token_bytes(16))
    self.created = now()
    self.counter = self.bytes_used = 0
    self.revision = -1
    self.encoder = StateEncoder()

  def expired(self, size=0):
    return self.now() - self.created >= 86400 or self.counter >= (1 << 20) - 1 or self.bytes_used + size > 1 << 30

  def installation(self, mac):
    body = {'type': 'status-key', 'epoch': self.epoch, 'key': b64(self.key), 'mac': b64(mac), 'after': self.counter,
            'expires': max(0, 86400 - (self.now() - self.created))}
    if self.encoder.previous is not None:
      body['baseline'] = {'stream': self.encoder.stream, 'sequence': self.encoder.sequence, 'base': None,
                          'full': True, 'changes': self.encoder.previous}
    return body

  def encrypt(self, snapshot, revision):
    body = canonical({'type': 'event', 'name': 'liveState', 'payload': self.encoder.encode(snapshot, self.now())})
    if self.expired(len(body)):
      raise ValueError('Status key usage limit reached')
    self.counter += 1  # Reserve before encryption. Never retry this nonce.
    self.bytes_used += len(body)
    self.revision = revision
    header = {'v': 1, 'type': 'status', 'from': self.owner, 'epoch': self.epoch, 'n': self.counter}
    return {**header, 'ciphertext': b64(AESGCM(self.key).encrypt(message_nonce(self.counter), body, canonical(header)))}
