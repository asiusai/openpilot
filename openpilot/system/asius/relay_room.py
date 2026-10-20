"""Device-owned relay room. The server never receives session or status keys."""
from collections import deque
from dataclasses import dataclass
import json
import secrets
import threading
import time

from cryptography.exceptions import InvalidSignature, InvalidTag

from openpilot.system.asius.live_state import StateEncoder
from openpilot.system.asius.session import SecureSession, b64, canonical, encrypt_data

# Bound each key well before nonce exhaustion. Reserve space for one maximum relay frame.
STATUS_KEY_MAX_MESSAGES = 1 << 20
STATUS_KEY_MAX_BYTES = 1 << 30
MAX_FRAME_BYTES = 256 * 1024


class StatusPublisher:
  def __init__(self, identity, owner):
    self.identity, self.owner = identity, owner
    self.key, self.epoch, self.counter = secrets.token_bytes(32), b64(secrets.token_bytes(32)), 0
    self.encrypted_bytes = 0
    self.encoder = StateEncoder()

  def key_message(self):
    return {"type": "stream-key", "epoch": self.epoch, "key": b64(self.key)}

  def exhausted(self):
    return self.counter >= STATUS_KEY_MAX_MESSAGES or self.encrypted_bytes >= STATUS_KEY_MAX_BYTES - MAX_FRAME_BYTES

  def encrypt(self, snapshot):
    if self.exhausted():
      raise ValueError("Status key exhausted")
    body = canonical({"type": "event", "name": "liveState", "payload": self.encoder.encode(snapshot)})
    if self.encrypted_bytes + len(body) > STATUS_KEY_MAX_BYTES:
      raise ValueError("Status key byte limit exceeded")
    self.counter += 1
    self.encrypted_bytes += len(body)
    message = encrypt_data(self.key, self.owner, "status", "web", self.epoch, self.counter, body)
    return {**message, "sig": b64(self.identity.sign(b"asius-status-v1\n" + canonical(message)))}


@dataclass
class Member:
  peer: str
  session: SecureSession
  deadline: float
  authorized: bool = False


class RelayRoom:
  def __init__(self, identity, authorized, invitation, dispatch, now=time.monotonic):
    self.identity = identity
    self.owner = SecureSession(identity, "web").public_key
    self.authorized, self.invitation, self.dispatch, self.now = authorized, invitation, dispatch, now
    self.members: dict[str, Member] = {}
    self.queue = deque()
    self.queued_bytes = 0
    self.lock = threading.RLock()
    self.sequence = 0
    self.snapshot = None
    self.group = ()
    self.publisher = None
    self.keys_pending = False
    self.closed = False

  def clear(self):
    with self.lock:
      self.closed = True
      for member in self.members.values():
        member.session.clear()
      self.members.clear()
      self.queue.clear()
      self.queued_bytes = 0
      self.publisher, self.snapshot, self.group = None, None, ()
      self.keys_pending = False

  def _queue(self, message, connection=None, sid=None):
    size = len(canonical(message))
    if len(self.queue) >= 128 or self.queued_bytes + size > 2 * 1024 * 1024:
      raise BufferError("Relay output queue is full")
    self.queue.append((message, connection, sid, size))
    self.queued_bytes += size

  def _remove(self, connection):
    member = self.members.pop(connection, None)
    if member:
      member.session.clear()
      self._queue({"type": "remove", "connection": connection})

  def receive(self, message):
    with self.lock:
      if self.closed:
        return
      kind, connection = message.get("type"), message.get("connection")
      if kind == "join":
        peer, admission = message.get("from"), message.get("admission", "")
        if (not isinstance(connection, str) or connection in self.members or len(self.members) >= 64 or
            not isinstance(peer, str) or (peer not in self.authorized() and not self.invitation(admission))):
          self._queue({"type": "remove", "connection": connection})
          return
        self.members[connection] = Member(peer, SecureSession(self.identity, "web"), self.now() + 15, peer in self.authorized())
        self._queue({"type": "admit", "connection": connection})
        return
      if kind == "leave":
        self._remove(connection)
        return
      if kind != "peer":
        return
      # Acknowledge even invalid ciphertext. Bad traffic cannot fill the server queue.
      if type(message.get("delivery")) is int:
        self._queue({"type": "ack", "delivery": message["delivery"]})
      member = self.members.get(connection)
      if not member or message.get("from") != member.peer or message.get("to") != self.owner:
        return
      try:
        wire = json.loads(message["payload"])
        if not isinstance(wire, dict) or wire.get("from") != member.peer or wire.get("to") != self.owner:
          raise ValueError("Session envelope mismatch")
        result = member.session.receive(wire)
        body = json.loads(result["text"]) if "text" in result else None
        if "text" in result and not isinstance(body, dict):
          raise ValueError("Invalid private body")
      except (ValueError, TypeError, KeyError, InvalidSignature, InvalidTag):
        self._remove(connection)
        return
      if "reply" in result:
        self._queue({"type": "peer", "from": self.owner, "to": member.peer, "connection": connection,
                     "payload": json.dumps(result["reply"])}, connection)
      if "text" not in result:
        return
      # Bind a delayed response to this exact connection and authenticated session.
      sid = member.session.session_id(member.peer)
    self.dispatch(connection, member.peer, body, sid)

  def send(self, recipient, body, expected_sid=None):
    with self.lock:
      if self.closed:
        return
      authorized = self.authorized()
      for connection, member in self.members.items():
        if recipient not in (connection, member.peer) or member.peer not in authorized or not member.session.ready(member.peer):
          continue
        sid = member.session.session_id(member.peer)
        if expected_sid is None or sid == expected_sid:
          self._queue(body, connection, sid)

  def publish(self, snapshot):
    with self.lock:
      self.snapshot = snapshot  # Replace unsent state. Never queue old positions.

  def refresh(self):
    authorized = self.authorized()
    for connection, member in list(self.members.items()):
      if member.authorized and member.peer not in authorized:
        self._remove(connection)
        continue
      member.authorized = member.peer in authorized
      if member.peer not in authorized or not member.session.ready(member.peer):
        if self.now() >= member.deadline:
          self._remove(connection)
    group = tuple(sorted((c, m.session.session_id(m.peer)) for c, m in self.members.items()
                         if m.peer in authorized and m.session.ready(m.peer)))
    if group != self.group or (self.publisher and self.publisher.exhausted()):
      self.group = group
      self.publisher = StatusPublisher(self.identity, self.owner) if group else None
      # Wrap only the latest key, together with its first status. Joins need no key acknowledgement.
      self.keys_pending = bool(group)
    return authorized

  def has_subscribers(self):
    with self.lock:
      self.refresh()
      return bool(self.group)

  def next_message(self):
    with self.lock:
      if self.closed:
        return None
      authorized = self.refresh()
      while self.queue:
        message, connection, sid, size = self.queue.popleft()
        self.queued_bytes -= size
        member = self.members.get(connection) if connection else None
        if connection and member is None:
          continue
        if sid:
          if member.session.session_id(member.peer) != sid or member.peer not in authorized:
            continue
          message = {"type": "peer", "from": self.owner, "to": member.peer, "connection": connection,
                     "payload": json.dumps(member.session.encrypt(member.peer, json.dumps(message), expected_sid=sid))}
        break
      else:
        if self.snapshot is None or self.publisher is None:
          return None
        payload = self.publisher.encrypt(self.snapshot)
        self.snapshot = None
        message = {"type": "broadcast", "from": self.owner, "payload": json.dumps(payload)}
        if self.keys_pending:
          key = json.dumps(self.publisher.key_message())
          message["keys"] = {
            connection: json.dumps(self.members[connection].session.encrypt(self.members[connection].peer, key, expected_sid=sid))
            for connection, sid in self.group
          }
          self.keys_pending = False
      self.sequence += 1
      return json.dumps({**message, "sequence": self.sequence})
