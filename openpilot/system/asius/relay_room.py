"""Device-owned relay room. The server never receives session keys."""
from collections import deque
from dataclasses import dataclass
import json
import secrets
import threading
import time

from cryptography.exceptions import InvalidSignature, InvalidTag

from openpilot.system.asius.session import SecureSession, canonical
from openpilot.system.asius.status_stream import StatusStream, status_tag

@dataclass
class Member:
  peer: str
  session: SecureSession
  deadline: float
  authorized: bool = False
  status_sid: str | None = None
  status_epoch: str | None = None
  status_mac: bytes | None = None


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
    self.snapshot_revision = 0
    self.closed = False
    self.allowlist = frozenset(authorized())
    self.status = StatusStream(self.owner, now)

  def clear(self):
    with self.lock:
      self.closed = True
      for member in self.members.values():
        member.session.clear()
      self.members.clear()
      self.queue.clear()
      self.queued_bytes = 0
      self.snapshot = None
      self.status = None

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
      self.snapshot = snapshot  # Each connection sends only the latest unsent state.
      self.snapshot_revision += 1

  def refresh(self):
    authorized = frozenset(self.authorized())
    if authorized != self.allowlist or self.status.expired():
      self.allowlist = authorized
      self.status = StatusStream(self.owner, self.now)
    for connection, member in list(self.members.items()):
      if member.authorized and member.peer not in authorized:
        self._remove(connection)
        continue
      member.authorized = member.peer in authorized
      if member.peer not in authorized or not member.session.ready(member.peer):
        if self.now() >= member.deadline:
          self._remove(connection)
    return authorized

  def has_subscribers(self):
    with self.lock:
      self.refresh()
      return any(m.authorized and m.session.ready(m.peer) for m in self.members.values())

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
        if self.snapshot is None:
          return None
        # A full snapshot bounds the next delta's plaintext size, including removed keys.
        if self.status.expired(len(canonical(self.snapshot)) + len(canonical(self.status.encoder.previous)) + 1024):
          self.status = StatusStream(self.owner, self.now)
        recipients = {}
        for connection, member in self.members.items():
          if member.peer not in authorized or not member.session.ready(member.peer):
            continue
          sid = member.session.session_id(member.peer)
          if sid != member.status_sid or member.status_epoch != self.status.epoch:
            mac = secrets.token_bytes(32)
            payload = member.session.encrypt(member.peer, canonical(self.status.installation(mac)).decode(), expected_sid=sid)
            member.status_sid, member.status_epoch, member.status_mac = sid, self.status.epoch, mac
            message = {"type": "peer", "from": self.owner, "to": member.peer, "connection": connection,
                       "payload": json.dumps(payload)}
            break
          recipients[connection] = member.status_mac
        else:
          if not recipients or self.status.revision == self.snapshot_revision:
            return None
          packet = self.status.encrypt(self.snapshot, self.snapshot_revision)
          message = {'type': 'status', 'from': self.owner, 'payload': packet,
                     'auth': {connection: status_tag(mac, packet) for connection, mac in recipients.items()}}
      self.sequence += 1
      return json.dumps({**message, "sequence": self.sequence})
