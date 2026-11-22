"""Authenticated session messages over TCP. Each message is a 4-byte length and a session envelope."""
import json
import socket
import struct
import threading
import time

from openpilot.system.asius.session import SecureSession, canonical

MAX_MESSAGE = 4 * 1024 * 1024
# Renew below the session limits: 1 GiB combined plaintext and 2^20 messages per direction.
RENEW_BYTES, RENEW_MESSAGES, RENEW_SECONDS = 900 * 1024 * 1024, 900_000, 23 * 3600

FRAME, HELLO, PING, PONG = b'F', b'H', b'P', b'Q'


def recv_exact(sock, size):
  out = bytearray()
  while len(out) < size:
    chunk = sock.recv(size - len(out))
    if not chunk:
      raise EOFError('Camera disconnected')
    out += chunk
  return bytes(out)


def read_wire(sock):
  size, = struct.unpack('!I', recv_exact(sock, 4))
  if not 1 <= size <= MAX_MESSAGE:
    raise ValueError('Camera message too large')
  return json.loads(recv_exact(sock, size))


def write_wire(sock, message):
  data = canonical(message)
  if len(data) > MAX_MESSAGE:
    raise ValueError('Camera message too large')
  sock.sendall(struct.pack('!I', len(data)) + data)


class Connection:
  """One authenticated `camera` session per socket. Sends are serialized for the session counter."""

  def __init__(self, sock, identity):
    self.sock = sock
    sock.settimeout(5)
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    # A small send buffer keeps queued video from adding latency.
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 512 * 1024)
    self.session = SecureSession(identity, 'camera')
    self.peer = None
    self.started = time.monotonic()
    self.bytes = self.sent = self.received = 0
    self.lock = threading.Lock()

  def initiate(self, peer):
    self.peer = peer
    write_wire(self.sock, self.session.begin(peer))
    reply = read_wire(self.sock)
    if not isinstance(reply, dict) or reply.get('from') != peer or reply.get('type') != 'welcome':
      raise ValueError('Unexpected camera host')
    self.session.receive_bytes(reply)

  def accept(self, allowed):
    hello = read_wire(self.sock)
    if not isinstance(hello, dict) or hello.get('from') not in allowed or hello.get('type') != 'hello':
      raise ValueError('Unauthorized camera')
    self.peer = hello['from']
    write_wire(self.sock, self.session.receive_bytes(hello)['reply'])

  def needs_renewal(self):
    return (self.bytes >= RENEW_BYTES or max(self.sent, self.received) >= RENEW_MESSAGES
            or time.monotonic() - self.started >= RENEW_SECONDS)

  def send(self, kind, payload=b''):
    with self.lock:
      write_wire(self.sock, self.session.encrypt_bytes(self.peer, kind + payload))
      self.bytes += 1 + len(payload)
      self.sent += 1

  def receive(self):
    wire = read_wire(self.sock)
    if not isinstance(wire, dict) or wire.get('from') != self.peer or wire.get('type') != 'data':
      raise ValueError('Unexpected camera message')
    data = self.session.receive_bytes(wire)['data']
    self.bytes += len(data)
    self.received += 1
    if not data:
      raise ValueError('Empty camera message')
    return data[:1], data[1:]

  def close(self):
    self.session.clear()
    try:
      self.sock.shutdown(socket.SHUT_RDWR)
    except OSError:
      pass
    self.sock.close()
