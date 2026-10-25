import concurrent.futures
import copy
import pytest
from openpilot.system.asius.pqc import RootIdentity
from cryptography.exceptions import InvalidTag, InvalidSignature
from openpilot.system.asius.session import SecureSession, SessionChanged


def sessions():
  a = SecureSession(RootIdentity.generate(), 'bluetooth')
  b = SecureSession(RootIdentity.generate(), 'bluetooth')
  hello = a.begin(b.public_key)
  welcome = b.receive(hello)['reply']
  a.receive(welcome)
  return a, b, hello


def test_duplex_replay_and_header_authentication():
  a, b, _ = sessions()
  wire = a.encrypt(b.public_key, 'hello')
  invalid = {**wire, 'n': 100}
  with pytest.raises(InvalidTag):
    b.receive(invalid)
  assert b.receive(wire)['text'] == 'hello'
  with pytest.raises(ValueError):
    b.receive(wire)
  assert a.receive(b.encrypt(a.public_key, 'reply'))['text'] == 'reply'
  b.clear()
  with pytest.raises(ValueError):
    b.receive(a.encrypt(b.public_key, 'old connection'))


def test_handshake_replay_cannot_replace_confirmed_session():
  a, b, hello = sessions()
  b.receive(a.encrypt(b.public_key, 'first'))
  replay = b.receive(hello)['reply']
  with pytest.raises(ValueError):
    a.receive(replay)
  assert b.receive(a.encrypt(b.public_key, 'second'))['text'] == 'second'
  modified = copy.deepcopy(hello)
  modified['nonce'] = 'A' * 43
  with pytest.raises(InvalidSignature):
    b.receive(modified)


def test_nonce_reservation_is_atomic():
  a, b, _ = sessions()
  with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
    values = list(pool.map(lambda n: a.encrypt(b.public_key, str(n)), range(128)))
  assert len({value['n'] for value in values}) == 128
  for wire in sorted(values, key=lambda value: value['n']):
    b.receive(wire)


def test_transport_binding_and_counter_types():
  a, b, hello = sessions()
  other = SecureSession(b.identity, 'web')
  with pytest.raises(ValueError):
    other.receive(hello)
  for n in (True, 0, -1, 1.5, 0xffffffff):
    with pytest.raises(ValueError):
      b.receive({**a.encrypt(b.public_key, 'hello'), 'n': n})


def test_old_queued_response_cannot_cross_session_replacement():
  a, b, _ = sessions()
  b.receive(a.encrypt(b.public_key, 'first'))
  old = b.session_id(a.public_key)
  a.receive(b.receive(a.begin(b.public_key))['reply'])
  b.receive(a.encrypt(b.public_key, 'new connection'))
  with pytest.raises(SessionChanged):
    b.encrypt(a.public_key, 'old reply', expected_sid=old)
  assert a.receive(b.encrypt(a.public_key, 'current reply'))['text'] == 'current reply'


def test_completed_sessions_are_bounded_and_forgotten_keys_cannot_resume():
  device = SecureSession(RootIdentity.generate(), 'web')
  clients = []
  for _ in range(129):
    client = SecureSession(RootIdentity.generate(), 'web')
    client.receive(device.receive(client.begin(device.public_key))['reply'])
    device.receive(client.encrypt(device.public_key, 'confirm'))
    clients.append(client)
  assert len(device.active) == 128
  with pytest.raises(ValueError):
    device.receive(clients[0].encrypt(device.public_key, 'evicted'))
  device.forget(clients[-1].public_key)
  with pytest.raises(ValueError):
    device.receive(clients[-1].encrypt(device.public_key, 'forgotten'))


def test_session_lifetime_requires_fresh_authenticated_keys():
  clock = [100.0]
  a = SecureSession(RootIdentity.generate(), 'web', now=lambda: clock[0])
  b = SecureSession(RootIdentity.generate(), 'web', now=lambda: clock[0])
  a.receive(b.receive(a.begin(b.public_key))['reply'])
  b.receive(a.encrypt(b.public_key, 'confirm'))
  old = a.encrypt(b.public_key, 'delayed')
  clock[0] += 86400
  assert not a.ready(b.public_key)
  with pytest.raises(ValueError):
    a.encrypt(b.public_key, 'expired')
  with pytest.raises(ValueError):
    b.receive(old)
  a.receive(b.receive(a.begin(b.public_key))['reply'])
  assert b.receive(a.encrypt(b.public_key, 'fresh'))['text'] == 'fresh'


def test_message_and_byte_limits_do_not_consume_a_nonce_on_failure():
  a, b, _ = sessions()
  channel = a.active[b.public_key]
  channel.bytes_used = (1 << 30) - 1
  before = channel.tx
  with pytest.raises(ValueError):
    a.encrypt(b.public_key, 'two')
  assert channel.tx == before
  a.encrypt(b.public_key, 'x')
  with pytest.raises(ValueError):
    a.encrypt(b.public_key, 'x')
  channel.bytes_used = 0
  channel.tx = (1 << 20) - 1
  with pytest.raises(ValueError):
    a.encrypt(b.public_key, 'exhausted')
