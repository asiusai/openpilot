import json

import pytest
from openpilot.system.asius.pqc import RootIdentity

from openpilot.system.asius.relay_room import RelayRoom
from openpilot.system.asius.session import SecureSession, canonical, unb64, b64, message_nonce
from openpilot.system.asius.status_stream import status_tag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
import hmac


class Client:
  def __init__(self, identity):
    self.session = SecureSession(identity, 'web')
    self.keys = None

  def __getattr__(self, name):
    return getattr(self.session, name)

  def receive(self, wire):
    if wire.get('type') == 'status':
      keys = self.keys
      assert keys and keys['sid'] == self.session_id(wire['from'])
      assert wire['epoch'] == keys['epoch'] and wire['n'] > keys['after']
      packet = {k: v for k, v in wire.items() if k != 'tag'}
      assert hmac.compare_digest(wire['tag'], status_tag(unb64(keys['mac']), packet))
      header = {k: wire[k] for k in ('v', 'type', 'from', 'epoch', 'n')}
      text = AESGCM(unb64(keys['key'])).decrypt(message_nonce(wire['n']), unb64(wire['ciphertext']), canonical(header)).decode()
      keys['after'] = wire['n']
      return {'text': text}
    result = self.session.receive(wire)
    if 'text' in result:
      try:
        body = json.loads(result['text'])
      except ValueError:
        return result
      if body.get('type') == 'status-key':
        self.keys = {**body, 'sid': wire['sid']}
        result.pop('text')
        if 'baseline' in body:
          result['text'] = json.dumps({'type': 'event', 'name': 'liveState', 'payload': body['baseline']})
    return result


def harness():
  identity = RootIdentity.generate()
  app = RootIdentity.generate()
  clients = {c: Client(app) for c in ('tab-a', 'tab-b')}
  allowed = {clients['tab-a'].public_key}
  calls = []
  room = RelayRoom(identity, lambda: allowed, lambda value: value == 'invitation', lambda *args: calls.append(args))
  bodies = {c: [] for c in clients}

  def receive(connection, wire):
    room.receive({'type': 'peer', 'from': clients[connection].public_key, 'to': room.owner,
                  'connection': connection, 'delivery': 1, 'payload': json.dumps(wire)})

  def drain():
    messages = []
    while raw := room.next_message():
      message = json.loads(raw)
      messages.append(message)
      if message['type'] == 'status':
        for connection, tag in message['auth'].items():
          result = clients[connection].receive({**message['payload'], 'tag': tag})
          bodies[connection].append(json.loads(result['text']))
      if message['type'] == 'peer':
        connection = message['connection']
        result = clients[connection].receive(json.loads(message['payload']))
        if 'text' in result:
          bodies[connection].append(json.loads(result['text']))
    return messages

  def join(connection):
    client = clients[connection]
    room.receive({'type': 'join', 'connection': connection, 'from': client.public_key})
    assert drain()[0]['type'] == 'admit'
    receive(connection, client.begin(room.owner))
    drain()
    receive(connection, client.encrypt(room.owner, json.dumps({'type': 'event', 'name': 'session'})))
    room.publish({'speed': 0})
    drain()

  return room, clients, allowed, calls, bodies, receive, drain, join


def test_tabs_have_independent_commands_and_status_sessions():
  room, clients, _, calls, bodies, receive, drain, join = harness()
  join('tab-a')
  join('tab-b')
  assert calls[0][3] != calls[1][3]
  for connection, client in clients.items():
    receive(connection, client.encrypt(room.owner, json.dumps({'method': 'echo', 'id': connection})))
  for connection, _, body, sid in calls[2:]:
    room.send(connection, {'id': body['id'], 'result': 'private'}, sid)
  drain()
  assert bodies['tab-a'][-1]['id'] == 'tab-a' and bodies['tab-b'][-1]['id'] == 'tab-b'
  room.publish({'speed': 1})
  room.publish({'speed': 2})
  messages = drain()
  assert len(messages) == 1 and messages[0]['type'] == 'status'
  assert set(messages[0]['auth']) == set(clients)
  assert bodies['tab-a'][-1] == bodies['tab-b'][-1]
  assert drain() == []


def test_leave_and_reconnect_preserve_group_key_and_other_connection():
  room, clients, _, _, bodies, _, drain, join = harness()
  join('tab-a')
  join('tab-b')
  previous = room.status
  mac = room.members['tab-a'].status_mac
  old_b_mac = room.members['tab-b'].status_mac
  room.receive({'type': 'leave', 'connection': 'tab-b'})
  room.publish({'speed': 10})
  messages = [m for m in drain() if m['type'] == 'status']
  assert len(messages) == 1 and set(messages[0]['auth']) == {'tab-a'}
  assert room.status is previous
  clients['tab-b'].clear()
  join('tab-b')
  assert room.status is previous and room.members['tab-a'].status_mac == mac
  assert room.members['tab-b'].status_mac != old_b_mac
  assert any(b.get('payload', {}).get('full') is True for b in bodies['tab-b'])


def test_revoked_app_has_no_grace_period_or_status_access():
  room, _, allowed, _, _, _, drain, join = harness()
  join('tab-a')
  allowed.clear()
  assert not room.has_subscribers() and not room.members
  assert drain()[0]['type'] == 'remove'


def test_status_authentication_replay_tampering_and_member_forgery():
  room, clients, _, _, _, _, drain, join = harness()
  join('tab-a')
  join('tab-b')
  room.publish({'speed': 1})
  message = json.loads(room.next_message())
  wire = {**message['payload'], 'tag': message['auth']['tab-b']}
  forged = {**wire, 'n': 500}
  with pytest.raises(AssertionError):
    clients['tab-b'].receive(forged)
  # A member knows the content key but lacks the other connection's MAC key.
  header = {k: wire[k] for k in ('v', 'type', 'from', 'epoch', 'n')}
  ciphertext = AESGCM(unb64(clients['tab-a'].keys['key'])).encrypt(
    message_nonce(wire['n']), b'{"type":"event","name":"liveState","payload":{}}', canonical(header))
  forged = {**header, 'ciphertext': b64(ciphertext)}
  forged['tag'] = status_tag(unb64(clients['tab-a'].keys['mac']), forged)
  with pytest.raises(AssertionError):
    clients['tab-b'].receive(forged)
  clients['tab-b'].receive(wire)
  with pytest.raises(AssertionError):
    clients['tab-b'].receive(wire)


def test_allowlist_and_usage_limits_rotate_without_private_session_replacement():
  room, clients, allowed, _, _, _, drain, join = harness()
  join('tab-a')
  join('tab-b')
  sid = room.members['tab-a'].session.session_id(clients['tab-a'].public_key)
  previous = room.status
  allowed.add(RootIdentity.generate().identity())
  drain()
  assert room.status is not previous
  for limit in ('time', 'counter', 'bytes'):
    previous = room.status
    if limit == 'time':
      previous.created -= 86400
    elif limit == 'counter':
      previous.counter = (1 << 20) - 1
    else:
      previous.bytes_used = (1 << 30) - 10
    room.publish({'speed': 1})
    drain()
    assert room.status is not previous
    assert room.members['tab-a'].session.session_id(clients['tab-a'].public_key) == sid


def test_reconnect_installs_baseline_and_rejects_previous_authentication():
  room, clients, _, _, bodies, receive, drain, join = harness()
  join('tab-a')
  room.publish({'speed': 1})
  message = json.loads(room.next_message())
  previous = {**message['payload'], 'tag': message['auth']['tab-a']}
  client = clients['tab-a']
  receive('tab-a', client.begin(room.owner))
  drain()
  receive('tab-a', client.encrypt(room.owner, '{}'))
  drain()
  assert bodies['tab-a'][-1]['payload']['full'] is True
  with pytest.raises(AssertionError):
    client.receive(previous)


def test_revocation_rekeys_remaining_app_and_new_app_cannot_read_old_epoch():
  room, clients, allowed, _, _, _, drain, join = harness()
  join('tab-a')
  old_key = room.status.key
  room.publish({'speed': 1})
  old_packet = json.loads(room.next_message())['payload']
  clients['tab-b'] = Client(RootIdentity.generate())
  allowed.add(clients['tab-b'].public_key)
  join('tab-b')
  new_key = unb64(clients['tab-b'].keys['key'])
  assert new_key != old_key
  from cryptography.exceptions import InvalidTag
  header = {k: old_packet[k] for k in ('v', 'type', 'from', 'epoch', 'n')}
  with pytest.raises(InvalidTag):
    AESGCM(new_key).decrypt(message_nonce(old_packet['n']), unb64(old_packet['ciphertext']), canonical(header))
  removed_key = room.status.key
  allowed.remove(clients['tab-a'].public_key)
  room.publish({'speed': 2})
  messages = drain()
  assert 'tab-a' not in room.members and 'tab-b' in room.members
  current = next(m for m in messages if m['type'] == 'status')
  assert set(current['auth']) == {'tab-b'}
  header = {k: current['payload'][k] for k in ('v', 'type', 'from', 'epoch', 'n')}
  with pytest.raises(InvalidTag):
    AESGCM(removed_key).decrypt(message_nonce(header['n']), unb64(current['payload']['ciphertext']), canonical(header))


def test_bad_client_cannot_disconnect_another_tab():
  room, clients, _, _, bodies, receive, drain, join = harness()
  join('tab-a')
  join('tab-b')
  wire = clients['tab-a'].encrypt(room.owner, '{}')
  receive('tab-a', {**wire, 'n': 500})
  assert 'tab-a' not in room.members and 'tab-b' in room.members
  room.send('tab-b', {'result': 'still online'})
  drain()
  assert any(b.get('result') == 'still online' for b in bodies['tab-b'])


def test_stale_private_reply_never_crosses_session_replacement():
  room, clients, _, calls, bodies, receive, drain, join = harness()
  join('tab-a')
  old_sid = calls[-1][3]
  room.send('tab-a', {'result': 'old'}, old_sid)
  client = clients['tab-a']
  receive('tab-a', client.begin(room.owner))
  # A pending hello does not replace the old session until an authenticated frame.
  for raw in drain():
    assert raw['type'] in ('peer', 'ack')
  receive('tab-a', client.encrypt(room.owner, '{}'))
  room.send('tab-a', {'result': 'too late'}, old_sid)
  drain()
  assert not any(b.get('result') == 'too late' for b in bodies['tab-a'])


def test_unknown_identity_requires_invitation_and_times_out_without_authorization():
  room, clients, allowed, _, _, _, drain, _ = harness()
  allowed.clear()
  room.receive({'type': 'join', 'connection': 'tab-a', 'from': clients['tab-a'].public_key})
  assert drain()[0]['type'] == 'remove'
  room.receive({'type': 'join', 'connection': 'tab-a', 'from': clients['tab-a'].public_key, 'admission': 'invitation'})
  assert drain()[0]['type'] == 'admit'
  room.now = lambda: 1e20
  assert drain()[0]['type'] == 'remove'
  assert not room.has_subscribers()
