import json

import pytest
from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.asymmetric import ed25519
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from openpilot.system.asius.relay_room import RelayRoom, STATUS_KEY_MAX_MESSAGES, STATUS_KEY_MAX_BYTES
from openpilot.system.asius.session import SecureSession, canonical, unb64


def harness():
  identity = ed25519.Ed25519PrivateKey.generate()
  app = ed25519.Ed25519PrivateKey.generate()
  clients = {c: SecureSession(app, 'web') for c in ('tab-a', 'tab-b')}
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
      if message['type'] == 'peer':
        connection = message['connection']
        result = clients[connection].receive(json.loads(message['payload']))
        if 'text' in result:
          bodies[connection].append(json.loads(result['text']))
      if message['type'] == 'broadcast':
        for connection, key in message.get('keys', {}).items():
          result = clients[connection].receive(json.loads(key))
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


def plaintext(room, key, message):
  data = json.loads(message['payload'])
  signed = {k: v for k, v in data.items() if k != 'sig'}
  room.identity.public_key().verify(unb64(data['sig']), b'asius-status-v1\n' + canonical(signed))
  header = {k: v for k, v in signed.items() if k != 'ciphertext'}
  return json.loads(AESGCM(unb64(key['key'])).decrypt(data['n'].to_bytes(12, 'big'), unb64(data['ciphertext']), canonical(header)))


def test_same_app_tabs_have_independent_private_sessions_and_one_shared_status():
  room, clients, _, calls, bodies, receive, drain, join = harness()
  join('tab-a')
  first_key = bodies['tab-a'][-1]
  join('tab-b')
  assert bodies['tab-a'][-1] == bodies['tab-b'][-1]
  assert first_key['epoch'] != bodies['tab-a'][-1]['epoch']
  assert calls[0][3] != calls[1][3]
  for connection, client in clients.items():
    receive(connection, client.encrypt(room.owner, json.dumps({'method': 'echo', 'id': connection})))
  for connection, _, body, sid in calls[2:]:
    room.send(connection, {'id': body['id'], 'result': 'private'}, sid)
  drain()
  assert bodies['tab-a'][-1]['id'] == 'tab-a'
  assert bodies['tab-b'][-1]['id'] == 'tab-b'
  room.publish({'speed': 1})
  room.publish({'speed': 2})
  messages = drain()
  assert len(messages) == 1 and messages[0]['type'] == 'broadcast'
  assert plaintext(room, bodies['tab-a'][-2], messages[0])['name'] == 'liveState'


def test_leave_rotates_keys_and_old_key_cannot_decrypt_new_status():
  room, _, _, _, bodies, _, drain, join = harness()
  join('tab-a')
  join('tab-b')
  old_key = bodies['tab-a'][-1]
  room.receive({'type': 'leave', 'connection': 'tab-b'})
  room.publish({'speed': 10})
  broadcast = next(message for message in drain() if message['type'] == 'broadcast')
  new_key = bodies['tab-a'][-1]
  assert old_key['epoch'] != new_key['epoch']
  assert set(broadcast['keys']) == {'tab-a'}
  with pytest.raises(InvalidTag):
    plaintext(room, old_key, broadcast)
  assert plaintext(room, new_key, broadcast)['payload']


def test_revoked_app_has_no_grace_period_and_no_status_key():
  room, _, allowed, _, _, _, drain, join = harness()
  join('tab-a')
  allowed.clear()
  assert not room.has_subscribers()
  assert not room.members
  assert drain()[0]['type'] == 'remove'


def test_unchanged_members_reuse_one_key_and_only_send_one_ciphertext_per_update():
  room, _, _, _, bodies, _, drain, join = harness()
  join('tab-a')
  join('tab-b')
  key = bodies['tab-a'][-1]
  counters = []
  for n in range(10):
    room.publish({'speed': n})
    messages = drain()
    assert len(messages) == 1
    assert 'keys' not in messages[0]
    data = json.loads(messages[0]['payload'])
    assert data['sid'] == key['epoch']
    assert data['v'] == 5 and data['type'] == 'data' and data['to'] == 'status'
    counters.append(data['n'])
    assert plaintext(room, key, messages[0])['name'] == 'liveState'
  assert len(set(counters)) == 10 and counters == sorted(counters)


@pytest.mark.parametrize('attribute,limit', [('counter', STATUS_KEY_MAX_MESSAGES), ('encrypted_bytes', STATUS_KEY_MAX_BYTES)])
def test_usage_limits_rotate_before_sending_and_start_with_full_baseline(attribute, limit):
  room, _, _, _, bodies, _, drain, join = harness()
  join('tab-a')
  old_key = bodies['tab-a'][-1]
  setattr(room.publisher, attribute, limit)
  room.publish({'speed': 10})
  messages = drain()
  assert len(messages) == 1 and set(messages[0]['keys']) == {'tab-a'}
  key = bodies['tab-a'][-1]
  assert key['epoch'] != old_key['epoch']
  assert json.loads(messages[0]['payload'])['n'] == 1
  assert plaintext(room, key, messages[0])['payload']['full'] is True
  with pytest.raises(InvalidTag):
    plaintext(room, old_key, messages[0])


def test_keys_are_wrapped_for_exact_private_session_even_for_same_identity_tabs():
  room, clients, _, _, _, _, _, join = harness()
  join('tab-a')
  # Replace the second join's output handler so we can inspect its undelivered keys.
  client = clients['tab-b']
  room.receive({'type': 'join', 'connection': 'tab-b', 'from': client.public_key})
  room.next_message()  # admit
  room.receive({'type': 'peer', 'from': client.public_key, 'to': room.owner,
                'connection': 'tab-b', 'payload': json.dumps(client.begin(room.owner))})
  welcome = json.loads(room.next_message())
  client.receive(json.loads(welcome['payload']))
  room.receive({'type': 'peer', 'from': client.public_key, 'to': room.owner,
                'connection': 'tab-b', 'payload': json.dumps(client.encrypt(room.owner, '{}'))})
  room.publish({'speed': 10})
  broadcast = json.loads(room.next_message())
  assert set(broadcast['keys']) == {'tab-a', 'tab-b'}
  for connection in clients:
    other = 'tab-a' if connection == 'tab-b' else 'tab-b'
    with pytest.raises(ValueError):
      clients[connection].receive(json.loads(broadcast['keys'][other]))
    assert json.loads(clients[connection].receive(json.loads(broadcast['keys'][connection]))['text'])['type'] == 'stream-key'
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
