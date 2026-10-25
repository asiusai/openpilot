import json

import pytest
from openpilot.system.asius.pqc import RootIdentity

from openpilot.system.asius.relay_room import RelayRoom
from openpilot.system.asius.session import SecureSession


def harness():
  identity = RootIdentity.generate()
  app = RootIdentity.generate()
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
  join('tab-a'); join('tab-b')
  assert calls[0][3] != calls[1][3]
  for connection, client in clients.items():
    receive(connection, client.encrypt(room.owner, json.dumps({'method': 'echo', 'id': connection})))
  for connection, _, body, sid in calls[2:]:
    room.send(connection, {'id': body['id'], 'result': 'private'}, sid)
  drain()
  assert bodies['tab-a'][-1]['id'] == 'tab-a' and bodies['tab-b'][-1]['id'] == 'tab-b'
  room.publish({'speed': 1}); room.publish({'speed': 2})
  messages = drain()
  assert len(messages) == 2
  for message in messages:
    assert message['type'] == 'peer'
    assert bodies[message['connection']][-1]['name'] == 'liveState'
    other = 'tab-b' if message['connection'] == 'tab-a' else 'tab-a'
    with pytest.raises(ValueError):
      clients[other].receive(json.loads(message['payload']))
  assert drain() == []


def test_leave_stops_status_without_rekeying_other_apps():
  room, clients, _, _, _, _, drain, join = harness()
  join('tab-a'); join('tab-b')
  previous = room.members['tab-a'].session.session_id(clients['tab-a'].public_key)
  room.receive({'type': 'leave', 'connection': 'tab-b'})
  room.publish({'speed': 10})
  messages = [m for m in drain() if m['type'] == 'peer']
  assert len(messages) == 1 and messages[0]['connection'] == 'tab-a'
  assert json.loads(messages[0]['payload'])['sid'] == previous
  with pytest.raises(ValueError):
    clients['tab-b'].receive(json.loads(messages[0]['payload']))


def test_revoked_app_has_no_grace_period_or_status_access():
  room, _, allowed, _, _, _, drain, join = harness()
  join('tab-a'); allowed.clear()
  assert not room.has_subscribers() and not room.members
  assert drain()[0]['type'] == 'remove'


def test_status_uses_symmetric_messages_without_signatures_or_shared_keys():
  room, clients, _, _, bodies, _, drain, join = harness()
  join('tab-a'); join('tab-b')
  counters = []
  for n in range(10):
    room.publish({'speed': n})
    messages = drain()
    assert len(messages) == 2
    for message in messages:
      data = json.loads(message['payload'])
      assert data['v'] == 6 and data['to'] == clients[message['connection']].public_key
      assert 'sig' not in data and 'keys' not in message
      assert bodies[message['connection']][-1]['name'] == 'liveState'
    counters.append(json.loads(messages[0]['payload'])['n'])
  assert len(set(counters)) == 10 and counters == sorted(counters)


def test_reconnect_starts_status_with_full_baseline_and_rejects_previous_session():
  room, clients, _, _, bodies, receive, drain, join = harness()
  join('tab-a')
  room.publish({'speed': 1})
  previous = json.loads(drain()[0]['payload'])
  client = clients['tab-a']
  receive('tab-a', client.begin(room.owner)); drain()
  receive('tab-a', client.encrypt(room.owner, '{}')); drain()
  assert bodies['tab-a'][-1]['payload']['full'] is True
  with pytest.raises(ValueError):
    client.receive(previous)


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
