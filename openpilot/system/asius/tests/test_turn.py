import json
from unittest.mock import Mock

import pytest
import requests

from openpilot.system.asius import turn


@pytest.fixture
def provider(monkeypatch):
  monkeypatch.setattr(turn.time, 'time', lambda: 1800000000)
  monkeypatch.setattr(turn, 'sign_jwt', Mock(return_value='device-proof'))
  monkeypatch.setattr(turn, 'get_device_public_key', lambda: 'key')
  monkeypatch.setattr(turn, 'dongle_id_from_public_key', lambda _: 'registered-device')
  monkeypatch.setattr(turn, 'HELPER', Mock(is_file=lambda: True))
  response = Mock(status_code=200)
  response.__enter__ = Mock(return_value=response)
  response.__exit__ = Mock(return_value=False)
  response.iter_content.return_value = [json.dumps({
    'iceServers': [{'urls': ['turn:turn.example:3478'], 'username': 'user', 'credential': 'temporary'}],
    'expiresAt': 1800003600000,
  }).encode()]
  post = Mock(return_value=response)
  monkeypatch.setattr(turn.requests, 'post', post)
  return post, response


def test_fetches_credentials_using_registered_device_identity(provider):
  post, _ = provider
  result = turn.prepare_live()
  assert result['mode'] == 'internet'
  assert result['iceServers'][0]['credential'] == 'temporary'
  assert turn.sign_jwt.call_args.args[0] == {
    'type': 'turn', 'aud': 'asius-turn', 'identity': 'registered-device', 'method': 'POST', 'path': turn.TURN_PATH,
  }
  assert post.call_args.kwargs == {'headers': {'Authorization': 'Bearer device-proof'}, 'timeout': (3, 6), 'allow_redirects': False, 'stream': True}


def test_offline_phone_skips_gateway_and_requires_local_support(provider, monkeypatch):
  post, _ = provider
  assert turn.prepare_live(False, True) == {'mode': 'localWifi'}
  assert turn.prepare_live(False, False)['mode'] == 'unavailable'
  monkeypatch.setattr(turn, 'HELPER', Mock(is_file=lambda: False))
  assert turn.prepare_live(False, True)['mode'] == 'unavailable'
  post.assert_not_called()


@pytest.mark.parametrize('error', [requests.ConnectionError(), requests.Timeout()])
def test_device_network_failure_selects_local_wifi(provider, error):
  post, _ = provider
  post.side_effect = error
  assert turn.prepare_live(True, True) == {'mode': 'localWifi'}
  assert turn.prepare_live(True, False)['mode'] == 'unavailable'


@pytest.mark.parametrize('status', [301, 401, 403, 429, 502, 503])
def test_http_errors_do_not_misreport_device_as_offline(provider, status):
  _, response = provider
  response.status_code = status
  result = turn.prepare_live(True, True)
  assert result['mode'] == 'unavailable' and result['error']
  response.iter_content.assert_not_called()


def test_invalid_or_expired_credentials_and_tls_errors_fail_closed(provider):
  post, response = provider
  for body in [b'x' * 32769, b'bad json', b'{}', b'{"iceServers":[],"expiresAt":0}',
               json.dumps({'iceServers': [{'urls': 'turn:example'}], 'expiresAt': 1800003600000}).encode()]:
    response.iter_content.return_value = [body]
    assert turn.prepare_live(True, True)['mode'] == 'unavailable'
  post.side_effect = requests.exceptions.SSLError('private details')
  result = turn.prepare_live(True, True)
  assert result['mode'] == 'unavailable' and 'private details' not in result['error']


def test_preparation_is_exposed_on_authorized_dispatcher(provider):
  from openpilot.system.asius import methods
  result = json.loads(methods.handle({'jsonrpc': '2.0', 'id': 1, 'method': 'prepareLive',
                                     'params': {'phoneOnline': False, 'localWifiSupported': True}}, methods.dispatcher_for_peer('app')))
  assert result['result'] == {'mode': 'localWifi'}
  assert 'prepareLive' not in methods.NETWORK_ONLY_METHODS


def test_relay_rejects_unpaired_preparation(provider, monkeypatch):
  from openpilot.system.asius import methods
  monkeypatch.setattr(methods, 'relay_room', Mock())
  monkeypatch.setattr(methods, 'Params', lambda: Mock(get=lambda _: 'device'))
  monkeypatch.setattr(methods, 'load_authorized_peers', dict)
  dispatch = Mock()
  monkeypatch.setattr(methods, 'handle_rpc', dispatch)
  methods.handle_peer_message('connection', 'unpaired', {'method': 'prepareLive', 'id': 1}, 'session')
  dispatch.assert_not_called()
  provider[0].assert_not_called()
