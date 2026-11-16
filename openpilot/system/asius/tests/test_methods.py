import asyncio
import datetime
import json
import threading
from unittest.mock import AsyncMock, Mock

import pytest

from openpilot.common.params import Params, UnknownKeyName
from openpilot.system.asius import methods


@pytest.mark.parametrize('method,params', [
  ('setDeviceName', {'name': 1}),
  ('configureTailscale', {'disconnect': 'false'}),
  ('connectNetwork', {'ssid': 'test', 'hidden': 1}),
  ('completeTraining', {'version': '0', 'recordFront': 1, 'shareDrivingData': False}),
  ('setGithubUsername', {'username': '../other'}),
  ('getNavigationRoute', {'revision': 'test', 'offset': True}),
  ('saveParams', {'params_to_update': {'IsMetric': float('nan')}}),
  ('startStream', {'sdp': '', 'cameras': ['invalid'], 'bridge_services_out': []}),
])
def test_rpc_rejects_invalid_parameters_before_side_effects(monkeypatch, method, params):
  monkeypatch.setattr(methods, 'Params', Mock(side_effect=AssertionError('unexpected parameter access')))
  monkeypatch.setattr(methods.subprocess, 'run', Mock(side_effect=AssertionError('unexpected command')))
  result = json.loads(methods.handle({'id': 1, 'method': method, 'params': params}, methods.dispatcher_for_peer('test')))
  assert result['error']['code'] == -32602


def test_rpc_validates_name_parameter_and_preserves_nested_json(tmp_path, monkeypatch):
  params = Params(str(tmp_path))
  monkeypatch.setattr(methods, 'Params', lambda: params)
  dispatcher = methods.dispatcher_for_peer('test')
  result = json.loads(methods.handle({'id': 1, 'method': 'setDeviceName', 'params': {'name': 'Bench'}}, dispatcher))
  assert result['result'] == {'name': 'Bench'}
  assert params.get('DeviceName') == 'Bench'
  value = {'nested': [None, True, 2, 3.5, 'text']}
  result = json.loads(methods.handle({'id': 2, 'method': 'echo', 'params': {'s': value}}, dispatcher))
  assert result['result'] == value


def test_version_includes_stable_fork_identity(monkeypatch):
  monkeypatch.setattr(methods.upstream_athena, 'getVersion', lambda: {'version': 'test', 'remote': 'custom', 'branch': 'test', 'commit': 'abc'})
  assert methods.getVersion() == {'version': 'test', 'remote': 'custom', 'branch': 'test', 'commit': 'abc', 'fork': methods.FORK,
                                   'termsVersion': methods.terms_version, 'trainingVersion': methods.training_version}


@pytest.mark.parametrize('ap_first', [False, True])
@pytest.mark.parametrize('station_connected', [False, True])
def test_wifi_state_uses_station_when_hotspot_is_present(monkeypatch, ap_first, station_connected):
  station = 'wlan0:wifi:connected:openpilot connection Office' if station_connected else 'wlan0:wifi:disconnected:'
  # An externally managed AP can be listed before the station interface.
  ap = 'asiusap0:wifi:connected (externally):Asius-Live-test'
  rows = [ap, station] if ap_first else [station, ap]
  monkeypatch.setattr(methods, '_nmcli', lambda args: '\n'.join(rows))
  expected = {'ssid': 'Office', 'status': 2} if station_connected else {'ssid': None, 'status': 0}
  assert methods._nmcli_wifi_state() == expected


@pytest.mark.parametrize('station_connected', [False, True])
def test_metering_uses_station_connection_with_concurrent_hotspot(monkeypatch, station_connected):
  def nmcli(args):
    if args[-2:] == ['show', '--active']:
      rows = ['Camera hotspot:802-11-wireless:asiusap0']
      if station_connected:
        rows.append('openpilot connection Office:802-11-wireless:wlan0')
      return '\n'.join(rows)
    assert args[-1] == 'openpilot connection Office'
    return 'yes'

  monkeypatch.setattr(methods, '_nmcli', nmcli)
  assert methods._connection_metered() == (1 if station_connected else 0)


def test_wifi_scan_uses_station_interface(monkeypatch):
  monkeypatch.setattr(methods, '_saved_wifi_connections', lambda: {'Office'})

  def nmcli(args):
    if args[-2:] == ['ifname', 'wlan0']:
      return 'yes:Office:75:WPA2'
    return 'yes:Asius-Live-test:100:WPA2'

  monkeypatch.setattr(methods, '_nmcli', nmcli)
  assert [(network['ssid'], network['connected']) for network in methods._nmcli_wifi_networks()] == [('Office', True)]


@pytest.mark.parametrize('transport', ['web', 'bluetooth'])
@pytest.mark.parametrize('ignition', [False, True, None])
def test_sound_preview_checks_ignition_on_each_transport(tmp_path, monkeypatch, transport, ignition):
  from openpilot.system.asius import access_policy
  params = Params(str(tmp_path))
  monkeypatch.setattr(methods, 'Params', lambda: params)
  monkeypatch.setattr(methods.HARDWARE, 'get_device_type', lambda: 'v0')
  monkeypatch.setattr(access_policy, 'ignition_state', lambda: ignition)
  request = {'jsonrpc': '2.0', 'method': 'previewSound', 'params': {}, 'id': 1}
  if transport == 'bluetooth':
    from openpilot.system.asius.bluetoothd import BlePeerEngine
    engine = BlePeerEngine.__new__(BlePeerEngine)
    engine.rpc_active = 0
    engine.send_body = AsyncMock()
    engine.peer_devices = {'test': '/test'}
    engine.sessions = {'/test': Mock(session_id=lambda _peer: 'test-session')}
    asyncio.run(engine.handle_rpc('test', request))
    response = engine.send_body.call_args.args[1]
  else:
    response = json.loads(methods.handle(request, methods.dispatcher_for_peer('test')))
  if ignition is False:
    assert response['result'] == {'queued': True}
    assert params.get('SoundVolumePreview')['id']
  else:
    assert 'error' in response
    assert params.get('SoundVolumePreview') is None


def test_get_all_params_serializes_current_api_without_protected_values(tmp_path, monkeypatch):
  params = Params(str(tmp_path))
  values = {
    'DeviceName': 'Test device',
    'IsMetric': True,
    'VoiceAlertsEnabled': True,
    'LongitudinalPersonality': 1,
    'LastUpdateUptimeOnroad': 1.25,
    'Offroad_CarUnrecognized': {'text': 'Test alert'},
    'LastUpdateTime': datetime.datetime(2026, 9, 29, tzinfo=datetime.UTC),
    'CarParams': b'binary',
    'CloudUploadState': {'keys': [{'key': 'private test value'}]},
  }
  for key, value in values.items():
    params.put(key, value, block=True)
  monkeypatch.setattr(methods, 'Params', lambda: params)
  response = json.loads(methods.handle({'jsonrpc': '2.0', 'method': 'getAllParams', 'id': 1}, methods.dispatcher_for_peer('test')))
  result = response['result']
  for key in ('DeviceName', 'IsMetric', 'LongitudinalPersonality', 'LastUpdateUptimeOnroad', 'Offroad_CarUnrecognized'):
    assert result[key] == values[key]
  assert result['LastUpdateTime'] == values['LastUpdateTime'].timestamp()
  assert result['InstallDate'] is None
  assert result['VoiceAlertsEnabled'] is True
  assert 'CarParams' not in result
  assert not methods.READ_PARAMS_BLOCKED_KEYS.intersection(result)
  assert 'private test value' not in json.dumps(response)


@pytest.mark.parametrize('name', ['uploadFileToUrl', 'uploadFilesToUrls', 'listUploadQueue', 'cancelUpload'])
def test_legacy_upload_methods_are_not_callable(name):
  response = json.loads(methods.handle({'jsonrpc': '2.0', 'method': name, 'id': 1}, methods.dispatcher_for_peer('test')))
  assert response['error']['code'] == -32601
  assert 'requestRouteUpload' in methods.dispatcher_for_peer('test')


def test_relay_does_not_start_legacy_upload_workers(monkeypatch):
  from openpilot.system.asius.pqc import RootIdentity
  monkeypatch.setattr(methods, "get_device_private_key", RootIdentity.generate)
  threads = []

  def thread(*, target, **kwargs):
    threads.append(target)
    return Mock()

  monkeypatch.setattr(methods.threading, 'Thread', thread)
  stop = threading.Event()
  stop.set()
  methods.handle_long_poll(Mock(), stop, "wss://relay.example")
  assert threads == [methods.upstream_athena.ws_manage, methods.ws_recv, methods.ws_send, methods.live_state_handler]


@pytest.mark.parametrize('transport', ['web', 'bluetooth'])
def test_selected_params_read_only_requested_values(tmp_path, monkeypatch, transport):
  params = Params(str(tmp_path))
  params.put_bool('IsMetric', True, block=True)
  params.put('GithubUsername', 'test-user', block=True)
  params.put('GithubSshKeys', 'ssh-ed25519 test-public-key', block=True)
  params.put('HasAcceptedTerms', methods.terms_version, block=True)
  params.put('DeviceName', 'Must not be included', block=True)
  params.put('CloudUploadState', {'key': 'must stay private'}, block=True)
  monkeypatch.setattr(methods, 'Params', lambda: params)
  names = ['IsMetric', 'GithubUsername', 'GithubSshKeys', 'HasAcceptedTerms', 'InstallDate', 'CloudUploadState', 'CarParams', 'IsMetric']
  request = {'jsonrpc': '2.0', 'method': 'getParams', 'params': {'names': names}, 'id': 1}
  if transport == 'bluetooth':
    from openpilot.system.asius.bluetoothd import BlePeerEngine
    engine = BlePeerEngine.__new__(BlePeerEngine)
    engine.rpc_active = 0
    engine.send_body = AsyncMock()
    engine.peer_devices = {'test': '/test'}
    engine.sessions = {'/test': Mock(session_id=lambda _peer: 'test-session')}
    asyncio.run(engine.handle_rpc('test', request))
    response = engine.send_body.call_args.args[1]
  else:
    response = json.loads(methods.handle(request, methods.dispatcher_for_peer('test')))
  assert response['result'] == {'IsMetric': True, 'GithubUsername': 'test-user', 'GithubSshKeys': 'ssh-ed25519 test-public-key',
                                'HasAcceptedTerms': methods.terms_version, 'InstallDate': None}
  assert methods.getParams([]) == {}
  assert methods.getParams(['GithubUsername']) == {'GithubUsername': methods.getAllParams()['GithubUsername']}
  for name in ('GithubUsername', 'GithubSshKeys', 'HasAcceptedTerms', 'CompletedTrainingVersion'):
    assert methods.saveParams({name: 'unvalidated'})[name] == 'error: blocked'


@pytest.mark.parametrize('names', [None, 'IsMetric', {}, [1], ['IsMetric'] * 257, ['../secret'], ['UnknownParameter']])
def test_selected_params_reject_invalid_requests(names):
  with pytest.raises((ValueError, TypeError, UnknownKeyName)):
    methods.getParams(names)


@pytest.mark.parametrize('name', ['getOnboardingStatus', 'getSoftwareUpdateState', 'getGithubUsername', 'getSshAuthorizedKeys', 'getDeviceName'])
def test_parameter_read_wrappers_are_not_callable(name):
  response = json.loads(methods.handle({'jsonrpc': '2.0', 'method': name, 'id': 1}, methods.dispatcher_for_peer('test')))
  assert response['error']['code'] == -32601


@pytest.mark.parametrize('enabled,host', [(False, 'wss://old.example'), (True, 'wss://new.example')])
def test_relay_closes_when_services_change_after_connect(tmp_path, monkeypatch, enabled, host):
  from openpilot.system.asius.pqc import RootIdentity
  params = Params(str(tmp_path))
  params.put_bool('RelayEnabled', enabled, block=True)
  params.put('RelayHost', host, block=True)
  monkeypatch.setattr(methods, 'Params', lambda: params)
  monkeypatch.setattr(methods, 'get_device_private_key', RootIdentity.generate)
  end = Mock()
  end.wait.side_effect = [False, True]
  monkeypatch.setattr(methods.threading, 'Event', lambda: end)
  monkeypatch.setattr(methods.threading, 'Thread', lambda **kwargs: Mock())
  methods.handle_long_poll(Mock(), None, 'wss://old.example')
  end.set.assert_called_once()
