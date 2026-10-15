import datetime
import json
import threading
from unittest.mock import Mock

import pytest

from openpilot.common.params import Params
from openpilot.system.asius import methods


def test_version_includes_stable_fork_identity(monkeypatch):
  monkeypatch.setattr(methods.upstream_athena, 'getVersion', lambda: {'version': 'test', 'remote': 'custom', 'branch': 'test', 'commit': 'abc'})
  assert methods.getVersion() == {'version': 'test', 'remote': 'custom', 'branch': 'test', 'commit': 'abc', 'fork': 'openpilot'}


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
  assert not methods.SAVE_PARAMS_BLOCKED_KEYS.intersection(result)
  assert 'private test value' not in json.dumps(response)


@pytest.mark.parametrize('name', ['uploadFileToUrl', 'uploadFilesToUrls', 'listUploadQueue', 'cancelUpload'])
def test_legacy_upload_methods_are_not_callable(name):
  response = json.loads(methods.handle({'jsonrpc': '2.0', 'method': name, 'id': 1}, methods.dispatcher_for_peer('test')))
  assert response['error']['code'] == -32601
  assert 'requestRouteUpload' in methods.dispatcher_for_peer('test')


def test_relay_does_not_start_legacy_upload_workers(monkeypatch):
  from cryptography.hazmat.primitives.asymmetric import ed25519
  monkeypatch.setattr(methods, "get_device_private_key", ed25519.Ed25519PrivateKey.generate)
  threads = []

  def thread(*, target, **kwargs):
    threads.append(target)
    return Mock()

  monkeypatch.setattr(methods.threading, 'Thread', thread)
  stop = threading.Event()
  stop.set()
  methods.handle_long_poll(Mock(), stop)
  assert threads == [methods.upstream_athena.ws_manage, methods.ws_recv, methods.ws_send, methods.live_state_handler]
