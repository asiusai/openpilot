import json

import pytest

from openpilot.system.asius import access_policy, live_lighting as lights, methods


@pytest.fixture
def bench(tmp_path, monkeypatch):
  monkeypatch.setattr(lights, 'STATE', tmp_path / 'lease.json')
  monkeypatch.setattr(lights, 'LEDS', tmp_path / 'leds')
  ignition = {'state': False}
  monkeypatch.setattr(lights, 'ignition_state', lambda: ignition['state'])
  monkeypatch.setattr(access_policy, 'ignition_state', lambda: ignition['state'])
  for cam in (2, 3):
    for channel in ['ir'] + [f'{color}:{package}' for color in ('red', 'green', 'blue') for package in (0, 2)]:
      path = lights.LEDS / f'asius:cam{cam}:{channel}/brightness'
      path.parent.mkdir(parents=True)
      path.write_text('0')
  return ignition


def call(peer='a', **params):
  body = {'jsonrpc': '2.0', 'id': 1, 'method': 'setLiveLighting', 'params': {'lease': 'a' * 32, **params}}
  return json.loads(methods.handle(body, methods.dispatcher_for_peer(peer)))


def test_capabilities_and_independent_manual_controls(bench):
  state = lights.get_state()
  assert state['canEdit']
  assert state['capabilities'] == {'leds': True, 'ir': True, 'maxLedBrightness': 127, 'maxIrMa': 30}
  assert call(led={'color': '#1234ab', 'brightness': 6}, irMa=10)['result']['expiresIn'] == 5
  assert lights.read_state()['irMa'] == 10
  assert call(led=None, irMa=0)['result']['expiresIn'] == 5
  assert lights.read_state()['led'] is None
  assert lights.read_state()['irMa'] == 0
  assert call()['result']['expiresIn'] == 0
  assert lights.read_state() == {}


@pytest.mark.parametrize('value', [True, None])
def test_ignition_restrictions_apply_to_both_transports_and_reset_is_always_allowed(bench, value):
  call(irMa=30)
  bench['state'] = value
  assert not lights.get_state()['canEdit']
  assert 'ignition' in call(irMa=30)['error']['message']
  assert call()['result']['expiresIn'] == 0


def test_other_peer_or_tab_cannot_renew_or_cancel_an_active_lease(bench):
  call(irMa=20)
  assert 'Another' in call(peer='b', irMa=10)['error']['message']
  assert 'Another' in call(lease='b' * 32)['error']['message']
  assert lights.read_state()['irMa'] == 20
  assert 'Another' in call(peer='b')['error']['message']


def test_expired_or_invalid_lease_does_not_survive_disconnection_or_clock_changes(bench, monkeypatch):
  now = 100.
  monkeypatch.setattr(lights.time, 'monotonic', lambda: now)
  call(irMa=10)
  now = 106.
  assert lights.read_state() == {}
  assert call(peer='b', irMa=20)['result']['expiresIn'] == 5
  now = 90.
  assert lights.read_state() == {}
  for contents in ('invalid', 'null', '[]', '{"expires":"later"}'):
    lights.STATE.write_text(contents)
    assert lights.read_state() == {}


@pytest.mark.parametrize('params', [
  {'irMa': 31}, {'irMa': -1}, {'irMa': True}, {'irMa': 1.5}, {'irMa': float('nan')},
  {'led': {'color': '#ff0000', 'brightness': 128}}, {'led': {'color': '#ff0000', 'brightness': True}},
  {'led': {'color': 'red', 'brightness': 6}}, {'led': {'color': '#ff0000', 'brightness': 6, 'camera': 1}},
  {'lease': '../invalid'},
])
def test_invalid_or_over_limit_requests_do_not_change_output(bench, params):
  assert 'error' in call(**params)
  assert lights.read_state() == {}


def test_missing_road_emitters_are_reported_without_using_cam1(bench):
  (lights.LEDS / 'asius:cam2:ir/brightness').unlink()
  assert not lights.get_state()['capabilities']['ir']
  assert 'not available' in call(irMa=20)['error']['message']
