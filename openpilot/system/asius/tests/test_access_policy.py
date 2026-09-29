import base64
import json
import os
import subprocess
import threading
import time
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from openpilot.common.params import Params, ParamKeyFlag
from openpilot.system.asius import access_policy as policy, methods, bluetoothd, relayd
from openpilot.system.asius.param_editor import ParameterEditor
from openpilot.system.asius.terminal import TerminalManager


@pytest.fixture
def setup(tmp_path, monkeypatch):
  p = Params(str(tmp_path))
  ignition = {'value': False}
  monkeypatch.setattr(policy, 'ignition_state', lambda: ignition['value'])
  monkeypatch.setattr(methods, 'Params', lambda: p)
  monkeypatch.setattr(bluetoothd, 'Params', lambda: p)
  monkeypatch.setattr(bluetoothd, 'ignition_state', lambda: ignition['value'])
  monkeypatch.setattr(relayd, 'Params', lambda: p)
  return p, ignition


@pytest.mark.parametrize('state', [True, None])
def test_restricted_rpc_and_param_paths_reject_ignition_or_unknown(setup, state):
  p, ignition = setup
  ignition['value'] = state
  for name, params in [('setGithubUsername', {'username': ''}), ('getPairingUrl', {}),
                       ('setUpdateBranch', {'branch': 'other'}), ('installSoftwareUpdate', {})]:
    response = json.loads(methods.handle({'jsonrpc': '2.0', 'id': 1, 'method': name, 'params': params}, methods.dispatcher_for_peer('test')))
    assert 'ignition' in response['error']['message']
  for key in policy.IGNITION_RESTRICTED_PARAMS:
    assert 'ignition' in methods.saveParams({key: None})[key]
  assert methods.saveParams({'AppTerminalEnabled': True})['AppTerminalEnabled'] == 'error: blocked'
  assert not p.get_bool('AppTerminalEnabled')
  with pytest.raises(PermissionError):
    bluetoothd.enable_pairing_mode()
  with pytest.raises(PermissionError):
    relayd.authorize_peer('test')


def test_regular_writes_and_calibration_reboot_shutdown_allowed_during_ignition(setup):
  p, ignition = setup
  ignition['value'] = True
  values = {'DoReboot': True, 'DoShutdown': True, 'CalibrationParams': None, 'LiveTorqueParameters': None,
            'LiveParametersV2': None, 'LiveDelay': None, 'OnroadCycleRequested': True, 'ExperimentalMode': True,
            'Offroad_CarUnrecognized': {'text': 'test'}}
  result = methods.saveParams(values)
  assert all(v.startswith('ok') for v in result.values()), result
  assert p.get_bool('DoReboot') and p.get_bool('DoShutdown')
  assert p.get('Offroad_CarUnrecognized') == {'text': 'test'}


def test_restricted_chunked_write_rechecks_ignition_and_allows_reads(setup):
  p, ignition = setup
  editor = ParameterEditor(methods.SAVE_PARAMS_BLOCKED_KEYS, lambda: p)
  entry, _ = editor._value(p, 'SshEnabled')
  assert entry['requiresIgnitionOff']
  revision = entry['revision']
  editor.write('peer', 'SshEnabled', revision, 'upload', 0, base64.b64encode(b'tr').decode(), False)
  ignition['value'] = True
  assert editor.read('SshEnabled', revision)['done']
  with pytest.raises(PermissionError):
    editor.write('peer', 'SshEnabled', revision, 'upload', 2, base64.b64encode(b'ue').decode(), True)
  assert not p.get_bool('SshEnabled')
  regular, _ = editor._value(p, 'IsMetric')
  editor.write('peer', 'IsMetric', regular['revision'], 'regular', 0, base64.b64encode(b'true').decode(), True)
  assert p.get_bool('IsMetric')
  ignition['value'] = False
  editor.write('peer', 'SshEnabled', revision, 'upload', 2, base64.b64encode(b'ue').decode(), True)
  assert p.get_bool('SshEnabled')


def test_pairing_window_closes_at_ignition(setup):
  p, ignition = setup
  bluetoothd.enable_pairing_mode()
  assert bluetoothd.pairing_mode_active()
  ignition['value'] = True
  assert not bluetoothd.pairing_mode_active()
  ignition['value'] = False
  assert not bluetoothd.pairing_mode_active()


def test_terminal_opt_in_survives_ignition_close_and_reconnect_but_not_manager_reset(setup, monkeypatch):
  p, ignition = setup
  output = []
  terminal = TerminalManager(lambda _, body: output.append(body['payload']), p)
  monkeypatch.setattr(terminal, '_read', lambda *_: None)
  monkeypatch.setattr(terminal, '_spawn', lambda *_: (os.open(os.devnull, os.O_RDWR), Mock(poll=lambda: 0)))
  assert p.get('AppTerminalEnabled', return_default=True) is False
  ignition['value'] = True
  terminal.handle('peer', {'action': 'open', 'sessionId': 'blocked'})
  assert output[-1]['action'] == 'error' and not terminal.sessions
  ignition['value'] = False
  terminal.handle('peer', {'action': 'open', 'sessionId': 'first'})
  assert p.get_bool('AppTerminalEnabled')
  ignition['value'] = True
  p.clear_all(ParamKeyFlag.CLEAR_ON_IGNITION_ON)
  p.clear_all(ParamKeyFlag.CLEAR_ON_ONROAD_TRANSITION)
  terminal.handle('peer', {'action': 'close', 'sessionId': 'first'})
  terminal.handle('peer', {'action': 'open', 'sessionId': 'second'})
  assert output[-1]['action'] == 'opened'
  p.clear_all(ParamKeyFlag.CLEAR_ON_MANAGER_START)
  terminal.handle('peer', {'action': 'input', 'sessionId': 'second', 'data': 'YQ=='})
  assert output[-1]['action'] == 'error' and not terminal.sessions
  terminal.handle('peer', {'action': 'open', 'sessionId': 'third'})
  assert output[-1]['action'] == 'error'


def test_idle_terminal_closes_when_manager_clears_permission(setup, monkeypatch):
  p, _ = setup
  closed = threading.Event()
  terminal = TerminalManager(lambda _, body: closed.set() if body['payload']['action'] == 'closed' else None, p)
  writer = None

  def spawn(*_):
    nonlocal writer
    reader, writer = os.pipe()
    return reader, subprocess.Popen(['sleep', '30'], start_new_session=True)

  monkeypatch.setattr(terminal, '_spawn', spawn)
  try:
    terminal.handle('peer', {'action': 'open', 'sessionId': 'idle'})
    assert p.get_bool('AppTerminalEnabled')
    p.clear_all(ParamKeyFlag.CLEAR_ON_MANAGER_START)
    assert closed.wait(3)
    assert not terminal.sessions
  finally:
    if writer is not None:
      os.close(writer)
    for peer, session in list(terminal.sessions.items()):
      terminal._close(peer, session)


def test_ignition_uses_fresh_panda_signals_not_engagement(monkeypatch):
  panda = SimpleNamespace(pandaType='tres', ignitionLine=True, ignitionCan=False)
  sm = Mock()
  sm.__getitem__ = Mock(return_value=[panda])
  sm.all_checks.return_value = True
  sm.logMonoTime = {'pandaStates': time.monotonic_ns()}
  monkeypatch.setattr(policy, '_state', sm)
  assert policy.ignition_state() is True
  panda.ignitionLine = False
  assert policy.ignition_state() is False
  panda.ignitionCan = True
  assert policy.ignition_state() is True
  sm.all_checks.return_value = False
  assert policy.ignition_state() is None
  sm.all_checks.return_value = True
  sm.logMonoTime['pandaStates'] -= 2_000_000_000
  assert policy.ignition_state() is None


@pytest.mark.parametrize('ignition', [False, True, None])
def test_webrtc_joystick_checks_ignition_for_each_message(monkeypatch, ignition):
  from openpilot.system.webrtc import webrtcd
  monkeypatch.setattr(webrtcd, 'ignition_state', lambda: ignition)
  session = SimpleNamespace(logger=Mock(), incoming_bridge_services=['testJoystick'], incoming_bridge=Mock())
  message = json.dumps({'type': 'testJoystick', 'data': {'axes': [0, 0]}}).encode()
  webrtcd.StreamSession.message_handler(session, message)
  assert session.incoming_bridge.send.called is (ignition is False)
  session.logger.exception.assert_not_called()
