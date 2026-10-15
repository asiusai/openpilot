import subprocess
from unittest.mock import Mock, PropertyMock, patch

import pytest

from openpilot.common.params import Params
from openpilot.system.updated import updated


@pytest.fixture
def updater(tmp_path, monkeypatch):
  updated_params = Params(str(tmp_path))
  monkeypatch.setattr(updated, 'Params', lambda: updated_params)
  updated_params.put('UpdaterTargetBranch', 'master', block=True)
  return updated.Updater()


def test_offline_check_does_not_run_git(updater, monkeypatch):
  monkeypatch.setattr(updated.HARDWARE, 'get_network_type', lambda: updated.NetworkType.none)
  run = Mock()
  monkeypatch.setattr(updated, 'run', run)
  updater._has_internet = True
  assert updater.check_for_update() is False
  assert not updater.has_internet
  run.assert_not_called()


@pytest.mark.parametrize('error', [
  subprocess.CalledProcessError(128, ['git'], output='fatal: Could not resolve host: github.com'),
  subprocess.CalledProcessError(128, ['git'], output='Failed to connect to github.com port 443'),
  subprocess.TimeoutExpired(['git'], 30),
])
def test_unreachable_server_waits_without_changing_failure_counters(updater, monkeypatch, error):
  monkeypatch.setattr(updated.HARDWARE, 'get_network_type', lambda: updated.NetworkType.wifi)
  monkeypatch.setattr(updated, 'setup_git_options', Mock())
  monkeypatch.setattr(updated, 'run', Mock(side_effect=error))
  updater.params.put('UpdateFailedCount', 4, block=True)
  assert updater.check_for_update() is False
  assert not updater.has_internet
  assert updater.params.get('UpdateFailedCount') == 4


def test_repository_errors_remain_errors(updater, monkeypatch):
  monkeypatch.setattr(updated.HARDWARE, 'get_network_type', lambda: updated.NetworkType.wifi)
  monkeypatch.setattr(updated, 'setup_git_options', Mock())
  error = subprocess.CalledProcessError(128, ['git'], output='fatal: Authentication failed')
  monkeypatch.setattr(updated, 'run', Mock(side_effect=error))
  with pytest.raises(subprocess.CalledProcessError):
    updater.check_for_update()


def test_interrupted_download_invalidates_staging_without_recording_failure(updater, tmp_path, monkeypatch):
  class StopLoop(BaseException):
    pass

  helper = Mock(user_request=updated.UserRequest.FETCH)
  helper.sleep.side_effect = [None, StopLoop()]
  monkeypatch.setattr(updated, 'WaitTimeHelper', lambda: helper)
  monkeypatch.setattr(updated, 'Updater', lambda: updater)
  monkeypatch.setattr(updated, 'LOCK_FILE', str(tmp_path / 'lock'))
  overlay = tmp_path / '.overlay_init'
  overlay.touch()
  monkeypatch.setattr(updated, 'OVERLAY_INIT', overlay)
  monkeypatch.setattr(updated, 'init_overlay', Mock())
  monkeypatch.setattr(updated, 'set_consistent_flag', Mock())
  monkeypatch.setattr(updated, 'system_time_valid', lambda: True)
  monkeypatch.setattr(updated, 'should_skip_noop_vamos_fetch', lambda *args: False)
  monkeypatch.setattr(updater, 'check_for_update', Mock(return_value=True))
  monkeypatch.setattr(updater, 'fetch_update', Mock(side_effect=subprocess.CalledProcessError(128, ['git'], output='Connection reset')))
  set_params = Mock()
  monkeypatch.setattr(updater, 'set_params', set_params)
  with pytest.raises(StopLoop):
    updated.main()
  assert not overlay.exists()
  assert updater.params.get('UpdaterState') == 'waiting for internet'
  assert updater.params.get('LastUpdateException') is None
  assert all(call.args == (False, 0, None) for call in set_params.call_args_list)


def test_online_check_uses_one_bounded_request(updater, monkeypatch):
  monkeypatch.setattr(updated.HARDWARE, 'get_network_type', lambda: updated.NetworkType.wifi)
  monkeypatch.setattr(updated, 'setup_git_options', Mock())
  monkeypatch.setattr(updater, 'get_branch', lambda _: 'master')
  monkeypatch.setattr(updater, 'get_commit_hash', lambda _: 'a' * 40)
  run = Mock(return_value=f'{"a" * 40}\trefs/heads/master\n')
  monkeypatch.setattr(updated, 'run', run)
  assert updater.check_for_update() is True
  assert updater.has_internet
  run.assert_called_once_with(['git', 'ls-remote', '--heads', 'origin'], updated.OVERLAY_MERGED, timeout=30)


def test_fetch_gets_only_selected_submodule_revisions(updater, monkeypatch):
  commands = []
  monkeypatch.setattr(updated, 'run', lambda cmd, *args, **kwargs: commands.append(cmd) or '')
  for name in ['set_consistent_flag', 'setup_git_options', 'finalize_update']:
    monkeypatch.setattr(updated, name, Mock())
  monkeypatch.setattr(updated, 'AGNOS', False)
  monkeypatch.setattr(updated, 'vamos_update_supported', lambda: False)
  updater.fetch_update()
  fetch = ['git', 'fetch', '--no-recurse-submodules', 'origin', 'master']
  checkout = ['git', 'checkout', '--force', '--no-recurse-submodules', '-B', 'master', 'FETCH_HEAD']
  submodules = ['git', 'submodule', 'update', '--init', '--recursive']
  assert commands.index(fetch) < commands.index(checkout) < commands.index(submodules)


def test_persistent_offroad_failure_keeps_upstream_threshold(updater, monkeypatch):
  monkeypatch.setattr(updated.os.path, 'exists', lambda _: False)
  monkeypatch.setattr(updated, 'parse_release_notes', lambda _: b'')
  monkeypatch.setattr(updated, 'get_build_metadata', lambda: Mock(tested_channel=True))
  alert = Mock()
  monkeypatch.setattr(updated, 'set_offroad_alert', alert)
  updater._has_internet = True
  with patch.object(updated.Updater, 'update_ready', new_callable=PropertyMock, return_value=False):
    updater.set_params(False, 4, 'broken repository')
    assert not any(call.args[:2] == ('Offroad_UpdateFailed', True) for call in alert.call_args_list)
    alert.reset_mock()
    updater.set_params(False, 16, 'broken repository')
    assert any(call.args[:2] == ('Offroad_UpdateFailed', True) for call in alert.call_args_list)
