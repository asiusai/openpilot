import asyncio
import json
import pytest

from openpilot.system.asius.pairing_confirmation import Confirmation, read_confirmation, send_button


@pytest.mark.asyncio
async def test_rejected_or_unfinished_enrollment_discards_its_new_bond(tmp_path):
  now = [10.]
  rejected = []
  control = Confirmation(lambda: True, tmp_path, lambda: now[0], on_reject=rejected.append)
  task = asyncio.create_task(control.request('/phone', 42))
  await asyncio.sleep(0)
  control.button('press', 10.)
  with pytest.raises(PermissionError):
    await task
  assert rejected == ['/phone']
  task = asyncio.create_task(control.request('/other', 43))
  await asyncio.sleep(0)
  now[0] = 14.
  control.button('hold', 11.)
  await task
  now[0] = 45.
  control.expire()
  assert rejected == ['/phone', '/other']
  assert not control.consume('/other')


@pytest.mark.asyncio
async def test_short_press_after_confirmation_cancels_unfinished_enrollment(tmp_path):
  from openpilot.system.asius.pairing_confirmation import read_announcement
  now, rejected = [10.], []
  control = Confirmation(lambda: True, tmp_path, lambda: now[0], on_reject=rejected.append)
  task = asyncio.create_task(control.request('/phone', 42))
  await asyncio.sleep(0)
  now[0] = 14.
  control.button('hold', 11.)
  await task
  assert control.button('press', 14.)
  assert not control.consume('/phone')
  assert rejected == ['/phone']
  assert read_announcement(tmp_path, 14.)['outcome'] == 'failed'


@pytest.mark.asyncio
async def test_local_screen_confirmation_binds_request_and_is_disabled_on_v0(tmp_path):
  control = Confirmation(lambda: True, tmp_path, lambda: 10.)
  task = asyncio.create_task(control.request('/phone', 42))
  await asyncio.sleep(0)
  request_id = read_confirmation(tmp_path, 10.)['id']
  control.screen(request_id, True)
  assert not control.pending.result.done()
  control.allow_ui = True
  control.screen('previous-request', True)
  assert not control.pending.result.done()
  control.screen(request_id, True)
  await task
  assert control.consume('/phone')


@pytest.mark.asyncio
async def test_confirmation_requires_new_hold_and_binds_one_peer(tmp_path):
  now = [10.]
  control = Confirmation(lambda: True, tmp_path, lambda: now[0])
  task = asyncio.create_task(control.request('/phone', 42))
  await asyncio.sleep(0)
  value = read_confirmation(tmp_path, 10.)
  assert value['digits'] == '000042'
  assert not control.consume('/phone')
  with pytest.raises(PermissionError):
    await control.request('/attacker', 123456)
  assert read_confirmation(tmp_path, 10.) == value
  now[0] = 14.
  assert control.button('hold', 7.)  # The initial hold must never approve a later request.
  assert not task.done()
  assert control.button('hold', 11.)
  await task
  assert read_confirmation(tmp_path, 14.) is None
  assert not control.consume('/attacker')
  assert control.consume('/phone')
  assert not control.consume('/phone')


@pytest.mark.asyncio
async def test_short_press_cancels_only_current_request(tmp_path):
  control = Confirmation(lambda: True, tmp_path, lambda: 10.)
  task = asyncio.create_task(control.request('/phone', 123456))
  await asyncio.sleep(0)
  control.button('press', 10.)
  with pytest.raises(PermissionError):
    await task
  second = asyncio.create_task(control.request('/phone', 654321))
  await asyncio.sleep(0)
  assert read_confirmation(tmp_path, 10.)['digits'] == '654321'
  control.cancel()
  with pytest.raises(PermissionError):
    await second


@pytest.mark.asyncio
@pytest.mark.parametrize('reason', ['expiry', 'ignition', 'disconnect'])
async def test_pending_request_cannot_survive_expiry_ignition_or_disconnect(tmp_path, reason):
  now, active = [10.], [True]
  control = Confirmation(lambda: active[0], tmp_path, lambda: now[0])
  task = asyncio.create_task(control.request('/phone', 123456))
  await asyncio.sleep(0)
  if reason == 'expiry':
    now[0] = 40.
  elif reason == 'ignition':
    active[0] = False
  else:
    control.cancel('/phone')
  control.expire()
  with pytest.raises(PermissionError):
    await task
  assert not control.consume('/phone')
  assert read_confirmation(tmp_path, now[0]) is None


@pytest.mark.asyncio
async def test_approval_expires_and_process_restart_clears_visible_state(tmp_path):
  now = [10.]
  control = Confirmation(lambda: True, tmp_path, lambda: now[0])
  task = asyncio.create_task(control.request('/phone', 42))
  await asyncio.sleep(0)
  now[0] = 14.
  control.button('hold', 11.)
  await task
  now[0] = 44.
  assert not control.consume('/phone')
  task = asyncio.create_task(control.request('/phone', 42))
  await asyncio.sleep(0)
  assert read_confirmation(tmp_path, 44.)
  restarted = Confirmation(lambda: True, tmp_path, lambda: now[0])
  assert read_confirmation(tmp_path, 44.) is None
  assert not restarted.consume('/phone')
  control.cancel()
  with pytest.raises(PermissionError):
    await task


@pytest.mark.asyncio
async def test_socket_routes_button_actions_without_persistent_parameters(tmp_path):
  now = [10.]
  opened = []
  control = Confirmation(lambda: True, tmp_path, lambda: now[0])
  server = asyncio.create_task(control.buttons(lambda: opened.append(True)))
  await asyncio.sleep(0)
  assert send_button('hold', 7., tmp_path)
  for _ in range(20):
    if opened:
      break
    await asyncio.sleep(.01)
  assert opened == [True]
  request = asyncio.create_task(control.request('/phone', 123456))
  await asyncio.sleep(0)
  assert send_button('press', 10., tmp_path)
  with pytest.raises(PermissionError):
    await asyncio.wait_for(request, 1.)
  assert opened == [True]
  server.cancel()
  with pytest.raises(asyncio.CancelledError):
    await server
  assert not (tmp_path / 'button.sock').exists()


def test_stale_state_cannot_keep_led_or_speech_running(tmp_path):
  (tmp_path / 'status.json').write_text(json.dumps({'phase': 'confirm', 'created': 10., 'expires': 40.}))
  assert read_confirmation(tmp_path, 10.)
  assert read_confirmation(tmp_path, 40.) is None
  assert read_confirmation(tmp_path, 9.) is None
  (tmp_path / 'status.json').write_text('{bad')
  assert read_confirmation(tmp_path, 10.) is None
