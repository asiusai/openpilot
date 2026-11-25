from openpilot.selfdrive.alpamayo.planner import GAP_FAIL_S, LOAD_TIMEOUT_S, BigModelState


def test_loading_then_active():
  state = BigModelState(0.)
  assert not state.update(1., False, False) and state.state == BigModelState.LOADING
  assert state.update(60., True, False) and state.state == BigModelState.ACTIVE


def test_load_timeout_fails_and_recovers_when_not_engaged():
  state = BigModelState(0.)
  assert not state.update(LOAD_TIMEOUT_S+1, False, False) and state.state == BigModelState.FAILED
  assert not state.update(LOAD_TIMEOUT_S+2, True, True) and state.state == BigModelState.FAILED  # never takes over while engaged
  assert state.update(LOAD_TIMEOUT_S+3, True, False) and state.state == BigModelState.ACTIVE


def test_short_gap_keeps_active_and_uses_the_small_model():
  state = BigModelState(0.)
  assert state.update(10., True, True)
  assert not state.update(10.+GAP_FAIL_S/2, False, True) and state.state == BigModelState.ACTIVE
  assert state.update(10.+GAP_FAIL_S/2+.05, True, True) and state.state == BigModelState.ACTIVE


def test_long_gap_fails_until_disengaged():
  state = BigModelState(0.)
  assert state.update(10., True, True)
  assert not state.update(10.+GAP_FAIL_S+.1, False, True) and state.state == BigModelState.FAILED
  assert not state.update(20., True, True) and state.state == BigModelState.FAILED
  assert state.update(21., True, False) and state.state == BigModelState.ACTIVE
