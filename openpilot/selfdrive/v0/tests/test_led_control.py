from types import SimpleNamespace

import pytest

from openpilot.common.params import ParamKeyFlag, Params
from openpilot.selfdrive.v0.led_control import MANUAL_LED_PARAM, get_led_state, manual_led_channels, set_led_state
from openpilot.selfdrive.v0.tests.test_ledd import FakeSubMaster
from openpilot.system.app import methods

COLORS = ["#ff0000", "#00ff00", "#0000ff", "#ffffff", "#804020", "#000000"]


@pytest.fixture
def params(tmp_path):
  params = Params(str(tmp_path))
  params.put_bool("IsOffroad", True, block=True)
  return params


def parked_sm():
  sm = FakeSubMaster()
  sm.set('deviceState', SimpleNamespace(started=False))
  return sm


def test_individual_colors_map_to_six_leds_and_leave_driver_camera_off(params):
  state = set_led_state(params, True, True, COLORS, 100)
  assert state == {"supported": True, "parked": True, "manual": True, "colors": COLORS, "brightness": 100}
  assert manual_led_channels(parked_sm(), params) == {
    1: [0] * 9,
    2: [255, 0, 0, 0, 255, 0, 0, 0, 255],
    3: [255, 255, 255, 128, 64, 32, 0, 0, 0],
  }


def test_brightness_scales_all_channels_and_zero_is_off_without_losing_colors(params):
  set_led_state(params, True, True, ["#804020"] * 6, 50)
  assert manual_led_channels(parked_sm(), params) == {1: [0] * 9, 2: [64, 32, 16] * 3, 3: [64, 32, 16] * 3}
  set_led_state(params, True, True, COLORS, 0)
  assert manual_led_channels(parked_sm(), params) == {1: [0] * 9, 2: [0] * 9, 3: [0] * 9}
  assert get_led_state(params, True)["colors"] == COLORS
  assert get_led_state(params, True)["manual"] is True


@pytest.mark.parametrize("brightness", [-1, 101, 0.5, True, "10", None, float('nan')])
def test_rejects_invalid_brightness_without_replacing_existing_settings(params, brightness):
  set_led_state(params, True, True, COLORS, 10)
  with pytest.raises(ValueError):
    set_led_state(params, True, True, COLORS, brightness)
  assert get_led_state(params, True)["brightness"] == 10


@pytest.mark.parametrize("colors", [[], COLORS[:5], COLORS + ["#ffffff"], ["red"] * 6, ["#fff"] * 6, [None] * 6, "#ffffff"])
def test_rejects_invalid_colors(params, colors):
  with pytest.raises(ValueError):
    set_led_state(params, True, True, colors, 10)
  assert params.get(MANUAL_LED_PARAM) is None


@pytest.mark.parametrize("manual", [True, False])
def test_rpc_rejects_onroad_or_unknown_parked_state(params, monkeypatch, manual):
  monkeypatch.setattr(methods, 'Params', lambda: params)
  monkeypatch.setattr(methods, 'ASIUS_HARDWARE', True)
  for offroad in (False, None):
    if offroad is None:
      params.remove("IsOffroad")
    else:
      params.put_bool("IsOffroad", offroad, block=True)
    with pytest.raises(RuntimeError, match="parked"):
      methods.setLedState(manual, COLORS, 10)
    assert params.get(MANUAL_LED_PARAM) is None


def test_rpc_registration_and_generic_param_writes_cannot_bypass_guard(params, monkeypatch):
  monkeypatch.setattr(methods, 'Params', lambda: params)
  monkeypatch.setattr(methods, 'ASIUS_HARDWARE', True)
  assert methods.dispatcher["getLedState"]() == get_led_state(params, True)
  assert methods.dispatcher["setLedState"](True, COLORS, 30)["brightness"] == 30
  assert methods.saveParams({MANUAL_LED_PARAM: {}, "IsOffroad": False}) == {
    MANUAL_LED_PARAM: "error: blocked", "IsOffroad": "error: blocked",
  }
  assert params.get_bool("IsOffroad")
  assert get_led_state(params, True)["brightness"] == 30


def test_unsupported_device_and_invalid_manual_flag_are_rejected(params):
  assert get_led_state(params, False)["supported"] is False
  with pytest.raises(RuntimeError, match="Asius v0"):
    set_led_state(params, False, True, COLORS, 10)
  with pytest.raises(ValueError):
    set_led_state(params, True, 1, COLORS, 10)
  assert params.get(MANUAL_LED_PARAM) is None


@pytest.mark.parametrize("status", ["onroad", "unknown", "stale", "invalid", "manager_onroad"])
def test_runtime_revokes_manual_lighting_when_parked_status_is_lost(params, status):
  set_led_state(params, True, True, COLORS, 100)
  sm = parked_sm()
  if status == "onroad":
    sm['deviceState'].started = True
  elif status == "unknown":
    sm.seen['deviceState'] = False
  elif status == "stale":
    sm.alive['deviceState'] = False
  elif status == "invalid":
    sm.valid['deviceState'] = False
  else:
    params.put_bool("IsOffroad", False, block=True)
  assert manual_led_channels(sm, params) is None
  assert params.get(MANUAL_LED_PARAM) is None
  params.put_bool("IsOffroad", True, block=True)
  assert manual_led_channels(parked_sm(), params) is None


def test_reading_state_onroad_discards_override(params):
  set_led_state(params, True, True, COLORS, 10)
  params.put_bool("IsOffroad", False, block=True)
  state = get_led_state(params, True)
  assert state["parked"] is False
  assert state["manual"] is False
  assert params.get(MANUAL_LED_PARAM) is None


def test_transition_during_write_cannot_leave_manual_override(params, monkeypatch):
  original_put = params.put

  def write_then_start(*args, **kwargs):
    original_put(*args, **kwargs)
    params.put_bool("IsOffroad", False, block=True)

  monkeypatch.setattr(params, 'put', write_then_start)
  with pytest.raises(RuntimeError, match="parked"):
    set_led_state(params, True, True, COLORS, 10)
  assert params.get(MANUAL_LED_PARAM) is None


@pytest.mark.parametrize("flag", [ParamKeyFlag.CLEAR_ON_MANAGER_START, ParamKeyFlag.CLEAR_ON_ONROAD_TRANSITION])
def test_manager_clears_override(params, flag):
  set_led_state(params, True, True, COLORS, 10)
  params.clear_all(flag)
  assert params.get(MANUAL_LED_PARAM) is None


def test_automatic_restores_status_lighting(params):
  set_led_state(params, True, True, COLORS, 0)
  assert set_led_state(params, True, False, COLORS, 0)["manual"] is False
  assert manual_led_channels(parked_sm(), params) is None


@pytest.mark.parametrize("value", [{}, [], {"colors": COLORS, "brightness": -1}])
def test_invalid_stored_settings_are_cleared(params, value):
  params.put(MANUAL_LED_PARAM, value, block=True)
  assert manual_led_channels(parked_sm(), params) is None
  assert params.get(MANUAL_LED_PARAM) is None
