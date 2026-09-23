"""Parked-only manual control of the six road-facing RGB LEDs."""

import re

MANUAL_LED_PARAM = "ManualLedState"
LED_COUNT = 6


def validate_led_settings(colors, brightness) -> dict:
  if not isinstance(colors, list) or len(colors) != LED_COUNT:
    raise ValueError("Choose a color for each of the six LEDs")
  if any(not isinstance(color, str) or re.fullmatch(r"#[0-9a-fA-F]{6}", color) is None for color in colors):
    raise ValueError("LED colors must use #RRGGBB")
  if type(brightness) is not int or not 0 <= brightness <= 100:
    raise ValueError("LED brightness must be an integer from 0 to 100")
  return {"colors": [color.lower() for color in colors], "brightness": brightness}


def read_manual_leds(params) -> dict | None:
  value = params.get(MANUAL_LED_PARAM)
  if value is None:
    return None
  try:
    return validate_led_settings(value["colors"], value["brightness"])
  except (KeyError, TypeError, ValueError):
    params.remove(MANUAL_LED_PARAM)
    return None


def get_led_state(params, supported: bool) -> dict:
  parked = params.get_bool("IsOffroad")
  settings = read_manual_leds(params) if supported else None
  if settings is not None and not parked:
    params.remove(MANUAL_LED_PARAM)
    settings = None
  return {
    "supported": supported,
    "parked": parked,
    "manual": settings is not None,
    **(settings or {"colors": ["#0000ff"] * LED_COUNT, "brightness": 10}),
  }


def set_led_state(params, supported: bool, manual: bool, colors, brightness) -> dict:
  if not supported:
    raise RuntimeError("Manual LED control is only available on Asius v0")
  if type(manual) is not bool:
    raise ValueError("Manual control must be true or false")
  settings = validate_led_settings(colors, brightness)
  if not params.get_bool("IsOffroad"):
    raise RuntimeError("LEDs can only be changed while parked")
  if manual:
    params.put(MANUAL_LED_PARAM, settings, block=True)
    # The manager can start driving while the write is in progress.
    if not params.get_bool("IsOffroad"):
      params.remove(MANUAL_LED_PARAM)
      raise RuntimeError("LEDs can only be changed while parked")
  else:
    params.remove(MANUAL_LED_PARAM)
  return get_led_state(params, supported)


def manual_led_channels(sm, params) -> dict[int, list[int]] | None:
  settings = read_manual_leds(params)
  if settings is None:
    return None
  # Require fresh device status as well as the manager's offroad flag.
  parked = (sm.seen['deviceState'] and sm.alive['deviceState'] and sm.valid['deviceState'] and
            not sm['deviceState'].started and params.get_bool("IsOffroad"))
  if not parked:
    params.remove(MANUAL_LED_PARAM)
    return None
  channels = [round(int(color[offset:offset + 2], 16) * settings["brightness"] / 100)
              for color in settings["colors"] for offset in (1, 3, 5)]
  return {1: [0] * 9, 2: channels[:9], 3: channels[9:]}
