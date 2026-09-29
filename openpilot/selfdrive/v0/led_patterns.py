"""Asius v0 LEDs, numbered left to right when facing the light windows."""
import math

WHITE_RGB = (255, 195, 120)
ORANGE_RGB = (255, 32, 0)
LED_BRIGHTNESS = (1., 0.1, 1., 1., 0.1, 1.)
STARTUP_PERIOD = 3.
BOOT_BRIGHTNESS = 25  # At most 10% before camera exposure is available.
MIN_AUTO_BRIGHTNESS = 26  # 10%, rounded up to the next 8-bit PWM level.
MAX_AUTO_BRIGHTNESS = 127  # 50%, rounded down to stay within the cap.


def camera_channels(colors: list[list[int]]) -> dict[int, list[int]]:
  # Logical LEDs 1..6 run opposite to the camera board/package numbering.
  scaled = [[round(channel * scale) for channel in color] for color, scale in zip(colors, LED_BRIGHTNESS, strict=True)]
  channels = [channel for color in reversed(scaled) for channel in color]
  return {1: [0] * 9, 2: channels[:9], 3: channels[9:]}


def startup_levels(elapsed: float) -> list[float]:
  """Fade all six LEDs in and out together over three seconds."""
  phase = (elapsed % STARTUP_PERIOD) / STARTUP_PERIOD
  return [(1. - math.cos(math.tau * phase)) / 2.] * 6


def startup_channels(elapsed: float, brightness: int = BOOT_BRIGHTNESS) -> dict[int, list[int]]:
  brightness = max(0, min(MAX_AUTO_BRIGHTNESS, brightness))
  return camera_channels([[round(channel * level * brightness / 255.) for channel in WHITE_RGB]
                          for level in startup_levels(elapsed)])


def calibration_channels(percent: float, brightness: int) -> dict[int, list[int]]:
  brightness = max(0, min(MAX_AUTO_BRIGHTNESS, brightness))
  percent = max(0., min(100., percent)) if math.isfinite(percent) else 0.
  colors = [[0, 0, 0] for _ in range(6)]
  # Center LEDs fill halfway between their neighbors, at 10% output.
  for index, step in enumerate((0., 0.5, 1., 2., 2.5, 3.)):
    level = max(0., min(1., percent / 25. - step))
    # The first orange light identifies calibration even before it progresses.
    if index == 0:
      level = 1.
    colors[index] = [round(channel * level * brightness / 255.) for channel in ORANGE_RGB]
  return camera_channels(colors)
