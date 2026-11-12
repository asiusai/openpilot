"""Asius v0 LEDs, numbered left to right when facing the light windows."""
import math

ORANGE_RGB = (255, 32, 0)
LED_BRIGHTNESS = (1.,) * 8
STARTUP_PERIOD = 3.
STARTUP_RISE = 1.2
BOOT_BRIGHTNESS = 6  # Default before camera exposure is available.
MIN_AUTO_BRIGHTNESS = 6  # Approximately 2.4%.
MAX_AUTO_BRIGHTNESS = 127  # 50%, rounded down to stay within the cap.


def camera_channels(colors: list[list[int]]) -> dict[int, list[int]]:
  # Logical LEDs 1..8 run opposite to the camera board/package numbering.
  scaled = [[round(channel * scale) for channel in color] for color, scale in zip(colors, LED_BRIGHTNESS, strict=True)]
  channels = [channel for color in reversed(scaled) for channel in color]
  return {1: [0] * 12, 2: channels[:12], 3: channels[12:]}


def startup_levels(elapsed: float) -> list[float]:
  """Breathe in for 1.2s and out for 1.8s, retaining a faint blue glow."""
  phase = elapsed % STARTUP_PERIOD
  if phase < STARTUP_RISE:
    level = (1. - math.cos(math.pi * phase / STARTUP_RISE)) / 2.
  else:
    level = (1. + math.cos(math.pi * (phase - STARTUP_RISE) / (STARTUP_PERIOD - STARTUP_RISE))) / 2.
  return [(1. + 5. * level) / 6.] * 8


def startup_channels(elapsed: float, brightness: int = BOOT_BRIGHTNESS) -> dict[int, list[int]]:
  brightness = max(0, min(MAX_AUTO_BRIGHTNESS, brightness))
  return camera_channels([[0, 0, math.floor(level * brightness + .5 + 1e-9)] for level in startup_levels(elapsed)])


def calibration_channels(percent: float, brightness: int) -> dict[int, list[int]]:
  brightness = max(0, min(MAX_AUTO_BRIGHTNESS, brightness))
  percent = max(0., min(100., percent)) if math.isfinite(percent) else 0.
  colors = [[0, 0, 0] for _ in range(8)]
  for step, index in enumerate(range(8)):
    level = max(0., min(1., percent / 12.5 - step))
    # The first orange light identifies calibration even before it progresses.
    if index == 0:
      level = 1.
    colors[index] = [round(channel * level * brightness / 255.) for channel in ORANGE_RGB]
  return camera_channels(colors)
