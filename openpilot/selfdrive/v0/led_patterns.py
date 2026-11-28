"""Asius v0 LEDs, numbered left to right when facing the light windows."""
import math

ORANGE_RGB = (255, 32, 0)
LED_BRIGHTNESS = (1.,) * 8
STARTUP_PERIOD = 4.4
BOOT_BRIGHTNESS = 5  # Default before camera exposure is available.
MIN_AUTO_BRIGHTNESS = 5  # Approximately 2%.
MAX_AUTO_BRIGHTNESS = 127  # 50%, rounded down to stay within the cap.


def camera_channels(colors: list[list[float]]) -> dict[int, list[float]]:
  # Logical LEDs 1..8 run opposite to the camera board/package numbering.
  scaled = [[channel * scale for channel in color] for color, scale in zip(colors, LED_BRIGHTNESS, strict=True)]
  channels = [channel for color in reversed(scaled) for channel in color]
  return {1: [0] * 12, 2: channels[:12], 3: channels[12:]}


def startup_levels(elapsed: float) -> list[float]:
  """Wide blue comet with smooth turns and a trailing glow across both windows."""
  # Keep the curve aligned with VamOS tools/build/generate_boot_led_pattern.py.
  phase = elapsed % STARTUP_PERIOD
  positions = [3.5 - 3.5 * math.cos(2. * math.pi * (phase - delay) / STARTUP_PERIOD)
               for delay in (0., .14, .28)]
  levels = [sum(weight * math.exp(-.5 * ((index - position) / .82) ** 2)
                for weight, position in zip((1., .35, .12), positions, strict=True)) for index in range(8)]
  scale = max(1., max(levels))
  # Match the 16-bit driver, including truly dark outputs beyond the tail.
  peak = BOOT_BRIGHTNESS * 257
  return [round(level / scale * peak) / peak for level in levels]


def startup_channels(elapsed: float, brightness: int = BOOT_BRIGHTNESS) -> dict[int, list[float]]:
  brightness = max(0, min(MAX_AUTO_BRIGHTNESS, brightness))
  return camera_channels([[0, 0, level * brightness] for level in startup_levels(elapsed)])


def calibration_channels(percent: float, brightness: int) -> dict[int, list[float]]:
  brightness = max(0, min(MAX_AUTO_BRIGHTNESS, brightness))
  percent = max(0., min(100., percent)) if math.isfinite(percent) else 0.
  colors: list[list[float]] = [[0, 0, 0] for _ in range(8)]
  for step, index in enumerate(range(8)):
    level = max(0., min(1., percent / 12.5 - step))
    # The first orange light identifies calibration even before it progresses.
    if index == 0:
      level = 1.
    # Preserve the orange ratio while an LED fills; integer rounding erased green.
    colors[index] = [channel * level * brightness / 255. for channel in ORANGE_RGB]
  return camera_channels(colors)
