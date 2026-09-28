"""Asius v0 LEDs, numbered left to right when facing the light windows."""
import math

WHITE_RGB = (255, 195, 120)
ORANGE_RGB = (255, 32, 0)
LED_POSITIONS_MM = (0., 7.5, 15., 45., 52.5, 60.)
LED_BRIGHTNESS = (1., 0.1, 1., 1., 0.1, 1.)
STARTUP_PERIOD = 3.


def camera_channels(colors: list[list[int]]) -> dict[int, list[int]]:
  # Logical LEDs 1..6 run opposite to the camera board/package numbering.
  scaled = [[round(channel * scale) for channel in color] for color, scale in zip(colors, LED_BRIGHTNESS, strict=True)]
  channels = [channel for color in reversed(scaled) for channel in color]
  return {1: [0] * 9, 2: channels[:9], 3: channels[9:]}


def smooth(value: float) -> float:
  value = max(0., min(1., value))
  return value * value * (3. - 2. * value)


def startup_levels(elapsed: float) -> list[float]:
  """Three-second Knight Rider sweep across the physical LED positions."""
  def position(t: float) -> float:
    return 60. - abs((t * 40.) % 120. - 60.)

  def glow(x: float, t: float) -> float:
    head = position(t)
    width = 5. + 7. * smooth(min(head, 60. - head) / 15.)
    return math.exp(-0.5 * ((x - head) / width) ** 2)

  head = position(elapsed)
  tail_fade = smooth(min(head, 60. - head) / 15.)
  levels = [0.] * 6
  for index, x in enumerate(LED_POSITIONS_MM):
    tail = max(math.exp(-age / 0.45) * glow(x, elapsed - age) for age in (0.1, 0.2, 0.3, 0.4, 0.6, 0.8))
    levels[index] = max(glow(x, elapsed), tail_fade * tail)
  return levels


def startup_channels(elapsed: float, brightness: int = 255) -> dict[int, list[int]]:
  return camera_channels([[round(channel * level * brightness / 255.) for channel in WHITE_RGB]
                          for level in startup_levels(elapsed)])


def calibration_channels(percent: float, brightness: int) -> dict[int, list[int]]:
  percent = max(0., min(100., percent)) if math.isfinite(percent) else 0.
  colors = [[0, 0, 0] for _ in range(6)]
  # Center LEDs fill halfway between their neighbors, at 10% output.
  for index, step in enumerate((0., 0.5, 1., 2., 2.5, 3.)):
    level = max(0., min(1., percent / 25. - step))
    colors[index] = [round(channel * level * brightness / 255.) for channel in ORANGE_RGB]
  return camera_channels(colors)
