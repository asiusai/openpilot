"""Explicit experimental camera budgets shared by both inference adapters."""
CAMERAS = (0, 1, 2, 3, 5, 6)


def pixel_bounds(defaults, scale=1.0, camera_scales=None):
  if not 0 < scale <= 1:
    raise ValueError('pixel_scale must be in (0, 1]')
  scales = (1.0,)*len(CAMERAS) if camera_scales is None else tuple(camera_scales)
  if len(scales) != len(CAMERAS) or any(not 0 < value <= 1 for value in scales):
    raise ValueError('Six camera scales must each be in (0, 1]')
  factors = dict(zip(CAMERAS, scales, strict=True))
  return {camera: tuple(round(value*scale*factors.get(camera, 1.0)) for value in bounds)
          for camera, bounds in defaults.items()}
