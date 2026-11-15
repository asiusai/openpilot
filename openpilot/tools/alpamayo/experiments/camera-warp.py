"""Experimental equidistant-to-pinhole projection; nominal intrinsics, not calibration."""
import math
from functools import lru_cache
import torch
import torch.nn.functional as F


@lru_cache(maxsize=4)
def projection_grid(height, width, focal, horizontal_fov):
  if height < 2 or width < 2 or focal <= 0 or not 0 < horizontal_fov < 180:
    raise ValueError('Invalid projection geometry')
  target_focal = (width-1)/(2*math.tan(math.radians(horizontal_fov/2)))
  yy, xx = torch.meshgrid(torch.arange(height), torch.arange(width), indexing='ij')
  x, y = (xx-(width-1)/2)/target_focal, (yy-(height-1)/2)/target_focal
  radius = torch.sqrt(x.square()+y.square())
  factor = torch.atan(radius)/radius.clamp_min(1e-12)
  factor = torch.where(radius == 0, torch.ones_like(factor), factor)
  source_x, source_y = focal*x*factor, focal*y*factor
  grid = torch.stack((2*source_x/(width-1), 2*source_y/(height-1)), dim=-1)
  if not torch.isfinite(grid).all() or grid.abs().max() > 1:
    raise ValueError('Requested projection extends outside the source image')
  return grid.unsqueeze(0)


def rectify(frame, focal, horizontal_fov=120):
  if frame.ndim != 3 or frame.shape[0] != 3 or frame.dtype != torch.uint8 or frame.device.type != 'cpu':
    raise ValueError('Expected a CPU RGB uint8 image')
  grid = projection_grid(*frame.shape[-2:], focal, horizontal_fov)
  image = F.grid_sample(frame.unsqueeze(0).float(), grid, mode='bilinear', align_corners=True)
  return image[0].round().clamp(0,255).to(torch.uint8)
