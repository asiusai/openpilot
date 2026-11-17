"""Experimental two-evaluation integrators; require paired trajectory validation."""
import math
from types import MethodType
from tinygrad import Tensor, dtypes


def install(model, method):
  assert method in ('midpoint', 'heun')
  assert not model.trajectory_jits, 'Choose the solver before compiling trajectory graphs'

  def trajectory(self, actions, cos, sin, length, steps, offset):
    assert steps == 2, 'This experiment uses exactly two expert evaluations'
    cos, sin = (t[:, :, offset:offset+64] for t in (cos, sin))
    capacity = self.cache[0].shape[3]
    keys = Tensor.arange(capacity+64)
    mask = ((keys < length) | (keys >= capacity)).where(0, -math.inf)
    flat_cache = [t[i] for t in self.cache for i in range(2)]
    zero = Tensor.full((1,), 0., dtype=dtypes.float32)
    first = self.expert_step(actions, zero, cos, sin, flat_cache, mask)
    if method == 'midpoint':
      midpoint = (actions + first*.5).realize()
      second = self.expert_step(midpoint, Tensor.full((1,), .5, dtype=dtypes.float32), cos, sin, flat_cache, mask)
      result = (actions + second).realize()
    else:
      endpoint = (actions + first).realize()
      second = self.expert_step(endpoint, Tensor.full((1,), 1., dtype=dtypes.float32), cos, sin, flat_cache, mask)
      result = (actions + (first+second)*.5).realize()
    return (result.to('CPU') if self.cuda_kernels else result).realize()

  model._trajectory = MethodType(trajectory, model)
