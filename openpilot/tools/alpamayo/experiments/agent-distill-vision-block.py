"""Differentiable native FP32 layer12. Teacher target is the output of layer15."""
from tinygrad import Tensor, dtypes

WIDTH, MLP, HEADS, HEAD_DIM, PATCHES = 1152, 4304, 16, 72, 720
PREFIX = 'vlm.model.visual.encoder.layers.12.'


def shapes():
  result = {}
  for name in ('layer_norm1', 'layer_norm2'):
    result.update({name+'.weight': (WIDTH,), name+'.bias': (WIDTH,)})
  for name, outgoing, incoming in [*(('self_attn.'+n+'_proj', WIDTH, WIDTH) for n in ('q', 'k', 'v', 'out')),
                                   ('mlp.fc1', MLP, WIDTH), ('mlp.fc2', WIDTH, MLP)]:
    result.update({name+'.weight': (outgoing, incoming), name+'.bias': (outgoing,)})
  return dict(sorted(result.items()))


class Block:
  def __init__(self, arrays, device='CPU'):
    import numpy as np
    expected = shapes()
    if set(arrays) != set(expected): raise ValueError('Unexpected vision parameter names')
    for name, value in arrays.items():
      if value.shape != expected[name] or value.dtype != np.float32 or not np.isfinite(value).all():
        raise ValueError('Invalid vision parameter '+name)
    self.weights = {name: Tensor(arrays[name].copy(), device=device).is_param_(True).realize() for name in expected}

  def __call__(self, x):
    if x.ndim != 3 or x.shape[-1] != WIDTH or x.dtype != dtypes.float32: raise ValueError('Expected FP32 B,T,1152')
    def linear(value, name):
      shape = value.shape[:-1]
      return value.reshape(-1, value.shape[-1]).linear(self.weights[name+'.weight'].T,
                                                    self.weights[name+'.bias']).reshape(*shape, -1)
    def norm(value, name):
      return value.float().layernorm(eps=1e-6) * self.weights[name+'.weight'] + self.weights[name+'.bias']
    h = norm(x, 'layer_norm1')
    q, k, v = (linear(h, 'self_attn.'+n+'_proj').reshape(*h.shape[:2], HEADS, HEAD_DIM).transpose(1, 2)
               for n in ('q', 'k', 'v'))
    scores = (q.matmul(k.transpose(-2, -1)) * (HEAD_DIM ** -0.5)).float()
    h = scores.softmax(-1).matmul(v).transpose(1, 2).reshape(x.shape)
    x = x + linear(h, 'self_attn.out_proj')
    h = linear(norm(x, 'layer_norm2'), 'mlp.fc1').float().gelu()
    # Keep the expression lazy until gradients exist. This pinned tinygrad bufferizes realized outputs.
    return x + linear(h, 'mlp.fc2')


def mse(output, target):
  if output.shape != target.shape: raise ValueError('MSE shape mismatch')
  return (output - target).square().mean()
