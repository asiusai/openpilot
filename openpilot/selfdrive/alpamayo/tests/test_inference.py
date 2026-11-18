import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np

from openpilot.selfdrive.alpamayo.feature_cache import FeatureCache
from openpilot.selfdrive.alpamayo.inference import checkpoint_identity


class InferenceTests(unittest.TestCase):
  @classmethod
  def setUpClass(cls):
    if os.environ.get('DEV') != 'CPU':
      raise RuntimeError('Run this suite with DEV=CPU; it must never select a GPU')

  def test_checkpoint_identity_hashes_actual_shards(self):
    with tempfile.TemporaryDirectory() as folder:
      root = Path(folder)
      (root/'config.json').write_text('{}')
      (root/'model.safetensors.index.json').write_text(json.dumps({'weight_map': {'layer': 'weights.safetensors'}}))
      (root/'weights.safetensors').write_bytes(b'original weights')
      np.save(root/'noise.npy', np.zeros((1, 64, 2), dtype=np.float32))
      before = checkpoint_identity(root, root/'noise.npy')
      (root/'weights.safetensors').write_bytes(b'changed weights')
      after = checkpoint_identity(root, root/'noise.npy')
      self.assertNotEqual(before['checkpoint_weights_sha256'], after['checkpoint_weights_sha256'])
      self.assertEqual(before['checkpoint_weight_index_sha256'], after['checkpoint_weight_index_sha256'])

  def test_ten_hz_full_history_cache_needs_two_new_frames(self):
    cache = FeatureCache(lambda p, v, g: p+v, np.copy)
    offsets = (-20, -10, -5, -1, 0)
    for query in range(20, 70):
      misses = cache.misses
      for camera in (1, 6):
        for offset in offsets:
          pixels = np.full((2, 4), (query+offset)*10+camera, dtype=np.float32)
          actual = cache(pixels, np.zeros_like(pixels), np.array([1, 1, 2]))
          np.testing.assert_array_equal(actual, pixels)
      if query >= 40:
        self.assertEqual(cache.misses-misses, 2)
      self.assertLessEqual(len(cache.cache), 96)

  def test_bf16_linear_norm_and_masked_attention(self):
    from tinygrad import Tensor, dtypes
    from openpilot.selfdrive.alpamayo.model import EdgeModel
    model = EdgeModel.__new__(EdgeModel)
    model.amd = False
    model.weights = {'linear.weight': Tensor([[1., 2.], [-1., 0.]], dtype=dtypes.bfloat16),
                     'linear.bias': Tensor([1., -1.], dtype=dtypes.bfloat16),
                     'norm.weight': Tensor([1., 1.], dtype=dtypes.bfloat16),
                     'norm.bias': Tensor([0., 0.], dtype=dtypes.bfloat16)}
    actual = model.linear(Tensor([[[2., 3.]]], dtype=dtypes.bfloat16), 'linear').float().numpy()
    np.testing.assert_array_equal(actual, [[[9., -3.]]])
    np.testing.assert_allclose(model.norm(Tensor([[1., 3.]], dtype=dtypes.bfloat16), 'norm').float().numpy(), [[-1., 1.]], atol=.01)
    q = Tensor(np.ones((1, 2, 2, 4), dtype=np.float32), dtype=dtypes.bfloat16)
    k = Tensor(np.ones((1, 1, 3, 4), dtype=np.float32), dtype=dtypes.bfloat16)
    v = Tensor(np.array([1., 3., 9999.], dtype=np.float32).reshape(1, 1, 3, 1).repeat(4, axis=3), dtype=dtypes.bfloat16)
    mask = Tensor([0., 0., -np.inf])
    np.testing.assert_allclose(model.attention(q, k, v, mask).float().numpy(), np.full((1, 2, 2, 4), 2.), atol=.01)

  def test_expert_uses_all_ten_euler_steps(self):
    from tinygrad import Tensor
    from openpilot.selfdrive.alpamayo.model import EdgeModel
    model = EdgeModel.__new__(EdgeModel)
    model.cache = [Tensor.zeros(2, 1, 8, 4, 128)]
    seen = []
    def expert(actions, timestep, cos, sin, cache, mask):
      seen.append(float(timestep.numpy()[0]))
      return actions
    model.expert_step = expert
    noise = np.random.default_rng(1).standard_normal((1, 64, 2)).astype(np.float32)
    result = model._trajectory(Tensor(noise), Tensor.ones(1, 1, 64, 128), Tensor.zeros(1, 1, 64, 128), 3, 10, 0)
    np.testing.assert_allclose(result.numpy(), noise*1.1**10, rtol=1e-6)
    np.testing.assert_allclose(seen, np.arange(10)/10, atol=1e-7)

  def test_serial_block_updates_only_its_six_kv_rows(self):
    from functools import partial
    from tinygrad import Tensor, TinyJit, dtypes
    from tinygrad.uop.ops import UOp
    from openpilot.selfdrive.alpamayo.selected.serial_coordinates import coords_kernel, store_kernel

    @TinyJit
    def write(cache, keys, values, pos, step):
      coords = Tensor.custom_kernel(Tensor.empty(2, dtype=dtypes.int32), Tensor(pos), Tensor(step), fxn=coords_kernel)[0].realize()
      for j in range(6):
        cache = Tensor.custom_kernel(cache, keys[j], values[j], coords, fxn=partial(store_kernel, delta=j))[0].realize()
      return cache, coords

    expected = np.zeros((2, 1, 8, 2304, 128), dtype=np.uint16)
    cache = Tensor(expected.copy()).bitcast(dtypes.bfloat16).realize()
    rng = np.random.default_rng(1)
    for start, offset in ((0, 0), (113, 12), (2298, 58)):
      bits = rng.integers(0, 65536, (2, 6, 1, 8, 1, 128), dtype=np.uint16)
      keys, values = (Tensor(a.copy()).bitcast(dtypes.bfloat16).realize() for a in bits)
      pos = UOp.variable('block_position', 0, 2298).bind(start)
      step = UOp.variable('block_step', 0, 58).bind(offset)
      updated, coords = write(cache, keys, values, pos, step)
      expected[:, :, :, start:start+6] = bits[:, :, :, :, 0, :].transpose(0, 2, 3, 1, 4)
      np.testing.assert_array_equal(coords.numpy(), [start, offset])
      np.testing.assert_array_equal(updated.bitcast(dtypes.uint16).numpy(), expected)

  def test_runtime_fixed_noise_outputs_and_reused_cpu_buffers(self):
    import torch
    from tinygrad import Tensor, TinyJit
    from openpilot.selfdrive.alpamayo.inference import Runtime
    runner = Runtime.__new__(Runtime)
    runner.noise = np.full((1, 64, 2), .001, dtype=np.float32)
    runner.input_signature = runner.inputs = runner.vision = None
    runner.upload = TinyJit(runner.upload_arrays)
    config = SimpleNamespace(traj_ids={'future_start': 2}, vlm_config=SimpleNamespace(image_token_id=999),
                             expert_config=SimpleNamespace(action_space_cfg={'_target_': 'fixture', 'n_waypoints': 64}))
    xyz = torch.zeros((1, 1, 20, 3))
    xyz[0, 0, :, 0] = torch.arange(-19, 1)*.1
    prepared = {'tokenized_data': {'pixel_values': torch.zeros(40, 3), 'image_grid_thw': torch.tensor([[1, 2, 2]]*10)},
                'ego_history_xyz': xyz, 'ego_history_rot': torch.eye(3).expand(1, 1, 20, 3, 3).contiguous()}
    ids = np.array([[0]+[999]*10+[0]], dtype=np.int64)
    packed = (ids, np.zeros((3, 1, 12), dtype=np.int64), 0, np.zeros((40, 3), dtype=np.float32))
    runner.adapter = SimpleNamespace(config=config, tokenizer=SimpleNamespace(decode=lambda ids: 'fixture reasoning'),
                                     prepare=lambda data: (prepared, packed))
    seen = []
    def trajectory(actions, cos, sin, length, steps, offset):
      self.assertEqual((steps, offset), (10, 2))
      seen.append(actions.float().numpy().copy())
      return actions
    runner.model = SimpleNamespace(synchronize=lambda: None, encode=lambda *args: Tensor.zeros(1, 12, 8),
                                   prefill=lambda *args: Tensor.zeros(1, 1, 8), trajectory=trajectory,
                                   greedy_decode=lambda logits, cos, sin, position, step: (logits, Tensor([float(step+1)])))
    results = [runner.predict({}) for _ in range(3)]
    for result, noise in zip(results, seen, strict=True):
      np.testing.assert_array_equal(noise, runner.noise)
      self.assertEqual(result['generated_token_ids'], [1, 2])
      self.assertEqual(result['reasoning'], 'fixture reasoning')
      self.assertEqual(np.shape(result['xyz']), (1, 64, 3))
      self.assertEqual(np.shape(result['rot']), (1, 64, 3, 3))
      np.testing.assert_array_equal(result['xyz'], results[0]['xyz'])


if __name__ == '__main__':
  unittest.main()
