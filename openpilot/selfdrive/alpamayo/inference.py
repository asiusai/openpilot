"""Persistent BF16, ten-step, greedy Alpamayo runtime for live input and replay."""
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import subprocess
import time

import numpy as np

TINYGRAD_REVISION = '89846dcfb383a98ee01cfec3bba440ae7dcf9b1e'
NVIDIA_REVISION = '02f55a7aeabc1a2b9f7a80445c08e40bddc89f5e'


def sha(path):
  with Path(path).open('rb') as stream:
    return hashlib.file_digest(stream, 'sha256').hexdigest()


def checkpoint_identity(checkpoint, noise):
  root = Path(checkpoint)
  index = root/'model.safetensors.index.json'
  shards = {name: sha(root/name) for name in sorted(set(json.loads(index.read_text())['weight_map'].values()))}
  return {'initial_noise_sha256': sha(noise), 'checkpoint_config_sha256': sha(root/'config.json'),
          'checkpoint_weight_index_sha256': sha(index), 'checkpoint_shards': shards,
          'checkpoint_weights_sha256': hashlib.sha256(json.dumps(shards, sort_keys=True).encode()).hexdigest()}


def verify_runtime():
  import alpamayo2_edge
  import tinygrad
  sources = ((Path(tinygrad.__file__).resolve().parents[1], TINYGRAD_REVISION),
             (Path(alpamayo2_edge.__file__).resolve().parents[2], NVIDIA_REVISION))
  for root, expected in sources:
    revision = subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip()
    if revision != expected:
      raise RuntimeError(f'Alpamayo requires isolated source revision {expected}')
    if subprocess.check_output(['git', '-C', str(root), 'status', '--porcelain', '--untracked-files=no'], text=True).strip():
      raise RuntimeError(f'Modified runtime source: {root.name}')


class Runtime:
  def __init__(self, checkpoint, initial_noise, *, cache_vision=True):
    verify_runtime()
    import torch
    from tinygrad import Device, TinyJit
    from tinygrad.device import Compiled
    from openpilot.selfdrive.alpamayo.feature_cache import FeatureCache
    from openpilot.selfdrive.alpamayo.inputs import InputAdapter
    from openpilot.selfdrive.alpamayo.model import EdgeModel
    self.noise = np.load(initial_noise, allow_pickle=False)
    if self.noise.shape != (1, 64, 2) or self.noise.dtype != np.float32 or not np.isfinite(self.noise).all():
      raise ValueError('Use fixed finite float32 diffusion noise with shape (1,64,2)')
    torch.set_num_threads(2)
    Compiled.wait_timeout_ms = 120000
    self.manifest = {'variant': 'default', 'precision': 'bf16', 'diffusion_steps': 10, 'solver': 'euler', 'greedy': True,
                     'max_tokens': 256, 'vision_batch': 1, 'cache_vision': cache_vision, 'tinygrad_revision': TINYGRAD_REVISION,
                     'nvidia_revision': NVIDIA_REVISION, 'identities': checkpoint_identity(checkpoint, initial_noise)}
    self.adapter = InputAdapter(checkpoint)
    self.model = EdgeModel(checkpoint)
    self.model.synchronize()
    self.device = Device.DEFAULT
    self.manifest.update(device=self.device, arch=getattr(Device[self.device], 'arch', None))
    self.encoders = {}
    self.vision = FeatureCache(self.encode_group, lambda x: x.clone().realize()) if cache_vision else None
    self.input_signature = self.inputs = None
    self.upload = TinyJit(self.upload_arrays)

  @staticmethod
  def upload_arrays(*arrays):
    from tinygrad import Device, dtypes
    types = [dtypes.bfloat16]*len(arrays)
    types[2], types[-1] = dtypes.int32, dtypes.float32
    return tuple(x.to(Device.DEFAULT).cast(dtype).contiguous().realize() for x, dtype in zip(arrays, types, strict=True))

  def encode_group(self, pixels, positions, grid):
    from tinygrad import Device, Tensor, TinyJit, dtypes
    count = len(grid)
    p, v = (Tensor(x, device='CPU').to(Device.DEFAULT).cast(dtypes.bfloat16).reshape(count, -1, x.shape[-1]).contiguous().realize()
            for x in (pixels, positions))
    key = (p.shape, v.shape)
    if key not in self.encoders:
      if len(self.encoders) >= 4:
        raise RuntimeError('Unexpected vision layouts exceeded the streaming bound')
      self.encoders[key] = TinyJit(self.model._vision)
    return self.encoders[key](p, v)

  def predict(self, data):
    """Consume the six canonical input fields and return all decoded model outputs."""
    import torch
    from tinygrad import Tensor
    from alpamayo2_edge.action_space import UnicycleAccelCurvatureCourseActionSpace
    from openpilot.selfdrive.alpamayo.inputs import rotary
    start, times, generated = time.monotonic_ns(), {}, []

    @contextmanager
    def timed(name):
      self.model.synchronize()
      begin = time.monotonic_ns()
      yield
      self.model.synchronize()
      times[name] = (time.monotonic_ns()-begin)/1e6

    with timed('preprocess'):
      prepared, packed = self.adapter.prepare(data)
      ids, positions, delta, vpos = packed
      cpu_pixels = prepared['tokenized_data']['pixel_values'].numpy()
      arrays = [cpu_pixels, vpos, ids.astype(np.int32), *rotary(positions)]
      for length in (256, 320):
        pos = np.arange(length)+ids.shape[1]+delta
        arrays.extend(rotary(np.broadcast_to(pos, (3, 1, length))))
      arrays.append(self.noise)
      signature = tuple((a.shape, a.dtype.str) for a in arrays)
      if self.input_signature != signature:
        self.inputs = [Tensor(x, device='CPU').contiguous().realize() for x in arrays]
        self.input_signature = signature
        self.upload.reset()
      else:
        for tensor, array in zip(self.inputs, arrays, strict=True):
          tensor._buffer().as_memoryview(allow_zero_copy=True)[:] = np.ascontiguousarray(array).tobytes()
      grids = prepared['tokenized_data']['image_grid_thw'].numpy()
      offsets = np.r_[0, np.cumsum(grids.prod(axis=1))]
      batches, i = [], 0
      while i < len(grids):
        end = i+1
        batches.append((int(offsets[i]), int(offsets[end]), end-i, i, end))
        i = end
      image_indices = np.flatnonzero(ids[0] == self.adapter.config.vlm_config.image_token_id)
      spans = tuple((int(s[0]), int(s[-1])+1) for s in np.split(image_indices, np.flatnonzero(np.diff(image_indices) != 1)+1))
    with timed('input_upload'):
      pixels, gpu_vpos, gpu_ids, cos, sin, decode_cos, decode_sin, action_cos, action_sin, actions = self.upload(*self.inputs)
    with timed('vision'):
      if self.vision is None:
        embedding = self.model.encode(pixels, gpu_vpos, gpu_ids, tuple(b[:3] for b in batches), spans)
      else:
        features = [self.vision(cpu_pixels[a:b], vpos[a:b], grids[i:j]) for a, b, _, i, j in batches]
        embedding = self.model.embed(gpu_ids, Tensor.cat(*features, dim=0), spans)
    with timed('prefill'):
      logits = self.model.prefill(embedding, cos, sin, 256)
    with timed('reasoning'):
      for step in range(256):
        logits, token_tensor = self.model.greedy_decode(logits, decode_cos, decode_sin, ids.shape[1]+step, step)
        token = int(token_tensor.float().numpy().item())
        generated.append(token)
        if token == self.adapter.config.traj_ids['future_start']:
          break
      else:
        raise RuntimeError('No trajectory anchor after 256 reasoning tokens')
    with timed('diffusion'):
      actions = self.model.trajectory(actions, action_cos, action_sin, ids.shape[1]+len(generated), 10, len(generated))
      action_array = actions.float().numpy()
    with timed('trajectory_decode'):
      action_config = dict(self.adapter.config.expert_config.action_space_cfg)
      action_config.pop('_target_')
      action_space = UnicycleAccelCurvatureCourseActionSpace(**action_config)
      with torch.no_grad():
        xyz, rot = action_space.action_to_traj(torch.from_numpy(action_array.copy()),
                                             prepared['ego_history_xyz'][:, 0], prepared['ego_history_rot'][:, 0])
      if xyz.shape != (1, 64, 3) or rot.shape != (1, 64, 3, 3) or not torch.isfinite(xyz).all() or not torch.isfinite(rot).all():
        raise RuntimeError('Invalid decoded trajectory')
    return {'xyz': xyz.numpy().tolist(), 'rot': rot.numpy().tolist(), 'reasoning': self.adapter.tokenizer.decode(generated),
            'generated_token_ids': generated, 'generated_tokens': len(generated), 'timings_ms': times,
            'runtime_ms': (time.monotonic_ns()-start)/1e6, 'prompt_tokens': int(ids.shape[1])}

  def synchronize(self):
    self.model.synchronize()

  def close(self):
    self.synchronize()


class ShadowAdapter:
  def __init__(self, config):
    self.runtime = Runtime(**config)
    self.manifest = self.runtime.manifest

  def predict(self, arrays, metadata):
    return {'query_ns': metadata['query_ns'], 'prediction': self.runtime.predict(arrays)}

  def synchronize(self):
    self.runtime.synchronize()

  def close(self):
    self.runtime.close()
