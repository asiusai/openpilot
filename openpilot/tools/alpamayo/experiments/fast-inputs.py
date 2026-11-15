"""Reuse immutable CPU adapters; preserve Harald's input math and image processing."""
import json
import importlib.util
import ctypes
from functools import partial
from collections import OrderedDict, deque
from pathlib import Path
from types import SimpleNamespace, FunctionType

import torch
from safetensors import safe_open
import inputs as reference
from camera_budgets import pixel_bounds
from alpamayo2_edge.sample import canonical_frame_order, validate_sample


_memcmp = ctypes.CDLL(None).memcmp
_memcmp.argtypes = (ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t)
_memcmp.restype = ctypes.c_int


def same_image(a, b):
  """Compare every CPU uint8 byte without a parallel tensor reduction."""
  if a.shape != b.shape or a.dtype != b.dtype:
    return False
  if a.device.type == b.device.type == 'cpu' and a.dtype == torch.uint8 and a.is_contiguous() and b.is_contiguous():
    return _memcmp(a.data_ptr(), b.data_ptr(), a.numel()) == 0
  return torch.equal(a, b)


class FastInputs:
  def __init__(self, checkpoint, pixel_scale=1.0, cache_resize=False, narrow_crop=1.0,
               wide_equidistant_focal=0.0, wide_target_fov=120.0, camera_profile='openpilot', camera_pixel_scales=None):
    if camera_profile not in ('openpilot', 'nvidia'):
      raise ValueError('Unsupported camera profile')
    self.camera_profile = camera_profile
    if not 0 < pixel_scale <= 1:
      raise ValueError('pixel_scale must be in (0, 1]')
    if not 0 < narrow_crop <= 1:
      raise ValueError('narrow_crop must be in (0, 1]')
    self.narrow_crop = narrow_crop
    self.wide_equidistant_focal = wide_equidistant_focal
    self.wide_target_fov = wide_target_fov
    self.warp_cache = OrderedDict()
    if wide_equidistant_focal:
      spec = importlib.util.spec_from_file_location('camera_warp', Path(__file__).with_name('camera-warp.py'))
      self.warp = importlib.util.module_from_spec(spec)
      spec.loader.exec_module(self.warp)
    if camera_pixel_scales is not None and camera_profile != 'nvidia':
      raise ValueError('Per-camera budgets require the NVIDIA profile')
    self.pixel_bounds = pixel_bounds(reference.CAMERA_PIXEL_BOUNDS, pixel_scale, camera_pixel_scales)
    self.checkpoint = Path(checkpoint).resolve()
    self.config = reference.Alpamayo2EdgeConfig(**json.loads((self.checkpoint / 'config.json').read_text()))
    self.config._name_or_path = str(self.checkpoint)
    self.tokenizer = reference.build_alpamayo2_edge_tokenizer(str(self.checkpoint))
    self.processor = reference.get_processor(self.tokenizer, self.config)
    self.history = reference.DeltaTrajectoryTokenizer(pad_origin_at_beginning=False)
    self.rope = SimpleNamespace(config=self.config.vlm_config)
    self.rope.get_vision_position_ids = partial(reference.Cosmos3EdgeModel.get_vision_position_ids, self.rope)
    name = 'vlm.model.visual.embeddings.position_embedding.weight'
    index = json.loads((self.checkpoint / 'model.safetensors.index.json').read_text())['weight_map']
    with safe_open(self.checkpoint / index[name], framework='pt') as weights:
      self.position_weight = weights.get_tensor(name).reshape(16, 16, -1).clone()
    self.position_cache = {}
    self.resize_cache = OrderedDict()
    self.resize_keys = deque()
    self.tokenize = reference._tokenize_messages
    if cache_resize:
      original = self.tokenize
      resize = original.__globals__['resize']
      def cached_resize(frame, size, **kwargs):
        key = (*self.resize_keys.popleft(), tuple(size), tuple(sorted(kwargs.items())))
        old = self.resize_cache.get(key)
        # Timestamps only select candidates. Compare every source byte before reuse.
        if old is not None and same_image(frame, old[0]):
          self.resize_cache.move_to_end(key)
          return old[1]
        result = resize(frame, size, **kwargs)
        self.resize_cache[key] = (frame.clone(), result)
        while len(self.resize_cache) > (144 if self.camera_profile == 'nvidia' else 48):
          self.resize_cache.popitem(last=False)
        return result
      self.tokenize = FunctionType(original.__code__, dict(original.__globals__, resize=cached_resize), original.__name__, original.__defaults__)
      self.tokenize.__kwdefaults__ = original.__kwdefaults__

  def prepare(self, data):
    if self.camera_profile == 'nvidia':
      validate_sample(data)
      order = canonical_frame_order(data['camera_indices'], data['absolute_timestamps'])
      if not torch.equal(order, torch.arange(len(order))):
        data = dict(data, **{key: data[key][order] for key in ('image_frames', 'camera_indices', 'absolute_timestamps')})
      data = dict(data, absolute_timestamps=data['absolute_timestamps'].reshape(-1))
    else:
      assert data['camera_indices'].tolist() == [1]*5 + [6]*5
      assert data['image_frames'].dtype == torch.uint8 and data['image_frames'].shape[:3] == (10, 1, 3)
    assert data['ego_history_xyz'].shape == (1, 20, 3) and data['ego_history_rot'].shape == (1, 20, 3, 3)
    canonical = dict(data, camera_pixel_bounds=self.pixel_bounds)
    self.resize_keys = deque(zip(data['camera_indices'].tolist(), data['absolute_timestamps'].tolist(), strict=True))
    messages = reference.build_conversation(canonical)
    cameras = iter(zip(data['camera_indices'].tolist(), data['absolute_timestamps'].tolist(), strict=True))
    for message in messages:
      for entry in message['content']:
        if entry['type'] != 'image':
          continue
        camera, timestamp = next(cameras)
        if camera == 1 and self.wide_equidistant_focal:
          frame = entry['image']
          old = self.warp_cache.get(timestamp)
          if old is not None and same_image(frame,old[0]):
            self.warp_cache.move_to_end(timestamp)
            entry['image'] = old[1]
          else:
            entry['image'] = self.warp.rectify(frame,self.wide_equidistant_focal,self.wide_target_fov)
            self.warp_cache[timestamp] = (frame.clone(),entry['image'])
            while len(self.warp_cache) > 24:
              self.warp_cache.popitem(last=False)
        if camera == 6 and self.narrow_crop < 1:
          frame = entry['image']
          h, w = frame.shape[-2:]
          ch, cw = round(h*self.narrow_crop), round(w*self.narrow_crop)
          entry['image'] = frame[..., (h-ch)//2:(h-ch)//2+ch, (w-cw)//2:(w-cw)//2+cw]
    tokens = self.tokenize(messages, self.processor, include_closed_thinking=True)
    prepared = {'tokenized_data': tokens, 'ego_history_xyz': data['ego_history_xyz'].unsqueeze(0),
                'ego_history_rot': data['ego_history_rot'].unsqueeze(0)}
    return self.config, self.tokenizer, prepared

  def native_inputs(self, prepared):
    tokens = prepared['tokenized_data']
    ids = tokens['input_ids'].clone()
    history = self.history.encode(None, None, prepared['ego_history_xyz'][:, 0], prepared['ego_history_rot'][:, 0])
    ids[ids == self.config.traj_ids['history_pad']] = (history + self.config.traj_ids['history_id0']).flatten()
    positions, delta = reference.Cosmos3EdgeModel.get_rope_index(
      self.rope, ids, tokens['mm_token_type_ids'], image_grid_thw=tokens['image_grid_thw'])
    key = tuple(map(tuple, tokens['image_grid_thw'].tolist()))
    if key not in self.position_cache:
      # Bound storage when callers change image sizes.
      if len(self.position_cache) >= 4:
        self.position_cache.pop(next(iter(self.position_cache)))
      self.position_cache[key] = reference.Cosmos3EdgeVisionEmbeddings.resize_positional_embeddings(
        self.position_weight, tokens['image_grid_thw'], 2).float().numpy()
    return ids.numpy(), positions.numpy(), int(delta.item()), self.position_cache[key]
