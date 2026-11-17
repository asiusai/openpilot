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
                wide_equidistant_focal=0.0, wide_target_fov=120.0, camera_profile='openpilot', camera_pixel_scales=None,
                  allow_missing_cameras=False, history_pixel_scale=1.0, history_feature_scale=1.0, direct_byte_inputs=False,
                  front_history_indices=None):
    if camera_profile not in ('openpilot', 'nvidia'):
      raise ValueError('Unsupported camera profile')
    self.camera_profile = camera_profile
    self.allow_missing_cameras = allow_missing_cameras
    if allow_missing_cameras and camera_profile != 'nvidia':
      raise ValueError('Missing-camera replay uses the NVIDIA camera IDs and budgets')
    if not 0 < pixel_scale <= 1:
      raise ValueError('pixel_scale must be in (0, 1]')
    if not 0 < history_pixel_scale <= 1:
      raise ValueError('history_pixel_scale must be in (0, 1]')
    self.history_pixel_scale = history_pixel_scale
    if not 0 < history_feature_scale <= 1 or (history_feature_scale < 1 and history_pixel_scale < 1):
      raise ValueError('Feature compression requires unchanged image pixel budgets and a scale in (0, 1]')
    self.history_feature_scale = history_feature_scale
    self.front_history_indices = None if front_history_indices is None else tuple(front_history_indices)
    if self.front_history_indices is not None and (not self.front_history_indices or
        tuple(sorted(set(self.front_history_indices))) != self.front_history_indices or
        self.front_history_indices[0] < 0 or self.front_history_indices[-1] != 4):
      raise ValueError('Front history indices must be a sorted subset of 0..4 including the current frame')
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
    if direct_byte_inputs:
      image = self.processor.image_processor
      if (list(image.image_mean) != [.5]*3 or list(image.image_std) != [.5]*3 or
          image.rescale_factor != 1/255 or image.temporal_patch_size != 1):
        raise ValueError('Direct byte inputs require the exact checkpoint pixel palette')
      def retain_bytes(patches, image_processor):
        if patches.dtype != torch.uint8 or patches.ndim != 2 or patches.shape[1] != 3*image_processor.patch_size**2:
          raise ValueError('Expected unnormalized uint8 image patches')
        return patches
      original = self.tokenize
      self.tokenize = FunctionType(original.__code__, dict(original.__globals__, _normalize_patches=retain_bytes), original.__name__, original.__defaults__)
      self.tokenize.__kwdefaults__ = original.__kwdefaults__

  def prepare(self, data):
    if self.camera_profile == 'nvidia':
      if self.allow_missing_cameras:
        validate_available_sample(data)
      else:
        validate_sample(data)
      order = canonical_frame_order(data['camera_indices'], data['absolute_timestamps'])
      if not torch.equal(order, torch.arange(len(order))):
        data = dict(data, **{key: data[key][order] for key in ('image_frames', 'camera_indices', 'absolute_timestamps')})
      data = dict(data, absolute_timestamps=data['absolute_timestamps'].reshape(-1))
    else:
      assert data['camera_indices'].tolist() == [1]*5 + [6]*5
      assert data['image_frames'].dtype == torch.uint8 and data['image_frames'].shape[:3] == (10, 1, 3)
    assert data['ego_history_xyz'].shape == (1, 20, 3) and data['ego_history_rot'].shape == (1, 20, 3, 3)
    if self.front_history_indices is not None:
      if data['camera_indices'].tolist() != [1]*5+[6]*5:
        raise ValueError('Temporal thinning currently requires exactly the two complete front-camera histories')
      selected = torch.tensor([i+base for base in (0,5) for i in self.front_history_indices])
      data = dict(data, **{key:data[key][selected] for key in ('image_frames','camera_indices','absolute_timestamps')})
    canonical = dict(data, camera_pixel_bounds=self.pixel_bounds)
    self.resize_keys = deque(zip(data['camera_indices'].tolist(), data['absolute_timestamps'].tolist(), strict=True))
    messages = reference.build_conversation(canonical)
    latest = {}
    for camera, timestamp in zip(data['camera_indices'].tolist(), data['absolute_timestamps'].tolist(), strict=True):
      latest[camera] = max(latest.get(camera, timestamp), timestamp)
    cameras = iter(zip(data['camera_indices'].tolist(), data['absolute_timestamps'].tolist(), strict=True))
    for message in messages:
      for entry in message['content']:
        if entry['type'] != 'image':
          continue
        camera, timestamp = next(cameras)
        if timestamp < latest[camera] and self.history_pixel_scale < 1:
          for key in ('min_pixels', 'max_pixels'):
            entry[key] = max(1, round(entry[key]*self.history_pixel_scale))
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
    if self.history_feature_scale < 1:
      # Encode every image at the original resolution. Compress only historical
      # feature grids after vision encoding; retain actual timestamps and text.
      grids = tokens['image_grid_thw'].clone()
      for i, (camera, timestamp) in enumerate(zip(data['camera_indices'].tolist(), data['absolute_timestamps'].tolist(), strict=True)):
        if timestamp < latest[camera]:
          grids[i, 1:] = torch.clamp((grids[i, 1:]*self.history_feature_scale**.5/2).round().to(grids.dtype)*2, min=2)
      ids = tokens['input_ids'][0]
      indices = torch.nonzero(ids == self.config.vlm_config.image_token_id).flatten()
      boundaries = torch.nonzero(indices[1:]-indices[:-1] != 1).flatten()+1
      spans = torch.tensor_split(indices, boundaries.tolist())
      if len(spans) != len(grids):
        raise ValueError('Image spans do not match feature grids')
      for key in ('input_ids', 'attention_mask', 'mm_token_type_ids'):
        chunks, cursor = [], 0
        for span, grid in zip(spans, grids, strict=True):
          begin, end = int(span[0]), int(span[-1])+1
          count = int(grid.prod())//4
          chunks.extend((tokens[key][:, cursor:begin], tokens[key][:, begin:begin+1].expand(-1, count)))
          cursor = end
        chunks.append(tokens[key][:, cursor:])
        tokens[key] = torch.cat(chunks, dim=1)
      tokens['embedding_grid_thw'] = grids
    prepared = {'tokenized_data': tokens, 'ego_history_xyz': data['ego_history_xyz'].unsqueeze(0),
                'ego_history_rot': data['ego_history_rot'].unsqueeze(0)}
    return self.config, self.tokenizer, prepared

  def native_inputs(self, prepared):
    tokens = prepared['tokenized_data']
    ids = tokens['input_ids'].clone()
    history = self.history.encode(None, None, prepared['ego_history_xyz'][:, 0], prepared['ego_history_rot'][:, 0])
    ids[ids == self.config.traj_ids['history_pad']] = (history + self.config.traj_ids['history_id0']).flatten()
    positions, delta = reference.Cosmos3EdgeModel.get_rope_index(
      self.rope, ids, tokens['mm_token_type_ids'], image_grid_thw=tokens.get('embedding_grid_thw', tokens['image_grid_thw']))
    key = tuple(map(tuple, tokens['image_grid_thw'].tolist()))
    if key not in self.position_cache:
      # Bound storage when callers change image sizes.
      if len(self.position_cache) >= 4:
        self.position_cache.pop(next(iter(self.position_cache)))
      self.position_cache[key] = reference.Cosmos3EdgeVisionEmbeddings.resize_positional_embeddings(
        self.position_weight, tokens['image_grid_thw'], 2).float().numpy()
    return ids.numpy(), positions.numpy(), int(delta.item()), self.position_cache[key]


def validate_available_sample(data):
  """Omit whole unavailable views, retaining the normal history for present views."""
  cameras, times, images = (data[key] for key in ('camera_indices', 'absolute_timestamps', 'image_frames'))
  if cameras.dtype != torch.int64 or cameras.ndim != 1 or cameras.device.type != 'cpu':
    raise ValueError('Camera IDs must be a CPU int64 vector')
  ids, counts = cameras.unique(return_counts=True)
  available = dict(zip(ids.tolist(), counts.tolist(), strict=True))
  if not {1, 6}.issubset(available) or any(reference.CAMERA_FRAME_COUNTS.get(c) != n for c, n in available.items()):
    raise ValueError('Both front views and complete histories for each included view are required')
  n = cameras.numel()
  if (images.dtype != torch.uint8 or images.device.type != 'cpu' or images.ndim != 5
      or images.shape[:3] != (n, 1, 3) or not images.is_contiguous() or min(images.shape[-2:]) <= 0):
    raise ValueError('Images must be contiguous CPU uint8 [N,1,3,H,W]')
  if times.dtype != torch.int64 or times.shape not in ((n,), (n, 1)) or times.device.type != 'cpu':
    raise ValueError('Image timestamps must be CPU int64 microseconds')
  for key, shape in (('ego_history_xyz', (1, 20, 3)), ('ego_history_rot', (1, 20, 3, 3))):
    value = data[key]
    if value.shape != shape or not value.is_floating_point() or not torch.isfinite(value).all():
      raise ValueError('Invalid measured ego history')
  nav = data.get('nav_text')
  if not isinstance(nav, (list, tuple)) or len(nav) != 1 or not isinstance(nav[0], str) or not nav[0].strip():
    raise ValueError('A nonempty navigation instruction is required')
