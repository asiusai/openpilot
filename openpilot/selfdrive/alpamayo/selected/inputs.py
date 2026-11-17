# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Modified: selected resize caching, byte patches, shared adapter, and canonical input validation.
"""Exact selected CPU inputs, including cached resize and uint8 patch retention."""
from collections import OrderedDict
import ctypes
from functools import partial
from types import SimpleNamespace
import numpy as np
import torch
from torchvision.transforms.functional import InterpolationMode, resize
from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize
from transformers.models.cosmos3_edge.modeling_cosmos3_edge import Cosmos3EdgeModel, Cosmos3EdgeVisionEmbeddings
from alpamayo2_edge.chat_template import CLOSED_THINKING, build_conversation
from alpamayo2_edge.models.delta_tokenizer import DeltaTrajectoryTokenizer
from alpamayo2_edge.sample import CAMERA_PIXEL_BOUNDS
from openpilot.selfdrive.alpamayo.inputs import InputAdapter

_memcmp = ctypes.CDLL(None).memcmp
_memcmp.argtypes = (ctypes.c_void_p,ctypes.c_void_p,ctypes.c_size_t)
_memcmp.restype = ctypes.c_int


def same_image(a, b):
  if a.shape != b.shape or a.dtype != b.dtype:
    return False
  if a.device.type == b.device.type == 'cpu' and a.dtype == torch.uint8 and a.is_contiguous() and b.is_contiguous():
    return _memcmp(a.data_ptr(),b.data_ptr(),a.numel()) == 0
  return torch.equal(a,b)


class SelectedInputs(InputAdapter):
  def __init__(self, checkpoint):
    super().__init__(checkpoint)
    self.history = DeltaTrajectoryTokenizer(pad_origin_at_beginning=False)
    self.rope = SimpleNamespace(config=self.config.vlm_config)
    self.rope.get_vision_position_ids = partial(Cosmos3EdgeModel.get_vision_position_ids,self.rope)
    self.resize_cache,self.position_cache = OrderedDict(),{}
    image = self.processor.image_processor
    if (list(image.image_mean) != [.5]*3 or list(image.image_std) != [.5]*3 or
        image.rescale_factor != 1/255 or image.temporal_patch_size != 1):
      raise ValueError('Selected byte inputs require the exact checkpoint pixel palette')

  def reset(self):
    self.resize_cache.clear()
    self.position_cache.clear()

  def tokenize(self, messages, identities):
    # Tokenization body follows NVIDIA helper._tokenize_messages (Apache-2.0).
    # Only resize reuse and retained unnormalized uint8 patches differ.
    processor = self.processor
    text = processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,add_vision_id=False,
                                         continue_final_message=False,enable_thinking=False)
    if not text.endswith(CLOSED_THINKING):
      raise ValueError('Unexpected checkpoint chat suffix')
    image_processor = processor.image_processor
    images = []
    identity = iter(identities)
    for message in messages:
      for entry in message['content']:
        if entry['type'] != 'image':
          continue
        frame = entry['image'].squeeze(0)
        height,width = smart_resize(*frame.shape[-2:],factor=image_processor.patch_size*image_processor.merge_size,
                                    min_pixels=entry['min_pixels'],max_pixels=entry['max_pixels'])
        key = (*next(identity),(height,width),(('interpolation',InterpolationMode.BICUBIC),))
        old = self.resize_cache.get(key)
        if old is not None and same_image(frame,old[0]):
          self.resize_cache.move_to_end(key)
          image = old[1]
        else:
          image = resize(frame,[height,width],interpolation=InterpolationMode.BICUBIC)
          self.resize_cache[key] = (frame.clone(),image)
          while len(self.resize_cache) > 144:
            self.resize_cache.popitem(last=False)
        images.append(image)
    tokens = dict(processor(text=text,images=images,padding=False,return_tensors='pt',return_mm_token_type_ids=True,
                            do_resize=False,do_rescale=False,do_normalize=False))
    patches = tokens['pixel_values']
    if patches.dtype != torch.uint8 or patches.ndim != 2 or patches.shape[1] != 3*image_processor.patch_size**2:
      raise ValueError('Expected unnormalized uint8 image patches')
    if tokens['input_ids'].shape[0] != 1:
      raise ValueError('One sample is required')
    return tokens

  def prepare(self, arrays):
    fields = {'image_frames','camera_indices','absolute_timestamps','ego_history_xyz','ego_history_rot','nav_text'}
    if set(arrays) != fields:
      raise ValueError('Expected the six canonical model inputs')
    data = {k:v if k == 'nav_text' or isinstance(v,torch.Tensor) else torch.from_numpy(np.asarray(v)) for k,v in arrays.items()}
    cameras,times,images = data['camera_indices'],data['absolute_timestamps'],data['image_frames']
    if (cameras.dtype != torch.int64 or cameras.tolist() != [1]*5+[6]*5 or times.dtype != torch.int64 or times.shape != (10,)
        or images.dtype != torch.uint8 or images.shape != (10,1,3,432,768) or not images.is_contiguous()
        or any(value.device.type != 'cpu' for value in (cameras,times,images))):
      raise ValueError('Require the complete calibrated two-front CPU input')
    if any(not bool(torch.all(times[i+1:i+5] > times[i:i+4])) for i in (0,5)):
      raise ValueError('History timestamps must advance')
    for key,shape in [('ego_history_xyz',(1,20,3)),('ego_history_rot',(1,20,3,3))]:
      value = data[key]
      if value.shape != shape or value.dtype != torch.float32 or value.device.type != 'cpu' or not torch.isfinite(value).all():
        raise ValueError('Require finite measured float32 ego history')
    nav = data['nav_text']
    if not isinstance(nav,(list,tuple)) or len(nav) != 1 or not isinstance(nav[0],str):
      raise ValueError('Require one navigation string; empty means unavailable')
    canonical = dict(data,camera_pixel_bounds=CAMERA_PIXEL_BOUNDS)
    tokens = self.tokenize(build_conversation(canonical),zip(cameras.tolist(),times.tolist(),strict=True))
    grids = tokens['image_grid_thw']
    if grids.shape != (10,3) or not torch.all(grids.prod(dim=1) == 720):
      raise ValueError('Selected input requires 720 patches for every image')
    prepared = {'tokenized_data':tokens,'ego_history_xyz':data['ego_history_xyz'].unsqueeze(0),
                'ego_history_rot':data['ego_history_rot'].unsqueeze(0)}
    ids = tokens['input_ids'].clone()
    history = self.history.encode(None,None,prepared['ego_history_xyz'][:,0],prepared['ego_history_rot'][:,0])
    ids[ids == self.config.traj_ids['history_pad']] = (history+self.config.traj_ids['history_id0']).flatten()
    positions,delta = Cosmos3EdgeModel.get_rope_index(self.rope,ids,tokens['mm_token_type_ids'],image_grid_thw=grids)
    key = tuple(map(tuple,grids.tolist()))
    if key not in self.position_cache:
      if len(self.position_cache) >= 4:
        self.position_cache.pop(next(iter(self.position_cache)))
      self.position_cache[key] = Cosmos3EdgeVisionEmbeddings.resize_positional_embeddings(self.position_weight,grids,2).float().numpy()
    return prepared,(ids.numpy(),positions.numpy(),int(delta.item()),self.position_cache[key])
