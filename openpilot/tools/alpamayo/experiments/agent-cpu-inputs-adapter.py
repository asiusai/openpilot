"""Optional exact CPU input adapter: cached patches and structural RoPE indices.

The original FastInputs still owns validation, camera ordering, timestamps,
history policies, bicubic resize, text processing, and learned-history encoding.
"""
from collections import Counter, OrderedDict
import copy
from functools import lru_cache
import hashlib
import importlib.util
import inspect
import json
from pathlib import Path
from types import MethodType
import numpy as np
import torch


@lru_cache(None)
def local(name):
  spec=importlib.util.spec_from_file_location(name.replace('-','_'),Path(__file__).with_name(name+'.py'))
  value=importlib.util.module_from_spec(spec);spec.loader.exec_module(value);return value


BaseFastInputs=local('fast-inputs').FastInputs


class FastInputs(BaseFastInputs):
  def __init__(self,*args,patch_cache_entries=144,patch_cache_bytes=96*1024*1024,rope_cache_entries=16,text_cache_entries=64,**kwargs):
    if not kwargs.get('cache_resize',False) or not kwargs.get('direct_byte_inputs',False):
      raise ValueError('Standalone CPU adapter requires cache_resize=True and direct_byte_inputs=True')
    if min(patch_cache_entries,patch_cache_bytes,rope_cache_entries)<=0:raise ValueError('Cache bounds must be positive')
    if text_cache_entries<0:raise ValueError('Text cache bound must be nonnegative')
    super().__init__(*args,**kwargs)
    image=self.processor.image_processor
    if (image.patch_size,image.merge_size,image.temporal_patch_size)!=(16,2,1):
      raise ValueError('Standalone packing supports the current checkpoint layout only')
    self.patch_cache=OrderedDict();self.patch_cache_entries=patch_cache_entries;self.patch_cache_bytes=patch_cache_bytes
    self.patch_bytes=0;self.rope_structures=OrderedDict();self.rope_cache_entries=rope_cache_entries
    self.text_cache=OrderedDict();self.text_cache_entries=text_cache_entries
    self.cpu_input_statistics=Counter()
    paths={Path(__file__),Path(local('fast-inputs').__file__),
           Path(inspect.getfile(type(image))),Path(inspect.getfile(type(self.processor))),
           Path(inspect.getfile(type(self.processor).__mro__[1])),
           Path(inspect.getfile(local('fast-inputs').reference.Cosmos3EdgeModel)),
           Path(inspect.getfile(local('fast-inputs').reference._tokenize_messages)),
           Path(inspect.getfile(local('fast-inputs').reference.build_conversation)),
           Path(inspect.getfile(type(self.history))),
           self.checkpoint/'config.json',self.checkpoint/'processor_config.json',
           self.checkpoint/'tokenizer.json',self.checkpoint/'tokenizer_config.json',self.checkpoint/'chat_template.jinja'}
    self.cpu_input_provenance={str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in paths if path.is_file()}
    self._processor_image_path=self.processor._process_images
    self.processor._process_images=MethodType(lambda processor,images,**options:self.process_images(images,options),self.processor)
    tokenizer=self.processor.tokenizer
    owner=self
    class ExactTokenizerCache:
      def __getattr__(self,name):return getattr(tokenizer,name)
      def __call__(self,text,**options):return owner.tokenize_text(tokenizer,text,options)
    self.processor.tokenizer=ExactTokenizerCache()

  def tokenize_text(self,tokenizer,text,options):
    def freeze(value):
      if isinstance(value,dict):return tuple(sorted((key,freeze(item)) for key,item in value.items()))
      if isinstance(value,(list,tuple)):return tuple(freeze(item) for item in value)
      if value is None or isinstance(value,(str,int,float,bool)):return value
      raise TypeError('Unexpected mutable tokenizer option')
    def fresh(values):return {key:value.clone() if isinstance(value,torch.Tensor) else copy.deepcopy(value) for key,value in values.items()}
    key=(freeze(text),freeze(options),len(tokenizer),tokenizer.padding_side,tokenizer.truncation_side)
    cached=self.text_cache.get(key)
    if cached is not None:
      self.cpu_input_statistics['text_hits']+=1;self.text_cache.move_to_end(key)
      return fresh(cached)
    self.cpu_input_statistics['text_misses']+=1
    result=tokenizer(text,**options)
    if self.text_cache_entries:
      self.text_cache[key]=fresh(result)
      while len(self.text_cache)>self.text_cache_entries:
        self.text_cache.popitem(last=False);self.cpu_input_statistics['text_evictions']+=1
    return result

  def process_images(self,images,options):
    # Keep the original ProcessorMixin token expansion, tokenizer, special-token
    # checks, and mm_token_type_ids construction. Replace only image processing.
    if any(options.get(key) is not False for key in ('do_resize','do_rescale','do_normalize')):
      raise ValueError('Expected already resized uint8 images without normalization')
    if not isinstance(images,(list,tuple)):raise ValueError('Expected the original flat image list')
    patches,grids,replacements=[],[],[]
    image_processor=self.processor.image_processor
    layout=(image_processor.patch_size,image_processor.merge_size,image_processor.temporal_patch_size)
    if layout!=(16,2,1):raise ValueError('Checkpoint patch layout changed after adapter initialization')
    for image in images:
      if image.dtype!=torch.uint8 or image.device.type!='cpu' or image.ndim!=3 or image.shape[0]!=3:
        raise ValueError('Expected CPU uint8 RGB images from the checked bicubic resize')
      # The original resize cache validates every source byte before returning
      # this exact resized Tensor object. Retain that object to prevent ID reuse.
      # No timestamp alone authorizes a patch hit.
      key=(id(image),image._version,tuple(image.shape),tuple(image.stride()),layout)
      cached=self.patch_cache.get(key)
      if cached is not None and cached[0] is image:
        self.cpu_input_statistics['patch_hits']+=1;self.patch_cache.move_to_end(key)
        packed,grid=cached[1],cached[2]
      else:
        self.cpu_input_statistics['patch_misses']+=1
        packed,height,width=image_processor.patchify(image.unsqueeze(0),*layout)
        packed=packed.reshape(-1,3*layout[0]**2).contiguous()
        grid=(1,height,width)
        size=image.numel()*image.element_size()+packed.numel()*packed.element_size()
        if size<=self.patch_cache_bytes:
          self.patch_cache[key]=(image,packed,grid,size);self.patch_bytes+=size
          while len(self.patch_cache)>self.patch_cache_entries or self.patch_bytes>self.patch_cache_bytes:
            _,removed=self.patch_cache.popitem(last=False);self.patch_bytes-=removed[3]
            self.cpu_input_statistics['patch_evictions']+=1
      patches.append(packed);grids.append(grid)
      replacements.append(self.processor.image_token*(grid[0]*grid[1]*grid[2]//layout[1]**2))
    # A fresh output buffer prevents a caller mutation from poisoning the cache.
    pixels=torch.cat(patches,dim=0)
    return {'pixel_values':pixels,'image_grid_thw':torch.tensor(grids)},replacements

  def native_inputs(self,prepared):
    reference=local('fast-inputs').reference
    tokens=prepared['tokenized_data']
    ids=tokens['input_ids'].clone()
    # Recompute the learned-history values for every query. Only positional
    # structure is cached; history token contents never authorize a cache hit.
    history=self.history.encode(None,None,prepared['ego_history_xyz'][:,0],prepared['ego_history_rot'][:,0])
    ids[ids==self.config.traj_ids['history_pad']]=(history+self.config.traj_ids['history_id0']).flatten()
    grids=tokens.get('embedding_grid_thw',tokens['image_grid_thw'])
    types=tokens['mm_token_type_ids']
    key=(tuple(ids.shape),str(ids.dtype),types.contiguous().numpy().tobytes(),tuple(map(tuple,grids.tolist())),
         self.config.vlm_config.vision_config.spatial_merge_size)
    cached=self.rope_structures.get(key)
    if cached is None:
      positions,delta=reference.Cosmos3EdgeModel.get_rope_index(self.rope,ids,types,image_grid_thw=grids)
      cached=(positions.numpy().copy(),int(delta.item()))
      self.rope_structures[key]=cached;self.cpu_input_statistics['rope_misses']+=1
      while len(self.rope_structures)>self.rope_cache_entries:
        self.rope_structures.popitem(last=False);self.cpu_input_statistics['rope_evictions']+=1
    else:
      self.rope_structures.move_to_end(key);self.cpu_input_statistics['rope_hits']+=1
    image_key=tuple(map(tuple,tokens['image_grid_thw'].tolist()))
    if image_key not in self.position_cache:
      if len(self.position_cache)>=4:self.position_cache.pop(next(iter(self.position_cache)))
      self.position_cache[image_key]=reference.Cosmos3EdgeVisionEmbeddings.resize_positional_embeddings(
        self.position_weight,tokens['image_grid_thw'],2).float().numpy()
    return ids.numpy(),cached[0].copy(),cached[1],self.position_cache[image_key]

  def cache_state(self):
    return dict(statistics=dict(self.cpu_input_statistics),patch_entries=len(self.patch_cache),
                patch_bytes=self.patch_bytes,patch_limit_bytes=self.patch_cache_bytes,
                rope_entries=len(self.rope_structures),text_entries=len(self.text_cache),
                provenance=self.cpu_input_provenance)
