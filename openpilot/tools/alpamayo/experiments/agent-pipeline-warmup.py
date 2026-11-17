"""Bound startup graphs using current input bytes and symbolic destination rows.

This helper never opens sample files. Its caller supplies the current input.
Warm outputs are discarded. Real decoding still performs every target check.
"""
from __future__ import annotations

import functools
import weakref

import numpy as np
from tinygrad import Device, Tensor, TinyJit, dtypes
from tinygrad.uop.ops import UOp


def upload_dynamic(owner, sizes, *args):
  """Copy current source patches into bound destination rows, preserving bytes."""
  count = len(sizes)
  sources, rest, destinations = args[:count], args[count:-count] if count else args, args[-count:] if count else ()
  for source, rows, begin in zip(sources,sizes,destinations,strict=True):
    source = source.to(Device.DEFAULT)
    pixels = owner.pixel_table_gpu[source.cast(dtypes.int32)] if owner.pixel_table_gpu is not None else source.bitcast(dtypes.bfloat16)
    owner.pixel_storage[begin:begin+rows].assign(pixels).realize()
  other = tuple(owner.positions[owner.layout_key] if i == 1 else x.to(Device.DEFAULT).bitcast(dtypes.bfloat16).contiguous().realize()
                if i not in (2,len(owner.cpu)-1) else x.to(Device.DEFAULT).contiguous().realize()
                for i,x in enumerate(rest,1))
  return (owner.pixel_storage,*other)


def selection_call(owner, missing):
  if owner.pixel_storage is None:
    owner.pixel_storage = Tensor.empty(owner.cpu[0].shape,dtype=dtypes.bfloat16).realize()
  sizes = tuple(owner.batches[index][1]-owner.batches[index][0] for index in missing)
  key = ('dynamic_destinations',sizes)
  if key not in owner.selective_jits:
    reference = weakref.proxy(owner)
    owner.selective_jits[key] = TinyJit(lambda *args:upload_dynamic(reference,sizes,*args))
  sources = [owner.rows(owner.cpu[0],owner.batches[index][0],owner.batches[index][1]) for index in missing]
  rest = [owner.gpu_rotary.get(index,value) for index,value in enumerate(owner.cpu)][1:]
  destinations = [UOp.variable(f'current_pixel_destination_{slot}',0,owner.cpu[0].shape[0]-size).bind(owner.batches[index][0])
                  for slot,(index,size) in enumerate(zip(missing,sizes,strict=True))]
  return owner.selective_jits[key], (*sources,*rest,*destinations)


def warm_decoder(decoder, model, args, config, logits, cos, sin, prompt, graph_budget=32):
  """Warm serial tails, all verifier widths, selectors, and serial input aliases."""
  fallback = getattr(decoder,'fallback_block',decoder.block)
  widths = [width for width in (16,32,48,64) if width <= args.max_tokens]
  serial = list(range(1,min(fallback,args.max_tokens)+1))
  required = len(serial)+2*len(widths)+int(decoder.compact_io)
  if required > graph_budget:
    raise ValueError(f'Startup needs {required} graphs; budget is {graph_budget}. Reasoning budget was not reduced.')
  if model.cache[0].shape[3]-prompt < max(widths+[max(serial)]):
    raise ValueError('Current KV allocation cannot cover the requested startup shapes')
  # Selection uses actual current target logits. No warm token is accepted.
  first = decoder.first_token(logits)
  if decoder.compact_io:
    while decoder.choose_jit.cnt < 2:
      decoder.first_token(logits)
  for count in widths:
    tokens = Tensor(np.full(count,first,dtype=np.int32),device='CPU').to(logits.device).realize()
    position = UOp.variable('draft_position',0,model.cache[0].shape[3]-count).bind(prompt)
    step = UOp.variable('draft_step',0,cos.shape[2]-count).bind(0)
    key = (count,cos.shape,model.cache[0].shape[3])
    if key not in decoder.verifiers:
      decoder.verifiers[key] = TinyJit(decoder.verify)
    while decoder.verifiers[key].cnt < 2:
      Tensor.realize(*decoder.verifiers[key](tokens,cos,sin,position,step))
    verified, _ = decoder.verifiers[key](tokens,cos,sin,position,step)
    if decoder.compact_io:
      while count not in decoder.select_jits or decoder.select_jits[count].cnt < 2:
        decoder.select_logits(verified,0)
      # One symbolic selector covers every accepted-prefix row at this width.
      decoder.select_logits(verified,count-1)
  for count in serial:
    key = (cos.shape,count,model.cache[0].shape[3])
    while key not in decoder.jits or decoder.jits[key].cnt < 2:
      output, _ = decoder.decode(logits,cos,sin,prompt,0,count)
      output.realize()
    output, _ = decoder.decode(logits,cos,sin,prompt,0,count)
    # Repeated fallback consumes its own previous output. Link that alias now.
    Tensor.realize(*decoder.decode(output,cos,sin,prompt,0,count))
  # Warm writes touch only positions at/after prompt. Genuine generation starts
  # from unchanged prompt logits and overwrites each accepted token position.
  return {'serial_counts':serial,'verifier_widths':widths,'graph_count':required}


class StartupWarmup:
  def __init__(self, active, record=lambda *args,**kwargs:None, graph_budget=32):
    self.active, self.record, self.graph_budget = active, record, graph_budget

  def patch_decoder(self, cls):
    original = cls.generate
    helper = self
    @functools.wraps(original)
    def generate(decoder,model,args,config,tokenizer,logits,cos,sin,prompt):
      # Captured graphs retain buffer UOps. Tensor wrapper addresses can be reused.
      identity = tuple(id(value.uop.base) for value in model.cache)
      previous = getattr(decoder,'_agent_cache_identity',None)
      if previous is not None and previous != identity:
        # Existing keys include capacity, but not the captured allocation.
        # Reusing those graphs after A -> B -> A would read the retired cache.
        decoder.jits.clear()
        decoder.verifiers.clear()
        helper.record('conditioning_cache_reallocated',previous=list(previous),current=list(identity))
      decoder._agent_cache_identity = identity
      if helper.active():
        key = (identity,cos.shape,args.max_tokens,decoder.compact_io,
               getattr(decoder,'fallback_block',decoder.block))
        completed = getattr(decoder,'_agent_current_shapes',set())
        if key not in completed:
          if len(completed) >= 4:
            raise ValueError('Startup warmup supports at most four cache and rotary layouts')
          plan = warm_decoder(decoder,model,args,config,logits,cos,sin,prompt,helper.graph_budget)
          completed.add(key)
          decoder._agent_current_shapes = completed
          helper.record('current_decoder_shapes_warmed',**plan)
      return original(decoder,model,args,config,tokenizer,logits,cos,sin,prompt)
    cls.generate = generate

  def patch_inputs(self, cls):
    original_upload = cls.upload
    helper = self
    def upload(owner,noise):
      if owner.selective_cache is None:
        return original_upload(owner,noise)
      owner.noise[:] = noise
      cache = owner.selective_cache
      if len(cache.keys) != len(owner.batches) or not all(count == 1 for _,_,count in owner.batches):
        raise ValueError('Dynamic selection requires one image per batch')
      missing = tuple(index for index,key in enumerate(cache.keys) if cache.verify or key not in cache.cache)
      owner.uploaded_pixel_bytes = sum((owner.batches[index][1]-owner.batches[index][0])*owner.cpu[0].shape[1]*owner.cpu[0].dtype.itemsize
                                      for index in missing)
      jit, arguments = selection_call(owner,missing)
      return jit(*arguments)
    def warmup_selections(owner):
      sizes = {end-begin for begin,end,_ in owner.batches}
      if len(owner.batches) != 10 or len(sizes) != 1:
        raise ValueError('Complete bounded selection warmup currently requires ten equal-grid front-history images')
      if len(owner.batches)+1 > helper.graph_budget:
        raise ValueError('Selection graph budget exceeded')
      for count in range(len(owner.batches)+1):
        jit, arguments = selection_call(owner,tuple(range(count)))
        while jit.cnt < 2:
          Tensor.realize(*jit(*arguments))
      helper.record('current_selection_shapes_warmed',image_count=10,graph_count=11)
    cls.upload, cls.warmup_selections = upload, warmup_selections

  def patch_vision(self, cls):
    original = cls.encode
    helper = self
    @functools.wraps(original)
    def encode(owner,pixels,positions,ids,batches,spans):
      if helper.active() and owner.grouped:
        grids = {}
        for key,(begin,end,count) in zip(owner.keys,batches,strict=True):
          if count != 1:
            raise ValueError('Current-image warmup requires independent image batches')
          grids.setdefault(key[0],[]).append((begin,end))
        planned = sum(min(owner.group_limit or len(ranges),len(ranges)) for ranges in grids.values())
        if planned > helper.graph_budget:
          raise ValueError('Vision startup graph budget exceeded')
        for ranges in grids.values():
          for count in range(1,min(owner.group_limit or len(ranges),len(ranges))+1):
            pairs = [(owner.image_view(value,begin,end) if owner.views else value[begin:end].unsqueeze(0).clone())
                     for begin,end in ranges[:count] for value in (pixels,positions)]
            Tensor.realize(*pairs)
            shape = tuple(value.shape for value in pairs)
            if shape not in owner.encoders:
              owner.encoders[shape] = TinyJit(owner.batch_encoder or
                                            (lambda *values:tuple(owner.model._vision(p,v) for p,v in zip(values[::2],values[1::2]))))
            while owner.encoders[shape].cnt < 2:
              Tensor.realize(*owner.encoders[shape](*pairs))
        helper.record('current_vision_shapes_warmed',grid_count=len(grids),graph_count=planned)
      return original(owner,pixels,positions,ids,batches,spans)
    cls.encode = encode
