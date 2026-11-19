"""Selected mixed-precision specialization of the shared BF16 core."""
from contextlib import contextmanager
from functools import partial
import json
import math
from pathlib import Path
from tinygrad import Device, Tensor, TinyJit, dtypes
from tinygrad.helpers import USE_TC, TC_OPT, round_up
from tinygrad.renderer.llvmir import AMDLLVMRenderer
from tinygrad.uop.ops import UOp
from openpilot.selfdrive.alpamayo.model import EdgeModel
from openpilot.selfdrive.alpamayo.amd_attention import _pack, _matmul, _softmax

PROMPT_TILES = 2304//16  # Expert attention keys: 2304 prompt rows, then 64 action rows.
PREFILL_ROWS = 2176  # Padded prompt rows: 2025 base tokens plus navigation text and the verified draft (17 x 128).
from . import amd_gemm, amd_decode, weights
from .serial_attention import direct as serial_cache_attention
from .serial_coordinates import cache_attention
from .kernels import bf16_linear, frozen_ops, serial_mlp, tiled_gemm, vision_tiles


class PackedActivation:
  def __init__(self, value, rows, width):
    self.value,self.rows,self.width = value,rows,width


class SelectedModel(EdgeModel):
  def __init__(self, checkpoint, fp8_cache, fixed_scales, pv_program):
    # Initialize shared state without loading the full BF16 weights first.
    self.config = json.loads((Path(checkpoint)/'config.json').read_text())
    self.weights,self.weight_scales = {},{}
    self.dtype,self.amd = dtypes.bfloat16,True
    self.cache = self.prompt_length = self.greedy_shape = None
    self.encode_jits,self.trajectory_jits = {},{}
    self.prefill_jit = TinyJit(self._language)
    self.decode_jit = TinyJit(self._greedy_decode)
    self.images,self.verifying,self.trajectory_active = 0,False,False
    self.prefill_prefix,self.quantizer_slot = None,0
    self.pv_program = pv_program
    self.trace_counts = {'pv':0,'qk':0,'fused_mlp':0}
    import time
    marks = [('start',time.monotonic())]
    def mark(name):
      Device[Device.DEFAULT].synchronize(); marks.append((name,time.monotonic()))
    self.weight_digest = weights.load_bulk(self,checkpoint,fp8_cache)
    mark('bulk')
    self.restored = weights.restore_bf16(self,checkpoint)
    mark('restore_bf16')
    self.bf16_weights = {name[:-7]:bf16_linear.pack_weight(value) for name,value in self.weights.items()
                        if name.startswith(('vlm.model.visual.embeddings.','vlm.model.visual.encoder.layers.'))
                        and name.endswith('.weight') and value.ndim == 2 and value.dtype == dtypes.bfloat16}
    mark('bf16_pack')
    self.vocabulary_size = weights.reduce_head(self)
    mark('reduce_head')
    # One schedule for all packs: per-tensor realizes cost ~20 ms of host scheduling each on the Dragon.
    expert = {name:amd_gemm.pack(value) for name,value in self.weights.items()
              if name.startswith('expert.') and value.dtype == dtypes.fp8e4m3}
    self.packed_weights = {name[:-7]:amd_gemm.pack(value) for name,value in self.weights.items()
                           if name.startswith('vlm.model.language_model.layers.') and name.endswith('.weight')
                           and ('.mlp.' in name or '.self_attn.' in name) and value.dtype == dtypes.fp8e4m3}
    Tensor.realize(*expert.values(),*self.packed_weights.values())
    self.expert_weights = set()
    for name,packed in expert.items():
      self.weights[name] = amd_gemm.unpack(packed)
      self.expert_weights.add(name.removesuffix('.weight'))
    mark('expert_and_language_pack')
    expected = {f'vlm.model.language_model.layers.{i}/{j}' for i in range(28) for j in range(4)}
    if set(fixed_scales) != expected or not all(math.isfinite(v) and v > 0 for v in fixed_scales.values()):
      raise ValueError('Require all 112 positive finite fixed language scales')
    if self.config['vlm_config']['text_config']['num_hidden_layers'] != 28:
      raise ValueError('Unsupported selected model depth')
    # One upload for the 112 fixed scales; each scale is an independent 4-byte view of it.
    keys = sorted(fixed_scales)
    scales = Tensor([float(fixed_scales[k]) for k in keys],dtype=dtypes.float32).realize()
    base = scales.uop.base.buffer
    self.fixed = {key:Tensor(UOp.from_buffer(base.view(4,4*i,).ensure_allocated(),dtypes.float32)).reshape(())
                  for i,key in enumerate(keys)}
    self._fixed_storage = scales
    projector = {name:amd_gemm.pack(self.weights[name+'.weight']) for name in ('vlm.model.projector.linear_fc1','vlm.model.projector.linear_fc2')}
    Tensor.realize(*projector.values())
    self.projector_weights = {name:amd_gemm.unpack(packed) for name,packed in projector.items()}
    mark('projector')
    self.w4_weights = serial_mlp.pack_weights(self)
    mark('w4_pack')
    self.load_ms = {b[0]:round((b[1]-a[1])*1e3) for a,b in zip(marks,marks[1:])}

  def reset(self):
    self.cache = self.prompt_length = self.greedy_shape = None
    self.encode_jits.clear()
    self.trajectory_jits.clear()
    self.prefill_jit.reset()
    self.decode_jit.reset()
    self.images,self.verifying,self.trajectory_active = 0,False,False
    self.prefill_prefix,self.quantizer_slot = None,0

  @contextmanager
  def verification(self):
    if self.verifying or self.trajectory_active or self.images:
      raise RuntimeError('Nested verification is unsupported')
    self.verifying = True
    try:
      yield
    finally:
      self.verifying = False

  def quantize(self, x):
    if self.verifying:
      value = x.reshape(-1,x.shape[-1]).float()
      scale = (value.abs().max(axis=1,keepdim=True)/448).clamp(min_=1e-12)
      return (value/scale).clamp(-448,448).cast(dtypes.fp8e4m3).contiguous(),scale
    if self.images:
      return frozen_ops.grouped_quantize(x,self.images)
    rows,width = x.numel()//x.shape[-1],x.shape[-1]
    if self.prefill_prefix is not None and x.dtype == dtypes.bfloat16 and rows >= 128:
      scale = self.fixed[f'{self.prefill_prefix}/{self.quantizer_slot}']
      self.quantizer_slot += 1
      value = (x.float()/scale).clamp(-448,448).cast(dtypes.fp8e4m3).reshape(-1,width)
      return PackedActivation(amd_gemm.pack(value).realize(),rows,width),scale
    if x.dtype == dtypes.bfloat16 and x.numel() >= 131072:
      return frozen_ops.quantize(x)
    scale = (x.float().abs().max()/448).clamp(min_=1e-12)
    return (x.float()/scale).clamp(-448,448).cast(dtypes.fp8e4m3).reshape(-1,x.shape[-1]).contiguous(),scale

  def packed_linear(self, packed, scale, name, shape):
    weight = self.weights[name+'.weight']
    bias = self.weights.get(name+'.bias')
    out = Tensor.empty(packed.rows,weight.shape[0],device=packed.value.device,dtype=dtypes.bfloat16)
    value = Tensor.custom_kernel(out,packed.value,self.packed_weights[name],scale,self.weight_scales[name+'.weight'],
                                *(() if bias is None else (bias,)),fxn=partial(amd_gemm._gemm,packed_input=True))[0]
    return value.reshape(*shape,weight.shape[0])

  def base_linear(self, x, name, quantized_input=None, heads=None):
    shape = x.shape[:-1]
    x = x.reshape(-1,x.shape[-1])
    single = x.numel() == x.shape[-1]
    weight,bias = self.weights[name+'.weight'],self.weights.get(name+'.bias')
    if (scale := self.weight_scales.get(name+'.weight')) is not None:
      quantized,x_scale = self.quantize(x) if quantized_input is None else quantized_input
      if single and weight.shape[1]%128 == 0 and isinstance(Device[x.device].renderer,AMDLLVMRenderer):
        packed = name.startswith('vlm.') and ('.mlp.' in name or '.self_attn.' in name)
        return amd_decode.gemv(quantized,weight,(x_scale,scale),bias,packed=packed,dtype=x.dtype).reshape(*shape,weight.shape[0])
      if not single and USE_TC.value == 1 and (x.shape[0]%16 == 0 or TC_OPT >= 2):
        if x.shape[0] >= 128 and name.startswith('vlm.') and ('.mlp.' in name or '.self_attn.' in name):
          return amd_gemm.gemm(quantized,weight,(x_scale,scale),bias,dtype=x.dtype).reshape(*shape,weight.shape[0])
        y = quantized.dot(weight.T,dtype=dtypes.float32)
      else:
        y = quantized.float().dot(weight.float().T)
      y = y*(x_scale*scale)
      if bias is not None:
        y = y+bias.float()
      return y.cast(x.dtype).reshape(*shape,weight.shape[0])
    return super().linear(x,name).reshape(*shape,weight.shape[0])

  def linear(self, x, name, quantized_input=None, heads=None):
    shape = x.shape[:-1]
    weight,bias = self.weights[name+'.weight'],self.weights.get(name+'.bias')
    scale = self.weight_scales.get(name+'.weight')
    if self.verifying and scale is not None:
      quantized,x_scale = self.quantize(x) if quantized_input is None else quantized_input
      return tiled_gemm.gemm(quantized,weight,(x_scale,scale),tile=(1,4),row_scale=True,bias=bias).reshape(*shape,-1)
    match = vision_tiles.NAME.fullmatch(name)
    if (match is not None and int(match[1]) >= 2 and x.ndim == 3 and x.shape[0] in (1,2) and x.shape[1] == 720
        and x.dtype == dtypes.bfloat16 and heads is None and scale is not None and weight.dtype == dtypes.fp8e4m3):
      fc,images = match[2],x.shape[0]
      assert (x.shape[1],weight.shape[0],x.shape[-1]) == vision_tiles.SHAPES[fc]
      assert bias is not None and bias.dtype == dtypes.bfloat16
      quantized,x_scale = self.quantize(x) if quantized_input is None else quantized_input
      return vision_tiles.gemm(quantized.reshape(images,720,x.shape[-1]),weight,(x_scale,scale),bias,vision_tiles.TILES[fc]).reshape(*shape,weight.shape[0])
    if self.images and scale is not None:
      if name.startswith(('vlm.model.visual.embeddings.','vlm.model.projector.')):
        count = self.images
        grouped = x.reshape(count,-1,x.shape[-1])
        self.images = 0
        try:
          parts = [self.linear(grouped[i:i+1],name,heads=heads).realize() for i in range(count)]
          return Tensor.cat(*parts,dim=0).reshape(*shape,parts[0].shape[-1])
        finally:
          self.images = count
      quantized,x_scale = self.quantize(x) if quantized_input is None else quantized_input
      return tiled_gemm.gemm(quantized,weight,(x_scale,scale),row_scale=True,bias=bias).reshape(*shape,weight.shape[0])
    rows = x.numel()//x.shape[-1]
    if (name in self.w4_weights and x.shape == (1,1,weight.shape[1]) and x.dtype == dtypes.bfloat16 and heads is None
        and not (self.images or self.verifying or self.trajectory_active) and self.prefill_prefix is None):
      q,x_scale = self.quantize(x.reshape(1,-1)) if quantized_input is None else quantized_input
      return serial_mlp.gemv(q,self.w4_weights[name],x_scale,scale,weight.shape)
    if self.prefill_prefix is not None and name in self.packed_weights and rows >= 128:
      packed,x_scale = self.quantize(x) if quantized_input is None else quantized_input
      assert isinstance(packed,PackedActivation) and (packed.rows,packed.width) == (rows,x.shape[-1])
      return self.packed_linear(packed,x_scale,name,shape)
    if name in self.expert_weights and rows == 64:
      quantized,x_scale = self.quantize(x) if quantized_input is None else quantized_input
      return amd_gemm.gemm(quantized,weight,(x_scale,scale),bias,dtype=x.dtype).reshape(*shape,weight.shape[0])
    if name in self.bf16_weights and rows >= 128:
      assert quantized_input is None and x.dtype == dtypes.bfloat16
      if x.ndim == 3 and x.shape[0] > 1 and x.shape[1]%16 and weight.shape[0]%64:
        parts = [bf16_linear.linear(x[i],self.bf16_weights[name],weight.shape[0],bias).unsqueeze(0) for i in range(x.shape[0])]
        return parts[0].cat(*parts[1:],dim=0)
      return bf16_linear.linear(x.reshape(-1,x.shape[-1]),self.bf16_weights[name],weight.shape[0],bias).reshape(*shape,weight.shape[0])
    return self.base_linear(x,name,quantized_input,heads)

  def norm(self, x, name, rms=False, eps=None):
    if x.dtype != dtypes.bfloat16 or x.numel()//x.shape[-1] < 64:
      return super().norm(x,name,rms,eps)
    return frozen_ops.norm(x,self.weights[name+'.weight'],None if rms else self.weights[name+'.bias'],
                           rms,eps if eps is not None else 1e-5 if rms else 1e-6)

  def expert_kv(self, k, v):
    """Packed expert K/V: the prompt tiles were stored once per diffusion call; store only the 64 action rows."""
    layer = self.expert_layer % len(self.cache)
    self.expert_layer += 1
    pk,pv = self.expert_packs[layer]
    pk = Tensor(pk.uop.after(pk[:,:,PROMPT_TILES:].uop.store(_pack(k[:,:,PROMPT_TILES*16:]).uop)))
    pv = Tensor(pv.uop.after(pv[:,:,:,PROMPT_TILES:].uop.store(_pack(v[:,:,PROMPT_TILES*16:].transpose(-2,-1)).uop)))
    return pk,pv

  def masked_attention(self, q, k, v, mask, *, expert=False):
    batch,heads,queries,width = q.shape
    kv_heads,keys = k.shape[1:3]
    assert queries in (16,32,48,64) and mask.shape in ((keys,),(queries,keys))
    packed_k,packed_v = self.expert_kv(k,v) if expert else (_pack(k),None)
    scores = Tensor.empty(batch,heads,queries,keys,device=q.device,dtype=q.dtype)
    if expert:
      # The expert mask is one FP32 value per key; apply it in the score kernel's epilogue (bit-identical, one launch fewer).
      assert mask.shape == (keys,)  # 0 or -inf per key: exact in FP32 whatever its weak float dtype
      scores = Tensor.custom_kernel(scores,_pack(q),packed_k,mask.float().contiguous(),
                                    fxn=partial(_matmul,shape=(batch,heads,kv_heads,queries,keys,width),scaled=True))[0]
    else:
      scores = Tensor.custom_kernel(scores,_pack(q),packed_k,fxn=partial(_matmul,shape=(batch,heads,kv_heads,queries,keys,width),scaled=True))[0]
      scores = (scores.float()+mask).cast(q.dtype).contiguous()
    probs = Tensor.empty(batch,heads,round_up(queries,16),round_up(keys,16),device=q.device,dtype=q.dtype)
    probs = Tensor.custom_kernel(probs,scores,fxn=_softmax)[0]
    out = Tensor.empty(q.shape,device=q.device,dtype=q.dtype)
    if not expert:
      packed_v = _pack(v.transpose(-2,-1))
    if expert:
      assert self.trajectory_active and not self.verifying
      assert q.shape == (1,16,64,128) and keys == 2368
      assert probs.shape == (1,16,64,2368) and packed_v.shape == (1,8,8,148,4,16,4)
      assert out.dtype == probs.dtype == packed_v.dtype == dtypes.bfloat16
      self.trace_counts['pv'] += 1
      self.trace_counts['qk'] += 1
      return Tensor.custom_kernel(out,probs,packed_v,fxn=lambda *args: self.pv_program)[0]
    return Tensor.custom_kernel(out,probs,packed_v,fxn=partial(_matmul,shape=(batch,heads,kv_heads,queries,width,keys)))[0]

  def attention(self, q, k, v, mask=None, causal=False):
    if self.verifying:
      return self.masked_attention(q,k,v,mask)
    if mask is None and q.shape[2] >= 128 and q.shape[-1] in (72,128):
      return frozen_ops.attention(q,k,v,causal)
    if q.dtype == dtypes.bfloat16 and q.shape[2] == 64 and mask is not None and mask.numel() == k.shape[2] and not causal:
      return self.masked_attention(q,k,v,mask,expert=self.trajectory_active)
    return super().attention(q,k,v,mask,causal)

  def vision_layer(self, x, i):
    prefix = f'vlm.model.visual.encoder.layers.{i}'
    h = self.norm(x,f'{prefix}.layer_norm1')
    quantized = self.quantize(h) if f'{prefix}.self_attn.q_proj.weight' in self.weight_scales else None
    q,k,v = (self.linear(h,f'{prefix}.self_attn.{n}_proj',quantized,heads=16).reshape(*h.shape[:2],16,72).transpose(1,2) for n in ('q','k','v'))
    h = self.attention(q,k,v).transpose(1,2).reshape(x.shape)
    x = x+self.linear(h,f'{prefix}.self_attn.out_proj')
    h = self.linear(self.norm(x,f'{prefix}.layer_norm2'),f'{prefix}.mlp.fc1').float().gelu().cast(self.dtype)
    return (x+self.linear(h,f'{prefix}.mlp.fc2')).realize()

  def text_layer(self, x, cos, sin, prefix, cache=None, mask=None, cache_pos=None, serial_position=None):
    previous = self.prefill_prefix,self.quantizer_slot
    prefill = x.shape[1] >= 128
    self.prefill_prefix,self.quantizer_slot = (prefix if prefill else None),0
    try:
      h = self.norm(x,f'{prefix}.input_layernorm',rms=True)
      quantized = self.quantize(h) if f'{prefix}.self_attn.q_proj.weight' in self.weight_scales else None
      q,k,v = (self.linear(h,f'{prefix}.self_attn.{n}_proj',quantized,heads=heads).reshape(1,h.shape[1],heads,128).transpose(1,2)
               for n,heads in (('q',16),('k',8),('v',8)))
      q,k = self.rope(q,k,cos,sin)
      if serial_position is not None:
        assert x.shape[1] == 1 and cache_pos is None and not self.verifying and not self.trajectory_active
        h = cache_attention(q,cache,k,v,*serial_position,mask)
      else:
        if cache_pos is not None:
          store = cache[:,:,:,cache_pos:cache_pos+x.shape[1]].uop.store(Tensor.stack(k,v).uop)
          assigned = Tensor(cache.uop.after(store))
          end = cache.shape[3] if not isinstance(cache_pos,int) else cache_pos+x.shape[1]
          k,v = (assigned[i,:,:,:end] for i in range(2))
        elif cache is not None:
          k,v = (old.cat(new,dim=2) for old,new in zip(cache,(k,v),strict=True))
        causal = isinstance(cache_pos,int) and cache_pos == 0
        if (not self.verifying and not self.trajectory_active and cache_pos is not None and not isinstance(cache_pos,int)
            and q.shape == (1,16,1,128) and cache.shape == (2,1,8,2304,128) and mask is not None and mask.numel() == 2304):
          h = serial_cache_attention(q,assigned,mask)
        else:
          h = self.attention(q,k,v,mask,causal)
      h = h.transpose(1,2).reshape(1,x.shape[1],2048)
      x = x+self.linear(h,f'{prefix}.self_attn.o_proj')
      h = self.norm(x,f'{prefix}.post_attention_layernorm',rms=True)
      if prefill:
        rows = x.shape[1]
        assert x.shape == (1,PREFILL_ROWS,2048) and type(cache_pos) is int and cache_pos == 0
        fc1,fc2 = prefix+'.mlp.fc1',prefix+'.mlp.fc2'
        assert self.quantizer_slot == 2 and fc1+'.bias' not in self.weights and fc2+'.bias' not in self.weights
        packed,scale = self.quantize(h)
        assert self.quantizer_slot == 3 and isinstance(packed,PackedActivation)
        output_scale = self.fixed[prefix+'/3']
        output = frozen_ops.fused_mlp(packed.value,self.packed_weights[fc1],scale,self.weight_scales[fc1+'.weight'],output_scale,rows)
        # The fused kernel already performed original BF16/ReLU/square/slot-3 conversion.
        self.quantizer_slot += 1
        residual = x.reshape(rows,2048).contiguous()
        result = Tensor.custom_kernel(Tensor.empty(residual.shape,dtype=x.dtype,device=x.device),output,self.packed_weights[fc2],
                                      output_scale,self.weight_scales[fc2+'.weight'],residual,fxn=amd_gemm.prefill_fc2)[0]
        self.trace_counts['fused_mlp'] += 1
        return result.reshape(x.shape).realize()
      else:
        raw = self.linear(h,f'{prefix}.mlp.fc1')
        quantized = amd_decode.quantize_relu_squared(raw) if not self.verifying and raw.shape == (1,1,9216) else None
        h = self.linear(raw.relu().square(),f'{prefix}.mlp.fc2',quantized_input=quantized)
      return (x+h).realize()
    finally:
      self.prefill_prefix,self.quantizer_slot = previous

  def _greedy_decode(self, logits, cos, sin, position, step):
    token = self.output_token_mapping[logits.reshape(-1).argmax()].realize()
    x = self.weights['vlm.model.language_model.embed_tokens.weight'][token].reshape(1,1,-1)
    logits = self._language(x,cos[:,:,step:step+1],sin[:,:,step:step+1],position)
    return logits,token.float().realize()

  def allocate_expert_packs(self):
    # Persistent packed expert K/V, allocated before any captured or cached diffusion graph binds it.
    # Not created in __init__: the frozen prefill graph pins that method's source.
    if getattr(self,'expert_packs',None) is None:
      self.expert_packs = [(Tensor.empty(1,8,PROMPT_TILES+4,8,4,16,4,dtype=self.dtype).contiguous().realize(),
                            Tensor.empty(1,8,8,PROMPT_TILES+4,4,16,4,dtype=self.dtype).contiguous().realize())
                           for _ in range(self.config['vlm_config']['text_config']['num_hidden_layers'])]

  def _trajectory(self, actions, cos, sin, length, steps, offset):
    if steps != 3 or self.cache[0].shape[3] != 2304 or self.trajectory_active or self.verifying:
      raise ValueError('Selected expert requires three Euler steps and 2304 KV rows')
    before = self.trace_counts['pv'],self.trace_counts['qk']
    # The prompt KV cache is constant during diffusion: pack it once per call, not once per Euler step and layer.
    self.allocate_expert_packs()
    for (pk,pv),plane in zip(self.expert_packs,self.cache,strict=True):
      Tensor(pk.uop.after(pk[:,:,:PROMPT_TILES].uop.store(_pack(plane[0]).uop))).realize()
      Tensor(pv.uop.after(pv[:,:,:,:PROMPT_TILES].uop.store(_pack(plane[1].transpose(-2,-1)).uop))).realize()
    self.expert_layer = 0
    self.trajectory_active = True
    try:
      result = super()._trajectory(actions,cos,sin,length,steps,offset)
    finally:
      self.trajectory_active = False
    if (self.trace_counts['pv']-before[0],self.trace_counts['qk']-before[1]) != (84,84):
      raise ValueError('Selected expert PV coverage changed')
    return result

  def encode_group(self, *pairs):
    count = len(pairs)//2
    assert count in (1,2) and not self.images
    self.images = count
    try:
      pixels,positions = Tensor.cat(*pairs[::2],dim=0),Tensor.cat(*pairs[1::2],dim=0)
      assert pixels.shape[1] == 720
      value = (self.linear(pixels,'vlm.model.visual.embeddings.patch_embedding')+positions).realize()
      for layer in range(self.config['vlm_config']['vision_config']['num_hidden_layers']):
        value = self.vision_layer(value,layer)
      self.images = 0
      parts = []
      for i in range(count):
        x = self.norm(value[i:i+1],'vlm.model.visual.post_layernorm')
        parts.append(self.norm(x,'vlm.model.projector.norm').reshape(-1,4608).realize())
      x = Tensor.cat(*parts,dim=0)
      for index,name in enumerate(self.projector_weights):
        quantized,scale = self.quantize(x) if count == 1 else frozen_ops.grouped_quantize(x,count)
        x = tiled_gemm.gemm(quantized,self.projector_weights[name],(scale,self.weight_scales[name+'.weight']),
                            row_scale=count > 1,bias=self.weights.get(name+'.bias'))
        if index == 0:
          x = x.float().gelu(approximate='none').cast(self.dtype)
      x = x.realize().reshape(count,-1,x.shape[-1])
      return tuple(x[i].realize() for i in range(count))
    finally:
      self.images = 0
