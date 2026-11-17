"""CPU differentiable native language-block surrogate. Captured selected-runtime targets remain external."""
import hashlib
import json
import os
from pathlib import Path
import sys

BASE=Path('/tmp/opencode/alpamayo-asius-20261114')
CHECKPOINT=BASE/'model'
NATIVE=BASE/'upstream/.venv/lib/python3.12/site-packages/transformers/models/cosmos3_edge/modeling_cosmos3_edge.py'


def torch_cpu():
  os.environ.update(CUDA_VISIBLE_DEVICES='',HIP_VISIBLE_DEVICES='',ROCR_VISIBLE_DEVICES='',
    HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false')
  import torch
  torch.set_num_threads(4)
  torch.set_default_device('cpu')
  return torch


def make(layer=12,expert=False):
  torch=torch_cpu()
  from transformers.models.cosmos3_edge.configuration_cosmos3_edge import Cosmos3EdgeTextConfig
  from transformers.models.cosmos3_edge.modeling_cosmos3_edge import Cosmos3EdgeTextDecoderLayer
  data=json.loads((CHECKPOINT/'config.json').read_text())
  source=data['expert_config']['llm_config'] if expert else data['vlm_config']['text_config']
  config=Cosmos3EdgeTextConfig(**source);config._attn_implementation='eager'
  block=Cosmos3EdgeTextDecoderLayer(config,layer).to(device='cpu',dtype=torch.float32)
  block.eval()
  assert all(parameter.device.type=='cpu' for parameter in block.parameters())
  return block,config


def initialize(block,layer=12,expert=False):
  """Load only one original BF16 block; no whole-model loader or rounded FP8 master."""
  from safetensors import safe_open
  index=json.loads((CHECKPOINT/'model.safetensors.index.json').read_text())['weight_map']
  prefix=('expert.expert.layers.' if expert else 'vlm.model.language_model.layers.')+str(layer)+'.'
  values={};hashes={}
  for name,wanted in block.state_dict().items():
    full=prefix+name
    with safe_open(CHECKPOINT/index[full],framework='pt',device='cpu') as shard:value=shard.get_tensor(full)
    assert value.shape==wanted.shape and str(value.dtype)=='torch.bfloat16'
    hashes[full]=hashlib.sha256(value.view(torch_cpu().uint16).numpy().tobytes()).hexdigest()
    values[name]=value.float()
  block.load_state_dict(values,strict=True)
  return hashes


def forward(block,config,x,cos,sin,causal=True,context=None):
  torch=torch_cpu()
  from transformers.cache_utils import DynamicCache
  assert x.device.type=='cpu' and cos.device.type==sin.device.type=='cpu'
  cache=DynamicCache(config=config)
  past=0
  if context is not None:
    key,value=context;past=key.shape[2]
    cache.update(key,value,block.self_attn.layer_idx)
  length=x.shape[1]
  mask=torch.zeros((1,1,length,past+length),device='cpu',dtype=x.dtype)
  if causal:
    visible=torch.arange(past+length,device='cpu')[None,:]<=past+torch.arange(length,device='cpu')[:,None]
    mask=mask.masked_fill(~visible,-float('inf'))
  y=block(x,attention_mask=mask,position_embeddings=(cos,sin),past_key_values=cache,use_cache=True)
  layer=cache.layers[block.self_attn.layer_idx]
  return y,layer.keys,layer.values


def recovery_loss(outputs,targets,valid_rows):
  torch=torch_cpu()
  y,k,v=outputs;ty,tk,tv=targets
  assert 0<valid_rows<=y.shape[1]
  # Untuned plumbing loss. Targets are immutable captured data, never a trainable teacher.
  residual=torch.nn.functional.mse_loss(y[:,:valid_rows],ty[:,:valid_rows].detach())
  key=torch.nn.functional.mse_loss(k[:,:,:valid_rows],tk[:,:,:valid_rows].detach())
  value=torch.nn.functional.mse_loss(v[:,:,:valid_rows],tv[:,:,:valid_rows].detach())
  return residual+.1*(key+value),dict(residual=residual,key=key,value=value)


def assert_cpu_only():
  assert not getattr(sys.modules.get('torch.cuda'),'_initialized',False)
  for path in Path('/proc/self/fd').iterdir():
    try:target=str(path.readlink())
    except FileNotFoundError:continue
    assert not target.startswith(('/dev/dri/','/dev/kfd','/dev/nvidia','/dev/bus/usb/'))
