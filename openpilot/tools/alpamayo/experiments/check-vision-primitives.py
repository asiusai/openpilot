"""Locate batching errors in the first visual transformer layer, before propagation."""
import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from tinygrad import Tensor
import benchmark


def module(name):
  spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(name+'.py'))
  value = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(value)
  return value


torch.set_num_threads(4)
module('benchmark-fast')
root = Path('/tmp/opencode/alpamayo-asius-20261114')
model = benchmark.EdgeModel(str(root/'model'), precision='fp8')
adapter = module('fast-inputs').FastInputs(str(root/'model'), pixel_scale=float(os.getenv('ALPAMAYO_PIXEL_SCALE','1')))
config, _, prepared = adapter.prepare(benchmark.load_sample(root/'sample-5.npz'))
inputs = module('compact-inputs').CompactInputs(SimpleNamespace(max_tokens=256, vision_batch=1), config, prepared,
                                               adapter.native_inputs(prepared))
pixels, positions, *_ = inputs.upload(np.zeros((1,64,2),dtype=np.float32))
view = module('vision-cache').VisionCache.image_view
pairs = [view(t,a,b) for a,b,_ in inputs.batches for t in (pixels,positions)]
records = []


def first_layer(x, index):
  prefix = 'vlm.model.visual.encoder.layers.0'
  record = {}
  def keep(name, t):
    t = t.realize()
    record[name] = t.float().numpy()
    return t
  x = keep('patch', x)
  h = keep('norm1', model.norm(x, prefix+'.layer_norm1'))
  qx, scale = model.quantize(h)
  qx, scale = keep('quantized1',qx), keep('scale1',scale)
  qkv = [keep(name,model.linear(h,prefix+'.self_attn.'+name+'_proj',(qx,scale))) for name in ('q','k','v')]
  q,k,v = (t.reshape(*h.shape[:2],16,72).transpose(1,2) for t in qkv)
  h = keep('attention', model.attention(q,k,v)).transpose(1,2).reshape(x.shape)
  x = keep('residual1',x+model.linear(h,prefix+'.self_attn.out_proj'))
  h = keep('norm2',model.norm(x,prefix+'.layer_norm2'))
  qx, scale = model.quantize(h)
  qx, scale = keep('quantized2',qx), keep('scale2',scale)
  h = keep('fc1',model.linear(h,prefix+'.mlp.fc1',(qx,scale))).float().gelu().cast(model.dtype)
  x = keep('residual2',x+model.linear(h,prefix+'.mlp.fc2'))
  records.append(record)
  return x


model.vision_layer = first_layer
model.config['vlm_config']['vision_config']['num_hidden_layers'] = 1
for i in range(len(pairs)//2):
  model._vision(*pairs[2*i:2*i+2])
expected = records.copy()
records.clear()
encode = module('batched-vision').install(model)
encode(*pairs)
actual = records[0]
report = []
for name, value in actual.items():
  refs = [r[name] for r in expected]
  if name.startswith('scale'):
    reference = np.repeat(np.array(refs).reshape(-1),value.size//len(refs)).reshape(value.shape)
  else:
    reference = np.concatenate(refs,axis=0)
  assert reference.shape == value.shape,(name,reference.shape,value.shape)
  delta = value-reference
  row = {'stage':name,'exact':bool(np.array_equal(value,reference)),
         'max_abs':float(np.abs(delta).max()),'different':int(np.count_nonzero(delta))}
  report.append(row)
  print(json.dumps(row),flush=True)
Path(os.getenv('ALPAMAYO_PRIMITIVES_OUTPUT','/tmp/opencode/alpamayo-vision-primitives.json')).write_text(json.dumps(report,indent=2))
