"""Find the first numerical difference between batched and independent image encoding."""
import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace, MethodType
import numpy as np
import torch
from tinygrad.device import Compiled
import benchmark


def module(name):
  spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(name+'.py'))
  value = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(value)
  return value


torch.set_num_threads(4)
Compiled.wait_timeout_ms = 300000
module('benchmark-fast')
if os.environ.get('ALPAMAYO_LLVM_WGP') == '1':
  from tinygrad import Device
  import wgp_compiler
  wgp_compiler.install(Device[Device.DEFAULT])
root = Path('/tmp/opencode/alpamayo-asius-20261114')
model = benchmark.EdgeModel(str(root/'model'), precision='fp8')
module('expert-kernels').install(model)
if os.environ.get('ALPAMAYO_ONLINE_ATTENTION') == '1':
  module('prefill-kernels').fix_dispatch_packet()
  module('online-attention').install(model)
adapter = module('fast-inputs').FastInputs(str(root/'model'), pixel_scale=float(os.environ.get('ALPAMAYO_PIXEL_SCALE','1')))
config, tokenizer, prepared = adapter.prepare(benchmark.load_sample(root/'sample-5.npz'))
packed = adapter.native_inputs(prepared)
inputs = module('compact-inputs').CompactInputs(SimpleNamespace(max_tokens=256, vision_batch=1), config, prepared, packed)
pixels, positions, *_ = inputs.upload(np.zeros((1,64,2), dtype=np.float32))
count = int(os.environ.get('ALPAMAYO_VISION_IMAGES', '2'))
view = module('vision-cache').VisionCache.image_view
pairs = [view(t,a,b) if os.environ.get('ALPAMAYO_VISION_VIEWS') == '1' else t[a:b].unsqueeze(0).contiguous().realize()
         for a,b,_ in inputs.batches[:count] for t in (pixels, positions)]
record = []
original = model.vision_layer


def layer(self, value, index):
  if index == 0:
    record.append(value.float().numpy())
  result = original(value, index)
  record.append(result.float().numpy())
  return result


model.vision_layer = MethodType(layer, model)
references = []
for i in range(count):
  record = []
  result = model._vision(*pairs[2*i:2*i+2])
  record.append(result.float().numpy()[None])
  references.append(record)
encode = module('batched-vision').install(model)
record = []
results = encode(*pairs)
record.append(np.stack([r.float().numpy() for r in results]))
report = []
for i, actual in enumerate(record):
  expected = np.concatenate([r[i] for r in references], axis=0)
  assert expected.shape == actual.shape, (i, expected.shape, actual.shape)
  delta = actual - expected
  row = {'stage': 'patch' if i == 0 else 'projector' if i == len(record)-1 else f'layer_{i-1}',
         'exact': bool(np.array_equal(actual, expected)), 'max_abs': float(np.abs(delta).max()),
         'rmse': float(np.sqrt(np.mean(delta**2)))}
  report.append(row)
  print(json.dumps(row), flush=True)
Path(os.environ.get('ALPAMAYO_BATCH_CHECK_OUTPUT','/tmp/opencode/alpamayo-batched-layers.json')).write_text(json.dumps(report, indent=2))
assert all(row['exact'] for row in report), 'Batched vision differs from independent image encoding'
