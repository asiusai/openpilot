"""Measure full draft verification and compare every output logit across tiles."""
import argparse
import importlib.util
import json
from pathlib import Path
import time
from types import SimpleNamespace
import numpy as np
import torch
from tinygrad import Device, Tensor, TinyJit
from tinygrad.device import Compiled
from tinygrad.uop.ops import UOp
import benchmark
from runtime import download


def module(name):
  spec = importlib.util.spec_from_file_location(name.replace('-', '_'), Path(__file__).with_name(name+'.py'))
  value = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(value)
  return value


parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint', required=True)
parser.add_argument('--sample', required=True)
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--counts', nargs='+', type=int, default=(16,32,48))
parser.add_argument('--tiles', nargs='+', default=('4x4','1x4','2x4','1x8','2x8','1x2'))
args = parser.parse_args()
torch.set_num_threads(4)
Compiled.wait_timeout_ms = 300000
module('wgp_compiler').install(Device[Device.DEFAULT])
module('benchmark-fast')
model = benchmark.EdgeModel(args.checkpoint, precision='fp8')
module('fast-quantize').install(model, materialize=True)
module('prefill-kernels').fix_dispatch_packet()
module('online-attention').install(model)
module('fused-norm').install(model)
module('reduced-vocab').install(model, Path('/tmp/opencode/alpamayo-coc-vocab.json'))
adapter = module('fast-inputs').FastInputs(args.checkpoint, pixel_scale=.5)
data = {p.stem: [str(np.load(p, allow_pickle=False).item())] if p.stem == 'nav_text'
        else torch.from_numpy(np.load(p, allow_pickle=False)) for p in Path(args.sample).glob('*.npy')}
config, tokenizer, prepared = adapter.prepare(data)
settings = SimpleNamespace(max_tokens=256, diffusion_steps=3, vision_batch=1, greedy=True, byte_pixels=True)
inputs = module('compact-inputs').CompactInputs(settings, config, prepared, adapter.native_inputs(prepared))
pixels, vpos, ids, cos, sin, dc, ds, ac, ass, actions = inputs.upload(np.zeros((1,64,2)))
embedding = model.encode(pixels, vpos, ids, inputs.batches, inputs.spans)
model.prefill(embedding, cos, sin, 256)
model.synchronize()
tokens = tokenizer.encode('Wait at the stop line for the red traffic light to turn green before proceeding straight through the intersection to follow our route.')
results = []
for count in args.counts:
  ids_draft = Tensor(np.asarray((tokens*4)[:count],dtype=np.int32),device='CPU').to(Device.DEFAULT).realize()
  position = UOp.variable('draft_position',0,model.cache[0].shape[3]-count).bind(ids.shape[1])
  step = UOp.variable('draft_step',0,dc.shape[2]-count).bind(0)
  reference = None
  for tile in [tuple(map(int, value.split('x'))) for value in args.tiles]:
    decoder = module('draft-decode').DraftDecoder(model)
    decoder.fast_attention = True
    decoder.tile = tile
    jit = TinyJit(decoder.verify)
    times = []
    for iteration in range(7):
      start = time.perf_counter()
      logits, choices = jit(ids_draft,dc,ds,position,step)
      Tensor.realize(logits,choices)
      model.synchronize()
      if iteration >= 2:
        times.append((time.perf_counter()-start)*1000)
    values = download(logits)
    if reference is None:
      reference = values.copy()
    np.testing.assert_array_equal(values,reference)
    result = dict(count=count,tile=tile,median_ms=float(np.median(times)),exact=True)
    print(json.dumps(result),flush=True)
    results.append(result)
    args.output.write_text(json.dumps(results,indent=2)+'\n')
