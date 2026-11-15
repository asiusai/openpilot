"""Prepare Harald's FP8 scaling and gfx1200 packing once, without GPU compilation."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import torch
from safetensors import safe_open

parser = argparse.ArgumentParser()
parser.add_argument('checkpoint', type=Path)
parser.add_argument('output', type=Path)
args = parser.parse_args()
torch.set_num_threads(4)
args.output.mkdir(parents=True, exist_ok=True)
index = json.loads((args.checkpoint / 'model.safetensors.index.json').read_text())['weight_map']
records = []
digest = hashlib.sha256()
started = time.perf_counter()
offset = 0
with (args.output / 'weights.bin.tmp').open('wb') as output:
  def emit(name, tensor, kind, packed=False):
    global offset
    padding = bytes((-offset) % 256)
    output.write(padding)
    digest.update(padding)
    offset += len(padding)
    raw = tensor.contiguous().view(torch.uint8).reshape(-1).numpy().tobytes()
    records.append(dict(name=name, kind=kind, packed=packed, shape=list(tensor.shape), offset=offset, nbytes=len(raw)))
    output.write(raw)
    digest.update(raw)
    offset += len(raw)

  for shard in sorted(set(index.values())):
    with safe_open(args.checkpoint / shard, framework='pt', device='cpu') as weights:
      for name in weights.keys():
        value = weights.get_tensor(name).to(torch.bfloat16)
        quantized = value.ndim == 2 and all(size % 16 == 0 for size in value.shape) and not name.endswith(
          ('embed_tokens.weight', 'position_embedding.weight'))
        packed = False
        if quantized:
          scale = (value.float().abs().max() / 448).clamp(min=1e-12)
          value = (value.float() / scale).clamp(-448, 448).to(torch.float8_e4m3fn)
          emit(name, scale.reshape(1), 'scale')
          if name.startswith('vlm.') and ('.mlp.' in name or '.self_attn.' in name):
            rows, width = value.shape
            value = value.view(torch.uint8).reshape(-1, 16, width // 16, 4, 4).permute(0, 2, 3, 1, 4).contiguous().reshape(rows, width)
            packed = True
        emit(name, value, 'fp8' if quantized else 'bf16', packed)
    print('Prepared', shard, offset, 'bytes', flush=True)
  output.flush()
  os.fsync(output.fileno())
(args.output / 'weights.bin.tmp').replace(args.output / 'weights.bin')
(args.output / 'index.json').write_text(json.dumps(dict(format=1, sha256=digest.hexdigest(), nbytes=offset, records=records), indent=2))
print('Preparation seconds:', time.perf_counter() - started, flush=True)
