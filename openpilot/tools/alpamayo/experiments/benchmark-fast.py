"""Bulk-load prepared weights, then run Harald's unchanged benchmark and kernels."""
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time

from tinygrad import Device, Tensor, dtypes
from tinygrad.device import Buffer
from tinygrad.uop.ops import UOp
from tinygrad.nn.state import safe_load
import benchmark
from model import EdgeModel
from amd_gemm import pack, unpack
from runtime import upload

original_init = EdgeModel.__init__

def load(self, checkpoint, dtype=dtypes.bfloat16, precision='bf16'):
  assert precision == 'fp8' and dtype == dtypes.bfloat16
  dev = Device[Device.DEFAULT]
  assert dev.arch in ('gfx1200', 'gfx1201'), dev.arch
  directory = Path(os.environ['ALPAMAYO_FP8_CACHE'])
  index = json.loads((directory / 'index.json').read_text())
  assert index['format'] == 1
  # Check conversion on the target architecture, including FP8 subnormal behavior.
  name = 'vlm.model.language_model.layers.0.mlp.fc1.weight'
  shard = json.loads(Path(checkpoint, 'model.safetensors.index.json').read_text())['weight_map'][name]
  value = upload(safe_load(Path(checkpoint, shard))[name], dtype)
  scale = (value.float().abs().max() / 448).clamp(min_=1e-12).realize()
  value = (value.float() / scale).clamp(-448, 448).cast(dtypes.fp8e4m3).realize()
  value = pack(value).realize()
  with (directory / 'weights.bin').open('rb') as source:
    for kind, tensor in [('scale', scale), ('fp8', value)]:
      record = next(r for r in index['records'] if r['name'] == name and r['kind'] == kind)
      source.seek(record['offset'])
      assert tensor.reshape(-1).bitcast(dtypes.uint8).numpy().tobytes() == source.read(record['nbytes']), f'Target FP8 parity failed: {kind}'
  del value, scale
  print('Target FP8 conversion and packing parity passed', flush=True)
  # Let the original initializer establish every inference field without loading shards.
  with tempfile.TemporaryDirectory() as empty:
    Path(empty, 'config.json').write_bytes(Path(checkpoint, 'config.json').read_bytes())
    Path(empty, 'model.safetensors.index.json').write_text('{"weight_map": {}}')
    original_init(self, empty, dtype, precision)
  started = time.perf_counter()
  storage = Buffer(Device.DEFAULT, index['nbytes']).ensure_allocated()
  digest = hashlib.sha256()
  with (directory / 'weights.bin').open('rb') as source:
    offset = 0
    while chunk := source.read(16 * 1024 * 1024):
      raw = memoryview(bytearray(chunk))
      target = storage.view(len(raw), offset).ensure_allocated()
      target.copy_from(Buffer('NPY', len(raw), opaque=raw))
      dev.synchronize()
      actual = target.as_memoryview()
      assert actual == raw, f'GPU upload mismatch at {offset}'
      digest.update(actual)
      offset += len(raw)
  assert offset == index['nbytes'] and digest.hexdigest() == index['sha256']
  for record in index['records']:
    kind = record['kind']
    dt = {'scale': dtypes.float32, 'fp8': dtypes.fp8e4m3, 'bf16': dtypes.bfloat16}[kind]
    view = storage.view(record['nbytes'], record['offset']).ensure_allocated()
    value = Tensor(UOp.from_buffer(view, dt)).reshape(record['shape'])
    if record['packed']:
      value = unpack(value)
    (self.weight_scales if kind == 'scale' else self.weights)[record['name']] = value
  self._bulk_storage = storage
  print(f'Bulk weights verified: {offset} bytes in {time.perf_counter() - started:.2f}s', flush=True)

EdgeModel.__init__ = load
if __name__ == '__main__':
  benchmark.main()
