#!/usr/bin/env python3
"""Actual tinygrad CPU proof of bounded FP8/BF16/FP32 cache readback semantics."""
import importlib.util
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace

import numpy as np

if os.environ.get('DEV') != 'CPU':
  raise RuntimeError('This proof requires DEV=CPU before importing tinygrad')
from tinygrad import Tensor, dtypes

EXP = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('runner_reader_cpu', EXP / 'agent-int4model-run.py')
r = importlib.util.module_from_spec(spec)
spec.loader.exec_module(r)
p = r.ptq


def main():
  artifacts = EXP / 'agent-int4model-run-artifacts'
  artifacts.mkdir(exist_ok=True)
  model = SimpleNamespace(weights={}, weight_scales={})
  records, chunks, offset = [], [], 0
  definitions = [('packed-weight', 'fp8', (16, 32), dtypes.fp8e4m3, True),
                 ('row-weight', 'fp8', (7, 32), dtypes.fp8e4m3, False),
                 ('embedding', 'bf16', (8, 8), dtypes.bfloat16, False),
                 ('packed-weight', 'scale', (1,), dtypes.float32, False)]
  for name, kind, shape, dtype, packed in definitions:
    values = (np.arange(np.prod(shape)).reshape(shape) % 31 - 15).astype(np.float32) / 8
    tensor = Tensor(values, device='CPU').cast(dtype).realize()
    raw = tensor.bitcast(dtypes.uint8).numpy().tobytes()
    if packed:
      # Independent ISA/F8-16x16 address map, rather than the runtime pack helper.
      logical = np.frombuffer(raw, np.uint8).reshape(shape)
      encoded = np.zeros(np.prod(shape), np.uint8)
      for row in range(shape[0]):
        for k in range(shape[1]):
          address = (row // 16 * (shape[1] // 16) + k // 16) * 256 + (k % 16 // 4) * 64 + row % 16 * 4 + k % 4
          encoded[address] = logical[row, k]
      raw = encoded.tobytes()
    padding = (-offset) % 256
    chunks.append(bytes(padding)); offset += padding
    records.append(dict(name=name, kind=kind, packed=packed, shape=list(shape), offset=offset, nbytes=len(raw)))
    chunks.append(raw); offset += len(raw)
    (model.weight_scales if kind == 'scale' else model.weights)[name] = tensor
  with tempfile.TemporaryDirectory(prefix='reader-', dir=artifacts) as directory:
    file = Path(directory) / 'weights.bin'
    file.write_bytes(b''.join(chunks))
    proof = r.verify_loaded_records({'records': records}, file, r.gpu_record_reader(model), chunk_bytes=37)
    assert proof['all_bytes_equal'] and len(proof['records']) == 4
    model.weights['packed-weight'] = Tensor.zeros(16, 32, device='CPU', dtype=dtypes.fp8e4m3).realize()
    try:
      r.verify_loaded_records({'records': records}, file, r.gpu_record_reader(model), chunk_bytes=37)
    except ValueError:
      mutation_rejected = True
    else:
      raise AssertionError('Changed actually bound FP8 tensor was not rejected')
  result = dict(cpu_only=True, all_pass=True, chunk_bytes=37, records=4, mutation_rejected=mutation_rejected,
                source_sha256=p.sha_file(EXP / 'agent-int4model-run.py'),
                proof='Actual CPU Tensor readback, packed logical FP8 mapping, raw BF16 bytes, float32 scales; no GPU execution')
  p.json_write(artifacts / 'reader-cpu-check.json', result)
  print(json.dumps(result, indent=2))


if __name__ == '__main__': main()
