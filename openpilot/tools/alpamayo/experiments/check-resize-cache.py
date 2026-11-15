"""Verify resize reuse on advancing inputs and changed pixels with reused timestamps."""
import importlib.util
import json
from pathlib import Path
import time
import numpy as np
import torch

spec = importlib.util.spec_from_file_location('fast', Path(__file__).with_name('fast-inputs.py'))
fast = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fast)
torch.set_num_threads(4)
checkpoint = '/tmp/opencode/alpamayo-asius-20261114/model'
reference = fast.FastInputs(checkpoint, pixel_scale=.25)
cached = fast.FastInputs(checkpoint, pixel_scale=.25, cache_resize=True)
results = []
paths = sorted(Path('/dev/shm/alpamayo-rolling').glob('sample-*'), key=lambda p: float(p.name[7:]))
for path in paths:
  data = {p.stem: [str(np.load(p).item())] if p.stem == 'nav_text' else torch.from_numpy(np.load(p)) for p in path.glob('*.npy')}
  start = time.perf_counter()
  _, _, expected = reference.prepare(data)
  mid = time.perf_counter()
  _, _, actual = cached.prepare(data)
  end = time.perf_counter()
  for key in expected['tokenized_data']:
    torch.testing.assert_close(actual['tokenized_data'][key], expected['tokenized_data'][key], rtol=0, atol=0)
  results.append({'sample': str(path), 'reference_ms': (mid-start)*1000, 'cached_ms': (end-mid)*1000})
# Same timestamps and shape must never cause reuse of different pixels.
data['image_frames'].fill_(128)
_, _, expected = reference.prepare(data)
_, _, actual = cached.prepare(data)
for key in expected['tokenized_data']:
  torch.testing.assert_close(actual['tokenized_data'][key], expected['tokenized_data'][key], rtol=0, atol=0)
report = {'advancing_inputs': len(results), 'changed_pixels_checked': True, 'exact': True, 'runs': results}
Path('/tmp/opencode/alpamayo-resize-cache-check.json').write_text(json.dumps(report, indent=2))
print('Exact:', len(results), 'advancing inputs plus changed pixels; median milliseconds',
      np.median([r['reference_ms'] for r in results]), np.median([r['cached_ms'] for r in results]))
