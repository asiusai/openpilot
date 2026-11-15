"""Restore selected tensors from the same BF16 checkpoint for accuracy studies.

This keeps the checkpoint's original values. It does not reproduce NVIDIA's
FP16 or calibrated FP8 deployment recipe.
"""
import hashlib
import json
from pathlib import Path

from tinygrad import dtypes
from tinygrad.nn.state import safe_load
from runtime import upload


def restore(model, checkpoint, vision_layers=0, expert_projections=False):
  assert 0 <= vision_layers <= model.config['vlm_config']['vision_config']['num_hidden_layers']
  assert not model.encode_jits, 'Restore precision before capturing inference graphs'
  prefixes = [f'vlm.model.visual.encoder.layers.{i}.' for i in range(vision_layers)]
  if vision_layers:
    prefixes.append('vlm.model.visual.embeddings.patch_embedding.')
  if expert_projections:
    prefixes.extend(('expert.action_in_proj.', 'expert.action_out_proj.'))
  names = sorted(name for name in model.weight_scales if name.startswith(tuple(prefixes)))
  assert names or not prefixes
  index = json.loads(Path(checkpoint, 'model.safetensors.index.json').read_text())['weight_map']
  shards = {}
  records = []
  for name in names:
    shard = index[name]
    if shard not in shards:
      shards[shard] = safe_load(Path(checkpoint, shard))
    source = shards[shard][name]
    assert source.dtype == dtypes.bfloat16 and source.shape == model.weights[name].shape
    expected = source.bitcast(dtypes.uint8).numpy().tobytes()
    value = upload(source, dtypes.bfloat16)
    actual = value.bitcast(dtypes.uint8).numpy().tobytes()
    assert actual == expected, f'BF16 upload mismatch: {name}'
    model.weights[name] = value
    del model.weight_scales[name]
    records.append({'name': name, 'shape': list(value.shape), 'bytes': len(actual),
                    'sha256': hashlib.sha256(actual).hexdigest(), 'exact_checkpoint_bytes': True})
  report = {'vision_layers': vision_layers, 'expert_projections': expert_projections,
            'dtype': 'bfloat16', 'tensors': records}
  print(f'Restored and verified {len(records)} BF16 tensors, {sum(r["bytes"] for r in records)} bytes', flush=True)
  return report
