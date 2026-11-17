"""Checked selected weight loading; no conversion or training jobs."""
import hashlib
import json
from pathlib import Path
from tinygrad import Device, Tensor, dtypes
from tinygrad.device import Buffer
from tinygrad.nn.state import safe_load
from tinygrad.uop.ops import UOp
from .amd_gemm import pack, unpack
from .frozen import ASSETS, checked, manifest
from .transport import upload


def load_bulk(model, checkpoint, directory):
  checkpoint, directory = Path(checkpoint), Path(directory)
  dev = Device[Device.DEFAULT]
  index = json.loads((directory/'index.json').read_text())
  if index['format'] != 1:
    raise ValueError('Unsupported FP8 weight format')
  name = 'vlm.model.language_model.layers.0.mlp.fc1.weight'
  shard = json.loads((checkpoint/'model.safetensors.index.json').read_text())['weight_map'][name]
  value = upload(safe_load(checkpoint/shard)[name], dtypes.bfloat16)
  scale = (value.float().abs().max()/448).clamp(min_=1e-12).realize()
  value = (value.float()/scale).clamp(-448,448).cast(dtypes.fp8e4m3).realize()
  value = pack(value).realize()
  with (directory/'weights.bin').open('rb') as source:
    for kind, tensor in [('scale', scale), ('fp8', value)]:
      record = next(r for r in index['records'] if r['name'] == name and r['kind'] == kind)
      source.seek(record['offset'])
      if tensor.reshape(-1).bitcast(dtypes.uint8).numpy().tobytes() != source.read(record['nbytes']):
        raise ValueError(f'Target FP8 conversion/packing differs: {kind}')
  del value, scale
  storage = Buffer(Device.DEFAULT,index['nbytes']).ensure_allocated()
  digest = hashlib.sha256()
  with (directory/'weights.bin').open('rb') as source:
    offset = 0
    while chunk := source.read(16*1024*1024):
      raw = memoryview(bytearray(chunk))
      target = storage.view(len(raw),offset).ensure_allocated()
      target.copy_from(Buffer('NPY',len(raw),opaque=raw))
      dev.synchronize()
      actual = target.as_memoryview()
      if actual != raw:
        raise ValueError('GPU weight upload differs')
      digest.update(actual)
      offset += len(raw)
  if offset != index['nbytes'] or digest.hexdigest() != index['sha256']:
    raise ValueError('FP8 weight content differs')
  for record in index['records']:
    kind = record['kind']
    dtype = {'scale':dtypes.float32,'fp8':dtypes.fp8e4m3,'bf16':dtypes.bfloat16}[kind]
    view = storage.view(record['nbytes'],record['offset']).ensure_allocated()
    value = Tensor(UOp.from_buffer(view,dtype)).reshape(record['shape'])
    if record['packed']:
      value = unpack(value)
    (model.weight_scales if kind == 'scale' else model.weights)[record['name']] = value
  model._bulk_storage = storage
  return index['sha256']


def restore_bf16(model, checkpoint):
  prefixes = ['vlm.model.visual.embeddings.patch_embedding.',
              'vlm.model.visual.encoder.layers.0.', 'vlm.model.visual.encoder.layers.1.',
              'expert.action_in_proj.', 'expert.action_out_proj.']
  names = sorted(name for name in model.weight_scales if name.startswith(tuple(prefixes)))
  if not names:
    raise ValueError('Missing mixed-precision restoration tensors')
  index = json.loads((Path(checkpoint)/'model.safetensors.index.json').read_text())['weight_map']
  shards = {}
  for name in names:
    shard = index[name]
    if shard not in shards:
      shards[shard] = safe_load(Path(checkpoint)/shard)
    source = shards[shard][name]
    assert source.dtype == dtypes.bfloat16 and source.shape == model.weights[name].shape
    expected = source.bitcast(dtypes.uint8).numpy().tobytes()
    value = upload(source,dtypes.bfloat16)
    if value.bitcast(dtypes.uint8).numpy().tobytes() != expected:
      raise ValueError(f'BF16 restoration differs: {name}')
    model.weights[name] = value
    del model.weight_scales[name]
  return names


def reduce_head(model):
  raw = checked(ASSETS/'coc_vocab.json',manifest()['vocabulary_sha256'])
  vocab = sorted(set(json.loads(raw)))
  config = model.config
  def allowed(i):
    return (not config['traj_ids']['history_id0'] <= i < config['traj_ids']['history_id0']+config['history_vocab_size']
            and not config['traj_ids']['future_id0'] <= i < config['traj_ids']['future_id0']+config['future_vocab_size']
            and i != config['vlm_config']['text_config']['eos_token_id'])
  vocab = [i for i in vocab if allowed(i)]
  assert config['traj_ids']['future_start'] in vocab
  assert vocab and min(vocab) >= 0 and max(vocab) < model.weights['vlm.lm_head.weight'].shape[0]
  padded = vocab+[vocab[-1]]*((-len(vocab))%64)
  model.output_token_mapping = Tensor(padded,dtype=dtypes.int32).realize()
  model.weights['vlm.lm_head.weight'] = model.weights['vlm.lm_head.weight'][model.output_token_mapping].contiguous().realize()
  return len(vocab)
