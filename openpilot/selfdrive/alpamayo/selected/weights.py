"""Checked selected weight loading; no conversion or training jobs."""
import hashlib
import json
import os
from functools import lru_cache
from pathlib import Path
from tinygrad import Device, Tensor, dtypes
from tinygrad.device import Buffer
from tinygrad.nn.state import safe_load
from tinygrad.uop.ops import KernelInfo, Ops, UOp
from .amd_gemm import pack, unpack
from .frozen import ASSETS, checked, manifest
from .transport import upload


_LEAF = 4096
_LANES = 64


@lru_cache(maxsize=1)
def _hash_kernel():
  stem = ASSETS/'gfx1200/weight_sha256'
  source = checked(stem.with_suffix('.cpp'), '32ae4eb358f9093bdc756e6787d148bc6440ce63438df71b6576a3862a35f354')
  binary = checked(stem.with_suffix('.hsaco'), '5122be76107cff16c3e3f5821079400cbd516e7e378a8813f5278e36fd4c4518')
  return source.decode(), binary


def _hash_program(out, data, nonce):
  source, binary = _hash_kernel()
  groups, leftover = divmod(data.numel()*4, _LEAF*_LANES)
  if (not groups or leftover or out.dtype != data.dtype or data.dtype != nonce.dtype or data.dtype != dtypes.uint32
      or out.numel() != groups*_LANES*8 or nonce.numel() != 4):
    raise ValueError('Invalid weight digest kernel buffers')
  sink = UOp.sink(out[0].store(data[0].load() ^ nonce[0].load()), UOp.special(groups, 'gidx0'),
                  UOp.special(_LANES, 'lidx0'), arg=KernelInfo(name='weight_sha256'))
  return UOp(Ops.PROGRAM, src=(sink, UOp(Ops.LINEAR, src=tuple(sink.toposort())),
                               UOp(Ops.SOURCE, arg=source), UOp(Ops.BINARY, arg=binary)))


def _check_destination(target, raw):
  size = len(raw) - len(raw)%(_LEAF*_LANES)
  if size:
    if target.device.split(':')[0] != 'AMD' or Device[target.device].arch != 'gfx1200':
      raise ValueError('Weight digest kernel requires gfx1200')
    # The expected proof stays on the host. A stale digest read cannot answer a fresh challenge.
    nonce = os.urandom(16)
    challenge = Tensor(nonce, device='CPU').bitcast(dtypes.uint32).to(target.device).realize()
    data = Tensor(UOp.from_buffer(target.view(size, 0).ensure_allocated(), dtypes.uint32))
    out = Tensor.empty(size//_LEAF*8, device=target.device, dtype=dtypes.uint32)
    proof = Tensor.custom_kernel(out, data, challenge, fxn=_hash_program)[0].realize()
    expected = bytearray()
    for i in range(0, size, _LEAF):
      h = hashlib.sha256(raw[i:i+_LEAF])
      h.update(nonce)
      expected.extend(h.digest())
    Device[target.device].synchronize()
    if proof._buffer().as_memoryview() != expected:
      raise ValueError(f'GPU weight upload digest differs at chunk offset {target.offset}')
  if size < len(raw) and target.view(len(raw)-size, size).ensure_allocated().as_memoryview() != raw[size:]:
    raise ValueError(f'GPU weight upload tail differs at offset {target.offset+size}')


def _upload_bulk(source, nbytes, expected):
  if nbytes <= 0 or os.fstat(source.fileno()).st_size != nbytes:
    raise ValueError('FP8 weight size differs')
  storage = Buffer(Device.DEFAULT, nbytes).ensure_allocated()
  staging = memoryview(bytearray(min(16*1024*1024, nbytes)))
  digest, offset = hashlib.sha256(), 0
  while count := source.readinto(staging):
    if offset + count > nbytes:
      raise ValueError('FP8 weight size differs')
    raw = staging[:count]
    target = storage.view(count, offset).ensure_allocated()
    target.copy_from(Buffer('NPY', count, opaque=raw))
    Device[target.device].synchronize()
    _check_destination(target, raw)
    digest.update(raw)
    offset += count
  if offset != nbytes or digest.hexdigest() != expected:
    raise ValueError('FP8 weight content differs')
  return storage


def load_bulk(model, checkpoint, directory):
  checkpoint, directory = Path(checkpoint), Path(directory)
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
  with (directory/'weights.bin').open('rb') as source:
    storage = _upload_bulk(source, index['nbytes'], index['sha256'])
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
