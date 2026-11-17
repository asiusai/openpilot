"""Optional language-prefill fc2 INT4 PTQ interfaces. Importing this file uses CPU only.

Initial contract: unsigned A4 [0,15], signed W4 [-7,7], no clipping/equalization,
FP32 per-token A scales (max/15, zero row uses 1), scalar/per-output W scales, output BF16.
GPU calls exist only in explicitly invoked collection/runtime adapters.
"""
from contextlib import contextmanager
import hashlib
import json
import math
from pathlib import Path
import re
import struct
from types import MethodType

import numpy as np

SCHEMA = 'agent-int4model-v1'
LAYOUT = 'rdna4-k32-row16-ktile-lane32-reg2'
FC2 = re.compile(r'^vlm\.model\.language_model\.layers\.(\d+)\.mlp\.fc2$')
INPUT_FILES = ('absolute_timestamps.npy', 'camera_indices.npy', 'ego_history_rot.npy',
               'ego_history_xyz.npy', 'image_frames.npy', 'nav_text.npy')
PREPARED_KEYS = ('input_ids', 'position_ids', 'pixel_values', 'ego_history_xyz', 'ego_history_rot',
                 'absolute_timestamps', 'camera_indices', 'nav_text')


def canonical(value):
  return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def sha_file(path):
  digest = hashlib.sha256()
  with Path(path).open('rb') as stream:
    while block := stream.read(8 * 1024 * 1024):
      digest.update(block)
  return digest.hexdigest()


def json_write(path, value):
  path = Path(path)
  temporary = path.with_suffix(path.suffix + '.pending')
  temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')
  temporary.replace(path)


def array_identity(values):
  values = np.asarray(values)
  if values.dtype.hasobject:
    raise ValueError('Object arrays cannot establish input identity')
  values = np.ascontiguousarray(values)
  return dict(shape=list(values.shape), dtype=values.dtype.str,
              sha256=hashlib.sha256(values.tobytes()).hexdigest())


def prepared_identity(arrays):
  if not set(PREPARED_KEYS).issubset(arrays):
    raise ValueError(f'Prepared input identity requires {PREPARED_KEYS}')
  selected = {name: array_identity(arrays[name]) for name in PREPARED_KEYS}
  return dict(arrays=selected, sha256=hashlib.sha256(canonical(selected)).hexdigest())


def checkpoint_headers(checkpoint):
  checkpoint = Path(checkpoint)
  index = json.loads((checkpoint / 'model.safetensors.index.json').read_text())['weight_map']
  headers = {}
  for shard in sorted(set(index.values())):
    path = checkpoint / shard
    if path.parent.resolve() != checkpoint.resolve():
      raise ValueError('Checkpoint shard escapes its directory')
    with path.open('rb') as stream:
      count = struct.unpack('<Q', stream.read(8))[0]
      header = json.loads(stream.read(count))
    for name, info in header.items():
      if name != '__metadata__':
        if name in headers or index.get(name) != shard:
          raise ValueError('Checkpoint header/index mismatch')
        headers[name] = dict(info, shard=shard, payload_offset=count + 8)
  if set(headers) != set(index):
    raise ValueError('Incomplete checkpoint index')
  return headers


def checkpoint_identity(checkpoint):
  checkpoint = Path(checkpoint).resolve()
  headers = checkpoint_headers(checkpoint)
  config = json.loads((checkpoint / 'config.json').read_text())
  result = dict(path=str(checkpoint), config=config,
                canonical_config_sha256=hashlib.sha256(canonical(config)).hexdigest(),
                files={name: dict(sha256=sha_file(checkpoint / name), bytes=(checkpoint / name).stat().st_size)
                       for name in ('config.json', 'model.safetensors.index.json',
                                    *sorted({h['shard'] for h in headers.values()}))})
  result['identity_sha256'] = hashlib.sha256(canonical(result)).hexdigest()
  return result


def reference_identity(checkpoint, fp8_cache, source_files, settings, calibration_files=()):
  """Hash the on-disk FP8 reference and original BF16 source; do not read GPU weights."""
  fp8_cache = Path(fp8_cache)
  metadata = json.loads((fp8_cache / 'index.json').read_text())
  cache_sha = sha_file(fp8_cache / 'weights.bin')
  if cache_sha != metadata['sha256'] or (fp8_cache / 'weights.bin').stat().st_size != metadata['nbytes']:
    raise ValueError('FP8 cache bytes do not match its index')
  result = dict(schema=SCHEMA, checkpoint=checkpoint_identity(checkpoint),
                fp8_cache=dict(path=str(fp8_cache.resolve()), index_sha256=sha_file(fp8_cache / 'index.json'),
                               weights_sha256=cache_sha, bytes=metadata['nbytes']),
                sources={str(Path(p).resolve()): sha_file(p) for p in source_files},
                calibration_files={str(Path(p).resolve()): sha_file(p) for p in calibration_files},
                settings=settings, gpu_weight_bytes_verified=False,
                note='Disk cache hashes bind the reference source. Parent must verify actual loaded GPU state.')
  result['identity_sha256'] = hashlib.sha256(canonical(result)).hexdigest()
  return result


def decode_bf16(words):
  words = np.asarray(words, dtype=np.uint16)
  return (words.astype(np.uint32) << 16).view(np.float32)


def encode_bf16_exact(values):
  values = np.asarray(values, dtype=np.float32)
  bits = values.view(np.uint32)
  if not np.isfinite(values).all() or np.any(bits & 0xffff):
    raise ValueError('Expected finite original BF16 values, widened exactly to float32')
  return (bits >> 16).astype(np.uint16)


def round_bf16(values):
  values = np.asarray(values, dtype=np.float32)
  bits = values.view(np.uint32)
  return decode_bf16(((bits + np.uint32(0x7fff) + ((bits >> 16) & 1)) >> 16).astype(np.uint16))


def read_bf16(checkpoint, info):
  if info['dtype'] != 'BF16':
    raise ValueError('Export requires original BF16 masters')
  start, end = info['data_offsets']
  if end - start != math.prod(info['shape']) * 2:
    raise ValueError('Invalid BF16 tensor byte extent')
  words = np.memmap(Path(checkpoint) / info['shard'], mode='r', dtype='<u2',
                    offset=info['payload_offset'] + start, shape=tuple(info['shape']))
  return words


def quantize_weight(weight, mode='peroutput'):
  weight = np.asarray(weight, dtype=np.float32)
  if weight.ndim != 2 or not np.isfinite(weight).all() or mode not in ('scalar', 'peroutput'):
    raise ValueError('Invalid weight quantization input')
  maximum = np.max(np.abs(weight), axis=1) if mode == 'peroutput' else np.array([np.max(np.abs(weight))])
  scale = np.maximum(maximum / np.float32(7), np.float32(1e-12)).astype(np.float32)
  divisor = scale[:, None] if mode == 'peroutput' else scale[0]
  codes = np.clip(np.rint(weight / divisor), -7, 7).astype(np.int8)
  return codes, scale


def quantize_activation(values):
  """CPU reference only: independent per-token scale; never fit from route outputs."""
  values = np.asarray(values, dtype=np.float32)
  if values.ndim != 2 or not np.isfinite(values).all() or np.any(values < 0):
    raise ValueError('Unsigned A4 requires finite, nonnegative post-ReLU² input')
  maximum = values.max(axis=1)
  scale = np.where(maximum == 0, np.float32(1), maximum / np.float32(15)).astype(np.float32)
  codes = np.clip(np.rint(values / scale[:, None]), 0, 15).astype(np.uint8)
  return codes, scale


def pack_codes(codes, signed):
  """[rows,K] -> [ceil(rows/16), K/32, 32 lanes, 2 int32 registers]."""
  codes = np.asarray(codes)
  if codes.ndim != 2 or min(codes.shape) <= 0 or codes.shape[1] % 32:
    raise ValueError('Expected nonempty K32-aligned matrix')
  if not np.issubdtype(codes.dtype, np.integer) or np.any(codes < (-8 if signed else 0)) or np.any(codes > (7 if signed else 15)):
    raise ValueError('Code outside native nibble domain')
  rows, width = codes.shape
  padded = np.pad(codes, ((0, (-rows) % 16), (0, 0)))
  fragments = padded.reshape(-1, 16, width // 32, 2, 2, 8).transpose(0, 2, 3, 1, 4, 5)
  bits = fragments.reshape(-1, width // 32, 32, 2, 8).astype(np.uint32) & 15
  return np.bitwise_or.reduce(bits << np.arange(0, 32, 4, dtype=np.uint32), axis=-1).view(np.int32)


def unpack_codes(packed, rows, signed):
  packed = np.asarray(packed)
  if packed.dtype != np.int32 or packed.ndim != 4 or packed.shape[2:] != (32, 2) or packed.shape[0] != (rows + 15) // 16:
    raise ValueError('Invalid native packed layout')
  bits = ((packed.view(np.uint32)[..., None] >> np.arange(0, 32, 4, dtype=np.uint32)) & 15)
  values = bits.reshape(packed.shape[0], packed.shape[1], 2, 16, 2, 8)
  values = values.transpose(0, 3, 1, 2, 4, 5).reshape(-1, packed.shape[1] * 32)[:rows].astype(np.int16)
  return np.where(values >= 8, values - 16, values).astype(np.int8) if signed else values.astype(np.uint8)


def scaled_reference(a_codes, w_codes, activation_scale, weight_scale):
  """Match original FP8 association: float32(dot) * (sA * sW), then BF16."""
  dot64 = a_codes.astype(np.int64) @ w_codes.astype(np.int64).T
  if np.max(np.abs(dot64), initial=0) > np.iinfo(np.int32).max:
    raise OverflowError('INT32 dot overflow')
  activation_scale, weight_scale = np.asarray(activation_scale), np.asarray(weight_scale)
  if activation_scale.dtype != np.float32 or weight_scale.dtype != np.float32:
    raise ValueError('Scale contract requires float32')
  product = activation_scale.reshape(-1, 1) * weight_scale.reshape(1, -1)
  return round_bf16(dot64.astype(np.int32).astype(np.float32) * product)


def export_bundle(checkpoint, output, mode='peroutput', layers=None):
  output = Path(output)
  output.mkdir(parents=True, exist_ok=False)
  headers = checkpoint_headers(checkpoint)
  identity = checkpoint_identity(checkpoint)
  config = identity['config']['vlm_config']['text_config']
  if config['hidden_act'] != 'relu2' or config.get('mlp_bias', False):
    raise ValueError('Initial hook requires bias-free ReLU² MLPs')
  layers = list(range(config['num_hidden_layers'])) if layers is None else sorted(set(layers))
  if not layers or any(i < 0 or i >= config['num_hidden_layers'] for i in layers):
    raise ValueError('Invalid layer selection')
  manifest = dict(schema=SCHEMA, complete=False, layout=LAYOUT, checkpoint=identity, mode=mode,
                  code_range=[-7, 7], activation_range=[0, 15], activation_scale='per-token max/15; zero row uses 1; preserve subnormals',
                  rounding='nearest-even', weight_scale='max(abs(master))/7; floor 1e-12',
                  master='original BF16 safetensors; never FP8 cache', equalization=None, clipping=None,
                  outlier_correction=None, collector_data_used=False, records=[],
                  exporter_sha256=sha_file(__file__))
  json_write(output / 'manifest.json', manifest)
  for layer in layers:
    name = f'vlm.model.language_model.layers.{layer}.mlp.fc2'
    info = headers[name + '.weight']
    words = read_bf16(checkpoint, info)
    raw_sha = hashlib.sha256(memoryview(words).cast('B')).hexdigest()
    weight = decode_bf16(words)
    codes, scale = quantize_weight(weight, mode)
    packed = pack_codes(codes, True)
    divisor = scale[:, None] if mode == 'peroutput' else scale[0]
    delta = codes.astype(np.float32) * divisor - weight
    relative = float(np.linalg.norm(delta.astype(np.float64)) / max(np.linalg.norm(weight.astype(np.float64)), 1e-12))
    packed_name, scale_name = f'layer-{layer:02d}-w4.npy', f'layer-{layer:02d}-scale.npy'
    np.save(output / packed_name, packed, allow_pickle=False)
    np.save(output / scale_name, scale, allow_pickle=False)
    manifest['records'].append(dict(name=name, shape=info['shape'], master_dtype=info['dtype'],
      master_shard=info['shard'], master_data_offsets=info['data_offsets'], master_bytes_sha256=raw_sha,
      packed=dict(file=packed_name, file_sha256=sha_file(output / packed_name), **array_identity(packed)),
      scale=dict(file=scale_name, file_sha256=sha_file(output / scale_name), **array_identity(scale)),
      weight_relative_rmse=relative, zero_codes=int(np.count_nonzero(codes == 0)),
      min_code=int(codes.min()), max_code=int(codes.max())))
    # The file hash covers the .npy header; array hash covers logical payload separately.
    json_write(output / 'manifest.json', manifest)
    del words, weight, codes, packed, delta
  manifest['complete'] = True
  manifest['identity_sha256'] = hashlib.sha256(canonical(manifest)).hexdigest()
  json_write(output / 'manifest.json', manifest)
  return manifest


def load_bundle(path, expected_checkpoint=None):
  path = Path(path)
  manifest = json.loads((path / 'manifest.json').read_text())
  claimed = manifest.pop('identity_sha256', None)
  if not manifest.get('complete') or manifest.get('schema') != SCHEMA or manifest.get('layout') != LAYOUT:
    raise ValueError('Incomplete or unsupported INT4 bundle')
  if hashlib.sha256(canonical(manifest)).hexdigest() != claimed:
    raise ValueError('Bundle manifest hash mismatch')
  manifest['identity_sha256'] = claimed
  if expected_checkpoint is not None and expected_checkpoint != manifest['checkpoint']['identity_sha256']:
    raise ValueError('Bundle checkpoint identity mismatch')
  result = {}
  for record in manifest['records']:
    name = record['name']
    if not FC2.fullmatch(name) or name in result:
      raise ValueError('Bundle scope or duplicate-name violation')
    arrays = {}
    for kind in ('packed', 'scale'):
      info = record[kind]
      file = path / info['file']
      if file.resolve().parent != path.resolve() or sha_file(file) != info['file_sha256']:
        raise ValueError('Bundle file identity mismatch')
      arrays[kind] = np.load(file, allow_pickle=False, mmap_mode='r')
      if array_identity(arrays[kind]) != {k: info[k] for k in ('shape', 'dtype', 'sha256')}:
        raise ValueError('Bundle array identity mismatch')
    n, k = record['shape']
    if (arrays['packed'].dtype != np.int32 or arrays['packed'].shape != ((n + 15) // 16, k // 32, 32, 2)
        or arrays['scale'].dtype != np.float32 or arrays['scale'].shape != ((n,) if manifest['mode'] == 'peroutput' else (1,))
        or not np.isfinite(arrays['scale']).all() or np.any(arrays['scale'] <= 0)):
      raise ValueError('Bundle tensor contract mismatch')
    result[name] = dict(**arrays, shape=(n, k))
  return manifest, result


class CalibrationSelection:
  """Only explicitly permitted calibration inputs may enter the collector."""
  def __init__(self, calibration_list, reserved_list, max_samples=32):
    self.calibration_list, self.reserved_list = Path(calibration_list), Path(reserved_list)
    allowed = [str(Path(p).resolve()) for p in json.loads(self.calibration_list.read_text())]
    reserved = [str(Path(p).resolve()) for p in json.loads(self.reserved_list.read_text())]
    if not 1 <= len(allowed) <= max_samples or len(set(allowed)) != len(allowed) or set(allowed) & set(reserved):
      raise ValueError('Calibration count, duplicates, or reserved-split overlap')
    self.allowed, self.reserved = allowed, set(reserved)

  def identity(self, sample):
    sample = str(Path(sample).resolve())
    if sample not in self.allowed or sample in self.reserved:
      raise ValueError('Sample is outside the explicit calibration allowlist')
    files = {name: dict(sha256=sha_file(Path(sample) / name), bytes=(Path(sample) / name).stat().st_size)
             for name in INPUT_FILES}
    return dict(path=sample, files=files, sha256=hashlib.sha256(canonical(files)).hexdigest())

  def provenance(self):
    return dict(calibration_list_sha256=sha_file(self.calibration_list),
                reserved_list_sha256=sha_file(self.reserved_list), allowed=self.allowed,
                count=len(self.allowed), reserved_inputs_read=False,
                note='Reuses prior FP8-calibration route windows. This route is not a blind test.')


def representative_rows(valid_rows, limit):
  if not 1 <= limit <= 64 or valid_rows <= 0:
    raise ValueError('Invalid row sampling bound')
  count = min(valid_rows, limit)
  return (np.arange(count, dtype=np.int64) * (valid_rows - 1) // max(count - 1, 1)).astype(np.int32)


def capture_numpy(values, valid_rows, indices):
  values = np.asarray(values, dtype=np.float32).reshape(-1, values.shape[-1])[:valid_rows]
  if values.shape[0] != valid_rows or not np.isfinite(values).all() or np.any(values < 0):
    raise ValueError('Collector observed invalid unsigned fc2 activations')
  return dict(rows_bf16=encode_bf16_exact(values[indices]), indices=indices,
              channel_max=values.max(axis=0), channel_min=values.min(axis=0),
              channel_sum=values.sum(axis=0, dtype=np.float32),
              channel_sum_squares=np.square(values).sum(axis=0, dtype=np.float32),
              zero_count=np.array(np.count_nonzero(values == 0), np.int64))


def capture_tinygrad(values, valid_rows, indices, readback):
  """Explicit GPU diagnostic, called only by the parent. Transfers bounded rows/statistics."""
  from tinygrad import Tensor, dtypes
  if values.dtype != dtypes.bfloat16:
    raise ValueError('Collector requires original BF16 fc2 input')
  flat = values.reshape(-1, values.shape[-1])[:valid_rows].float()
  if flat.shape[0] != valid_rows:
    raise ValueError('Invalid unpadded prompt length')
  statistics = Tensor.stack(flat.max(axis=0), flat.min(axis=0), flat.sum(axis=0), flat.square().sum(axis=0)).contiguous().realize()
  flags = Tensor.stack((~flat.isfinite()).sum().float(), (flat < 0).sum().float(), (flat == 0).sum().float()).contiguous().realize()
  flag_values = np.asarray(readback(flags)).reshape(-1)
  if flag_values[0] != 0 or flag_values[1] != 0:
    raise ValueError('Collector observed nonfinite or negative post-ReLU² values')
  # Gather only fixed token rows; never transfer the complete intermediate.
  row_ids = Tensor(indices.astype(np.int32), device=values.device)
  selected = flat[row_ids].contiguous().realize()
  stats = np.asarray(readback(statistics), dtype=np.float32)
  return dict(rows_bf16=encode_bf16_exact(readback(selected)), indices=indices,
              channel_max=stats[0], channel_min=stats[1], channel_sum=stats[2], channel_sum_squares=stats[3],
              zero_count=np.array(round(float(flag_values[2])), np.int64))


class EagerCall:
  def __init__(self, function):
    self.function = function

  def __call__(self, *args, **kwargs):
    return self.function(*args, **kwargs)

  def reset(self):
    pass


def prefill_context(prefix, cache_pos, rows):
  return prefix.startswith('vlm.model.language_model.layers.') and type(cache_pos) is int and cache_pos == 0 and rows >= 128


class Collector:
  def __init__(self, selection, output, reference, rows_per_layer=32, budget_bytes=960 * 1024 ** 2, capture=None):
    reference_body = {k: v for k, v in reference.items() if k != 'identity_sha256'}
    if (reference.get('schema') != SCHEMA
        or hashlib.sha256(canonical(reference_body)).hexdigest() != reference.get('identity_sha256')
        or reference.get('settings', {}).get('precision') != 'fp8'
        or reference.get('settings', {}).get('candidate_installed_during_collection') is not False):
      raise ValueError('Collector requires hashed original-FP8 reference provenance')
    config = reference['checkpoint']['config']['vlm_config']['text_config']
    if config['hidden_act'] != 'relu2' or not 1 <= rows_per_layer <= 64:
      raise ValueError('Unsupported collector architecture or bound')
    self.names = {f'vlm.model.language_model.layers.{i}.mlp.fc2' for i in range(config['num_hidden_layers'])}
    self.width = config['intermediate_size']
    worst = len(selection.allowed) * len(self.names) * (rows_per_layer * self.width * 2 + 4 * self.width * 4 + rows_per_layer * 4 + 4096)
    if not 0 < budget_bytes < 1024 ** 3 or worst >= budget_bytes:
      raise ValueError(f'Capture bound {worst} exceeds its sub-1-GiB budget')
    self.output, self.selection, self.reference = Path(output), selection, reference
    self.output.mkdir(parents=True, exist_ok=False)
    self.rows_per_layer, self.budget_bytes = rows_per_layer, budget_bytes
    self.capture = capture
    self.active, self.seen, self.written = None, set(), 0
    self.manifest = dict(schema=SCHEMA, complete=False, reference=reference, selection=selection.provenance(),
      rows_per_layer=rows_per_layer, budget_bytes=budget_bytes, worst_payload_bytes=worst,
      method='All valid prefill rows for statistics; evenly spaced fixed input rows for bounded storage',
      stats_arithmetic='FP32 GPU reductions; zero count can round above 2**24; not used for fitting',
      scope='Original FP8 prefill, language fc2 inputs after BF16 ReLU²; no candidate installed',
      source_sha256=sha_file(__file__), samples=[])
    json_write(self.output / 'manifest.json', self.manifest)

  @contextmanager
  def sample(self, model, sample, valid_rows, prepared_arrays):
    if self.active is not None or str(Path(sample).resolve()) in self.seen:
      raise ValueError('Nested or duplicate calibration sample')
    if getattr(model, 'agent_int4model_hook', False):
      raise ValueError('Collect original FP8 reference before installing candidate')
    if hashlib.sha256(canonical(model.config)).hexdigest() != self.reference['checkpoint']['canonical_config_sha256']:
      raise ValueError('Reference model config differs from provenance')
    if type(valid_rows) is not int or valid_rows < 128:
      raise ValueError('Collector requires the complete unpadded prefill length')
    identity = self.selection.identity(sample)
    prepared = prepared_identity(prepared_arrays)
    ids = np.asarray(prepared_arrays['input_ids'])
    if ids.ndim != 2 or ids.shape[0] != 1 or not valid_rows <= ids.shape[1]:
      raise ValueError('Collector supports one actual prompt per sample')
    entry = dict(input=identity, prepared=prepared, valid_rows=valid_rows, layers=[], complete=False)
    self.active = dict(entry=entry, names=set(), prefix=None, enabled=False)
    old_linear, old_text, old_jit = model.linear, model.text_layer, model.prefill_jit
    padded = getattr(model, 'padded_prefill', None)
    padded_jit = None if padded is None else padded.jit
    state = self.active

    def text_layer(this, x, cos, sin, prefix, cache=None, mask=None, cache_pos=None):
      previous = state['prefix'], state['enabled']
      state['prefix'] = prefix
      state['enabled'] = prefill_context(prefix, cache_pos, math.prod(x.shape[:-1]))
      try:
        return old_text(x, cos, sin, prefix, cache, mask, cache_pos)
      finally:
        state['prefix'], state['enabled'] = previous

    def linear(this, x, name, quantized_input=None, heads=None):
      if state['enabled'] and name in self.names:
        if name in state['names'] or name != state['prefix'] + '.mlp.fc2' or x.shape[-1] != self.width:
          raise ValueError('Duplicate or misassociated fc2 capture')
        if name + '.weight' not in this.weight_scales:
          raise ValueError('Reference fc2 is not the original FP8 linear')
        indices = representative_rows(valid_rows, self.rows_per_layer)
        if self.capture is None:
          from runtime import download
          values = capture_tinygrad(x, valid_rows, indices, download)
        else:
          values = self.capture(x, valid_rows, indices)
        self._save_capture(name, values, entry)
        state['names'].add(name)
      return old_linear(x, name, quantized_input, heads)

    model.text_layer, model.linear = MethodType(text_layer, model), MethodType(linear, model)
    model.prefill_jit = EagerCall(model._language)
    if padded is not None:
      padded.jit = EagerCall(padded.language)
    try:
      yield
      if state['names'] != self.names:
        raise ValueError(f'Incomplete actual prefill capture: {len(state["names"])} / {len(self.names)} layers')
      entry['complete'] = True
      self.seen.add(identity['path'])
      self.manifest['samples'].append(entry)
      json_write(self.output / 'manifest.json', self.manifest)
    finally:
      model.text_layer, model.linear, model.prefill_jit = old_text, old_linear, old_jit
      old_jit.reset()  # Original cached graphs can refer to a prior KV allocation.
      if padded is not None:
        padded.jit = padded_jit
        padded_jit.reset()
      self.active = None

  def _save_capture(self, name, values, entry):
    if (values['rows_bf16'].shape != (min(self.rows_per_layer, entry['valid_rows']), self.width)
        or values['rows_bf16'].dtype != np.uint16
        or not np.array_equal(values['indices'], representative_rows(entry['valid_rows'], self.rows_per_layer))):
      raise ValueError('Capture exceeded row bound or changed width')
    for key in ('channel_max', 'channel_min', 'channel_sum', 'channel_sum_squares'):
      if values[key].shape != (self.width,) or values[key].dtype != np.float32 or not np.isfinite(values[key]).all():
        raise ValueError('Invalid channel statistic')
    filename = f'sample-{len(self.manifest["samples"]):02d}-layer-{int(FC2.fullmatch(name)[1]):02d}.npz'
    path = self.output / filename
    predicted = sum(np.asarray(v).nbytes for v in values.values()) + 4096
    if self.written + predicted >= self.budget_bytes:
      raise ValueError('Capture disk budget exhausted')
    with path.open('xb') as stream:
      np.savez(stream, **values)
    self.written += path.stat().st_size
    if self.written >= self.budget_bytes:
      raise ValueError('Capture disk budget exhausted')
    entry['layers'].append(dict(name=name, file=filename, sha256=sha_file(path), bytes=path.stat().st_size,
                                maximum=float(values['channel_max'].max()), minimum=float(values['channel_min'].min())))

  def finish(self):
    if self.active is not None or self.seen != set(self.selection.allowed):
      raise ValueError('Collector did not complete exactly the calibration allowlist')
    self.manifest.update(complete=True, payload_bytes=self.written)
    self.manifest['identity_sha256'] = hashlib.sha256(canonical(self.manifest)).hexdigest()
    json_write(self.output / 'manifest.json', self.manifest)
    return self.manifest

  def prefill(self, model, sample, embedding, cos, sin, max_tokens, valid_rows, prepared_arrays):
    """Parent calls this after its unchanged original image encoder and input upload."""
    with self.sample(model, sample, valid_rows, prepared_arrays):
      if embedding.shape[1] == valid_rows:
        return model.prefill(embedding, cos, sin, max_tokens)
      if not hasattr(model, 'padded_prefill'):
        raise ValueError('Padded embedding requires the existing padded-prefill implementation')
      return model.padded_prefill(embedding, cos, sin, max_tokens, valid_rows)


def install_fc2(model, bundle_manifest, runtime_weights, quantize_unsigned, scaled_kernel):
  """Install after other optional linear/text hooks and before any graph capture.

  quantize_unsigned(x2d) -> (native_packed_A, float32_scale[M])
  scaled_kernel(A, W, (sA,sW), *, rows=M, columns=N, bias=None, dtype=x.dtype) -> BF16 [M,N]
  runtime_weights[name] = {packed: uploaded int32, scale: uploaded float32, shape: (N,K)}
  The callable must use unsigned A / signed B and dot * (sA*sW), without reassociation.
  """
  if getattr(model, 'agent_int4model_hook', False) or not runtime_weights:
    raise ValueError('Duplicate or empty candidate hook')
  if (bundle_manifest.get('schema') != SCHEMA or not bundle_manifest.get('complete')
      or bundle_manifest.get('layout') != LAYOUT or bundle_manifest.get('equalization') is not None
      or bundle_manifest.get('clipping') is not None or bundle_manifest.get('outlier_correction') is not None):
    raise ValueError('Initial hook accepts only the unmodified v1 bundle')
  if hashlib.sha256(canonical(model.config)).hexdigest() != bundle_manifest['checkpoint']['canonical_config_sha256']:
    raise ValueError('Candidate model config differs from bundle')
  record_shapes = {r['name']: tuple(r['shape']) for r in bundle_manifest['records']}
  if set(runtime_weights) != set(record_shapes):
    raise ValueError('Runtime weight names differ from bundle')
  for name, record in runtime_weights.items():
    if (not FC2.fullmatch(name) or tuple(record['shape']) != record_shapes[name]
        or tuple(model.weights[name + '.weight'].shape) != record_shapes[name]
        or name + '.bias' in model.weights or name + '.weight' not in model.weight_scales):
      raise ValueError('Candidate weight scope/shape/original FP8 precision differs')
  old_linear, old_text = model.linear, model.text_layer
  state = dict(enabled=False, prefix=None)

  def text_layer(this, x, cos, sin, prefix, cache=None, mask=None, cache_pos=None):
    previous = dict(state)
    state.update(enabled=prefill_context(prefix, cache_pos, math.prod(x.shape[:-1])), prefix=prefix)
    try:
      return old_text(x, cos, sin, prefix, cache, mask, cache_pos)
    finally:
      state.update(previous)

  def linear(this, x, name, quantized_input=None, heads=None):
    if not state['enabled'] or name not in runtime_weights:
      return old_linear(x, name, quantized_input, heads)
    if name != state['prefix'] + '.mlp.fc2' or quantized_input is not None or heads is not None:
      raise ValueError('Unexpected candidate fc2 association or prequantized input')
    record = runtime_weights[name]
    rows, width = math.prod(x.shape[:-1]), x.shape[-1]
    if width != record['shape'][1]:
      raise ValueError('Candidate input width differs from bundle')
    a, sa = quantize_unsigned(x.reshape(rows, width))
    if tuple(sa.shape) != (rows,) or getattr(sa.dtype, 'name', '') not in ('f32', 'float32'):
      raise ValueError('Activation quantizer violated the float32 per-token scale contract')
    output = scaled_kernel(a, record['packed'], (sa, record['scale']), rows=rows,
                           columns=record['shape'][0], bias=None, dtype=x.dtype)
    if tuple(output.shape) != (rows, record['shape'][0]) or output.dtype != x.dtype:
      raise ValueError('Scaled kernel violated the output shape/dtype contract')
    return output.reshape(*x.shape[:-1], record['shape'][0])

  model.text_layer, model.linear = MethodType(text_layer, model), MethodType(linear, model)
  model.agent_int4model_hook = True
  _reset_graphs(model)

  def uninstall():
    model.text_layer, model.linear = old_text, old_linear
    model.agent_int4model_hook = False
    _reset_graphs(model)

  return uninstall


def scaled_module_adapter(kernel_module, cpu_weights, upload):
  """Adapt the separate agent-int4-scaled-kernel.py API without editing its source.

  upload(np_array) must upload immutable weights/scales once, before graph capture.
  Its quantize_pack uses uint8 bytes; exported int32 fragment words have identical bits.
  """
  runtime = {}
  for name, record in cpu_weights.items():
    words = np.asarray(record['packed'])
    if words.dtype != np.int32 or words.shape[2:] != (32, 2):
      raise ValueError('Adapter requires native int32 fragment words')
    byte_view = words.astype('<i4', copy=False).view(np.uint8).reshape(*words.shape[:-1], 8).copy()
    scale = np.asarray(record['scale'], dtype=np.float32)
    n, _ = record['shape']
    if scale.shape == (1,):
      scale = np.repeat(scale, n)
    if scale.shape != (n,):
      raise ValueError('Scaled kernel requires one weight scale per output')
    runtime[name] = dict(packed=upload(byte_view), scale=upload(scale.copy()), shape=record['shape'])

  def scaled(a, w, scales, *, rows, columns, bias, dtype):
    return kernel_module.gemm(a, w, scales[0], scales[1], rows=rows, columns=columns, bias=bias)

  return runtime, kernel_module.quantize_pack, scaled


def _reset_graphs(model):
  for name in ('prefill_jit', 'decode_jit', 'greedy_decode_jit'):
    if (jit := getattr(model, name, None)) is not None:
      jit.reset()
  if getattr(model, 'padded_prefill', None) is not None:
    model.padded_prefill.jit.reset()
  for jit in getattr(model, 'trajectory_jits', {}).values():
    jit.reset()
