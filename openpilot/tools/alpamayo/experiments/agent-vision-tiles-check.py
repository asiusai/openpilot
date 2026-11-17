"""Independent image parity, serial matrix timing, and complete linear-stage costs.

Default mode is CPU only. GPU mode belongs to the serialized parent wrapper.
"""
import argparse
from functools import lru_cache, partial
import importlib.util
import json
import os
from pathlib import Path
import statistics
import sys
import time
from types import SimpleNamespace

import numpy as np

HERE = Path(__file__).resolve().parent
WORK = Path('/mnt/nvme-500/tmp/alpamayo-speed-20261116')
CALLS = 40


@lru_cache(None)
def load(name):
  spec = importlib.util.spec_from_file_location('agent_vision_tiles_check_'+name.replace('-', '_'), HERE/(name+'.py'))
  module = importlib.util.module_from_spec(spec)
  sys.modules[spec.name] = module
  spec.loader.exec_module(module)
  return module


def fixture(images, shape, state, integer=False):
  E = load('agent-expert-gemm')
  rows, columns, width = shape
  arrays = []
  for image in range(images):
    change = 2 if state == 4 else int(state >= 1 and image == images-1)
    rng = np.random.default_rng(906071+image*10+change)
    arrays.append(E.int_fp8(rng.integers(-3, 4, (rows, width))) if integer else
      rng.integers(0, 127, (rows, width), dtype=np.uint8) | (rng.integers(0, 2, (rows, width), dtype=np.uint8) << 7))
  rng = np.random.default_rng(906075+int(state == 4))
  weight = E.int_fp8(rng.integers(-3, 4, (columns, width))) if integer else (
    rng.integers(0, 127, (columns, width), dtype=np.uint8) | (rng.integers(0, 2, (columns, width), dtype=np.uint8) << 7))
  sx = np.array([.125, .25][:images], np.float32)
  if state >= 2: sx[0] = .5 if integer else .071234569
  if state == 4: sx *= np.float32(.5 if integer else .913713)
  ws = np.array([.25 if state < 4 else .125 if integer else .0291127], np.float32)
  bias = E.bf16(np.random.default_rng(906078+int(state >= 3)).integers(-8, 9, columns).astype(np.float32)/8)
  return {'a_unpacked': np.stack(arrays), 'w_unpacked': weight,
    'a': np.concatenate([E.pack_cpu(a) for a in arrays]), 'w': E.pack_cpu(weight),
    'sx': np.repeat(sx, rows), 'sw': ws, 'bias': bias}


class MatrixBatch:
  def __init__(self, images, shape, fn, candidate, count):
    from tinygrad import Tensor, TinyJit, dtypes
    self.images, self.shape, self.count, self.proofs = images, shape, count, []
    self.output = Tensor.empty(images, shape[0], shape[1], dtype=dtypes.bfloat16).realize()
    self.output.uop.buffer.ensure_allocated()
    def execute(a, w, sx, sw, bias, out):
      for _ in range(count):
        target = out if candidate else out.reshape(images*shape[0], shape[1])
        out = Tensor.custom_kernel(target, a, w, sx, sw, bias, fxn=fn)[0].reshape(images, shape[0], shape[1]).realize()
      return (out,)
    self.inner = TinyJit(execute)

  @property
  def captured(self): return self.inner.captured

  def __call__(self, *inputs):
    E = load('agent-expert-gemm')
    if self.count == CALLS and self.inner.cnt == 1:
      with E.observe_serial_queue(self.proofs): result = self.inner(*inputs, self.output)
    else: result = self.inner(*inputs, self.output)
    assert result[0].uop.buffer.base is self.output.uop.buffer.base
    assert result[0].uop.buffer.offset == self.output.uop.buffer.offset
    return result


def independent_prepared(images, rows, a, w, sx, sw, bias):
  from tinygrad import Tensor, dtypes
  from tinygrad.helpers import round_up
  from amd_gemm import _gemm
  results = []
  for image in range(images):
    one = a.reshape(images, round_up(rows, 16), a.shape[-1])[image]
    scale = sx.reshape(images, rows)[image, :1]
    out = Tensor.empty(rows, w.shape[0], device=a.device, dtype=dtypes.bfloat16)
    results.append(Tensor.custom_kernel(out, one, w, scale, sw, bias, fxn=_gemm)[0].realize())
  return tuple(results)


def integer_reference(data):
  E = load('agent-expert-gemm')
  sums = E.decode_fp8(data['a_unpacked']).astype(np.int64) @ E.decode_fp8(data['w_unpacked']).astype(np.int64).T
  assert np.abs(sums).max() < 2**24
  scales = data['sx'].reshape(sums.shape[0], sums.shape[1], 1)*data['sw'][0]
  return E.bf16(sums.astype(np.float32)*scales+E.unpack_bf16(data['bias']))


def cpu_checks():
  from tinygrad import Device, Tensor, dtypes
  from tinygrad.uop.ops import KernelInfo, UOp
  from amd_gemm import pack
  E, V = load('agent-expert-gemm'), load('agent-vision-tiles')
  checks = []
  for images in (1, 2):
    for shape in ((17, 48, 32), (65, 80, 64), *V.SHAPES.values()):
      rows, columns, width = shape
      source = (np.arange(images*rows*width, dtype=np.uint32) % 251).astype(np.uint8).reshape(images, rows, width)
      packed = V.pack_images(Tensor(source, device='CPU')).numpy()
      expected = np.concatenate([E.pack_cpu(image) for image in source])
      np.testing.assert_array_equal(packed, expected)
      if rows % 16 == 0:
        np.testing.assert_array_equal(packed, pack(Tensor(source.reshape(-1, width), device='CPU')).numpy())
      for tile in V.TILES.values():
        seen = []
        blocks = (rows+16*tile[0]-1)//(16*tile[0])
        for dispatch in range(images*blocks):
          image, block = divmod(dispatch, blocks)
          for m in range(block*16*tile[0], min(rows, (block+1)*16*tile[0])):
            seen.append(image*rows+m)
            k = np.arange(width)
            address = image*((rows+15)//16*16)*width+(m//16*(width//16)+k//16)*256+(k%16//4)*64+(m%16)*4+k%4
            np.testing.assert_array_equal(packed.reshape(-1)[address], source[image, m])
        assert seen == list(range(images*rows))
        cols = [n for block in range((columns+16*tile[1]-1)//(16*tile[1]))
                for n in range(block*16*tile[1], min(columns, (block+1)*16*tile[1]))]
        assert cols == list(range(columns))
      checks.append({'case': 'image-packing-offset-scale-row-output-partition', 'images': images, 'shape_mnk': shape, 'exact': True})
    previous = None
    for state in range(5):
      data = fixture(images, (65, 80, 64), state, True)
      expected = integer_reference(data)
      result = Tensor(E.decode_fp8(data['a_unpacked']), device='CPU').dot(Tensor(E.decode_fp8(data['w_unpacked']), device='CPU').T)
      result *= Tensor((data['sx'].reshape(images, 65, 1)*data['sw'][0]), device='CPU')
      result += Tensor(E.unpack_bf16(data['bias']), device='CPU')
      np.testing.assert_array_equal(result.cast(dtypes.bfloat16).bitcast(dtypes.uint16).numpy(), expected)
      if images == 2 and state == 1: np.testing.assert_array_equal(expected[0], previous[0])
      if images == 2 and state == 2: np.testing.assert_array_equal(expected[1], previous[1])
      if previous is not None: assert not np.array_equal(expected, previous)
      previous = expected
    checks.append({'case': 'independent-integer-bias-and-scale-mutations', 'images': images, 'states': 5, 'exact': True})
  # The real 40-call capture wrapper, using a CPU kernel with independent image scales.
  def fill(out, a, w, sx, sw, bias):
    r = UOp.range(out.numel(), 0)
    image = r//(out.shape[1]*out.shape[2])
    value = a.flatten().index(0).float()+sx.flatten().index(image*out.shape[1])
    return out.flatten().index(r).store(value.cast(out.dtype)).end(r).sink(arg=KernelInfo())
  arrays = [np.array([2], np.float32), np.array([0], np.float32), np.array([3, 3, 5, 5], np.float32),
            np.array([1], np.float32), np.array([0], np.float32)]
  inputs = tuple(Tensor(value, device='CPU').realize() for value in arrays)
  batch = MatrixBatch(2, (2, 16, 16), fill, True, CALLS)
  for _ in range(3): batch(*inputs)
  assert len(batch.captured._linear.src) == CALLS
  proof = E.prove_cpu_queue(batch.captured._linear.src)
  arrays[2][2:] = 7
  inputs[2].uop.buffer.host[:] = arrays[2].tobytes()
  actual = batch(*inputs)[0].bitcast(dtypes.uint16).numpy()
  np.testing.assert_array_equal(actual, E.bf16(np.stack([np.full((2, 16), 5), np.full((2, 16), 9)])))
  # Shared storage is necessary: independently allocated outputs must fail the proof.
  from tinygrad.engine.realize import get_call_outs_ins
  from tinygrad.runtime.support.hcq2 import unwrap_lane
  changed = []
  for i, call in enumerate(batch.captured._linear.src):
    position = 1+get_call_outs_ins(call)[0][0]
    sources = list(call.src)
    base, lane, offset = unwrap_lane(sources[position])
    assert lane is None and offset == 0
    sources[position] = sources[position].substitute({base: UOp.param(100+i, base.dtype, base.max_numel(), base.device)}, walk=True)
    changed.append(call.replace(src=tuple(sources)))
  try: E.prove_cpu_queue(changed)
  except AssertionError: pass
  else: raise AssertionError('Independent output allocations passed the serial proof')
  checks.append({'case': '40-call-capture-mutation-and-overlap-rejection', 'exact': True, 'queue_proof': proof})
  # Exercise the installer without executing its AMD kernel.
  old_gemm, calls, quant_calls = V.gemm, [], []
  weights = {'vlm.model.visual.embeddings.patch_embedding.weight': Tensor.empty(16, 16, device='CPU', dtype=dtypes.bfloat16)}
  scales = {}
  for layer in (0, 1, 2):
    for fc, (_, n, k) in V.SHAPES.items():
      prefix = f'vlm.model.visual.encoder.layers.{layer}.mlp.{fc}'
      weights[prefix+'.weight'] = Tensor.empty(n, k, device='CPU', dtype=dtypes.bfloat16 if layer < 2 else dtypes.fp8e4m3)
      weights[prefix+'.bias'] = Tensor.empty(n, device='CPU', dtype=dtypes.bfloat16)
      if layer == 2: scales[prefix+'.weight'] = Tensor.empty(1, device='CPU', dtype=dtypes.float32)
  def quantize(x):
    quant_calls.append(x.shape)
    return Tensor.empty(x.shape, device='CPU', dtype=dtypes.fp8e4m3), Tensor.empty(x.shape[0]*720, device='CPU', dtype=dtypes.float32)
  def old_linear(x, name, quantized_input=None, heads=None): calls.append(('old', name)); return x
  model = SimpleNamespace(dtype=dtypes.bfloat16, grouped_quantize=True, weights=weights, weight_scales=scales, linear=old_linear, quantize=quantize)
  def stub(x, weight, values, bias, tile): calls.append(('new', tile, x.shape)); return Tensor.empty(x.shape[0], x.shape[1], weight.shape[0], device='CPU', dtype=dtypes.bfloat16)
  try:
    V.gemm = stub
    V.install(model)
    for images in (1, 2):
      for fc, (_, _, width) in V.SHAPES.items(): model.linear(Tensor.empty(images, 720, width, device='CPU', dtype=dtypes.bfloat16), f'vlm.model.visual.encoder.layers.2.mlp.{fc}')
    assert len(quant_calls) == 4 and len([c for c in calls if c[0] == 'new']) == 4
    q = quantize(Tensor.empty(2, 720, 1152, device='CPU', dtype=dtypes.bfloat16))
    count = len(quant_calls)
    model.linear(Tensor.empty(2, 720, 1152, device='CPU', dtype=dtypes.bfloat16), 'vlm.model.visual.encoder.layers.2.mlp.fc1', q)
    assert len(quant_calls) == count
    for name in ('vlm.model.visual.encoder.layers.0.mlp.fc1', 'vlm.model.visual.encoder.layers.1.mlp.fc1',
                 'vlm.model.visual.encoder.layers.2.self_attn.q_proj', 'vlm.model.language_model.layers.2.mlp.fc1',
                 'vlm.model.visual.embeddings.patch_embedding', 'vlm.model.projector.linear_fc1'):
      model.linear(Tensor.empty(2, 720, 1152, device='CPU', dtype=dtypes.bfloat16), name)
      assert calls[-1] == ('old', name) and len(quant_calls) == count
  finally: V.gemm = old_gemm
  checks.append({'case': 'installer-vision-mlp-only-preserves-quantizer-index-and-bf16', 'exact': True})
  # The route comparator must reject every changed output field without tolerance.
  R = load('agent-vision-tiles-route')
  class Record:
    def __init__(self, value): self.data = json.dumps(value)
    def read_text(self): return self.data
    def read_bytes(self): return self.data.encode()
  saved = {'warmup': [], 'runs': [{'sample': 'sample-control', 'prediction':
    {'generated_token_ids': [1, 2], 'reasoning': 'control', 'xyz': [[0., 1., 2.]], 'rot': [[0., 0., 0.]]}}]}
  assert R.compare(Record(saved), Record(saved))['all_exact']
  for field in ('generated_token_ids', 'xyz', 'rot'):
    changed = json.loads(json.dumps(saved))
    value = changed['runs'][0]['prediction'][field]
    if field == 'generated_token_ids': value[-1] += 1
    else: value[0][0] = float(np.nextafter(np.float64(0), np.float64(1)))
    try: R.compare(Record(saved), Record(changed))
    except AssertionError: pass
    else: raise AssertionError(f'Changed {field} passed the exact route gate')
  checks.append({'case': 'route-tokens-xyz-rotation-mutation-rejection', 'exact': True})
  assert not any(d.startswith('AMD') for d in Device._opened_devices)
  return checks


def matrix_check(device, images, shape, tile, rounds, integer=False, transfer_options=None):
  from tinygrad import TinyJit, dtypes
  E, C = load('agent-expert-gemm'), load('agent-vision-tiles-compile')
  functions = C.functions(device.renderer, images, shape[0], tile)
  if integer: functions.pop('current')  # Tail controls have separate image padding.
  transfers = load('agent-vision-tiles-transfer-check').InputTransfers(device, **(transfer_options or {}))
  keys = ('a', 'w', 'sx', 'sw', 'bias')
  initial = fixture(images, shape, 0, integer)
  inputs = transfers.upload([initial[key] for key in keys],
    (dtypes.fp8e4m3, dtypes.fp8e4m3, dtypes.float32, dtypes.float32, dtypes.bfloat16))
  original = TinyJit(partial(independent_prepared, images, shape[0]))
  singles = {name: MatrixBatch(images, shape, fn, name == 'candidate', 1) for name, fn in functions.items()}
  for fn in (original, *singles.values()):
    for _ in range(3): fn(*inputs)
  report = {'images': images, 'shape_mnk': shape, 'tile': tile, 'checks': [], 'timing': None}
  previous = None
  for state in range(5):
    data = fixture(images, shape, state, integer)
    transfers.replace(inputs, [data[key] for key in keys])
    expected = np.stack([E.read_output(t, device) for t in original(*inputs)])
    if integer: np.testing.assert_array_equal(expected, integer_reference(data))
    if previous is not None: assert not np.array_equal(expected, previous)
    if images == 2 and state == 1: np.testing.assert_array_equal(expected[0], previous[0])
    if images == 2 and state == 2: np.testing.assert_array_equal(expected[1], previous[1])
    for name, fn in singles.items():
      E.poison((fn.output,), device)
      E.exact_output(fn(*inputs)[0], expected, device)
    previous = expected.copy()
    report['checks'].append({'state': state, 'independent_original_images_exact': True, 'independent_integer': integer,
      'unchanged_other_image_exact': images == 2 and state in (1, 2)})
  if integer:
    report['input_transfers'] = transfers.report()
    return report
  batches = {name: MatrixBatch(images, shape, fn, name == 'candidate', CALLS) for name, fn in functions.items()}
  for fn in batches.values():
    for _ in range(3): fn(*inputs)
    assert len(E.captured_kernels(fn)) == CALLS and E.shared_output_slot(fn) == 5
    E.poison((fn.output,), device)
    E.exact_output(fn(*inputs)[0], expected, device)
  def measure(fn):
    device.synchronize()
    start = time.perf_counter()
    fn(*inputs)
    device.synchronize()
    return (time.perf_counter()-start)*1000/CALLS
  pairs = []
  for iteration in range(rounds):
    order = ('current', 'candidate') if iteration % 2 == 0 else ('candidate', 'current')
    pairs.append({name: measure(batches[name]) for name in order})
  report['timing'] = {'paired_amortized_ms': pairs, 'medians_ms': {name: statistics.median(p[name] for p in pairs) for name in batches},
    'calls_per_batch': CALLS, 'profiling': False, 'host_inclusive': True, 'completion_fences_per_batch': 1,
    'queue_proofs': {name: fn.proofs for name, fn in batches.items()}, 'scope': 'Packed inputs through BF16 output; complete linear costs measured separately.'}
  report['input_transfers'] = transfers.report()
  return report


def cached_weights(cache, fc):
  E = load('agent-expert-gemm')
  index = json.loads((cache/'index.json').read_text())
  prefix = 'vlm.model.visual.encoder.layers.2.mlp.'+fc
  result, records = [], []
  for name, kind, dtype, packed in ((prefix+'.weight', 'fp8', np.uint8, True), (prefix+'.weight', 'scale', np.float32, False),
                                  (prefix+'.bias', 'bf16', np.uint16, False)):
    r = next(r for r in index['records'] if r['name'] == name and r['kind'] == kind and r['packed'] == packed)
    data = np.array(np.memmap(cache/'weights.bin', mode='r', dtype=dtype, offset=r['offset'], shape=tuple(r['shape'])), copy=True)
    result.append(data)
    records.append({'name': name, 'kind': kind, 'packed': packed, 'sha256': E.sha(data.tobytes())})
  return result, records


def full_values(images, shape, cache, fc, calibration, state):
  E = load('agent-expert-gemm')
  rows, columns, width = shape
  (w, ws, bias), provenance = cached_weights(cache, fc)
  x = np.random.default_rng(906080).normal(size=(images, rows, width)).astype(np.float32)
  if state >= 1:
    x[-1] = np.random.default_rng(906081).normal(size=(rows, width)).astype(np.float32)
    x[-1, 0, :4] = [0, 2**-30, -2**-30, 1000]
  if state >= 2:
    bias = E.bf16(E.unpack_bf16(bias)+np.float32(.03125))
  if state == 3:
    x = np.random.default_rng(906082).normal(size=x.shape).astype(np.float32)
    w = np.roll(w, 256, axis=None).reshape(w.shape).copy()
    ws *= np.float32(1.17)
  key = f'vlm.model.visual.encoder.layers.2/{2 if fc == "fc1" else 3}'
  scale = json.loads(calibration.read_text())['scales'][key]
  row_scale = np.full(images*rows, scale*(1.125 if state == 3 else 1), np.float32)
  return [E.bf16(x), w, row_scale, ws, bias], provenance


def full_operation(images, shape, tile, mode, candidate, x, w, fixed_rows, ws, bias):
  from tinygrad import Tensor, dtypes
  from amd_gemm import pack
  V, Q, T = (load(n) for n in ('agent-vision-tiles', 'quantize-reduction', 'tiled-gemm'))
  rows, columns, width = shape
  if mode == 'dynamic': q, scale = Q.grouped_quantize(x, images)
  else:
    q = (x.float()/fixed_rows[:1].reshape(())).clamp(-448, 448).cast(dtypes.fp8e4m3).reshape(-1, width).contiguous()
    scale = fixed_rows
  if candidate:
    packed_q = V.pack_images(q.reshape(images, rows, width))
    out = V.packed(packed_q, w, (scale, ws), bias, images, rows, tile)
  else:
    packed_q = pack(q)
    out = Tensor.custom_kernel(Tensor.empty(images*rows, columns, device=x.device, dtype=dtypes.bfloat16), packed_q, w, scale, ws, bias,
      fxn=partial(T.kernel, tile_m=4, tile_n=4, row_scale=True))[0].reshape(images, rows, columns)
  out.realize()
  return out, packed_q, scale


def independent_full(images, mode, x, w, fixed_rows, ws, bias):
  from tinygrad import Tensor, dtypes
  from amd_gemm import _gemm, pack
  Q = load('quantize-reduction')
  rows, width = x.shape[1:]
  results = []
  for image in range(images):
    one = x[image:image+1].contiguous().realize()
    if mode == 'dynamic': q, scale = Q.quantize(one)
    else:
      scale = fixed_rows[:1].reshape(())
      q = (one.float()/scale).clamp(-448, 448).cast(dtypes.fp8e4m3).reshape(rows, width).contiguous()
    packed_q = pack(q)
    out = Tensor.custom_kernel(Tensor.empty(rows, w.shape[0], device=x.device, dtype=dtypes.bfloat16), packed_q, w, scale, ws, bias, fxn=_gemm)[0].realize()
    results.extend((out, packed_q, scale))
  return tuple(results)


def full_check(device, timestamps, images, fc, mode, cache, calibration, transfer_options=None):
  from tinygrad import TinyJit, dtypes
  E, V, S = (load(n) for n in ('agent-expert-gemm', 'agent-vision-tiles', 'agent-spec-head-check'))
  shape, tile = V.SHAPES[fc], V.TILES[fc]
  values, provenance = full_values(images, shape, cache, fc, calibration, 0)
  transfers = load('agent-vision-tiles-transfer-check').InputTransfers(device, **(transfer_options or {}))
  inputs = transfers.upload(values, (dtypes.bfloat16, dtypes.fp8e4m3, dtypes.float32, dtypes.float32, dtypes.bfloat16))
  functions = {name: TinyJit(partial(full_operation, images, shape, tile, mode, name == 'candidate')) for name in ('current', 'candidate')}
  original = TinyJit(partial(independent_full, images, mode))
  for fn in (original, *functions.values()):
    for _ in range(3): fn(*inputs)
  report = {'images': images, 'fc': fc, 'quantization': mode, 'shape_mnk': shape, 'weights': provenance,
    'checks': [], 'samples': {name: [] for name in functions}}
  previous = None
  for state in range(4):
    values, _ = full_values(images, shape, cache, fc, calibration, state)
    transfers.replace(inputs, values)
    ref = original(*inputs)
    expected = np.stack([E.read_output(t, device) for t in ref[::3]])
    packed_expected = np.concatenate([S.read(t, device, np.uint8) for t in ref[1::3]])
    scales_expected = np.repeat(np.array([S.read(t, device, np.float32).reshape(-1)[0] for t in ref[2::3]], np.float32), shape[0])
    for name, fn in functions.items():
      out, q, scale = fn(*inputs)
      E.exact_output(out, expected, device)
      np.testing.assert_array_equal(S.read(q, device, np.uint8), packed_expected)
      np.testing.assert_array_equal(S.read(scale, device, np.float32).view(np.uint32), scales_expected.view(np.uint32))
    if images == 2 and state == 1: np.testing.assert_array_equal(expected[0], previous[0])
    if previous is not None: assert not np.array_equal(expected, previous)
    previous = expected.copy()
    report['checks'].append({'state': state, 'independent_original_images_exact': True, 'quantized_bytes_and_scale_bits_exact': True,
      'unchanged_other_image_exact': images == 2 and state == 1})
  values, _ = full_values(images, shape, cache, fc, calibration, 0)
  transfers.replace(inputs, values)
  for fn in functions.values():
    for _ in range(5): fn(*inputs)
  device.synchronize()
  for iteration in range(CALLS):
    for name in (('current', 'candidate') if iteration % 2 == 0 else ('candidate', 'current')):
      sample = timestamps.sample(functions[name], inputs, device)
      expected_name = f'agent_vision_tiles_{tile[0]}x{tile[1]}_b{images}' if name == 'candidate' else 'alpamayo_fp8_gemm_4_4'
      assert sum(e['name'] == expected_name for e in sample['events']) == 1, sample['events']
      report['samples'][name].append(sample)
  report['timing'] = {name: {'median_gpu_span_ms': statistics.median(s['gpu_span_ms'] for s in samples),
    'median_sum_kernel_ms': statistics.median(s['sum_kernel_ms'] for s in samples), 'calls': len(samples)} for name, samples in report['samples'].items()}
  report['scope'] = 'BF16 activation through existing dynamic/fixed quantization, activation packing, row scales, FP8 matrix, bias, BF16 output. Prepacked static weights and fixed row scales match model caching. LayerNorm, GELU, uploads, and full-pipeline timing are separate.'
  report['timing_limit'] = '40 individually fenced GPU-timestamp samples per variant. Profiling changes clocks/host overhead; route timing remains required.'
  report['input_transfers'] = transfers.report()
  return report


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--gpu', action='store_true')
  parser.add_argument('--output', type=Path, required=True)
  parser.add_argument('--rounds', type=int, default=3)
  parser.add_argument('--cache', type=Path, default=Path('/dev/shm/alpamayo-9060-fp8'))
  parser.add_argument('--calibration', type=Path, default=WORK/'vision-scale-calibration/report.vision-scales.json')
  parser.add_argument('--quantization', choices=('dynamic', 'fixed', 'both'), default='dynamic')
  parser.add_argument('--stage', choices=('all', 'matrix', 'complete-linear'), default='all')
  parser.add_argument('--batch', choices=('all', '1', '2'), default='all')
  parser.add_argument('--projection', choices=('all', 'fc1', 'fc2'), default='all')
  parser.add_argument('--transfer-mode', choices=('staging', 'direct'), default='staging')
  parser.add_argument('--no-transfer-shadow', action='store_true')
  args = parser.parse_args()
  assert 3 <= args.rounds <= 5
  os.environ.update(BEAM='0', DEBUG='0', PARALLEL='0', PROFILE='1' if args.gpu else '0')
  if not args.gpu: os.environ['DEV'] = 'CPU'
  E, V, C = (load(name) for name in ('agent-expert-gemm', 'agent-vision-tiles', 'agent-vision-tiles-transfer-check-coverage'))
  selected = C.selected_cases(args.stage, args.batch, args.projection, args.quantization)
  transfer_options = {'mode': args.transfer_mode, 'skip_identical': not args.no_transfer_shadow}
  report = {'complete': False, 'gpu_executed': args.gpu, 'sources': {str(p): E.sha(p.read_bytes()) for p in
    (Path(__file__), HERE/'agent-vision-tiles.py', HERE/'image-gemm.py', HERE/'tiled-gemm.py', HERE/'batched-vision.py', HERE/'quantize-reduction.py',
     HERE/'agent-spec-head-check.py', HERE/'agent-vision-tiles-transfer-check.py', HERE/'agent-vision-tiles-transfer-check-coverage.py')},
    'environment': {key: os.environ.get(key) for key in ('DEV', 'AM_POWER_LIMIT', 'AM_PERF_LEVEL', 'PROFILE', 'TC_OPT')}, 'matrix': [], 'full_linear': [],
    'complete_scope': 'Selected run only; consult coverage for full GPU qualification.', 'route_qualified': False,
    'selectors': {'stage': args.stage, 'batch': args.batch, 'projection': args.projection, 'quantization': args.quantization},
    'transfer_options': transfer_options, 'numeric_kernel_sources': C.source_manifest(Path(__file__))}
  def save():
    report['coverage'] = C.coverage(report, selected)
    report['full_kernel_suite_complete'] = report['coverage']['full_kernel_suite_complete']
    E.write_json(args.output, report)
  save()
  try:
    if not args.gpu:
      report['cpu_checks'] = cpu_checks()
      report['transfer_cpu_checks'] = load('agent-vision-tiles-transfer-check').cpu_checks()
      report['cache_weights'] = {fc: cached_weights(args.cache, fc)[1] for fc in V.TILES}
      for fc, shape in V.SHAPES.items():
        previous, _ = full_values(2, shape, args.cache, fc, args.calibration, 0)
        changed, _ = full_values(2, shape, args.cache, fc, args.calibration, 1)
        np.testing.assert_array_equal(previous[0][0], changed[0][0])
        assert not np.array_equal(previous[0][1], changed[0][1])
        for a, b in zip(previous[1:], changed[1:], strict=True): np.testing.assert_array_equal(a, b)
        assert np.isfinite(E.unpack_bf16(previous[-1])).all() and np.isfinite(previous[-2]).all()
        assert not np.any((previous[1] & 127) == 127)
      report['full_linear_fixture_cpu_checked'] = True
    else:
      from tinygrad import Device
      from tinygrad.device import Compiled
      from tinygrad.helpers import Context
      from tinygrad.renderer.llvmir import AMDLLVMRenderer
      assert os.environ.get('AM_POWER_LIMIT') == '144' and os.environ.get('AM_PERF_LEVEL') == 'auto'
      Compiled.wait_timeout_ms = 10000
      load('prefill-kernels').fix_dispatch_packet()
      device = Device[Device.DEFAULT]
      assert device.arch == 'gfx1200' and device.is_usb and isinstance(device.renderer, AMDLLVMRenderer)
      load('wgp_compiler').install(device)
      timestamps = load('agent-spec-head-check').Timestamps(device)
      report['calibration'] = {'path': str(args.calibration), 'sha256': E.sha(args.calibration.read_bytes())}
      report['device'] = {'arch': device.arch, 'renderer': type(device.renderer).__name__, 'timestamp_divider': device.timestamp_divider}
      batches = (1, 2) if args.batch == 'all' else (int(args.batch),)
      projections = tuple(V.TILES) if args.projection == 'all' else (args.projection,)
      if args.stage in ('all', 'matrix'):
        with Context(PROFILE=0):
          for images in batches:
            for fc in projections:
              tile = V.TILES[fc]
              report['active_case'] = {'stage': 'matrix-control', 'images': images, 'fc': fc, 'tile': tile}
              save()
              report['matrix'].append(matrix_check(device, images, (65, 80, 64), tile, args.rounds, integer=True, transfer_options=transfer_options))
              save()
            for fc in projections:
              report['active_case'] = {'stage': 'matrix-actual', 'images': images, 'fc': fc}
              save()
              report['matrix'].append(matrix_check(device, images, V.SHAPES[fc], V.TILES[fc], args.rounds, transfer_options=transfer_options))
              save()
      if args.stage in ('all', 'complete-linear'):
        for images in batches:
          for fc in projections:
            for mode in (('dynamic', 'fixed') if args.quantization == 'both' else (args.quantization,)):
              report['active_case'] = {'stage': 'complete-linear', 'images': images, 'fc': fc, 'quantization': mode}
              save()
              report['full_linear'].append(full_check(device, timestamps, images, fc, mode, args.cache, args.calibration, transfer_options))
              save()
    report.pop('active_case', None)
    report['complete'] = True
  except BaseException as exc:
    report['failure'] = {'type': type(exc).__name__, 'message': str(exc)}
    save()
    raise
  save()
  print(json.dumps({'complete': True, 'complete_scope': 'selected-run-only', 'gpu_executed': args.gpu,
    'selected_gpu_cases_complete': report['coverage']['selected_gpu_cases_complete'],
    'full_kernel_suite_complete': report['full_kernel_suite_complete'], 'output': str(args.output)}), flush=True)


if __name__ == '__main__': main()
