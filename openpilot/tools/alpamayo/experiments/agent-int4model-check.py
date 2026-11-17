#!/usr/bin/env python3
"""Independent CPU checks. No tinygrad, torch, GPU driver, or external agent import."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import struct
import tempfile

import numpy as np

spec = importlib.util.spec_from_file_location('agent_int4model', Path(__file__).with_name('agent-int4model.py'))
p = importlib.util.module_from_spec(spec)
spec.loader.exec_module(p)


def check(condition, message):
  if not condition:
    raise AssertionError(message)


def rejects(function):
  try:
    function()
  except (ValueError, OverflowError):
    return
  raise AssertionError('Expected rejection')


def independent_pack(values):
  rows, width = values.shape
  packed = np.zeros(((rows + 15) // 16, width // 32, 32, 2), np.uint32)
  for row in range(rows):
    for k in range(width):
      lane = ((k % 32) // 16) * 16 + row % 16
      register = (k % 16) // 8
      nibble = k % 8
      packed[row // 16, k // 32, lane, register] |= np.uint32((int(values[row, k]) & 15) << (4 * nibble))
  return packed.view(np.int32)


class Jit:
  def __init__(self, function=lambda *a, **k: None):
    self.function, self.resets = function, 0

  def __call__(self, *args, **kwargs):
    return self.function(*args, **kwargs)

  def reset(self):
    self.resets += 1


class Model:
  def __init__(self, weight):
    self.config = {'vlm_config': {'text_config': {'hidden_act': 'relu2', 'mlp_bias': False,
                   'num_hidden_layers': 1, 'hidden_size': weight.shape[0], 'intermediate_size': weight.shape[1]}}}
    self.weights = {'vlm.model.language_model.layers.0.mlp.fc2.weight': weight}
    self.weight_scales = {next(iter(self.weights)): np.array([1.], np.float32)}
    self.calls = []
    self.prefill_jit = Jit(self._language)
    self.decode_jit, self.greedy_decode_jit, self.trajectory_jits = Jit(), Jit(), {}

  def linear(self, x, name, quantized_input=None, heads=None):
    self.calls.append(name)
    return x @ self.weights[next(iter(self.weights))].T

  def text_layer(self, x, cos, sin, prefix, cache=None, mask=None, cache_pos=None):
    return self.linear(x, prefix + '.mlp.fc2')

  def _language(self, x):
    return self.text_layer(x, None, None, 'vlm.model.language_model.layers.0', cache_pos=0)


def fixture(root):
  sample, reserved = root / 'allowed', root / 'reserved'
  sample.mkdir()
  reserved.mkdir()
  for filename in p.INPUT_FILES:
    np.save(sample / filename, np.array([1.], np.float32), allow_pickle=False)
  calibration, holdout = root / 'calibration.json', root / 'holdout.json'
  calibration.write_text(json.dumps([str(sample)]))
  holdout.write_text(json.dumps([str(reserved)]))
  return sample, reserved, p.CalibrationSelection(calibration, holdout)


def run(root, tinygrad_cpu=False):
  rng = np.random.default_rng(20261116)
  checks = []
  for signed in (False, True):
    for shape in ((1, 32), (17, 64), (33, 96)):
      values = rng.integers(-8 if signed else 0, 8 if signed else 16, shape, dtype=np.int16)
      packed = p.pack_codes(values, signed)
      np.testing.assert_array_equal(packed, independent_pack(values))
      np.testing.assert_array_equal(p.unpack_codes(packed, shape[0], signed), values)
  checks.append('native_layout_independent_scalar_bit_mapping_signed_unsigned_partial_rows')
  a = np.array([[0., 15., .5, 1.5], [0., 0., 0., 0.], [1., 2., 1e6, 0.]], np.float32)
  aq, sa = p.quantize_activation(a)
  np.testing.assert_array_equal(aq[0], [0, 15, 0, 2])
  np.testing.assert_array_equal(aq[1], 0)
  check(sa[1] == 1 and aq[2, 2] == 15 and np.max(aq) == 15, 'Zero/outlier row scale')
  subnormal = p.decode_bf16(np.array([[1, 2, 3, 4]], np.uint16))
  qs, ss = p.quantize_activation(subnormal)
  check(0 < ss[0] < np.finfo(np.float32).tiny and qs[0, 3] == 15, 'Positive BF16 subnormal scale was floored')
  rejects(lambda: p.quantize_activation(np.array([[-1.]], np.float32)))
  rejects(lambda: p.quantize_activation(np.array([[np.inf]], np.float32)))
  rejects(lambda: p.quantize_activation(np.array([[np.nan]], np.float32)))
  checks.append('unsigned_quantization_ties_zero_large_outlier_negative_nonfinite')
  w = np.array([[0., 0., 0., 0.], [-7., 7., -.5, 1.5], [1., 2., 1e5, -1e5]], np.float32)
  for mode in ('scalar', 'peroutput'):
    wq, sw = p.quantize_weight(w, mode)
    np.testing.assert_array_equal(wq[0], 0)
    for row in range(len(w)):
      scale = float(sw[row] if mode == 'peroutput' else sw[0])
      expected = [min(7, max(-7, round(float(np.float32(v / np.float32(scale)))))) for v in w[row]]
      np.testing.assert_array_equal(wq[row], expected)
  checks.append('signed_weight_scalar_and_peroutput_original_float32_rounding')
  a_codes = rng.integers(0, 16, (7, 96), dtype=np.uint8)
  w_codes = rng.integers(-7, 8, (19, 96), dtype=np.int8)
  sa = np.array([2. ** (i - 8) for i in range(7)], np.float32)
  sw = np.array([2. ** ((i % 5) - 8) for i in range(19)], np.float32)
  actual = p.scaled_reference(a_codes, w_codes, sa, sw)
  expected = np.empty((7, 19), np.float32)
  for m in range(7):
    for n in range(19):
      integer = sum(int(a_codes[m, k]) * int(w_codes[n, k]) for k in range(96))
      value = np.float32(np.float32(integer) * np.float32(sa[m] * sw[n]))
      bits = struct.unpack('<I', struct.pack('<f', value))[0]
      top = (bits + 32767 + ((bits >> 16) & 1)) >> 16
      expected[m, n] = struct.unpack('<f', struct.pack('<I', (top & 65535) << 16))[0]
  np.testing.assert_array_equal(actual, expected)
  checks.append('independent_scaled_integer_dot_scale_association_bf16_rounding')
  sample, reserved, selection = fixture(root)
  rejects(lambda: selection.identity(reserved))
  rejects(lambda: selection.identity(root / 'remaining-route'))
  indices = p.representative_rows(2027, 32)
  check(indices.size == 32 and indices[0] == 0 and indices[-1] == 2026, 'Representative row coverage')
  np.testing.assert_array_equal(indices, p.representative_rows(2027, 32))
  checks.append('calibration_only_selection_reserved_remaining_rejection_deterministic_rows')
  weight = p.round_bf16(rng.normal(size=(8, 32)).astype(np.float32))
  model = Model(weight)
  config_hash = hashlib.sha256(p.canonical(model.config)).hexdigest()
  reference = {'schema': p.SCHEMA, 'checkpoint': {'config': model.config, 'canonical_config_sha256': config_hash},
               'settings': {'precision': 'fp8', 'candidate_installed_during_collection': False}}
  reference['identity_sha256'] = hashlib.sha256(p.canonical(reference)).hexdigest()
  collector = p.Collector(selection, root / 'capture', reference, rows_per_layer=4, capture=p.capture_numpy)
  prepared = {name: np.array([1.], np.float32) for name in p.PREPARED_KEYS}
  prepared['input_ids'] = np.ones((1, 160), np.int32)
  x = p.round_bf16(np.abs(rng.normal(size=(1, 160, 32))).astype(np.float32))
  old_jit = model.prefill_jit
  with collector.sample(model, sample, 129, prepared):
    result = model.prefill_jit(x)
  np.testing.assert_array_equal(result, x @ weight.T)
  capture_manifest = collector.finish()
  record = capture_manifest['samples'][0]['layers'][0]
  with np.load(root / 'capture' / record['file'], allow_pickle=False) as data:
    np.testing.assert_array_equal(data['indices'], [0, 42, 85, 128])
    np.testing.assert_array_equal(p.decode_bf16(data['rows_bf16']), x.reshape(160, 32)[data['indices']])
    np.testing.assert_array_equal(data['channel_max'], x.reshape(160, 32)[:129].max(axis=0))
  check(model.prefill_jit is old_jit and old_jit.resets == 1, 'Eager collector did not restore JIT')
  rejects(lambda: collector.sample(model, sample, 129, prepared).__enter__())
  checks.append('collector_prefill_eager_original_output_unpadded_stats_bounded_rows_restore_no_repeat')
  # Candidate hook dispatches only actual language prefill fc2, then restores cleanly.
  wq, sw = p.quantize_weight(weight)
  runtime = {'vlm.model.language_model.layers.0.mlp.fc2': {'packed': p.pack_codes(wq, True), 'scale': sw, 'shape': weight.shape}}
  manifest = dict(schema=p.SCHEMA, complete=True, layout=p.LAYOUT, equalization=None, clipping=None,
                  outlier_correction=None, checkpoint=reference['checkpoint'],
                  records=[dict(name=next(iter(runtime)), shape=list(weight.shape))])
  events = []

  def quantize(values):
    q, scale = p.quantize_activation(values)
    return p.pack_codes(q, False), scale

  def kernel(a, b, scales, *, rows, columns, bias, dtype):
    events.append((rows, columns))
    check(bias is None, 'Unexpected bias')
    return p.scaled_reference(p.unpack_codes(a, rows, False), p.unpack_codes(b, columns, True), *scales)

  uninstall = p.install_fc2(model, manifest, runtime, quantize, kernel)
  output = model._language(x)
  qa, sa = p.quantize_activation(x.reshape(160, 32))
  np.testing.assert_array_equal(output.reshape(160, 8), p.scaled_reference(qa, wq, sa, sw))
  check(events == [(160, 8)], 'Candidate missed intended prefill')
  for prefix, cache_pos, rows in (('vlm.model.language_model.layers.0', 10, 160),
                                 ('vlm.model.language_model.layers.0', None, 160),
                                 ('vlm.model.language_model.layers.0', 0, 1),
                                 ('expert.expert.layers.0', None, 160),
                                 ('vlm.model.visual.encoder.layers.0', 0, 160)):
    result = model.text_layer(x[:, :rows], None, None, prefix, cache_pos=cache_pos)
    np.testing.assert_array_equal(result, x[:, :rows] @ weight.T)
  check(events == [(160, 8)], 'Candidate changed decode/vision/expert/nonprefill')
  uninstall()
  np.testing.assert_array_equal(model._language(x), x @ weight.T)
  checks.append('optional_scaled_callable_hook_fc2_only_no_decode_vision_expert_effect_uninstall')
  # Adapter storage has the separate scaled kernel's byte view and output-scale shape.
  class ByteKernel:
    @staticmethod
    def quantize_pack(values):
      q, scale = p.quantize_activation(values)
      words = p.pack_codes(q, False)
      return words.view(np.uint8).reshape(*words.shape[:-1], 8), scale

    @staticmethod
    def gemm(a, w, sa, sw, *, rows, columns, bias):
      aw = np.ascontiguousarray(a).view(np.int32).reshape(*a.shape[:-1], 2)
      ww = np.ascontiguousarray(w).view(np.int32).reshape(*w.shape[:-1], 2)
      return p.scaled_reference(p.unpack_codes(aw, rows, False), p.unpack_codes(ww, columns, True), sa, sw)

  adapted = p.scaled_module_adapter(ByteKernel, runtime, lambda value: value.copy())
  uninstall = p.install_fc2(model, manifest, *adapted)
  np.testing.assert_array_equal(model._language(x).reshape(160, 8), p.scaled_reference(qa, wq, sa, sw))
  uninstall()
  checks.append('scaled_module_byte_view_adapter_and_peroutput_scales')
  # Tiny BF16 checkpoint exercises the complete exporter/loader without real-model assumptions.
  checkpoint = root / 'checkpoint'
  checkpoint.mkdir()
  name = 'vlm.model.language_model.layers.0.mlp.fc2.weight'
  raw = p.encode_bf16_exact(weight).astype('<u2').tobytes()
  header = p.canonical({name: dict(dtype='BF16', shape=list(weight.shape), data_offsets=[0, len(raw)])})
  header += b' ' * ((-len(header)) % 8)
  (checkpoint / 'weight.safetensors').write_bytes(struct.pack('<Q', len(header)) + header + raw)
  (checkpoint / 'config.json').write_bytes(p.canonical(model.config))
  (checkpoint / 'model.safetensors.index.json').write_bytes(p.canonical({'weight_map': {name: 'weight.safetensors'}}))
  exported = p.export_bundle(checkpoint, root / 'bundle')
  loaded, arrays = p.load_bundle(root / 'bundle')
  check(loaded['identity_sha256'] == exported['identity_sha256'], 'Bundle identity drift')
  np.testing.assert_array_equal(p.unpack_codes(arrays[name[:-7]]['packed'], 8, True), wq)
  np.testing.assert_array_equal(arrays[name[:-7]]['scale'], sw)
  scale_path = root / 'bundle' / exported['records'][0]['scale']['file']
  with scale_path.open('ab') as stream:
    stream.write(b'tamper')
  rejects(lambda: p.load_bundle(root / 'bundle'))
  checks.append('bf16_checkpoint_export_bundle_readback_tamper_rejection')
  rejects(lambda: p.Collector(selection, root / 'too-large', reference, rows_per_layer=64, budget_bytes=32))
  checks.append('capture_hard_budget_rejection')
  actual_scaled = Path(__file__).with_name('agent-int4-scaled-kernel.py')
  scaled_identity = None
  if actual_scaled.exists():
    kernel_spec = importlib.util.spec_from_file_location('agent_int4_scaled_cpu_only', actual_scaled)
    kernel_module = importlib.util.module_from_spec(kernel_spec)
    kernel_spec.loader.exec_module(kernel_module)
    for shape in ((1, 32), (17, 96)):
      a = rng.integers(0, 16, shape, dtype=np.uint8)
      w = rng.integers(-7, 8, shape, dtype=np.int8)
      for values, signed, actual in ((a, False, kernel_module.pack_unsigned_cpu(a)),
                                     (w, True, kernel_module.pack_weights_cpu(w))):
        words = independent_pack(values)
        expected_bytes = words.view(np.uint8).reshape(*words.shape[:-1], 8)
        np.testing.assert_array_equal(expected_bytes, actual)
    scaled_identity = dict(file=str(actual_scaled), sha256=p.sha_file(actual_scaled))
    checks.append('separate_scaled_kernel_actual_cpu_packers_exact_byte_compatibility')
  if tinygrad_cpu:
    import os
    check(os.environ.get('DEV') == 'CPU', 'Tinygrad CPU check requires DEV=CPU before import')
    from tinygrad import Tensor, dtypes
    tensor = Tensor(x, device='CPU').cast(dtypes.bfloat16).realize()
    indices = p.representative_rows(129, 4)
    observed = p.capture_tinygrad(tensor, 129, indices, lambda value: value.float().numpy())
    expected = p.capture_numpy(x, 129, indices)
    for key in ('rows_bf16', 'indices', 'channel_max', 'channel_min', 'zero_count'):
      np.testing.assert_array_equal(observed[key], expected[key])
    for key in ('channel_sum', 'channel_sum_squares'):
      np.testing.assert_allclose(observed[key], expected[key], rtol=2e-6, atol=2e-5)
    rejects(lambda: p.capture_tinygrad(Tensor(-x, device='CPU').cast(dtypes.bfloat16), 129, indices,
                                       lambda value: value.float().numpy()))
    checks.append('actual_tinygrad_cpu_collector_reductions_bounded_gather_and_negative_guard')
  return dict(cpu_only=True, all_pass=True, checks=checks, source_sha256=p.sha_file(p.__file__),
              separate_scaled_kernel_source=scaled_identity,
              limitations='CPU mock dispatch and numerical checks; no actual GPU collector or scaled kernel validation')


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--artifacts', type=Path, required=True)
  parser.add_argument('--tinygrad-cpu', action='store_true')
  args = parser.parse_args()
  args.artifacts.mkdir(parents=True, exist_ok=True)
  # Keep every generated fixture under the owned artifact prefix.
  with tempfile.TemporaryDirectory(prefix='checks-', dir=args.artifacts) as temp:
    result = run(Path(temp), args.tinygrad_cpu)
  p.json_write(args.artifacts / 'cpu-check.json', result)
  print(json.dumps(result, indent=2))


if __name__ == '__main__':
  main()
