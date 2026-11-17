#!/usr/bin/env python3
"""CPU controls by default. --gpu is ONLY for the parent's serialized GPU queue."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from types import MethodType, SimpleNamespace
import numpy as np

HERE = Path(__file__).resolve().parent


def load():
  spec = importlib.util.spec_from_file_location('activation_sparsity', HERE/'agent-activation-sparsity.py')
  m = importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def cases(m):
  rng = np.random.default_rng(4239)
  shape = (35, 48)
  dense = np.full(shape, 2, np.float32)
  sparse = np.zeros(shape, np.float32);sparse[:, ::4] = 2;sparse[:, 1::4] = .125
  mix = np.maximum(rng.standard_normal(shape).astype(np.float32), 0)**2
  # Scalar sparsity can be high while every full hardware tile is ineligible.
  scattered = np.zeros(shape, np.float32);scattered[::16, :3] = 1
  outlier = mix.copy();outlier[0, :6] = [np.inf, -np.inf, 1e30, -2, 448, 449]
  sub = np.zeros(shape, np.float32);words = m.bf16(sub);words[0, :8] = [0, 0x8000, 1, 0x8001, 0x7f7f, 0x7f80, 0xff80, 0]
  for name, value, valid, scale in [('dense', dense, 35, 1), ('2of4', sparse, 35, 1), ('relu2-random', mix, 31, .03125),
      ('high-scalar-ineligible', scattered, 35, 1), ('overflow-outlier', outlier, 33, 1e-12)]:
    yield name, m.bf16(value), valid, np.float32(scale)
  yield 'subnormal-signedzero', words, 17, np.float32(1)
  # Padded rows are deliberately dense; their exclusion must be exact.
  pad = np.zeros((48, 64), np.float32);pad[17:] = 444
  yield 'dense-padding-excluded', m.bf16(pad), 17, np.float32(1)
  yield 'zero', m.bf16(np.zeros((16, 32), np.float32)), 16, np.float32(1)


def main():
  p = argparse.ArgumentParser(description=__doc__)
  p.add_argument('--output', type=Path, required=True)
  p.add_argument('--compile', action='store_true')
  p.add_argument('--gpu', action='store_true')
  args = p.parse_args()
  if not args.gpu and os.environ.get('DEV', 'CPU') != 'CPU': raise ValueError('CPU checks require DEV=CPU')
  from tinygrad import Tensor, TinyJit, dtypes, Device
  from amd_gemm import pack
  m = load()
  gpu_device_identity = None
  if args.gpu:
    device_spec = importlib.util.spec_from_file_location('sparsity_device_guard', HERE/'agent-activation-sparsity-device.py')
    device_guard = importlib.util.module_from_spec(device_spec);device_spec.loader.exec_module(device_guard)
    gpu_device_identity = device_guard.require_gpu_device()
    fix_spec = importlib.util.spec_from_file_location('sparsity_existing_dispatch_fix', HERE/'prefill-kernels.py')
    fix = importlib.util.module_from_spec(fix_spec);fix_spec.loader.exec_module(fix);fix.fix_dispatch_packet()
    import wgp_compiler
    wgp_compiler.install(Device[Device.DEFAULT])
  selected_spec = importlib.util.spec_from_file_location('sparsity_actual_quantizer', HERE/'quantize-packed.py')
  selected_quantizer = importlib.util.module_from_spec(selected_spec);selected_spec.loader.exec_module(selected_quantizer)
  report = dict(cpu_only=not args.gpu, complete=False, all_pass=False, sources={
    f.name: sha(f) for f in HERE.glob('agent-activation-sparsity*.py')}, controls=[], compiler=[],
    gpu_device_identity=gpu_device_identity)
  report['quantizer_dependencies'] = {str(p): sha(p) for p in (
    HERE/'quantize-packed.py', HERE/'quantize-reduction.py', HERE/'wgp_compiler.py', HERE/'prefill-kernels.py', HERE.parent/'amd_gemm.py')}
  args.output.parent.mkdir(parents=True, exist_ok=True)
  def upload(a): return Tensor(np.asarray(a).copy(), device='CPU').to(Device.DEFAULT).realize()
  def quantize(x, scale):
    return selected_quantizer.quantize(x, scale)[0]
  def quantized(words, s, x, scale):
    if args.gpu: return quantize(x, scale)
    return upload(m.pack_cpu(m.selected_quantize_cpu(words, s))).bitcast(dtypes.fp8e4m3)
  try:
    # Exhaustive finite E4M3 boundaries, ties to even, clamp and signed zero.
    table = m.fp8_table()
    mid = (table[:-1]+table[1:])/2
    tests = np.concatenate((table, mid, np.nextafter(mid, -np.inf), np.nextafter(mid, np.inf),
                            -table, -mid, [-0., -500., 500., np.inf, -np.inf])).astype(np.float32)
    tie_words = m.bf16(mid)
    neighbors = np.concatenate((tie_words-1, tie_words, tie_words+1))
    tests = np.concatenate((tests, m.bf32(neighbors), m.bf32(neighbors | np.uint16(0x8000))))
    # The selected quantizer receives BF16. Match that input rounding first.
    words = m.bf16(tests)
    words = np.pad(words, (0, (-len(words)) % 16)).reshape(1, -1)
    x = upload(words).bitcast(dtypes.bfloat16)
    scale = upload(np.array([1], np.float32)).reshape(())
    wanted = m.pack_cpu(m.selected_quantize_cpu(words, 1))
    if args.gpu:
      actual = quantize(x, scale).bitcast(dtypes.uint8).numpy()
    else:
      import torch
      torch.set_num_threads(4)
      values = np.clip(m.bf32(words), -448, 448)
      actual = m.pack_cpu(torch.from_numpy(values).to(torch.float8_e4m3fn).view(torch.uint8).numpy())
      report['cpu_conversion_limit'] = 'CPU tinygrad flushes FP8 denormals. Independent Torch E4M3 controls validate NumPy; AMD conversion requires --gpu.'
    np.testing.assert_array_equal(actual, wanted)
    report['quantizer_boundary_values'] = words.size
    # Layout address equation is independent of transpose pack/unpack code.
    q = np.arange(35*48, dtype=np.uint32).reshape(35, 48).astype(np.uint8)
    packed = m.pack_cpu(q)
    for r in range(35):
      for k in range(48):
        address=(r//16*3+k//16)*256+(k%16//4)*64+r%16*4+k%4
        assert packed.ravel()[address] == q[r, k]
    report['packed_layout_independent_addresses'] = q.size
    for name, words, valid, scale_value in cases(m):
      x = upload(words).bitcast(dtypes.bfloat16)
      scale = upload(np.array([scale_value], np.float32)).reshape(())
      ctl = upload(np.array([valid, 1], np.uint32))
      scratch = Tensor.empty(m.BLOCKS, m.S, dtype=dtypes.uint32).realize()
      output = Tensor.empty(m.S+4, dtype=dtypes.uint32).realize()
      q = quantized(words, scale_value, x, scale)
      np.testing.assert_array_equal(q.bitcast(dtypes.uint8).numpy(), m.pack_cpu(m.selected_quantize_cpu(words, scale_value)))
      before_x = x.bitcast(dtypes.uint16).numpy().copy()
      before_q = q.bitcast(dtypes.uint8).numpy().copy()
      result = m.observe(x, q, scale, ctl, scratch, output, cpu=not args.gpu).numpy()
      expected = m.oracle(words, before_q, scale_value, valid)
      np.testing.assert_array_equal(result[:m.S].reshape(2, m.F), expected)
      assert result[-4:].tolist() == [valid, 1, int(np.asarray(scale_value).view(np.uint32)), m.MAGIC]
      np.testing.assert_array_equal(x.bitcast(dtypes.uint16).numpy(), before_x)
      np.testing.assert_array_equal(q.bitcast(dtypes.uint8).numpy(), before_q)
      description = m.describe(expected, valid, words.shape[1])
      if name == 'high-scalar-ineligible':
        assert description['bf16_before']['zero_fraction'] > .99
        assert description['bf16_before']['tile16_k32_eligible_fraction'] == 0
      report['controls'].append(dict(name=name, shape=list(words.shape), valid_rows=valid, all_exact=True,
                                     bf16=description['bf16_before'], fp8=description['fp8_after_actual']))
    # NaNs need counting even though the independent quantizer rejects payload conversion.
    words = np.zeros((17, 48), np.uint16);words[0, 0] = 0x7fc1
    packed = np.zeros((32, 48), np.uint8);packed.ravel()[0] = 0x7f
    expected = m.oracle(words, packed, 1, 17)
    result = m.observe(upload(words).bitcast(dtypes.bfloat16), upload(packed).bitcast(dtypes.fp8e4m3),
      upload(np.array([1], np.float32)), upload(np.array([17, 1], np.uint32)),
      Tensor.empty(m.BLOCKS, m.S, dtype=dtypes.uint32).realize(), Tensor.empty(m.S+4, dtype=dtypes.uint32).realize(),
      cpu=not args.gpu).numpy()
    np.testing.assert_array_equal(result[:m.S].reshape(2, m.F), expected)
    assert expected[:, 4].tolist() == [1, 1]
    report['nan_count_control'] = True
    # Real pinned TinyJit: two traces, then CURRENT x/scale/control mutation on replay.
    rows, width = 35, 48
    x = upload(m.bf16(np.ones((rows, width), np.float32))).bitcast(dtypes.bfloat16)
    scale = upload(np.array([1], np.float32)).reshape(())
    ctl = upload(np.array([35, 1], np.uint32))
    scratch = Tensor.empty(m.BLOCKS, m.S, dtype=dtypes.uint32).realize()
    output = Tensor.empty(m.S+4, dtype=dtypes.uint32).realize()
    q = upload(m.pack_cpu(m.selected_quantize_cpu(m.bf16(np.ones((rows, width), np.float32)), 1))).bitcast(dtypes.fp8e4m3)
    identities = [t.uop.buffer for t in (ctl, scratch, output)]
    traces = []
    def run(current_x, current_q, current_scale):
      traces.append(1)
      selected_q = quantize(current_x, current_scale) if args.gpu else current_q
      m.observe(current_x, selected_q, current_scale, ctl, scratch, output, cpu=not args.gpu)
      return (current_x.float()+1).realize()
    jit = TinyJit(run)
    mutations = []
    for i in range(6):
      words = m.bf16(np.zeros((rows, width), np.float32) if i % 2 else np.ones((rows, width), np.float32)*(i+1))
      valid, s = (17 if i % 2 else 35), np.float32(1+i*.25)
      x.assign(upload(words).bitcast(dtypes.bfloat16)).realize()
      scale.assign(upload(np.array(s, np.float32))).realize()
      q.assign(upload(m.pack_cpu(m.selected_quantize_cpu(words, s))).bitcast(dtypes.fp8e4m3)).realize()
      ctl.assign(upload(np.array([valid, i+10], np.uint32))).realize()
      output.assign(Tensor.full(output.shape, 0xdeadbeef, dtype=dtypes.uint32)).realize()
      y = jit(x, q, scale).numpy()
      raw = output.numpy()
      expected = m.oracle(words, m.pack_cpu(m.selected_quantize_cpu(words, s)), s, valid)
      np.testing.assert_array_equal(raw[:m.S].reshape(2, m.F), expected)
      np.testing.assert_array_equal(y, m.bf32(words)+1)
      assert raw[-4:].tolist() == [valid, i+10, int(s.view(np.uint32)), m.MAGIC]
      assert all(t.uop.buffer is identity for t, identity in zip((ctl, scratch, output), identities, strict=True))
      mutations.append(dict(iteration=i, replay=i >= 2, valid_rows=valid, epoch=i+10, exact=True))
    assert len(traces) == 2
    report['captured_current_mutations'] = mutations
    report['captured_python_traces'] = len(traces)
    report['persistent_allocations_unchanged'] = True
    if not args.gpu:
      # Exercise the complete Python hook over a real captured CPU graph. The
      # mock supplies the exact selected closure contract, not an eager callback.
      state = dict(prefill=False, prefix=None, index=0)
      calls = []
      model = SimpleNamespace(weights={'control': Tensor.zeros(1).realize()}, padded_prefill=object())
      def model_quantize(this, x):
        if not state['prefill'] or state['index'] != 3: raise ValueError('Wrong closure boundary')
        state['index'] += 1;calls.append(state['prefix'])
        value = pack(x.float().cast(dtypes.fp8e4m3).reshape(-1, x.shape[-1])).realize()
        return selected_quantizer.PackedActivation(value, x.numel()//x.shape[-1], x.shape[-1]), Tensor([1.], dtype=dtypes.float32).reshape(())
      def model_linear(this, x, name, quantized_input=None, heads=None):
        assert quantized_input is not None and isinstance(quantized_input[0], selected_quantizer.PackedActivation)
        return (x.float()+1).realize()
      model.quantize, model.linear = MethodType(model_quantize, model), MethodType(model_linear, model)
      original = model.linear
      hook = m.Observer(model, cpu=True)
      hook_x = Tensor.ones(1, 144, 48, dtype=dtypes.bfloat16).contiguous().realize()
      def forward(x):
        for layer in range(28):
          state.update(prefill=True, prefix=f'vlm.model.language_model.layers.{layer}', index=3)
          y = model.linear(x, state['prefix']+'.mlp.fc2')
          assert state['index'] == 4
        state.update(prefill=False, prefix=None, index=0)
        return y
      hooked_jit = TinyJit(forward)
      for iteration in range(5):
        value, valid = (0 if iteration % 2 else 1), (128 if iteration % 2 else 139)
        hook_x.assign(Tensor.full(hook_x.shape, value, dtype=dtypes.bfloat16)).realize()
        hook.prepare(valid)
        y = hooked_jit(hook_x).numpy()
        stats = hook.read(width=48)
        assert all(v['bf16_before']['zeros'] == (valid*48 if value == 0 else 0) for v in stats.values())
        np.testing.assert_array_equal(y, np.full(hook_x.shape, value+1, np.float32))
      assert len(calls) == 56 and set(hook.trace_counts.values()) == {2}
      bad_quantized = (None, None)
      state.update(prefill=True, prefix='vlm.model.language_model.layers.0', index=3)
      try: model.linear(hook_x, state['prefix']+'.mlp.fc2', bad_quantized)
      except ValueError: pass
      else: raise AssertionError('Duplicate quantized input accepted')
      hook.restore();assert model.linear == original
      report['full_observer_hook_cpu_jit'] = dict(layers=28, samples=5, captured_replays=3,
        original_quantizer_calls=56, duplicate_quantization_rejected=True, restoration_exact=True)
    # Bad masks and counter overflow/corruption fail closed.
    for invalid in (0, rows+1):
      try: m.oracle(words, m.pack_cpu(m.selected_quantize_cpu(words, 1)), 1, invalid)
      except ValueError: pass
      else: raise AssertionError('Invalid mask accepted')
      invalid_result = m.observe(x, q, scale, upload(np.array([invalid, 99], np.uint32)), scratch, output,
                                 cpu=not args.gpu).numpy()
      assert invalid_result[-1] == 0, 'Invalid mask must fail safely without out-of-bounds reads'
    damaged = expected.copy();damaged[0, 0] = np.uint32(0xffffffff)
    try: m.describe(damaged, valid, width)
    except ValueError: pass
    else: raise AssertionError('Overflow/corruption accepted')
    report['invalid_mask_and_counter_rejections'] = 3
    # The route identity includes all token metadata, rotary positions, full
    # pixel bytes, learned history, navigation, and every timestamp.
    route_spec = importlib.util.spec_from_file_location('activation_sparsity_runner_cpu', HERE/'agent-activation-sparsity-run.py')
    route = importlib.util.module_from_spec(route_spec);route_spec.loader.exec_module(route)
    prepared = dict(actual_prompt_length=17, tokenized_data=dict(pixel_values=np.arange(48, dtype=np.uint8).reshape(2, 24),
      input_ids=np.full((1, 32), 135087, np.int32), mm_token_type_ids=np.zeros((1, 32), np.int32), image_grid_thw=np.array([[1, 2, 2]])),
      ego_history_xyz=np.zeros((1, 1, 20, 3), np.float32), ego_history_rot=np.ones((1, 1, 20, 3, 3), np.float32),
      _activation_sparsity_source=dict(camera_indices=np.array([1, 6]), absolute_timestamps=np.array([100, 200]), nav_text=['Follow the road.']))
    packed = (np.zeros((1, 32), np.int32), np.zeros((3, 1, 32), np.int64), 0, np.zeros((2, 2), np.float32))
    original_identity = route.prepared_identity(prepared, packed)
    for name, array in (('ids', packed[0]), ('rotaries', packed[1]), ('pixels', prepared['tokenized_data']['pixel_values']),
       ('mm_ids', prepared['tokenized_data']['mm_token_type_ids']), ('history', prepared['ego_history_xyz']),
       ('timestamps', prepared['_activation_sparsity_source']['absolute_timestamps'])):
      old = array.flat[0].copy();array.flat[0] = old+1
      assert route.prepared_identity(prepared, packed) != original_identity, name
      array.flat[0] = old
    prepared['_activation_sparsity_source']['nav_text'] = ['Turn right.']
    assert route.prepared_identity(prepared, packed) != original_identity
    report['route_input_identity_mutations_detected'] = 7
    if args.compile:
      outdir = args.output.parent/(args.output.stem+'-compiled');outdir.mkdir(exist_ok=True)
      for rows, width in ((35, 48), (2112, 9216)):
        for stage in ('partial', 'final'):
          source, binary = m.compiled(rows, width, stage, cpu=False)
          stem = outdir/f'{rows}-{width}-{stage}'
          stem.with_suffix('.cpp').write_text(source);stem.with_suffix('.hsaco').write_bytes(binary)
          helper_spec = importlib.util.spec_from_file_location('sparsity_readonly_disassembler', HERE/'agent-int4k32-compile.py')
          helper = importlib.util.module_from_spec(helper_spec);helper_spec.loader.exec_module(helper)
          asm = helper.disassemble(binary)
          stem.with_suffix('.s').write_text(asm)
          report['compiler'].append(dict(shape=[rows, width], stage=stage, bytes=len(binary),
             source_sha256=hashlib.sha256(source.encode()).hexdigest(), binary_sha256=hashlib.sha256(binary).hexdigest(),
             resources=helper.metadata(binary),
             scratch_bytes_per_layer=m.BLOCKS*m.S*4, output_bytes_per_layer=(m.S+4)*4))
    report.update(complete=True, all_pass=True)
  except BaseException as e:
    report['error'] = dict(type=type(e).__name__, message=str(e));raise
  finally:
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
  print(json.dumps(dict(cpu_only=not args.gpu, all_pass=True, controls=len(report['controls']), output=str(args.output))))


if __name__ == '__main__': main()
