#!/usr/bin/env python3
"""CPU proof using ACTUAL frozen quantize-packed and batched-vision installers.

Only the GEMM execution endpoint and minimal model body are mocked. Quantizers,
their code, modules, closures, scale selection, and wrapper delegation are real.
"""
import argparse
from collections import Counter
import hashlib
import importlib.util
import inspect
import json
import os
from pathlib import Path
import sys
from types import MethodType, SimpleNamespace
from unittest.mock import patch
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = Path('/mnt/nvme-500/tmp/alpamayo-speed-20261116/activation-sparsity-devicefix-source/alpamayo/experiments')


def load(path, name):
  spec = importlib.util.spec_from_file_location(name, path)
  module = importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def main():
  parser = argparse.ArgumentParser(description=__doc__);parser.add_argument('--output', type=Path, required=True)
  parser.add_argument('--baseline-root', type=Path, default=ROOT);args = parser.parse_args()
  if os.environ.get('DEV', 'CPU') != 'CPU': raise ValueError('Run with DEV=CPU')
  from tinygrad import Tensor, TinyJit, dtypes, Device
  from amd_gemm import unpack
  Tensor.manual_seed(19)
  overlay = load(HERE/'agent-activation-sparsity-closurefix.py', 'sparsity_closure_adapter')
  core = load(args.baseline_root/'agent-activation-sparsity.py', 'sparsity_gpu_tested_core')
  # Exact runtime module names used by benchmark-pipeline.local_module.
  packed = load(args.baseline_root/'quantize-packed.py', 'quantize_packed')
  vision = load(args.baseline_root/'batched-vision.py', 'batched_vision')
  report = dict(cpu_only=True, complete=False, all_pass=False, controls=[], negatives=[],
    baseline_sources={name:overlay.sha(args.baseline_root/name) for name in overlay.APPROVED},
    numeric_source_sha256=overlay.sha(core.__file__), overlay_sources={
      p.name:overlay.sha(p) for p in sorted(HERE.glob('agent-activation-sparsity-closure*.py'))})
  names = [f'vlm.model.language_model.layers.{i}' for i in range(28)]
  model = SimpleNamespace(weights={}, weight_scales={})
  weight = Tensor(np.eye(32, dtype=np.float32)).cast(dtypes.fp8e4m3).realize()
  for prefix in names:
    for suffix in ('self_attn.q_proj', 'self_attn.o_proj', 'mlp.fc1', 'mlp.fc2'):
      model.weights[prefix+'.'+suffix+'.weight'] = weight
      model.weight_scales[prefix+'.'+suffix+'.weight'] = Tensor([1.], dtype=dtypes.float32).reshape(()).realize()
  def base_quantize(this, x): raise AssertionError('Language prefill escaped the actual packed quantizer')
  def base_linear(this, *a, **k): raise AssertionError('Language prefill escaped the actual packed linear')
  def base_norm(this, x, *a, **k): return x
  def body(this, x, cos, sin, prefix, *a, **k):
    # Four actual model quantization sites: shared QKV, output projection, fc1,
    # then post-ReLU-square fc2. State is managed only by the real installer.
    shared = this.quantize(x)
    this.linear(x, prefix+'.self_attn.q_proj', shared)
    value = this.linear(x, prefix+'.self_attn.o_proj')
    value = this.linear(value, prefix+'.mlp.fc1').relu().square()
    return this.linear(value, prefix+'.mlp.fc2').realize()
  model.quantize, model.linear = MethodType(base_quantize, model), MethodType(base_linear, model)
  model.norm, model.text_layer = MethodType(base_norm, model), MethodType(body, model)
  scales = {prefix+'/'+str(slot): float(2**(i%3)) if slot == 3 else 1.
            for i, prefix in enumerate(names) for slot in range(4)}
  installed = packed.install(model, fixed_scales=scales)
  vision.install(model)
  model.padded_prefill = object()
  assert installed == 112
  binding = overlay.Binding(model, args.baseline_root)
  assert binding.outer.__func__.__module__ == 'batched_vision'
  assert binding.inner.__func__.__module__ == 'quantize_packed'
  report['actual_binding'] = binding.evidence
  outer = model.quantize
  original_custom = Tensor.custom_kernel
  backend_calls = []
  def endpoint(*args, **kwargs): raise AssertionError('CPU endpoint must be intercepted before dispatch')
  def dispatch(out, *tensors, fxn, **kwargs):
    if getattr(fxn, 'func', None) is not endpoint:
      return original_custom(out, *tensors, fxn=fxn, **kwargs)
    q, w, xs, ws = tensors
    # Mock only the AMD GEMM endpoint. Real selected packed bytes/scales arrive
    # through the actual packed linear closure and original tuple delegation.
    backend_calls.append((q.uop, xs.uop))
    rows, columns = out.shape
    a = unpack(q)[:rows].float();b = unpack(w)[:columns].float()
    value = ((a @ b.T)*(xs*ws)).cast(dtypes.bfloat16)
    out.assign(value).realize()
    return [out, *tensors]
  counts = Counter()
  packed_fn, outer_fn = binding.inner.__func__.__code__, binding.outer.__func__.__code__
  quantize_fn = packed.quantize.__code__
  def profile(frame, event, arg):
    if event == 'call':
      if frame.f_code is packed_fn:
        state = frame.f_locals['state'];counts[('inner', state['prefix'], state['index'])] += 1
      elif frame.f_code is outer_fn: counts[('outer',)] += 1
      elif frame.f_code is quantize_fn: counts[('actual_quantize_function',)] += 1
  def forward(x):
    for prefix in names: value = model.text_layer(x, None, None, prefix, cache_pos=0)
    return value
  try:
    with patch.object(packed, '_gemm', endpoint), patch.object(Tensor, 'custom_kernel', dispatch):
      x = Tensor.full((1,144,32), 2, dtype=dtypes.bfloat16).contiguous().realize()
      reference = forward(x).bitcast(dtypes.uint16).numpy()
      before_linear = model.linear
      sys.setprofile(profile)
      observer = overlay.make_observer(model, core, args.baseline_root, cpu=True)
      assert model.quantize is outer and not counts, 'Constructor must restore the exact outer method without quantization'
      jit = TinyJit(forward)
      for iteration in range(5):
        value = (2, 0, 4, 0, 2)[iteration];valid = (139, 128, 139, 128, 139)[iteration]
        x.assign(Tensor.full(x.shape, value, dtype=dtypes.bfloat16)).realize()
        observer.prepare(valid)
        actual = jit(x).bitcast(dtypes.uint16).numpy()
        stats = observer.read(width=32)
        expected_output = core.bf16(np.full(x.shape, value*value, np.float32))
        np.testing.assert_array_equal(actual, expected_output)
        if iteration == 0: np.testing.assert_array_equal(actual, reference)
        for prefix in names:
          scale = np.float32(scales[prefix+'/3'])
          words = core.bf16(np.full((144,32), value*value, np.float32))
          expected = core.oracle(words, core.pack_cpu(core.selected_quantize_cpu(words, scale)), scale, valid)
          summary = core.describe(expected, valid, 32)
          assert stats[prefix+'.mlp.fc2'] == dict(actual_scale=float(scale), actual_scale_bits=int(scale.view(np.uint32)), **summary)
        report['controls'].append(dict(iteration=iteration, valid_rows=valid, value=value, captured_replay=iteration>=2,
          bf16_and_actual_fp8_stats_exact=True, output_bits_exact=True, current_epoch_and_poison_pass=True))
      sys.setprofile(None)
      assert all(counts[('inner', prefix, slot)] == 2 for prefix in names for slot in range(4)), counts
      assert counts[('outer',)] == counts[('actual_quantize_function',)] == 224, counts
      assert set(observer.trace_counts.values()) == {2} and len(observer.trace_counts) == 28
      report['exact_call_counts'] = dict(outer_batched_vision=counts[('outer',)], actual_packed_quantize=counts[('actual_quantize_function',)],
        fc2_slot3_calls=sum(counts[('inner', prefix, 3)] for prefix in names), python_traces_per_layer=2,
        duplicated_fc2_quantization=False, replay_python_callbacks=False)
      def reject(label, fn):
        try: fn()
        except ValueError: report['negatives'].append(label)
        else: raise AssertionError('Unsupported quantizer accepted: '+label)
      observer.restore();assert model.linear == before_linear and model.quantize is outer
      binding.vision_state['images'] = 1
      reject('active-vision-context', lambda: overlay.Binding(model, args.baseline_root))
      binding.vision_state['images'] = 0
      binding.state['index'] = 1
      reject('nonidle-packed-context', lambda: overlay.Binding(model, args.baseline_root))
      binding.state['index'] = 0
      original_code = outer.__func__.__code__
      # A same-name function with the wrong code is rejected, even if its metadata
      # and source filename are copied from the actual selected function.
      from types import FunctionType
      fake = FunctionType(compile('def quantize(self,x): return x', original_code.co_filename, 'exec').co_consts[0], outer.__func__.__globals__)
      fake.__module__, fake.__qualname__ = outer.__func__.__module__, outer.__func__.__qualname__
      model.quantize = MethodType(fake, model)
      reject('same-name-different-code', lambda: overlay.Binding(model, args.baseline_root))
      model.quantize = outer
      copied = SimpleNamespace(**model.__dict__)
      reject('method-bound-to-other-model', lambda: overlay.Binding(copied, args.baseline_root))
      saved = packed.quantize
      def replacement(x, scale=None): return saved(x, scale)
      replacement.__module__, replacement.__qualname__ = saved.__module__, saved.__qualname__
      packed.quantize = replacement
      reject('same-name-replaced-quantizer-global', lambda: overlay.Binding(model, args.baseline_root))
      packed.quantize = saved
      observer = overlay.make_observer(model, core, args.baseline_root, cpu=True)
      observer.prepare(139)
      binding2 = observer.quantize.__self__
      binding2.vision_state['images'] = 1
      binding2.state.update(prefill=True, prefix=names[0], index=3)
      reject('active-vision-before-delegation', lambda: binding2.quantize_once(x))
      binding2.vision_state['images'] = 0
      binding2.state['index'] = 2
      reject('wrong-language-slot', lambda: binding2.quantize_once(x))
      binding2.state.update(prefill=False, prefix=None, index=0)
      observer.restore()
    protected = Path(__file__).resolve()
    protected_hash = overlay.sha(protected)
    try: overlay.execute(SimpleNamespace(output=protected), {})
    except FileExistsError: pass
    else: raise AssertionError('Overlay can overwrite existing evidence')
    assert overlay.sha(protected) == protected_hash
    report['existing_output_rejected_without_writes'] = True
    report.update(complete=True, all_pass=True, unsupported_rejections=len(report['negatives']),
      execution_endpoint_only_mocked=True, actual_installers_and_all_quantizer_code_unmodified=True,
      opened_devices=sorted(Device._opened_devices),
      no_gpu_opened=all(name.split(':')[0] in ('CPU', 'NPY', 'PYTHON') for name in Device._opened_devices))
    assert report['no_gpu_opened']
  except BaseException as error:
    report['error'] = dict(type=type(error).__name__, message=str(error));raise
  finally:
    sys.setprofile(None)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
  print(json.dumps(dict(cpu_only=True, all_pass=True, actual_wrapper_controls=len(report['controls']), rejections=len(report['negatives']))))


if __name__ == '__main__': main()
