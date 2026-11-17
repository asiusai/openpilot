"""Independent CPU rounding, failure-isolation mocks, and gfx1200 compilation.

Never initializes AMD. Mock timing samples are discarded and are not GPU data.
"""
import argparse
from contextlib import contextmanager
import importlib.util
import json
import os
from pathlib import Path
import re
from types import SimpleNamespace

import numpy as np

HERE = Path(__file__).resolve().parent


def load(name):
  spec = importlib.util.spec_from_file_location(name.replace('-', '_'), HERE/(name+'.py'))
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module


check = load('agent-decode-gemm')


def threshold_case():
  # All multiplicands are exact E4M3 values. The first K16 supplies C=128.
  a, b = np.zeros(32, np.float32), np.zeros(32, np.float32)
  a[0], b[0] = 128, 1
  a[16:19], b[16:19] = [-64, -7, 2.**-9], [1, 9, 2.**-9]
  import torch
  for values in (a, b):
    tensor = torch.from_numpy(values)
    assert torch.equal(tensor.to(torch.float8_e4m3fn).float(), tensor)
  initial = np.float32(a[:16].astype(np.float64)@b[:16].astype(np.float64))
  partial = a[16:].astype(np.float64)@b[16:].astype(np.float64)
  fused = np.float32(np.float64(initial)+partial)
  separated = np.float32(initial+np.float32(partial))
  assert initial == 128 and fused == 1+2.**-18 and separated == 1
  scale = np.nextafter(np.float32(1.00390625), np.float32(-np.inf))
  outputs = np.array([fused, separated], np.float32)*scale
  bits = check.bf16(outputs)
  np.testing.assert_array_equal(bits, np.array([0x3f81, 0x3f80], np.uint16))
  np.testing.assert_array_equal(bits, torch.from_numpy(outputs).to(torch.bfloat16).view(torch.uint16).numpy())
  return dict(initial_accumulator=float(initial), exact_next_k16_dot=float(partial),
    running_accumulator_result=float(fused), zero_wmma_then_add_result=float(separated),
    same_scale_product=float(scale), bf16_bits=[hex(int(x)) for x in bits],
    one_bf16_ulp=True, inputs_exact_e4m3=True, hardware_internal_tree_not_emulated=True)


def model_threshold_case():
  model = check.model_fixture('q', Path('/dev/shm/alpamayo-9060-fp8'),
    Path('/tmp/opencode/alpamayo-asius-20261114/model'), HERE/'coc_vocab.json')
  data = check.fixture((16, 2048, 2048), 0, model)
  a = check.decode(data['logical_a']).astype(np.float64)
  w = check.decode(check.unpack_numpy(data['w'])).astype(np.float64)
  fused = np.zeros((16, 2048), np.float32)
  separated = fused.copy()
  for start in range(0, 2048, 16):
    dot = a[:, start:start+16]@w[:, start:start+16].T
    fused = (fused.astype(np.float64)+dot).astype(np.float32)
    separated = np.float32(separated+dot.astype(np.float32))
  scale = data['sx']*data['sw'][0]
  fused_bits, separated_bits = check.bf16(fused*scale), check.bf16(separated*scale)
  # This matches the first saved GPU discrepancy. The CPU recurrence does NOT
  # reproduce all 13 GPU mismatches or emulate native WMMA internal rounding.
  assert int(fused_bits[0, 725]) == 49007 and int(separated_bits[0, 725]) == 49006
  return dict(model_identity=model['identity'], fixture_state=0,
    input_sha256={key: check.sha(data[key].tobytes()) for key in check.KEYS},
    cpu_bf16_differences=int(np.count_nonzero(fused_bits != separated_bits)),
    saved_gpu_bf16_differences=13, full_gpu_failure_not_reproduced=True,
    first_saved_gpu_location=dict(index=[0, 725], running_accumulator=float(fused[0, 725]),
      separated_accumulator=float(separated[0, 725]), running_bf16=int(fused_bits[0, 725]),
      separated_bf16=int(separated_bits[0, 725])), hardware_internal_tree_not_emulated=True)


@contextmanager
def patched(module, **values):
  original = {key: getattr(module, key) for key in values}
  for key, value in values.items(): setattr(module, key, value)
  try: yield
  finally:
    for key, value in original.items(): setattr(module, key, value)


def failure_isolation_mocks():
  reports = []
  for failure in ('single', 'batch', 'nonfinite', 'blocked-control', 'baseline-reference'):
    state, constructed, timed, snapshots = [0], [], [], []
    shape = (16, 64, 64)
    labels = ('1x4', '1x2', '1x1', '1x1-accum')
    fns = {label: SimpleNamespace(label=label) for label in labels}
    def fixture(shape, value, model=None, control=None): return {'state': value}
    def expected():
      value = state[0]
      result = np.full(shape[:2], 0x3f80+value, np.uint16)
      if value == 2:
        result.fill(0x3f81)
        result[shape[0]//2] += 1
      return result
    class FakeJit:
      def __init__(self, shape, bias, fn, count):
        self.label, self.count = fn.label, count
        constructed.append((self.label, count))
      def __call__(self, *inputs):
        if self.count == check.CALLS: timed.append(self.label)
        return (None,)
    def checked(jit, inputs, device):
      result = expected()
      bad = ((failure == 'single' and jit.label == '1x1' and state[0] == 0) or
             (failure == 'batch' and jit.label == '1x1' and jit.count == check.CALLS))
      if failure == 'nonfinite' and jit.label == '1x1': raise AssertionError('nonfinite output')
      if bad: result[0, 0] += 1
      return result
    def reference(data, bias, sparse=False):
      result = expected()
      if failure == 'baseline-reference': result[0, 0] += 1
      return result
    def mutate(inputs, data, device): state[0] = data['state']
    device = SimpleNamespace(device='CPU-mock', synchronize=lambda: None)
    def progress(row): snapshots.append(json.loads(json.dumps(row, allow_nan=False)))
    with patched(check, fixture=fixture, upload=lambda *_: (), SerialJit=FakeJit,
        metadata=lambda jit: {'calls': jit.count, 'ordered_execution_barriers': jit.count-1, 'actual_queue_builder': True},
        mutate=mutate, checked_run=checked, reference=reference):
      try:
        row = check.gpu_case('mock', shape, False, {'identity': 'mock-only'}, fns, {}, 3, device,
          control='integer' if failure == 'baseline-reference' else None,
          blocked_variants=('1x1',) if failure == 'blocked-control' else (), on_progress=progress)
      except AssertionError as exc:
        assert failure == 'baseline-reference' and 'baseline gate failed' in str(exc)
        assert not any(count == check.CALLS for _, count in constructed)
        assert snapshots[-1]['baseline_failure']['stage'] == 'independent_reference_or_mutation'
        reports.append(dict(failure=failure, baseline_failure_fatal=True, no_timing=True))
        continue
    assert failure != 'baseline-reference'
    assert row['variants']['1x1']['status'] == ('blocked_control' if failure == 'blocked-control' else 'rejected')
    assert row['variants']['1x1-accum']['status'] == row['variants']['1x2']['status'] == 'passed'
    assert '1x1' not in row['timings']['median_ms_per_call']
    assert all('1x1' not in pair['order'] for pair in row['timings']['pairs_batch_ms'])
    if failure in ('single', 'nonfinite', 'blocked-control'):
      assert ('1x1', check.CALLS) not in constructed
    if failure == 'blocked-control': assert not any(label == '1x1' for label, _ in constructed)
    if failure != 'batch': assert not row['checks'][0]['all_tiles_exact_bf16']
    assert not row['mutated_batch_replay_exact'] and not row['post_timing_exact']
    assert snapshots
    if failure != 'blocked-control': assert snapshots[-1]['variants']['1x1']['failures']
    reports.append(dict(failure=failure, rejected_variant_not_timed=True, other_variants_independently_checked=True,
      legacy_hook_gate_remains_closed=True, json_finite=True, mock_samples_discarded=True))
  return reports


def compile_checks(artifacts):
  import tinygrad.codegen as codegen
  from tinygrad import Device, dtypes
  from tinygrad.helpers import Target
  from tinygrad.renderer.llvmir import AMDLLVMRenderer
  from tinygrad.uop.ops import Ops, UOp
  assert Device.DEFAULT.startswith('CPU')
  renderer = AMDLLVMRenderer(Target('AMD', 'LLVM', 'gfx1200'))
  renderer.compiler = check.local('wgp_compiler').WGPCompiler('gfx1200')
  fns = check.kernels(renderer)
  disassembler = check.local('agent-attention-check').disassemble
  old_matchers = codegen.pm_reduce_local, codegen.pm_expand_broadcast
  reports = []
  for shape, bias in [((16, 2048, 2048), False), ((48, 1024, 2048), False), ((16, 2048, 9216), False),
      ((17, 48, 32), False), ((17, 48, 32), True)]:
    for label, fn in fns.items():
      row = check.compile_one(shape, bias, fn, renderer)
      m, n, k = shape
      args = [UOp.placeholder(s, dt, i) for i, (s, dt) in enumerate([
        ((m, n), dtypes.bfloat16), (((m+15)//16*16, k), dtypes.fp8e4m3),
        (((n+15)//16*16, k), dtypes.fp8e4m3), ((m, 1), dtypes.float32), ((1,), dtypes.float32)])]
      if bias: args.append(UOp.placeholder((n,), dtypes.bfloat16, 5))
      program = fn(*args) if hasattr(fn, 'raw_kernel') else codegen.to_program(fn(*args), renderer)
      source = next(u.arg for u in program.src if u.op is Ops.SOURCE)
      assembly = disassembler(program.src[3].arg)
      wmma = [line.strip() for line in assembly.splitlines() if 'v_wmma_f32_16x16x16_fp8_fp8' in line]
      zero = [line for line in wmma if re.search(r',\s*0\s*//', line)]
      adds = sum(assembly.count(op) for op in ('v_add_f32_', 'v_dual_add_f32 '))
      if shape[2] >= 2048:
        if label == '1x1':
          assert zero and len(zero) == len(wmma) and adds
        else:
          assert wmma and not zero
          if not bias: assert adds == 0
      assert row['resources']['private_segment_fixed_size'] == row['resources']['vgpr_spill_count'] == 0
      row.update(shape_mnk=shape, bias=bias, variant=label, static_native_wmma=len(wmma),
        zero_accumulator_native_wmma=len(zero), assembly_sha256=check.sha(assembly.encode()))
      stem = artifacts/f"agent-decode-gemm-rounding-{'x'.join(map(str, shape))}-bias{int(bias)}-{label}"
      stem.with_suffix('.ll').write_text(source)
      stem.with_suffix('.asm').write_text(assembly)
      reports.append(row)
  assert (codegen.pm_reduce_local, codegen.pm_expand_broadcast) == old_matchers
  assert not any('AMD' in d for d in Device._opened_devices)
  return reports


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--output', required=True, type=Path)
  parser.add_argument('--artifacts', type=Path, default=HERE/'agent-decode-gemm-rounding-artifacts')
  args = parser.parse_args()
  os.environ.update(DEV='CPU', ALLOW_DEVICE_USAGE='0', BEAM='0', JITBEAM='0', PARALLEL='0', PROFILE='0', DEBUG='0')
  args.artifacts.mkdir(parents=True, exist_ok=True)
  report = dict(provenance=check.provenance(), cpu_check_source_sha256=check.sha(Path(__file__).read_bytes()),
    gpu_executed=False, threshold=threshold_case(),
    actual_fixture_cpu_witness=model_threshold_case(), failure_isolation_mocks=failure_isolation_mocks(),
    compilation=compile_checks(args.artifacts), global_lowering_matchers_unchanged=True,
    strict_gpu_equality_not_yet_verified=True, timing_or_route_quality_not_measured=True)
  check.save(args.output, report)
  print(json.dumps({'gpu_executed': False, 'compilations': len(report['compilation']),
    'mock_checks': len(report['failure_isolation_mocks']), 'threshold': report['threshold']}))


if __name__ == '__main__': main()
