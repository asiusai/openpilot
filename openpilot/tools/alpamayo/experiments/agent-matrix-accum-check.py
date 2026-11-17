"""Bounded compiler audit and parent-only running-C GPU checks.

Only prefill-down 3x3 has a repair candidate. Other tiles are CPU audit
controls. Existing model sources and frozen tests remain unchanged.
"""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import re
import statistics
import subprocess
import sys
import time
from types import FunctionType, SimpleNamespace

import numpy as np

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('agent_matrix_accum', HERE/'agent-matrix-accum.py')
A = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = A
spec.loader.exec_module(A)
CONTROLS = [('control-scalar', (17, 48, 32), False, False), ('control-row-bias', (49, 80, 64), True, True),
  ('control-long-k-tail', (49, 80, 9216), False, False)]


def cpu_rounding():
  """Independent E4M3 arithmetic witness and complete nonuniform reshape checks."""
  import tinygrad.codegen as codegen
  from tinygrad import dtypes
  from tinygrad.uop.ops import Ops, ProgramInfo, UOp
  E = A.helpers().E
  left, right = np.zeros(32, np.float32), np.zeros(32, np.float32)
  left[0], right[0] = 128, 1
  left[16:19], right[16:19] = [-64, -7, 2.**-9], [1, 9, 2.**-9]
  table = E.decode_fp8(np.arange(256, dtype=np.uint8))
  for values in (left, right):
    assert all(np.any(table == value) for value in values)
  first = np.float32(left[:16].astype(np.float64)@right[:16].astype(np.float64))
  dot = left[16:].astype(np.float64)@right[16:].astype(np.float64)
  fused, split = np.float32(np.float64(first)+dot), np.float32(first+np.float32(dot))
  assert fused == np.float32(1+2.**-18) and split == 1
  scale = np.nextafter(np.float32(1.00390625), np.float32(-np.inf))
  bits = E.bf16(np.array([fused, split], np.float32)*scale)
  np.testing.assert_array_equal(bits, np.array([0x3f81, 0x3f80], np.uint16))
  renderer = make_renderer()
  fn = A.functions(renderer)['3x3-accum']
  args, ast, normal = A.compile_program(A.ACTUAL, False, False, fn, renderer)
  lower, observations = A.private_lowering()
  sink = lower(ast, renderer, optimize=not ast.arg.optimized)
  private = codegen.to_program(UOp(Ops.PROGRAM, src=(sink,), arg=ProgramInfo.from_sink(sink, renderer.target)), renderer)
  assert normal.src[2].arg == private.src[2].arg and observations
  rng = np.random.default_rng(906033)
  checks = []
  for item in observations:
    a = rng.standard_normal(item['wmma_shape']).astype(np.float32)
    c = rng.standard_normal(item['accumulator_shape']).astype(np.float32)
    reference = np.float32(a.reshape(item['reshape_shape'])+c)
    moved = np.float32(a+c.reshape(item['wmma_shape'])).reshape(item['reshape_shape'])
    np.testing.assert_array_equal(reference.view(np.uint32), moved.view(np.uint32))
    checks.append(dict(**item, nonuniform_accumulator_mapping_exact=True))
  # Strong scope guards: neither upward 3x3 nor an unrelated tile can use this candidate.
  upward = [UOp.placeholder(s, dt, i) for i, (s, dt) in enumerate([
    ((2112, 9216), dtypes.bfloat16), ((2112, 2048), dtypes.fp8e4m3), ((9216, 2048), dtypes.fp8e4m3),
    ((1,), dtypes.float32), ((1,), dtypes.float32)])]
  try: fn(*upward)
  except AssertionError as exc: assert 'Unjustified repair scope' in str(exc)
  else: raise AssertionError('Upward shape admitted to downward repair')
  return dict(fused_result=float(fused), split_result=float(split), scale=float(scale), bf16_bits=[hex(int(x)) for x in bits],
    one_bf16_ulp=True, multiplicands_exact_e4m3=True, hardware_internal_tree_not_emulated=True,
    observed_reshape_checks=checks, upward_repair_scope_rejected=True)


def make_renderer():
  from tinygrad.helpers import Target
  from tinygrad.renderer.llvmir import AMDLLVMRenderer
  from tinygrad.runtime.support.compiler_llvm import LLVMCompiler
  renderer = AMDLLVMRenderer(Target('AMD', 'LLVM', 'gfx1200'))
  renderer.compiler = LLVMCompiler('AMDGPU', 'gfx1200', '-cumode')
  return renderer


def native_details(source, artifact):
  # The frozen assembly helper is CPU-only and sets compile-only environment.
  # This function is deliberately absent from the GPU execution path.
  from tinygrad.runtime.autogen import llvm
  compiler = make_renderer().compiler
  llvm.LLVMSetTargetMachineAsmVerbosity(compiler.target_machine, True)
  assembly = A.frozen('agent-matrix-tiles-assembly').assembly(compiler, source)
  artifact.with_suffix('.s').write_text(assembly)
  estimates = {key: int(value) for key, value in re.findall(
    r';\s*(Occupancy|NumVgprs|NumSgprs|ScratchSize|LDSByteSize):\s*(\d+)', assembly)}
  wmma = [line.strip() for line in assembly.splitlines() if line.strip().startswith('v_wmma_f32_16x16x16_fp8_fp8')]
  return dict(assembly_sha256=A.sha(assembly.encode()), compiler_estimates=estimates,
    native_wmma=len(wmma), native_zero_c_wmma=sum(bool(re.search(r',\s*0\s*$', line)) for line in wmma),
    native_fp32_add_operations=len(re.findall(r'\bv_(?:dual_)?add_f32(?:_e32|_e64)?\b', assembly)),
    wmma_instructions=wmma, estimates_not_measured_occupancy=True)


def audit(args):
  import tinygrad.codegen as codegen
  from tinygrad import Device
  assert Device.DEFAULT.startswith('CPU')
  original_matchers = codegen.pm_reduce_local, codegen.pm_expand_broadcast
  prior = json.loads((A.WORK/'agent-matrix-tiles-kernels/report.json').read_text())
  by_case = {row['case']: row for row in prior['cases']}
  renderer = make_renderer()
  result = []
  specifications = CONTROLS+[(name, A.helpers().CASES[name], False, False) for name in ('prefill-up', 'prefill-down')]
  for name, shape, bias, row_scale in specifications:
    actual = name.startswith('prefill')
    fns = A.functions(renderer, row_scale, repair=name != 'prefill-up',
      audit_tiles=((3, 3), (3, 4)) if actual else ((3, 3),))
    rows = {}
    for label, fn in fns.items():
      tile = (4, 4) if label in ('original', 'current') else tuple(map(int, label.split('-')[0].split('x')))
      stem = args.artifacts/f'agent-matrix-accum-{name}-{label}'
      row, program = A.record(shape, bias, row_scale, fn, renderer, tile, stem)
      row.update(native_details(program.src[2].arg, stem))
      assert row['compiler_estimates']['NumVgprs'] == row['resources']['vgpr_count']
      expected = by_case.get(name, {}).get('variants', {}).get(label)
      if expected:
        assert row['binary_sha256'] == expected['binary_sha256'], (name, label, row['binary_sha256'], expected['binary_sha256'])
        row['matches_tested_binary'] = True
      if actual:
        bad = name == 'prefill-down' and label == '3x3'
        assert row['llvm_zero_c_wmma'] == (9 if bad else 0)
        assert row['llvm_loop_fadd'] == (72 if bad else 0)
        assert row['native_zero_c_wmma'] == (9 if bad else 0)
        assert row['native_fp32_add_operations'] == (72 if bad else 0)
      rows[label] = row
      print(json.dumps(dict(case=name, variant=label, vgprs=row['resources']['vgpr_count'],
        llvm_zero_c=row['llvm_zero_c_wmma'], loop_fadd=row['llvm_loop_fadd'],
        native_zero_c=row['native_zero_c_wmma'], occupancy_estimate=row['compiler_estimates'].get('Occupancy'))), flush=True)
    if '3x3-accum' in rows:
      assert rows['3x3']['wmma_operand_pairs'] == rows['3x3-accum']['wmma_operand_pairs']
      assert rows['3x3']['epilogue_fmul_count'] == rows['3x3-accum']['epilogue_fmul_count']
      rows['3x3-accum'].update(unchanged_exact_ab_operand_graphs=True, unchanged_epilogue_multiply_count=True)
    result.append(dict(case=name, shape_mnk=shape, bias=bias, row_scales=row_scale, variants=rows))
  assert (codegen.pm_reduce_local, codegen.pm_expand_broadcast) == original_matchers
  assert not any(name.startswith('AMD') for name in Device._opened_devices)
  return result


def checked(jit, tensors, device):
  E = A.helpers().E
  E.poison(jit.captured.ret, device)
  values = E.read_output(jit(*tensors)[0], device)
  assert np.isfinite(E.unpack_bf16(values)).all()
  return values


def public_reference(tensors, shape, bias, expected, device):
  from amd_gemm import gemm, unpack
  E = A.helpers().E
  a, w, sx, sw, b = tensors
  public = gemm(unpack(a)[:shape[0]], unpack(w)[:shape[1]], (sx[:1], sw), b if bias else None).realize()
  E.exact_output(public, expected, device)


def gpu_case(name, shape, bias, row_scale, fns, resources, rounds, device, *, integer=False, blocked=(), progress=None):
  H, E = A.helpers(), A.helpers().E
  row = dict(case=name, shape_mnk=shape, bias=bias, row_scales=row_scale,
    variants={label: dict(info, checks=[], status='control-rejected' if label in blocked else 'pending')
      for label, info in resources.items()}, timing=[])
  def save():
    if progress: progress(row)
  def reject(label, stage, state, error, actual=None, expected=None):
    failure = dict(stage=stage, state=state, error=f'{type(error).__name__}: {error}')
    if actual is not None:
      failure['actual_sha256'] = A.sha(actual.tobytes())
    if actual is not None and expected is not None:
      differences = np.argwhere(actual != expected)
      failure.update(mismatched_bf16=len(differences), elements=expected.size,
        first_mismatches=[dict(index=index.tolist(), actual=int(actual[tuple(index)]), expected=int(expected[tuple(index)]))
          for index in differences[:16]], expected_sha256=A.sha(expected.tobytes()))
    row['variants'][label].update(status='rejected', failure=failure, eligible_for_timing=False)
    save()
    if label in ('current', 'original'):
      raise AssertionError(f'Original baseline gate failed: {label} {stage} state {state}') from error
  def compare(label, jit, tensors, expected, stage, state):
    actual = None
    try:
      actual = checked(jit, tensors, device)
      np.testing.assert_array_equal(actual, expected)
    except AssertionError as exc:
      reject(label, stage, state, exc, actual, expected)
      return False
    row['variants'][label]['checks'].append(dict(stage=stage, state=state, exact_bf16=True,
      reference_sha256=A.sha(expected.tobytes())))
    return True
  save()
  tensors = E.upload(H.fixture(shape, 0, integer, row_scale, nonuniform=integer and row_scale))
  singles = {}
  for label, fn in fns.items():
    if label in blocked: continue
    jit = singles[label] = E.make_jit(shape, bias, fn, 1)
    try:
      for _ in range(3): jit(*tensors)
      assert len(E.captured_kernels(jit)) == 1
    except AssertionError as exc:
      reject(label, 'single_capture', None, exc)
      del singles[label]
  previous = None
  for state in range(4):
    data = H.fixture(shape, state, integer, row_scale, nonuniform=integer and row_scale)
    E.change_inputs(tensors, data, device)
    try:
      expected = checked(singles['current'], tensors, device)
      if integer: np.testing.assert_array_equal(expected, H.integer_reference(data, bias, row_scale))
      if previous is not None: assert not np.array_equal(expected, previous)
      if state == 0 and not (integer and row_scale):
        public_reference(tensors, shape, bias, expected, device)
        row['original_public_wrapper_exact'] = True
    except AssertionError as exc:
      reject('current', 'reference_or_mutation', state, exc)
    previous = expected.copy()
    for label, jit in singles.items():
      if row['variants'][label]['status'] != 'rejected':
        compare(label, jit, tensors, expected, 'single_mutation', state)
    save()
  if integer:
    for info in row['variants'].values():
      if info['status'] == 'pending': info['status'] = 'passed-control'
    save()
    return row
  # The raw failing tile and duplicate baseline are diagnostic controls only.
  accepted = [label for label in ('current', '3x3-accum') if row['variants'][label]['status'] == 'pending']
  batches = {}
  for label in accepted:
    jit = batches[label] = E.make_jit(shape, bias, fns[label], 40)
    try:
      for _ in range(3): jit(*tensors)
      assert len(E.captured_kernels(jit)) == 40 and E.shared_output_slot(jit) == 5
      row['variants'][label]['serial_queue_proof'] = jit.proofs[0]
    except AssertionError as exc:
      reject(label, 'batch_capture_or_queue_proof', 3, exc)
      continue
    compare(label, jit, tensors, previous, 'batch_captured_state', 3)
  E.change_inputs(tensors, H.fixture(shape, 4, False, row_scale), device)
  try:
    expected = checked(singles['current'], tensors, device)
    assert not np.array_equal(expected, previous)
  except AssertionError as exc:
    reject('current', 'changed_batch_reference', 4, exc)
  for label, jit in batches.items():
    if row['variants'][label]['status'] != 'rejected':
      compare(label, jit, tensors, expected, 'batch_changed_all_inputs', 4)
  eligible = [label for label in accepted if row['variants'][label]['status'] != 'rejected']
  for label in eligible: row['variants'][label]['eligible_for_timing'] = True
  save()
  if '3x3-accum' not in eligible:
    row['timing_skipped'] = 'Repair failed a strict gate; baseline-only timing omitted'
    return row
  assert 'current' in eligible
  pairs = []
  for iteration in range(rounds):
    order = eligible if iteration % 2 == 0 else eligible[::-1]
    pair = dict(order=order)
    for label in order:
      device.synchronize()
      started = time.perf_counter()
      batches[label](*tensors)
      device.synchronize()
      pair[label] = (time.perf_counter()-started)*1000
    pairs.append(pair)
  for label in eligible:
    if not compare(label, batches[label], tensors, expected, 'post_timing', 4):
      row['invalidated_timing_samples'] = pairs
      row['timing'] = []
      save()
      return row
    row['variants'][label]['status'] = 'passed'
  times = {label: statistics.median(pair[label] for pair in pairs)/40 for label in eligible}
  row['timing'] = [dict(variant='3x3-accum', matrix_calls_per_batch=40, paired_batch_ms=pairs,
    current_amortized_ms=times['current'], candidate_amortized_ms=times['3x3-accum'],
    candidate_over_current=times['3x3-accum']/times['current'], host_inclusive=True, warm_repeated_weights=True,
    quantization_packing_uploads_excluded=True, device_only=False, model_streaming_reproduced=False)]
  save()
  return row


def failure_isolation_mocks():
  """Private function namespaces exercise admission; discard all mock times."""
  reports = []
  for scenario in ('raw-rejected', 'repair-single', 'repair-batch', 'queue-proof', 'nonfinite',
      'blocked-control', 'post-timing', 'baseline-reference'):
    state, constructed, snapshots = [0], [], []
    shape = (17, 48, 32)
    def expected(): return np.full(shape[:2], 0x3f80+state[0], np.uint16)
    class Jit:
      def __init__(self, fn, count):
        self.label, self.count, self.calls = fn.label, count, 0
        self.captured = SimpleNamespace(ret=(None,))
        self.proofs = [dict(captured_calls=40, ordered_execution_barriers=39, actual_queue_builder_observed=True)]
        constructed.append((self.label, count))
      def __call__(self, *inputs):
        self.calls += 1
        value = expected()
        bad = ((scenario == 'raw-rejected' and self.label == '3x3') or
          (scenario == 'repair-single' and self.label == '3x3-accum') or
          (scenario == 'repair-batch' and self.label == '3x3-accum' and self.count == 40) or
          (scenario == 'post-timing' and self.label == '3x3-accum' and self.count == 40 and self.calls >= 9))
        if bad: value[0, 0] += 1
        if scenario == 'nonfinite' and self.label == '3x3-accum': value[0, 0] = 0x7fc0
        return (SimpleNamespace(value=value),)
    def change(inputs, data, device): state[0] = data['state']
    def reference(*args):
      values = expected()
      if scenario == 'baseline-reference': values[0, 0] += 1
      return values
    def output_slot(jit):
      if scenario == 'queue-proof' and jit.label == '3x3-accum': raise AssertionError('Missing ordered WAW barrier')
      return 5
    E = SimpleNamespace(upload=lambda *_: (), make_jit=lambda shape, bias, fn, count: Jit(fn, count),
      change_inputs=change, captured_kernels=lambda jit: ['mock']*jit.count, poison=lambda *_: None,
      read_output=lambda tensor, device: tensor.value, unpack_bf16=A.helpers().E.unpack_bf16,
      shared_output_slot=output_slot)
    H = SimpleNamespace(E=E, fixture=lambda shape, state, *args, **kwargs: dict(state=state), integer_reference=reference)
    namespace = dict(gpu_case.__globals__, A=SimpleNamespace(helpers=lambda: H, sha=A.sha))
    namespace['checked'] = FunctionType(checked.__code__, namespace, checked.__name__)
    namespace['public_reference'] = lambda *_: None
    run = FunctionType(gpu_case.__code__, namespace, gpu_case.__name__, gpu_case.__defaults__)
    run.__kwdefaults__ = gpu_case.__kwdefaults__
    labels = ('current', 'original', '3x3', '3x3-accum')
    fns = {label: SimpleNamespace(label=label) for label in labels}
    def progress(row): snapshots.append(json.loads(json.dumps(row, allow_nan=False)))
    try:
      row = run('mock', shape, False, scenario == 'baseline-reference', fns,
        {label: {} for label in labels}, 3, SimpleNamespace(synchronize=lambda: None),
        integer=scenario == 'baseline-reference', blocked=('3x3-accum',) if scenario == 'blocked-control' else (),
        progress=progress)
    except AssertionError as exc:
      assert scenario == 'baseline-reference' and 'baseline gate failed' in str(exc)
      assert not any(count == 40 for _, count in constructed)
      reports.append(dict(scenario=scenario, baseline_failure_fatal=True, no_timing=True))
      continue
    assert scenario != 'baseline-reference' and snapshots
    if scenario == 'raw-rejected':
      assert row['variants']['3x3']['status'] == 'rejected' and row['variants']['3x3-accum']['status'] == 'passed'
      assert ('3x3', 40) not in constructed and row['timing']
      assert all(set(pair['order']) == {'current', '3x3-accum'} for pair in row['timing'][0]['paired_batch_ms'])
    else:
      assert not row['timing']
      assert row['variants']['3x3-accum']['status'] == ('control-rejected' if scenario == 'blocked-control' else 'rejected')
      if scenario not in ('repair-batch', 'queue-proof', 'post-timing'): assert ('3x3-accum', 40) not in constructed
      if scenario == 'post-timing': assert row['invalidated_timing_samples']
    reports.append(dict(scenario=scenario, rejected_variant_not_timed=scenario != 'post-timing', failure_retained=True,
      baseline_admission_preserved=True, json_finite=True, mock_times_discarded=True))
  return reports


def decoder_evidence():
  path = A.WORK/'agent-decode-gemm-rounding-retry/report.json'
  evidence = json.loads(path.read_text())
  row = next(row for row in evidence['cases'] if row['case'] == 'q' and row['shape_mnk'] == [16, 2048, 2048])
  assert evidence['gpu_executed'] and evidence['complete']
  variant = row['variants']['1x1-accum']
  assert variant['status'] == 'passed' and all(check['exact_bf16'] for check in variant['checks'])
  assert [check['state'] for check in variant['checks'] if check['stage'] == 'single_mutation'] == list(range(6))
  assert row['serial_proofs']['1x1-accum']['calls'] == 40
  assert row['serial_proofs']['1x1-accum']['ordered_execution_barriers'] == 39
  return dict(path=str(path), sha256=A.sha(path.read_bytes()), shape_mnk=row['shape_mnk'],
    original_1x1_status=row['variants']['1x1']['status'], repaired_1x1_status=variant['status'],
    median_ms_per_call=row['timings']['median_ms_per_call'], calls_per_batch=40, rotated_rounds=3,
    all_six_mutations_and_batch_post_timing_exact=True, host_inclusive=True, warm_repeated_weights=True,
    model_streaming_reproduced=False, supports_large_matrix_hypothesis_not_large_matrix_validation=True)


def frozen_cpu_checks():
  # Keep the compiler process device-disabled. The child permits CPU execution
  # and normal CPU finalization, with no AMD device opened by the frozen test.
  script = '''import importlib.util, json, sys
spec = importlib.util.spec_from_file_location('agent_matrix_accum_cpu_proof', sys.argv[1])
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
print(json.dumps(module.cpu_checks(), allow_nan=False))
'''
  output = subprocess.check_output([sys.executable, '-c', script, str(A.FROZEN/'agent-expert-gemm.py')],
    env=dict(os.environ, DEV='CPU', ALLOW_DEVICE_USAGE='1'), text=True)
  return json.loads(output)


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--mode', choices=('cpu', 'gpu'), default='cpu')
  parser.add_argument('--output', required=True, type=Path)
  parser.add_argument('--artifacts', type=Path, default=HERE/'agent-matrix-accum-artifacts')
  parser.add_argument('--rounds', type=int, choices=(3, 5), default=3)
  args = parser.parse_args()
  os.environ.update(BEAM='0', JITBEAM='0', PARALLEL='0', PROFILE='0', DEBUG='0')
  if args.mode == 'cpu': os.environ.update(DEV='CPU', ALLOW_DEVICE_USAGE='0')
  from tinygrad import Device
  H, E = A.helpers(), A.helpers().E
  report = dict(mode=args.mode, provenance=A.provenance(), prior_decoder_hardware_evidence=decoder_evidence(),
    gpu_executed=False, complete=False, cases=[],
    strict_all_bf16_equality=True, candidate_shapes={'3x3-accum': A.ACTUAL}, date='2026-11-17',
    scope='Kernel proof only. No model integration, calibration, route quality, or complete pipeline measurement.')
  args.artifacts.mkdir(parents=True, exist_ok=True)
  try:
    if args.mode == 'cpu':
      report['compiler_audit'] = audit(args)
      report['independent_rounding'] = cpu_rounding()
      report['failure_isolation_mocks'] = failure_isolation_mocks()
      report['frozen_cpu_queue_and_arithmetic_checks'] = frozen_cpu_checks()
      assert not Device._opened_devices
    else:
      assert os.environ.get('TC_OPT') == '2' and os.environ.get('LLVMOPT') in (None, '1')
      assert os.environ.get('AM_POWER_LIMIT') == '144' and os.environ.get('AM_PERF_LEVEL') == 'auto'
      from tinygrad.device import Compiled
      Compiled.wait_timeout_ms = 10000
      A.frozen('prefill-kernels').fix_dispatch_packet()
      device = Device[Device.DEFAULT]
      assert device.arch == 'gfx1200' and device.is_usb
      A.frozen('wgp_compiler').install(device)
      report['gpu_executed'] = True
      blocked = set()
      for name, shape, bias, row_scale in CONTROLS+[('prefill-down', A.ACTUAL, False, False)]:
        integer = name.startswith('control')
        fns = A.functions(device.renderer, row_scale)
        resources = {label: A.record(shape, bias, row_scale, fn, device.renderer,
          (4, 4) if label in ('current', 'original') else (3, 3))[0] for label, fn in fns.items()}
        prior = json.loads((A.WORK/'agent-matrix-tiles-kernels/report.json').read_text())
        historical = next((row for row in prior['cases'] if row['case'] == name), None)
        if historical:
          for label in ('original', 'current', '3x3'):
            if label in resources and label in historical['variants']:
              assert resources[label]['binary_sha256'] == historical['variants'][label]['binary_sha256'], (
                'Tested baseline/diagnostic binary changed', name, label)
              resources[label]['matches_tested_binary'] = True
        def progress(value):
          if not any(value is row for row in report['cases']): report['cases'].append(value)
          E.write_json(args.output, report)
        row = gpu_case(name, shape, bias, row_scale, fns, resources, args.rounds, device,
          integer=integer, blocked=() if integer else blocked, progress=progress)
        if integer:
          blocked.update(label for label, info in row['variants'].items() if info['status'] == 'rejected')
    report['complete'] = True
    report['rejected_variants'] = [dict(case=row['case'], variant=label, failure=info.get('failure'))
      for row in report['cases'] for label, info in row['variants'].items() if info['status'] in ('rejected', 'control-rejected')]
    E.write_json(args.output, report)
    print(json.dumps(dict(complete=True, gpu_executed=report['gpu_executed'],
      audited_cases=len(report.get('compiler_audit', [])), rejected_variants=len(report['rejected_variants']))))
    if report['rejected_variants']:
      raise AssertionError('Strict variants rejected; independent accepted results were retained')
  except BaseException as exc:
    report['failure'] = f'{type(exc).__name__}: {exc}'
    E.write_json(args.output, report)
    raise


if __name__ == '__main__': main()
