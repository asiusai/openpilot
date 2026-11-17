"""Parent-only arithmetic qualification of exact saved USB PV bindings.

No model load, route, warmup predictions, new kernels, or timing measurements.
"""
from collections import Counter
from contextlib import contextmanager
import argparse
import importlib.util
import json
import os
from pathlib import Path
import time


def common():
  spec = importlib.util.spec_from_file_location('pv_usb_check_common', Path(__file__).with_name('agent-expert-pv-usb-common.py'))
  value = importlib.util.module_from_spec(spec); spec.loader.exec_module(value)
  return value


@contextmanager
def observe(U, programs, report, save):
  from tinygrad.runtime.support import hcq2
  from tinygrad.uop.ops import Ops
  original = hcq2._build_queues
  seen = Counter()
  def build(ctx):
    changed = False
    for call, _, _ in ctx.batch:
      p = call.body
      if p.op is not Ops.PROGRAM or p.src[0].arg.function_name not in ('alpamayo_attention_values', 'alpamayo_expert_pv_2x2'): continue
      key = p.key.hex()
      assert key in programs, 'Unqualified runtime PV program'
      ast, expected = programs[key]
      U.validate_description(U.old().describe(p, ast), expected)
      seen[key] += 1
      if seen[key] == 1:
        report.setdefault('executed_bindings', {})[key] = expected
        changed = True
    if changed: save()
    return original(ctx)
  hcq2._build_queues = build
  try: yield seen
  finally: hcq2._build_queues = original


def gpu(U, capture, cpu, report, save, budget):
  import numpy as np
  from tinygrad import Device, dtypes
  from tinygrad.codegen import to_program_context
  from tinygrad.device import Compiled
  from tinygrad.helpers import PARALLEL
  B = U.old()
  E, C, T, K = (U.load(n) for n in ('agent-expert-gemm', 'agent-expert-pv-check',
    'agent-vision-tiles-transfer-check', 'agent-expert-pv-kernel'))
  Compiled.wait_timeout_ms = 10000
  U.load('prefill-kernels').fix_dispatch_packet()
  device = Device[Device.DEFAULT]
  assert device.arch == 'gfx1200' and device.is_usb
  U.load('wgp_compiler').install(device)
  U.validate_renderer(device.renderer, 'parent-wgp')
  context = {v.key: str(v.value) for v in to_program_context}
  assert context == capture['runtime']['compiler_context'], 'Runtime compiler options differ from actual capture'
  report['runtime'] = {'device': Device.DEFAULT, 'target': str(device.renderer.target),
    'renderer': type(device.renderer).__name__, 'compiler': type(device.compiler).__name__,
    'compiler_context': context, 'AM_POWER_LIMIT': os.environ.get('AM_POWER_LIMIT'),
    'AM_PERF_LEVEL': os.environ.get('AM_PERF_LEVEL'), 'GMMU': os.environ.get('GMMU'), 'parallel': PARALLEL.value}
  assert report['runtime']['AM_POWER_LIMIT'] == '144' and report['runtime']['AM_PERF_LEVEL'] == 'auto'
  assert report['runtime']['GMMU'] == '0' and PARALLEL.value == capture['runtime']['parallel']
  report['gpu_executed'] = True
  renderers, registry, compiled = U.renderers(), {}, {}
  for name, shape in K.CASES.items():
    compiled[name] = {}
    for mode, renderer in renderers.items():
      for label in U.LABELS:
        ast, program = U.compile_program(renderer, shape, label)
        expected = cpu['compiler_cases'][name][mode][label]
        U.validate_description(B.describe(program, ast), expected)
        if (name, mode, label) == ('expert', 'worker-cu', 'original'):
          program = U.captured_program(ast, program, capture)
          report['saved_actual_program_reconstructed_exactly'] = True
        compiled[name][mode+'/'+label] = program
        registry[program.key.hex()] = ast, expected
  assert len(registry) == 12
  report['cases'] = []
  with observe(U, registry, report, save) as seen:
    for name, shape in K.CASES.items():
      budget()
      row = {'case': name, 'shape': list(shape), 'checks': [], 'complete': False}
      report['cases'].append(row)
      transfers = T.InputTransfers(device)
      data = C.fixture(shape, 'base')
      inputs = transfers.upload(data['values'], (dtypes.bfloat16, dtypes.bfloat16))
      singles = {label: C.SerialPV(shape, lambda *_, p=p: p, 1) for label, p in compiled[name].items()}
      for jit in singles.values():
        for _ in range(3): jit(*inputs)
        assert len(E.captured_kernels(jit)) == 1
      base = previous = None
      def reference(data):
        ref = singles['parent-wgp/original']
        E.poison((ref.output,), device)
        expected = E.read_output(ref(*inputs)[0], device)
        assert np.isfinite(E.unpack_bf16(expected)).all(), 'Nonfinite reference'
        if data['integer']: np.testing.assert_array_equal(expected, C.integer_reference(data))
        return expected
      def exact_all(jits, expected):
        for label, jit in jits.items():
          E.poison((jit.output,), device)
          E.exact_output(jit(*inputs)[0], expected, device)
      for state in (*C.STATES, 'replay', 'post'):
        budget()
        data = C.fixture(shape, state)
        event = transfers.replace(inputs, data['values'])
        if state == 'a-only': assert event['writes'] == [True, False]
        if state == 'b-only-after-a': assert event['writes'] == [False, True]
        expected = reference(data)
        if state == 'base': base = expected.copy()
        elif state == 'restore': np.testing.assert_array_equal(expected, base)
        elif state == 'a-only':
          np.testing.assert_array_equal(expected[:-1], previous[:-1])
          np.testing.assert_array_equal(expected[-1, :-1], previous[-1, :-1])
          assert not np.array_equal(expected[-1, -1], previous[-1, -1])
        elif state == 'b-only-after-a':
          unaffected = shape[1]-(shape[1]//shape[2])
          np.testing.assert_array_equal(expected[:-1], previous[:-1])
          np.testing.assert_array_equal(expected[-1, :unaffected], previous[-1, :unaffected])
          assert not np.array_equal(expected[-1, unaffected:], previous[-1, unaffected:])
        exact_all(singles, expected)
        row['checks'].extend({'state': state, 'program': label, 'exact_bf16_bits': True,
          'independent_integer_reference': data['integer'], 'reference_sha256': U.sha(expected.tobytes())} for label in singles)
        previous = expected.copy()
        save()
      assert len(row['checks']) == 28
      if name == 'expert':
        assert compiled[name]['worker-cu/original'].key.hex() == capture['actual_reference']['program_key']
        changed = C.fixture(shape, 'post')
        changed['b'][..., 2017:2304] = E.bf16(np.float32(17))
        values = (changed['values'][0], C.pack_bits(changed['b']).reshape(changed['values'][1].shape))
        assert transfers.replace(inputs, values)['writes'] == [False, True]
        exact_all(singles, expected)
        row['masked_history_value_mutation_exact'] = True
        batches = {label: C.SerialPV(shape, lambda *_, p=p: p, 40) for label, p in compiled[name].items()}
        for label, jit in batches.items():
          budget()
          for _ in range(3): jit(*inputs)
          row.setdefault('serial_40_call_proofs', {})[label] = C.check_queue(jit)
          E.exact_output(jit.output, expected, device)
        # All four captured WAW graphs must see fresh mutations and restorations.
        row['captured_40_call_states'] = []
        for state in ('base', 'a-only', 'b-only-after-a', 'edges', 'restore', 'replay', 'post'):
          budget()
          data = C.fixture(shape, state)
          event = transfers.replace(inputs, data['values'])
          if state == 'a-only': assert event['writes'] == [True, False]
          if state == 'b-only-after-a': assert event['writes'] == [False, True]
          expected = reference(data)
          exact_all(batches, expected)
          row['captured_40_call_states'].append({'state': state, 'programs': list(batches),
            'exact_bf16_bits': True, 'reference_sha256': U.sha(expected.tobytes())})
          save()
        for jit in batches.values(): E.exact_output(jit.output, expected, device)
        row['final_fresh_replay_and_readback_exact'] = True
      row['transfers'], row['complete'] = transfers.report(), True
      save()
    assert set(seen) == set(registry), 'Missing execution of a qualified program'
  report['all_12_usb_programs_observed'] = True
  assert len(report['cases']) == 3 and all(c['complete'] for c in report['cases'])
  report['micro_full_bit_controls_passed'] = True
  report['arithmetic_qualified'] = True


def main():
  p = argparse.ArgumentParser(description=__doc__)
  p.add_argument('--output', type=Path, required=True)
  p.add_argument('--protocol', type=Path, required=True)
  p.add_argument('--protocol-sha', required=True)
  p.add_argument('--plan', action='store_true')
  args = p.parse_args()
  assert sys_optimization_off()
  U = common()
  protocol, provenance, capture, cpu = U.preflight(args.protocol, args.protocol_sha)
  report = dict(provenance, complete=False, gpu_executed=False, plan_only=args.plan, target=U.TARGET,
    model_predictions=0, maximum_model_predictions=0, saved_actual_capture_sha256=U.CAPTURE_SHA,
    reused_actual_capture_calls={'pv': 84, 'qk': 84, 'softmax': 84},
    arithmetic_qualified=False, micro_full_bit_controls_passed=False, numerical_candidate=False,
    route_equality_established=False, quality64_established=False, timings_valid=False,
    old_capture_report_complete=False, protocol_sha256=args.protocol_sha)
  assert not args.output.exists(), 'Never overwrite evidence'
  def save(): U.old().write(args.output, report)
  save()
  if args.plan: return
  started = time.perf_counter()
  def budget(): assert time.perf_counter()-started < 270, 'Internal 270-second budget exceeded'
  try:
    assert os.environ.get('DEV') == 'USB+AMD:LLVM', 'Run through the serialized parent wrapper'
    gpu(U, capture, cpu, report, save, budget)
    U.preflight(args.protocol, args.protocol_sha)
    report['complete'] = True
  except BaseException as exc:
    report['failure'] = {'type': type(exc).__name__, 'message': str(exc)}
    report['arithmetic_qualified'] = False
    raise
  finally:
    report['elapsed_seconds'] = time.perf_counter()-started
    save()


def sys_optimization_off():
  import sys
  return sys.flags.optimize == 0


if __name__ == '__main__': main()
