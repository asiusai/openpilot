"""Bounded parent-only GPU diagnostic. Never qualifies a route or changes its gate.

Run at most two original-policy predictions, save captured PV artifacts before
the old strict gate, then test those exact programs and both compiler modes.
No timing, candidate route, global compiler fix, or quality waiver is provided.
"""
from collections import Counter
from contextlib import contextmanager
import argparse
import importlib.util
import json
import os
from pathlib import Path
import runpy
import sys
import time
from types import MethodType


def common():
  spec = importlib.util.spec_from_file_location('pv_binding_diagnostic_common',
    Path(__file__).with_name('agent-expert-pv-binding-common.py'))
  value = importlib.util.module_from_spec(spec); spec.loader.exec_module(value)
  return value


class CaptureComplete(RuntimeError): pass


def validate_program(actual, expected):
  # No normalization of binary, AST, names, launch, or signatures is permitted.
  for key in ('name', 'lowered_ast_key', 'llvm_sha256', 'binary_sha256', 'global_size', 'local_size', 'target', 'params'):
    assert actual[key] == expected[key], f'Captured program differs: {key}'


def preflight(B, args):
  assert B.sha(args.protocol.read_bytes()) == args.protocol_sha
  protocol = json.loads(args.protocol.read_text())
  assert protocol['diagnostic_only'] and protocol['maximum_model_predictions'] == 2
  for path, digest in protocol['new_source_sha256'].items(): assert B.sha(Path(path).read_bytes()) == digest, path
  assert B.sha(Path(protocol['cpu_report']).read_bytes()) == protocol['cpu_report_sha256']
  assert B.sha(Path(protocol['cpu_check_report']).read_bytes()) == protocol['cpu_check_report_sha256']
  checks = json.loads(Path(protocol['cpu_check_report']).read_text())
  assert checks['complete'] and not checks['gpu_executed'] and checks['actual_program_saved_before_gate']
  cpu = json.loads(Path(protocol['cpu_report']).read_text())
  assert cpu['complete'] and not cpu['gpu_executed'] and cpu['serialized_worker_task_reproduces_cu_binary']
  assert cpu['parent_wgp_reproduces_passed_micro'] and cpu['in_memory_cache_compiler_collision']
  for path, digest in cpu['source_sha256'].items(): assert B.sha(Path(path).read_bytes()) == digest, path
  for path, digest in cpu['failure_evidence_sha256'].items(): assert B.sha(Path(path).read_bytes()) == digest, path
  sources = B.verify()
  sys.path.insert(0, str(B.SNAP))
  runner = B.load('agent-expert-pv-route-run')
  planned = runner.plan(B.ROOT/'invocations/quality-0.json', args.output.with_suffix('.model.json'))
  command = planned['effective_command'].copy()
  # Keep the first two calls in their original warmup state, including GC policy.
  command[command.index('--warmup')+1] = '2'
  command[command.index('--runs')+1] = '0'
  assert '--audit' not in command and planned['vision_policy'] == 'dynamic'
  return protocol, cpu, sources, planned, command


def capture(B, cpu, report, save, command, output):
  from tinygrad import Device
  from tinygrad.helpers import PARALLEL
  from tinygrad.runtime.support import hcq2
  from tinygrad.uop.ops import Ops
  from tinygrad.codegen import to_program_context
  H, R = B.load('agent-expert-pv-route-hook'), B.load('agent-expert-pv-route-run')
  handles, programs, input_asts, state = [], {}, {}, {}
  old_install, old_build = H.install, hcq2._build_queues
  def install(*values, **options):
    handle = old_install(*values, **options)
    state['handle'] = handle
    renderer = Device[Device.DEFAULT].renderer
    report['runtime'] = {'device': Device.DEFAULT, 'renderer_class': type(renderer).__name__,
      'renderer_target': str(renderer.target), 'compiler_class': type(renderer.compiler).__name__,
      'compiler_cachekey': renderer.compiler.cachekey, 'parallel': PARALLEL.value,
      'compiler_context': {v.key: str(v.value) for v in to_program_context},
      'environment': {key: os.environ.get(key) for key in ('DEV', 'PARALLEL', 'BEAM', 'JITBEAM', 'TC_OPT', 'LLVMOPT',
        'CCACHE', 'CACHEDB', 'AM_POWER_LIMIT', 'AM_PERF_LEVEL', 'GMMU')}}
    assert type(renderer.compiler).__name__ == 'WGPCompiler' and PARALLEL.value > 0
    original_dispatch = handle.dispatch
    def dispatch(out, a, b, **kw):
      ast = original_dispatch(out, a, b, **kw)
      if H.eligible(out, a, b, kw['shape'], kw.get('causal', False), kw.get('scaled', False), handle.depth > 0):
        record = {'shape': list(kw['shape']), 'input_ast_key': ast.key.hex(),
                  'input_specs': [{'shape': list(x.shape), 'dtype': str(x.dtype)} for x in (out, a, b)],
                  'kernel_info': repr(ast.arg)}
        input_asts[ast.key.hex()] = (ast, record)
        report['input_asts'] = [v[1] for v in input_asts.values()]
      return ast
    handle.dispatch = dispatch
    handle.module._matmul = dispatch
    old_inspect = handle.inspect_capture
    def inspect(self, jit):
      if jit.captured is None: return old_inspect(jit)
      records = [r for c in jit.captured._linear.src for r in c.without_after.arg.aux.kernels]
      counts = Counter(str(r[1]) for r in records)
      report['capture_counts'] = dict(counts)
      names = ('alpamayo_attention_values', 'alpamayo_attention_scores', 'alpamayo_attention_softmax')
      assert all(counts[n] == 84 for n in names) and counts[H.NAME] == 0
      keys = {r[4].hex() for r in records if str(r[1]) == names[0]}
      report['captured_pv_program_keys'] = sorted(keys)
      assert len(keys) == 1 and keys <= programs.keys()
      assert len(input_asts) == 1 and next(iter(input_asts.values()))[1]['shape'] == list(B.SHAPE)
      actual_program = programs[next(iter(keys))]
      ast, _ = next(iter(input_asts.values()))
      actual = B.artifacts(output.parent/'agent-expert-pv-binding-captured', 'actual-reference', actual_program, ast)
      report['actual_reference'] = actual
      report['captured_pv_call_arguments'] = [{'program_key': r[4].hex(), 'global_input_slots': list(r[5]),
        'output_argument_indices': list(r[6][0]), 'input_argument_indices': list(r[6][1])}
        for r in records if str(r[1]) == names[0]]
      save()  # Preserve evidence before every classification or old strict gate.
      matches = []
      for mode in ('parent-wgp', 'worker-cu'):
        expected = cpu['compiler_cases']['expert'][mode]['original']
        if actual['binary_sha256'] == expected['binary_sha256']:
          validate_program(actual, expected)
          assert actual['input_ast_key'] == expected['input_ast_key']
          matches.append(mode)
      assert len(matches) == 1, 'Captured program is neither source-bound compiler control'
      report['actual_reference_compiler_mode'] = matches[0]
      try: old_inspect(jit)
      except AssertionError as exc:
        assert matches == ['worker-cu'] and not str(exc), 'Unexpected old-gate failure'
        assert self.programs[next(iter(keys))]['binary_sha256'] == actual['binary_sha256']
        report['unchanged_old_binary_gate'] = {'passed': False, 'expected_micro_wgp': H.verify_evidence()['original_binary_sha256'],
          'actual': actual['binary_sha256'], 'failure': 'AssertionError at unchanged binary-set equality'}
      else:
        assert matches == ['parent-wgp']
        report['unchanged_old_binary_gate'] = {'passed': True}
      state['program'] = actual_program
      report['capture_diagnostic_complete'] = True
      save()
      raise CaptureComplete('First actual 84-PV capture recorded; model execution intentionally stops')
    handle.inspect_capture = MethodType(inspect, handle)
    return handle
  def build(ctx):
    changed = False
    for call, _, _ in ctx.batch:
      p = call.body
      if p.op is not Ops.PROGRAM or p.src[0].arg.function_name != 'alpamayo_attention_values': continue
      if p.key.hex() in programs: continue
      changed = True
      programs[p.key.hex()] = p
      report.setdefault('observed_pv_programs', {})[p.key.hex()] = B.artifacts(
        output.parent/'agent-expert-pv-binding-captured', 'observed-'+p.key.hex(), p)
      report['observed_pv_programs'][p.key.hex()]['call_argument_specs'] = [
        {'op': x.op.name, 'dtype': str(x.dtype), 'shape': list(x.max_shape), 'device': str(x.device)} for x in call.src[1:]]
    if changed: save()
    return old_build(ctx)
  H.install, hcq2._build_queues = install, build
  argv = sys.argv[:]
  try:
    sys.argv = command
    with R.interceptor(H, False, B.ROOT/'evidence/pv-gpu-report.json', handles):
      try: runpy.run_path(command[0], run_name='__main__')
      except CaptureComplete: pass
    assert 'program' in state, 'No actual trajectory capture within two predictions'
  finally:
    sys.argv = argv
    H.install, hcq2._build_queues = old_install, old_build
    report['hooks_on_exit'] = [h.info for h in handles]
    save()
  return state['program']


def micro(B, cpu, actual, report, save, budget):
  import numpy as np
  from tinygrad import Device, dtypes
  E, C, T, K = (B.load(n) for n in ('agent-expert-gemm', 'agent-expert-pv-check',
    'agent-vision-tiles-transfer-check', 'agent-expert-pv-kernel'))
  device = Device[Device.DEFAULT]
  assert device.arch == 'gfx1200' and device.is_usb
  renderers = B.renderers()
  report['micro_cases'] = []
  for name, shape in K.CASES.items():
    budget()
    programs = {}
    row = {'case': name, 'shape': list(shape), 'checks': [], 'complete': False}
    report['micro_cases'].append(row)
    for mode, renderer in renderers.items():
      for label in ('original', '2x2'):
        _, p = B.compile_program(renderer, shape, label)
        validate_program(B.describe(p), cpu['compiler_cases'][name][mode][label])
        programs[mode+'/'+label] = p
    if name == 'expert':
      mode = report['actual_reference_compiler_mode']
      p = programs[mode+'/original']
      assert actual.src[2].arg == p.src[2].arg and actual.src[3].arg == p.src[3].arg
      programs[mode+'/original'] = actual  # Execute the exact captured PROGRAM, not a replacement compile.
      row['actual_captured_program_executed'] = True
    transfers = T.InputTransfers(device)
    inputs = transfers.upload(C.fixture(shape, 'base')['values'], (dtypes.bfloat16, dtypes.bfloat16))
    singles = {label: C.SerialPV(shape, lambda *_, p=p: p, 1) for label, p in programs.items()}
    for jit in singles.values():
      for _ in range(3): jit(*inputs)
    base = previous = None
    for state in (*C.STATES, 'replay', 'post'):
      budget()
      data = C.fixture(shape, state)
      event = transfers.replace(inputs, data['values'])
      if state == 'a-only': assert event['writes'] == [True, False]
      if state == 'b-only-after-a': assert event['writes'] == [False, True]
      reference = singles['parent-wgp/original']
      E.poison((reference.output,), device)
      expected = E.read_output(reference(*inputs)[0], device)
      assert np.isfinite(E.unpack_bf16(expected)).all()
      if data['integer']: np.testing.assert_array_equal(expected, C.integer_reference(data))
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
      for label, jit in singles.items():
        E.poison((jit.output,), device)
        E.exact_output(jit(*inputs)[0], expected, device)
        row['checks'].append({'state': state, 'program': label, 'exact_bf16_bits': True,
          'independent_integer': data['integer'], 'reference_sha256': B.sha(expected.tobytes())})
      previous = expected.copy()
      save()
    if name == 'expert':
      changed = C.fixture(shape, 'post')
      changed['b'][..., 2017:2304] = E.bf16(np.float32(17))
      values = (changed['values'][0], C.pack_bits(changed['b']).reshape(changed['values'][1].shape))
      assert transfers.replace(inputs, values)['writes'] == [False, True]
      for jit in singles.values(): E.exact_output(jit(*inputs)[0], expected, device)
      row['masked_history_mutation_exact'] = True
      # Bounded replay proof only: no timings are published by this diagnostic.
      batches = {label: C.SerialPV(shape, lambda *_, p=p: p, 40) for label, p in programs.items()}
      for label, jit in batches.items():
        for _ in range(3): jit(*inputs)
        row.setdefault('serial_40_call_proofs', {})[label] = C.check_queue(jit)
      fresh = C.fixture(shape, 'replay')
      assert transfers.replace(inputs, fresh['values'])['writes'] == [True, True]
      expected = E.read_output(singles['parent-wgp/original'](*inputs)[0], device)
      assert not np.array_equal(expected, previous)
      for jit in batches.values():
        E.poison((jit.output,), device)
        E.exact_output(jit(*inputs)[0], expected, device)
      row['fresh_inputs_after_40_call_capture_exact'] = True
    row['transfers'], row['complete'] = transfers.report(), True
    save()
  report['micro_full_bit_controls_passed'] = True


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--output', type=Path, required=True)
  parser.add_argument('--protocol', type=Path, required=True)
  parser.add_argument('--protocol-sha', required=True)
  parser.add_argument('--plan', action='store_true')
  args = parser.parse_args()
  assert sys.flags.optimize == 0
  B = common()
  _, cpu, sources, planned, command = preflight(B, args)
  report = dict(sources, diagnostic_only=True, plan_only=args.plan, complete=False, gpu_executed=False,
    quality64_established=False, candidate_route_equality_established=False, timings_valid=False,
    maximum_model_predictions=2, micro_full_bit_controls_passed=False, command=command,
    policy=planned['vision_policy'], cpu_report_sha256=B.sha(B.CPU.read_bytes()))
  assert not args.output.exists(), 'Never overwrite diagnostic evidence'
  def save(): B.write(args.output, report)
  save()
  if args.plan: return
  started = time.perf_counter()
  def budget(): assert time.perf_counter()-started < 270, 'Diagnostic exceeded internal 270s budget'
  try:
    for key, value in json.loads((B.ROOT/'invocations/quality-0.json').read_text())['environment'].items():
      if value is None: os.environ.pop(key, None)
      else: os.environ[key] = value
    os.environ.update(PROFILE='0', PYTHONDONTWRITEBYTECODE='1')
    report['gpu_executed'] = True
    actual = capture(B, cpu, report, save, command, args.output)
    budget()
    micro(B, cpu, actual, report, save, budget)
    assert B.verify() == sources
    assert report['capture_diagnostic_complete'] and report['micro_full_bit_controls_passed']
    report['complete'] = True
  except BaseException as exc:
    report['failure'] = {'type': type(exc).__name__, 'message': str(exc)}
    raise
  finally:
    report['elapsed_seconds'] = time.perf_counter()-started
    save()


if __name__ == '__main__': main()
