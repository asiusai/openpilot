"""Optional per-instance draft GEMM hook, gated by this harness's GPU report."""
from contextlib import contextmanager
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace


def load_check():
  path = Path(__file__).with_name('agent-decode-gemm.py')
  spec = importlib.util.spec_from_file_location('agent_decode_gemm_gate', path)
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module


def validate(report, choices, current, wgp, tc_opt):
  assert report['complete'] and report['gpu_executed'] and report['mode'] == 'gpu'
  assert report['provenance'] == current and report['wgp'] == wgp and report['environment']['TC_OPT'] == tc_opt
  assert {(r['case'], r['bias']) for r in report['controls']} == {(n, b) for n in ('integer', 'integer-tail', 'sparse-outliers') for b in (False, True)}
  for row in report['controls']:
    assert len(row['checks']) == 6 and all(c['all_tiles_exact_bf16'] and c['independent_oracle'] for c in row['checks'])
  permitted = {}
  for key, n in choices.items():
    assert n in (1, 2), 'Only tested N alternatives are selectable'
    m, columns, width = map(int, key.split(','))
    matched = [r for r in report['cases'] if r['shape_mnk'] == [m, columns, width] and not r['bias']]
    assert len(matched) == 1, 'Selection requires exactly one checked actual model shape'
    row = matched[0]
    assert row['model_identity']['all_selected_fp8_finite'] and row['mutated_batch_replay_exact'] and row['post_timing_exact']
    assert len(row['checks']) == 6 and all(c['all_tiles_exact_bf16'] for c in row['checks'])
    assert row['checks'][2]['isolated_row_scale']
    for label in ('1x4', f'1x{n}'):
      proof = row['serial_proofs'][label]
      assert proof['calls'] == 40 and proof['ordered_execution_barriers'] == 39 and proof['actual_queue_builder']
    assert row['timings']['calls_per_batch'] == 40
    permitted[m, columns, width] = n
  assert permitted
  return permitted


@contextmanager
def scoped(decoder, permitted):
  """Patch only decoder.gemm; do not install model-wide dispatch or quantizers."""
  assert not decoder.verifiers, 'Install before any verifier trace or capture'
  assert tuple(decoder.tile) == (1, 4), 'This candidate compares against the selected 1x4 baseline'
  original = decoder.gemm
  stats = {'scope': 'DraftDecoder.verify GEMMs only', 'python_trace_calls': {}, 'selected': {','.join(map(str, k)): v for k, v in permitted.items()}}
  def gemm(x, weight, scales, tile=(4, 4), row_scale=False, bias=None, **kwargs):
    key = (x.shape[0], weight.shape[0], weight.shape[1])
    assert row_scale and tuple(tile) == (1, 4)
    assert scales[0].shape == (x.shape[0], 1), 'Keep one independent scale per verification row'
    chosen = permitted.get(key)
    if chosen is not None:
      assert bias is None, 'Actual language projections have no bias; optional bias is a kernel control only'
      tile = (1, chosen)
      label = ','.join(map(str, key))
      stats['python_trace_calls'][label] = stats['python_trace_calls'].get(label, 0)+1
    return original.gemm(x, weight, scales, tile=tile, row_scale=row_scale, bias=bias, **kwargs)
  decoder.gemm = SimpleNamespace(gemm=gemm)
  try: yield stats
  finally:
    decoder.gemm = original
    # Captured verifier programs contain the selected kernels. Remove only the
    # verifiers created while this initially empty experimental scope was active.
    for jit in decoder.verifiers.values(): jit.reset()
    decoder.verifiers.clear()
    stats['restored'] = True


@contextmanager
def install(decoder, report_path, choices, wgp=True):
  import os
  check = load_check()
  report = json.loads(Path(report_path).read_text())
  assert report['environment']['LLVMOPT'] == os.environ.get('LLVMOPT'), 'Compiler optimization setting changed'
  permitted = validate(report, choices, check.provenance(), wgp, os.environ.get('TC_OPT'))
  with scoped(decoder, permitted) as stats: yield stats


def cpu_check():
  calls = []
  original = SimpleNamespace(gemm=lambda *a, **kw: calls.append(kw) or 'sentinel')
  decoder = SimpleNamespace(gemm=original, tile=(1, 4), verifiers={})
  model_dispatch = object()
  decoder.model = SimpleNamespace(linear=model_dispatch)
  def tensor(shape): return SimpleNamespace(shape=shape)
  with scoped(decoder, {(16, 1024, 2048): 2}) as stats:
    args = (tensor((16, 2048)), tensor((1024, 2048)), (tensor((16, 1)), tensor((1,))))
    assert decoder.gemm.gemm(*args, tile=(1, 4), row_scale=True) == 'sentinel'
    assert calls[-1]['tile'] == (1, 2)
    decoder.gemm.gemm(tensor((32, 2048)), args[1], (tensor((32, 1)), args[2][1]), tile=(1, 4), row_scale=True)
    assert calls[-1]['tile'] == (1, 4) and decoder.model.linear is model_dispatch
    reset = []
    decoder.verifiers['test'] = SimpleNamespace(reset=lambda: reset.append(True))
  assert decoder.gemm is original and decoder.model.linear is model_dispatch and not decoder.verifiers and reset == [True]
  assert stats['restored']
  try:
    with scoped(SimpleNamespace(gemm=original, tile=(1, 4), verifiers={1: object()}), {}): pass
  except AssertionError as exc: assert 'before any verifier' in str(exc)
  else: raise AssertionError('Existing captured verifier accepted')
  checks = [{'decoder_only_dispatch_and_fallback': True, 'model_dispatch_untouched': True, 'hooks_and_verifiers_restored': True,
             'existing_verifier_rejected': True}]
  # Synthetic report exercises the admission contract only. It is never saved
  # or exposed as GPU evidence or a performance result.
  proof = {'calls': 40, 'ordered_execution_barriers': 39, 'actual_queue_builder': True}
  sample = {'shape_mnk': [16, 1024, 2048], 'bias': False, 'model_identity': {'all_selected_fp8_finite': True},
            'mutated_batch_replay_exact': True, 'post_timing_exact': True,
            'checks': [{'all_tiles_exact_bf16': True, 'isolated_row_scale': i == 2} for i in range(6)],
            'serial_proofs': {'1x4': proof, '1x2': proof.copy()}, 'timings': {'calls_per_batch': 40}}
  report = {'complete': True, 'gpu_executed': True, 'mode': 'gpu', 'provenance': 'synthetic-contract-test', 'wgp': True,
            'environment': {'TC_OPT': '2'}, 'cases': [sample], 'controls': [
              {'case': n, 'bias': b, 'checks': [{'all_tiles_exact_bf16': True, 'independent_oracle': True} for _ in range(6)]}
              for n in ('integer', 'integer-tail', 'sparse-outliers') for b in (False, True)]}
  selected = {'16,1024,2048': 2}
  assert validate(report, selected, 'synthetic-contract-test', True, '2') == {(16, 1024, 2048): 2}
  sample['serial_proofs']['1x2']['ordered_execution_barriers'] = 38
  try: validate(report, selected, 'synthetic-contract-test', True, '2')
  except AssertionError: pass
  else: raise AssertionError('Missing execution barrier passed installer gate')
  checks.append({'synthetic_report_admission_checked': True, 'missing_barrier_rejected': True})
  return checks
