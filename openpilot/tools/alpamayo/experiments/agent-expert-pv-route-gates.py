"""CPU-only negative controls for the final frozen queue and evidence gates."""
import argparse
import copy
import importlib.util
import json
import os
from pathlib import Path
import sys
from unittest.mock import patch

QUEUE = Path('/tmp/opencode/queue-alpamayo-expert-pv-route.py')


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--output', type=Path, required=True)
  args = parser.parse_args()
  os.environ.update(DEV='CPU', ALLOW_DEVICE_USAGE='0', PYTHONDONTWRITEBYTECODE='1')
  spec = importlib.util.spec_from_file_location('pv_queue_test', QUEUE)
  Q = importlib.util.module_from_spec(spec); spec.loader.exec_module(Q)
  Q.verify_sources()
  H = Q.load(Q.SNAP/'experiments/agent-expert-pv-route-hook.py', 'pv_gate_hook')
  R = Q.load(Q.RUNNER, 'pv_gate_runner_test')
  evidence = H.verify_evidence(Q.EVIDENCE)
  sources = {p.name: p for p in [*(Q.SNAP/'experiments').glob('*.py'), *(Q.SNAP/'experiments').glob('*.c'), Q.SNAP/'model.py']}
  real_bytes, real_text = Path.read_bytes, Path.read_text
  reports, sidecars = {}, {}
  def text(path, *a, **kw):
    if str(path) in reports: return json.dumps(reports[str(path)])
    if str(path) in sidecars: return json.dumps(sidecars[str(path)])
    return real_text(path, *a, **kw)
  def raw(path):
    if path.parent.name == 'report-sources' and path.parent.parent.name.startswith('agent-expert-pv-route-'):
      return real_bytes(sources[path.name])
    return real_bytes(path)
  original_hash = Q.file_sha
  def digest(path):
    path = Path(path)
    if path.parent.name == 'report-sources' and path.parent.parent.name.startswith('agent-expert-pv-route-'): return Q.sha(raw(path))
    return original_hash(path)
  rejected = []
  with patch.object(Path, 'read_text', text), patch.object(Path, 'read_bytes', raw), patch.object(Q, 'file_sha', digest):
    for label in Q.COUNTS:
      planned = R.plan(Q.protocol(label), Q.output(label, 'reference')/'report.json')
      baseline_name = f'agent-expert-tiles-{label}-reference'
      baseline = json.loads(real_text(Q.WORK/baseline_name/'report.json'))
      sidecar = json.loads(real_text(Q.WORK/baseline_name/'report.speculation.json'))
      baseline.pop('agent_expert_tiles')
      for variant in ('reference', 'candidate'):
        report = copy.deepcopy(baseline)
        report['source_sha256'] = {n: Q.sha(real_bytes(p)) for n, p in sources.items()}
        report['source_snapshot'] = str(Q.output(label, variant)/'report-sources')
        terminal = planned['terminal_command']
        for field, flag in (('manifest', '--native-manifest'), ('calibrations', '--native-calibrations')):
          path = R.option(terminal, flag)
          report['native_preprocessing'][field] = path
          report['native_preprocessing'][field+'_sha256'] = original_hash(path)
        own = copy.deepcopy(planned); cmd = own['effective_command']; cmd[cmd.index('--output')+1] = str(Q.output(label, variant)/'report.json')
        candidate = variant == 'candidate'
        name = H.NAME if candidate else 'alpamayo_attention_values'
        binary = evidence['binary_sha256'] if candidate else evidence['original_binary_sha256']
        hook = {'candidate': candidate, 'evidence': evidence, 'traces': [{'pv': 84, 'qk': 84}],
          'trajectory_calls': sum(Q.COUNTS[label]), 'unqualified_trajectory_calls': 0, 'outside_scope_selected': 0,
          'captures': [{'pv_calls': 84, 'qk_calls': 84, 'softmax_calls': 84, 'binaries': {
            name: [binary], 'alpamayo_attention_scores': ['qk'], 'alpamayo_attention_softmax': ['softmax']}}]}
        report['agent_expert_pv'] = {'variant': 'candidate' if candidate else 'current', 'plan_only': False,
          'plan': own, 'evidence': evidence, 'route_completed': True, 'coverage_validated': True,
          'exactness_validated': candidate, 'hooks': [hook], 'comparison': {'all_exact': True, 'samples': sum(Q.COUNTS[label])}}
        reports[str(Q.output(label, variant)/'report.json')] = report
        sidecars[str(Q.output(label, variant)/'report.speculation.json')] = copy.deepcopy(sidecar)
        Q.validate(report, label, variant, planned)
      candidate = reports[str(Q.output(label, 'candidate')/'report.json')]
      for field in ('missing_pv', 'extra_pv', 'wrong_binary', 'unqualified', 'wrong_qk', 'fixed_vision', 'no_exact', 'source', 'sixth_target'):
        bad = copy.deepcopy(candidate)
        hook = bad['agent_expert_pv']['hooks'][0]
        if field == 'missing_pv': hook['captures'][0]['pv_calls'] = 83
        elif field == 'extra_pv': hook['captures'][0]['pv_calls'] = 85
        elif field == 'wrong_binary': hook['captures'][0]['binaries'][H.NAME] = ['changed']
        elif field == 'unqualified': hook['unqualified_trajectory_calls'] = 1
        elif field == 'wrong_qk': hook['captures'][0]['binaries']['alpamayo_attention_scores'] = ['changed']
        elif field == 'fixed_vision': bad['vision_calibration_sha256'] = 'changed'
        elif field == 'no_exact': bad['agent_expert_pv']['exactness_validated'] = False
        elif field == 'source': bad['source_sha256']['expert-kernels.py'] = 'changed'
        else: bad['runs'][0]['draft']['prefill_draft']['targets'][5] += 1
        try: Q.validate(bad, label, 'candidate', planned)
        except AssertionError: rejected.append(label+'/'+field)
        else: raise AssertionError(f'Invalid final gate passed: {label}/{field}')
  # Evidence and runtime function mutations must fail independently of a fresh manifest.
  with patch.object(Path, 'read_bytes', lambda p: real_bytes(p)+b' ' if p == Q.EVIDENCE else real_bytes(p)):
    try: H.verify_evidence(Q.EVIDENCE)
    except AssertionError: pass
    else: raise AssertionError('Changed GPU report passed')
  import amd_attention
  original = amd_attention._matmul.__code__
  try:
    amd_attention._matmul.__code__ = (lambda *a, **kw: None).__code__
    try: H.verify_function(amd_attention._matmul, amd_attention.__file__)
    except (AssertionError, StopIteration): pass
    else: raise AssertionError('Changed in-memory arithmetic passed')
  finally: amd_attention._matmul.__code__ = original
  from tinygrad import Device
  assert not Device._opened_devices
  assert all(not Q.output(k, v).exists() for k in Q.COUNTS for v in ('reference', 'candidate'))
  result = {'complete': True, 'gpu_executed': False, 'synthetic_only': True,
    'queue_sha256': Q.file_sha(QUEUE), 'manifest_sha256': Q.EXPECTED_MANIFEST_SHA256,
    'all_four_full_report_gates_checked': True, 'rejected': rejected, 'report_mutation_rejected': True,
    'runtime_code_mutation_rejected': True, 'no_future_gpu_outputs_created': True, 'sequence': Q.queue_checks()}
  args.output.write_text(json.dumps(result, indent=2)+'\n')
  print(json.dumps({'complete': True, 'gpu_executed': False, 'rejected_reports': len(rejected)}))


if __name__ == '__main__': main()
