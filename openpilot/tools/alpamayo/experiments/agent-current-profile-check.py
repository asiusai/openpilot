"""CPU-only checks for the diagnostic wrapper. No tinygrad or model is initialized."""
import ast
import copy
import importlib.abc
import importlib.util
import json
import math
import os
from pathlib import Path
import runpy
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

EXP = Path(__file__).resolve().parent


def local(name):
  spec = importlib.util.spec_from_file_location(name.replace('-', '_'), EXP/(name+'.py'))
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module


runner = local('agent-current-profile-run')
summary = local('agent-current-profile-summary')


class PlanChecks(unittest.TestCase):
  def test_actual_plan_and_exact_delta(self):
    output = EXP/'agent-current-profile-unused-report.json'
    control, profiled, plan = runner.build_plan(output)
    self.assertNotIn('--vision-calibration', control['command'])
    self.assertNotIn('--vision-calibration', profiled['command'])
    expected = copy.deepcopy(control)
    expected['command'].remove('--coarse-timing')
    expected['command'].append('--profile-kernels')
    expected['environment']['PROFILE'] = '1'
    self.assertEqual(expected, profiled)
    self.assertEqual(len(plan['native_prefix']['queries']), 10)
    self.assertEqual(len({q['sample'] for q in plan['native_prefix']['queries']}), 10)
    self.assertEqual(runner.option(profiled['command'], '--pad-prompt-to'), '2048')
    self.assertFalse(plan['historical_outer']['used_as_base'])
    self.assertFalse(plan['gpu_executed'])

  def test_incompatible_base_rejected(self):
    recorded = json.loads(runner.BASE_INVOCATION.read_text())
    for change in ('nested', 'pixel', 'profile', 'bounds', 'solver'):
      data = copy.deepcopy(recorded)
      if change == 'nested': data['command'][1] = str(runner.LAUNCHER)
      if change == 'pixel': runner.replace(data['command'], '--pixel-scale', '.75')
      if change == 'profile': data['environment']['PROFILE'] = '1'
      if change == 'solver': data['command'] += ['--trajectory-solver', 'heun']
      with self.subTest(change=change), self.assertRaises(ValueError):
        runner.derive(data, Path('unused'), 9 if change == 'bounds' else 8, 2)

  def test_ast_only_profile_constant_changes(self):
    source = runner.LAUNCHER.read_text()
    original = ast.parse(source)
    tree, _ = runner.profile_ast(source)
    assertions = lambda t: [ast.dump(n) for n in ast.walk(t) if isinstance(n, (ast.Assert, ast.Raise))]
    self.assertEqual(assertions(original), assertions(tree))
    originals = [n.value for n in ast.walk(original) if isinstance(n, ast.Constant)]
    changed = [n.value for n in ast.walk(tree) if isinstance(n, ast.Constant)]
    self.assertEqual([(a, b) for a, b in zip(originals, changed) if a != b], [('0', '1')])
    with self.assertRaises(ValueError): runner.profile_ast(source.replace("os.environ['PROFILE'] = '0'", "os.environ['PROFILE'] = '2'"))
    with self.assertRaises(ValueError): runner.profile_ast(source + "\nos.environ['PROFILE'] = '0'\n")

  def test_import_order_guard(self):
    with patch.dict(sys.modules, {'tinygrad': ModuleType('tinygrad')}), self.assertRaises(RuntimeError):
      runner.execute_canonical([], {})

  def test_report_gate_and_mutations(self):
    # Historical outputs provide serialization structure only. Synthetic profiles are not current GPU evidence.
    control, profiled, plan = runner.build_plan(EXP/'agent-current-profile-unused-report.json')
    plan['profiled_invocation'] = profiled
    old = json.loads((runner.WORK/'combined-prefill-native-5hz/report.json').read_text())
    report = copy.deepcopy(old)
    entries = copy.deepcopy(old['warmup'][:10])
    report.update(profiling=True, pad_prompt_to=2048, vision_calibration_sha256=None,
                  warmup=entries[:8], runs=entries[8:], source_sha256={})
    report['settings']['coarse_timing'] = False
    report['summary']['profiling_enabled'] = True
    report['prefill_speculation'].update(timings_valid_for_throughput=False, padded_prompt_bound=2048)
    for entry in entries:
      stage = dict(sum_kernel_ms=1., kernels=[dict(name='cpu_mock_only', count=1, total_ms=1., max_ms=1.)])
      entry['device_profile'] = dict(invalid_samples=[], stages={k: copy.deepcopy(stage) for k in ('vision', 'prefill', 'diffusion')})
    runner.validate_report(report, plan)
    for mutation in ('nonfinite', 'unverified', 'missing-stage', 'coarse', 'fixed-vision', 'reordered'):
      modified = copy.deepcopy(report)
      if mutation == 'nonfinite': modified['runs'][0]['prediction']['xyz'][0][0][0] = float('nan')
      if mutation == 'unverified': modified['runs'][0]['draft']['prefill_draft']['all_targets_finite'] = False
      if mutation == 'missing-stage': del modified['runs'][0]['device_profile']['stages']['vision']
      if mutation == 'coarse': modified['settings']['coarse_timing'] = True
      if mutation == 'fixed-vision': modified['vision_calibration_sha256'] = 'unwanted'
      if mutation == 'reordered': modified['runs'].reverse()
      with self.subTest(mutation=mutation), self.assertRaises(ValueError): runner.validate_report(modified, plan)


class SummaryChecks(unittest.TestCase):
  @staticmethod
  def entry():
    return dict(sample='native-1', device_profile=dict(invalid_samples=[], stages={
      'vision': dict(sum_kernel_ms=5., kernels=[dict(name='\x1b[31malpamayo_fp8_gemm_4_4\x1b[0m', count=2, total_ms=3., max_ms=2.),
                                               dict(name='copy  1 KB, AMD <- CPU', count=1, total_ms=2., max_ms=2.)]),
      'prefill': dict(sum_kernel_ms=10., kernels=[dict(name='alpamayo_online', count=4, total_ms=10., max_ms=4.)]),
      'unclassified_stage': dict(sum_kernel_ms=1., kernels=[dict(name='some_new_kernel', count=1, total_ms=1., max_ms=1.)])}))

  def test_full_sums_families_unknowns_transfers(self):
    result = summary.aggregate([self.entry(), self.entry()])
    self.assertEqual(result['sum_gpu_events_ms'], 32.)
    self.assertEqual(result['categories']['vision']['mean_gpu_events_ms_per_query'], 5.)
    self.assertEqual(result['stages']['vision']['transfer_events_ms'], 4.)
    self.assertEqual(result['stages']['vision']['nontransfer_events_ms'], 6.)
    self.assertFalse(result['categories']['decoder']['observed'])
    self.assertEqual(result['categories']['unclassified_stage']['sum_gpu_events_ms'], 2.)
    self.assertEqual(result['stages']['vision']['kernels'][0]['family'], 'alpamayo_fp8_gemm')
    self.assertFalse(result['timings_valid_for_throughput'])
    json.dumps(result, allow_nan=False)

  def test_invalid_and_missing_coverage_are_explicit(self):
    entry = self.entry()
    entry['device_profile']['invalid_samples'].append(dict(stage='prefill', name='bad', elapsed_ms=-1))
    self.assertFalse(summary.aggregate([entry])['valid_timestamp_coverage'])
    entry = self.entry()
    entry['device_profile']['stages']['vision']['sum_kernel_ms'] += 1
    with self.assertRaises(ValueError): summary.aggregate([entry])
    entry = self.entry()
    entry['device_profile']['stages']['vision']['kernels'][0]['total_ms'] = float('nan')
    with self.assertRaises(ValueError): summary.aggregate([entry])

  def test_historical_fixture_totals_only(self):
    report = json.loads((runner.WORK/'profile-two/report.json').read_text())
    result = summary.summarize(report)['measured']
    independent = math.fsum(stage['sum_kernel_ms'] for entry in report['runs'] for stage in entry['device_profile']['stages'].values())
    self.assertAlmostEqual(result['sum_gpu_events_ms'], independent, places=7)
    self.assertGreater(result['categories']['vision']['sum_gpu_events_ms'], 0)
    self.assertGreater(result['categories']['expert']['sum_gpu_events_ms'], 0)


class CanonicalMockChecks(unittest.TestCase):
  def mock_run(self, enabled, fail=False):
    """Exercise the real canonical launcher and loader chain with independent CPU stubs."""
    events = []
    draft = (124549, 11135, 3174, 4286, 1561, 135084)
    original_spec = importlib.util.spec_from_file_location
    class Device:
      collect_prof = 'original-collect'
      _at_profile_finalize = 'original-finalizer'
    device = Device()
    benchmark = SimpleNamespace(timed='original-timer')
    original_environment = dict(os.environ)
    original_argv, original_path = sys.argv, sys.path[:]
    fake_device_module = ModuleType('tinygrad.device')
    fake_device_module.finalize_profile = lambda: self.fail('Generic GPU finalizer must not run')

    class Tokenizer:
      @classmethod
      def from_pretrained(cls, *args, **kwargs):
        events.append(('tokenizer', args, kwargs))
        return cls()
      def encode(self, *args, **kwargs): return list(draft)

    class Loader(importlib.abc.Loader):
      def __init__(self, name): self.name = name
      def create_module(self, spec): return None
      def exec_module(self, module):
        if self.name == 'agent-speculation-policy':
          module.DRAFT_TEXT, module.CHECKPOINT_DRAFT_IDS = 'complete draft', draft
          def validate(value, anchor):
            if tuple(value) != draft or anchor != draft[-1]: raise AssertionError('Draft changed')
            return tuple(value)
          module.validate_draft = validate
          module.padded_inputs = lambda *args: ('canonical-policy', args)
        elif self.name == 'agent-speculation-prefill':
          module.install_prefill = lambda model, d, anchor, audit: events.append(('prefill', model, d, anchor, audit))
          def decoder(obj, d, anchor, audit):
            events.append(('decoder-install', d, anchor, audit))
            obj.prefill_draft_records = [dict(accepted=True)]
          module.install_decoder = decoder
        elif self.name == 'online-attention':
          module.install = lambda *args, **kwargs: events.append(('original-attention', args, kwargs))
        elif self.name == 'agent-attention-rowsoftmax':
          module.install = lambda *args: events.append(('shuffle-attention', args))
        elif self.name == 'padded-prefill':
          module.install = lambda model: self.fail('Canonical candidate prefill hook was lost')
          module.pad_inputs = lambda *args: self.fail('Canonical candidate padding hook was lost')
        elif self.name == 'lookup-decode':
          class Decoder:
            def __init__(self, *args, **kwargs): events.append(('original-decoder', args, kwargs))
          module.LookupDecoder = Decoder
        elif self.name == 'profile-kernels':
          class Recorder:
            def __init__(self, dev, bench):
              dev.collect_prof, dev._at_profile_finalize, bench.timed = 'profile-collect', 'profile-finalizer', 'profile-timer'
          module.Recorder = Recorder
        else: raise AssertionError(self.name)

    def mocked_spec(name, location, *args, **kwargs):
      path = Path(location)
      return importlib.util.spec_from_loader(name, Loader(path.stem), origin=str(path))

    with tempfile.TemporaryDirectory(prefix='agent-current-profile-cpu-', dir=EXP) as directory:
      output = Path(directory)/'agent-current-profile-report.json'
      base = json.loads(runner.BASE_INVOCATION.read_text())
      control, profiled = runner.derive(base, output, 8, 2)
      selected = profiled if enabled else control
      invocation = Path(directory)/'agent-current-profile-invocation.json'
      invocation.write_text(json.dumps(selected))
      command = [str(runner.LAUNCHER), '--output', str(output), '--invocation', str(invocation),
                 '--mode', 'candidate', '--prefill-attention', 'shuffle-fma', '--pad-prompt-to', '2048', '--warmup', '8', '--runs', '2']

      def pipeline(path, run_name):
        self.assertEqual(os.environ['PROFILE'], '1' if enabled else '0')
        self.assertEqual(path, str(EXP/'benchmark-pipeline.py'))
        self.assertEqual('--profile-kernels' in sys.argv, enabled)
        self.assertEqual('--coarse-timing' in sys.argv, not enabled)
        attention, padding, lookup = [local(name) for name in ('online-attention', 'padded-prefill', 'lookup-decode')]
        attention.install('same-model', 'unchanged-original-args', unchanged=True)
        padding.install('same-model')
        padded = padding.pad_inputs('prepared', 'packed', 2048)
        self.assertEqual(padded, ('canonical-policy', ('prepared', 'packed', 2048, draft, draft[-1])))
        lookup.LookupDecoder('same-model', original_option=7)
        events.extend([('original-vision', 'full-five-frames', 'dynamic-fp8'),
                       ('original-expert', 'full-history', 'three-euler'), ('original-action-decode',)])
        if enabled:
          local('profile-kernels').Recorder(device, benchmark)
          sys.modules['tinygrad.device'] = fake_device_module
        if fail: raise RuntimeError('independent pipeline failure')
        prediction = dict(generated_token_ids=list(draft), xyz=[[[1., 2., 3.]]], rot=[[[0., 0., 0.]]])
        output.write_text(json.dumps(dict(summary=dict(profiling_enabled=enabled), prediction=prediction), allow_nan=False))

      fake_transformers = ModuleType('transformers')
      fake_transformers.AutoTokenizer = Tokenizer
      try:
        with patch.dict(sys.modules, {'transformers': fake_transformers}), \
             patch.object(importlib.util, 'spec_from_file_location', mocked_spec), \
             patch.object(runpy, 'run_path', pipeline), patch.object(atexit_module := runner.atexit, 'unregister') as unregister:
          if fail:
            with self.assertRaisesRegex(RuntimeError, 'independent pipeline failure'):
              runner.execute_canonical(command, selected['environment'], enabled=enabled)
          else:
            runner.execute_canonical(command, selected['environment'], enabled=enabled)
          self.assertIs(importlib.util.spec_from_file_location, mocked_spec)
          self.assertEqual(unregister.call_count, int(enabled))
        result = None if fail else json.loads(output.read_text())
      finally:
        sys.modules.pop('tinygrad.device', None)
      self.assertEqual(vars(device), {})
      self.assertEqual(device.collect_prof, 'original-collect')
      self.assertEqual(device._at_profile_finalize, 'original-finalizer')
      self.assertEqual(benchmark.timed, 'original-timer')
      self.assertEqual(dict(os.environ), original_environment)
      self.assertIs(sys.argv, original_argv)
      self.assertEqual(sys.path, original_path)
      self.assertIs(importlib.util.spec_from_file_location, original_spec)
      return result, events

  def test_paired_control_serialization_and_non_target_calls(self):
    control, control_events = self.mock_run(False)
    profiled, profile_events = self.mock_run(True)
    self.assertEqual(control_events, profile_events)
    self.assertEqual(json.dumps(control['prediction'], sort_keys=True, allow_nan=False),
                     json.dumps(profiled['prediction'], sort_keys=True, allow_nan=False))
    self.assertTrue(control['prefill_speculation']['timings_valid_for_throughput'])
    self.assertFalse(profiled['prefill_speculation']['timings_valid_for_throughput'])
    self.assertEqual(profiled['prefill_speculation']['draft'], list((124549, 11135, 3174, 4286, 1561, 135084)))

  def test_finally_restores_hooks_after_failure(self):
    self.mock_run(True, fail=True)


if __name__ == '__main__': unittest.main(verbosity=2)
