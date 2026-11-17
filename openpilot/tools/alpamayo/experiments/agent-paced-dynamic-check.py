"""CPU-only dynamic-policy, real-array, import-hook, and 10 Hz scheduling checks.

Existing check modules supply actual canonical helpers and explicit GPU doubles.
All tinygrad device opens and socket connections are blocked for the whole suite.
"""
import argparse
from contextlib import ExitStack, redirect_stdout
import copy
import importlib.machinery
import importlib.util
import io
import json
import os
from pathlib import Path
import runpy
import socket
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

os.environ['DEV'] = 'CPU'
os.environ['HF_HUB_OFFLINE'] = os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
sys.dont_write_bytecode = True
WORK = Path('/mnt/nvme-500/tmp/alpamayo-speed-20261116')


def load(path):
  spec = importlib.util.spec_from_file_location(path.stem.replace('-', '_'), path)
  module = importlib.util.module_from_spec(spec)
  sys.modules[spec.name] = module
  spec.loader.exec_module(module)
  return module


def forbidden(*args, **kwargs):
  raise AssertionError('CPU checks must not open devices or network connections')


class Checks(unittest.TestCase):
  @classmethod
  def setUpClass(cls):
    cls.policy = load(EXP/'agent-paced-dynamic-policy.py')
    cls.manifest = cls.policy.verify_snapshot()
    cls.pipeline = cls.policy.local('benchmark-pipeline')
    cls.live_checks = cls.policy.local('agent-live-replay-check')
    cls.live_checks.ARGS = SimpleNamespace(checkpoint=Path('/tmp/opencode/alpamayo-asius-20261114/model'),
      samples_json=WORK/'quality-samples.json', source_manifest=Path('/mnt/nvme-256/footage/f9ad9a96/exports/0000008e--321b2ddc03/alpamayo-4hz-calibrated-two-six/manifest.json'),
      invocation=ARGS.snapshot/'pipeline-invocation.json', prediction_report=WORK/'full-fixed-vision-quality/report.json')
    with cls.policy.imports():
      cls.live_checks.Checks.setUpClass()
    cls.live = cls.live_checks.Checks('test_actual_consumer_arrays_and_canonical_output_are_unchanged')
    cls.replay, cls.protocol = cls.live.replay, cls.live.protocol
    cls.pace_checks = cls.policy.local('agent-paced-replay-check')
    cls.pace_checks.ARGS = SimpleNamespace(manifest=cls.live_checks.ARGS.source_manifest,
      logs=Path('/mnt/nvme-256/footage/f9ad9a96/realdata'), raw_frames=WORK/'agent-paced-replay-cache-35-43/rgb')
    class TenHz(cls.pace_checks.Checks):
      def scenario(self, *args, **kwargs):
        original = self.bridge.Bridge
        def ceiling(*values, **options):
          if options.pop('hz') != 5:
            raise AssertionError('Frozen CPU scenario changed; review the explicit ceiling override')
          return original(*values, hz=10., **options)
        with patch.object(self.bridge, 'Bridge', side_effect=ceiling):
          return super().scenario(*args, **kwargs)
    TenHz.setUpClass()
    cls.pace = TenHz('test_elapsed_service_selects_newest_real_20hz_queries')
    cls.padding = []
    cls.hook_events = []

  def test_recipe_is_exact_combined_policy_with_declared_changes(self):
    original = json.loads(Path(self.manifest['origin']['nested_path']).read_text())
    prepared = json.loads((ARGS.snapshot/'pipeline-invocation.json').read_text())
    a, b = (self.pipeline.parse_args(v['command'][2:]) for v in (original, prepared))
    changed = {key for key in vars(a) if getattr(a, key) != getattr(b, key)}
    self.assertEqual(changed, {'pad_prompt_to', 'vision_calibration', 'reduced_vocab', 'output'})
    self.assertEqual(b.pad_prompt_to, 2048)
    self.assertIsNone(b.vision_calibration)
    self.assertEqual((b.warmup, b.runs, b.steps), (2, 62, 3))
    self.assertEqual(b.samples, self.live.dataset.samples)
    self.assertEqual({k:v for k,v in original['environment'].items() if k != 'PYTHONPATH'},
                     {k:v for k,v in prepared['environment'].items() if k != 'PYTHONPATH'})
    outer = json.loads((ARGS.snapshot/'combined-invocation.json').read_text())
    for flag, expected in (('--pad-prompt-to', '2048'), ('--mode', 'candidate'), ('--prefill-attention', 'shuffle-fma')):
      self.assertEqual(outer['command'][outer['command'].index(flag)+1], expected)
    self.assertEqual(Path(outer['command'][outer['command'].index('--invocation')+1]), ARGS.snapshot/'pipeline-invocation.json')
    self.assertEqual(outer['environment']['PYTHONPATH'], prepared['environment']['PYTHONPATH'])
    for item in (prepared, outer):
      text = json.dumps(item)
      self.assertNotIn('/combined-source/', text)
      self.assertNotIn('/live-parity-source/', text)
    self.assertEqual({d['flag'] for d in self.protocol['disabled_flags']}, {'--pipeline-inputs', '--freeze-warmup-gc'})
    self.assertIsNone(self.protocol['files']['vision_calibration'])
    self.assertEqual(self.protocol['flags']['pad_prompt_to'], 2048)
    self.assertEqual(self.protocol['dynamic_policy']['decoder'], 'speculative')

  def test_frozen_normalization_checks_remain_except_explicit_policy(self):
    frozen = self.policy.local('agent-live-replay')  # Outside adaptation context.
    prepared = json.loads((ARGS.snapshot/'pipeline-invocation.json').read_text())
    with self.assertRaisesRegex(ValueError, 'fixed vision calibration'):
      frozen.normalize(ARGS.snapshot/'pipeline-invocation.json', self.live.dataset, self.pipeline, Path('/virtual/out'))
    fixed = copy.deepcopy(prepared)
    vision = Path(self.manifest['origin']['nested_path'])
    old = json.loads(vision.read_text())['command']
    fixed['command'] += ['--vision-calibration', old[old.index('--vision-calibration')+1]]
    original_read = Path.read_text
    with patch.object(Path, 'read_text', lambda p, *a, **kw: json.dumps(fixed)
                      if p == ARGS.snapshot/'pipeline-invocation.json' else original_read(p, *a, **kw)):
      _, _, baseline = frozen.normalize(ARGS.snapshot/'pipeline-invocation.json', self.live.dataset, self.pipeline, Path('/virtual/out'))
    adapted = copy.deepcopy(self.protocol)
    del adapted['dynamic_policy']
    baseline['flags']['vision_calibration'] = None
    baseline['files']['vision_calibration'] = None
    self.assertEqual(adapted, baseline)
    bad_dataset = SimpleNamespace(samples=list(reversed(self.live.dataset.samples)), provenance=self.live.dataset.provenance)
    with self.assertRaisesRegex(ValueError, 'exact original 64'):
      self.replay.normalize(ARGS.snapshot/'pipeline-invocation.json', bad_dataset, self.pipeline, Path('/virtual/out'))
    for field, value in (('pad_prompt_to', 2112), ('vision_calibration', Path('/fixed/scales')), ('verify_vision_warmup', True),
                         ('front_history_indices', '0,4'), ('steps', 2)):
      args = copy.copy(self.live.args)
      setattr(args, field, value)
      with self.assertRaises(ValueError):
        self.policy.selected_args(args)
    with patch.object(self.policy, 'sha', return_value='different frozen source'):
      with self.assertRaisesRegex(ValueError, 'Frozen normalizer source'):
        self.policy.adapt(frozen)

  def test_all_original_64_have_actual_canonical_padding_and_metadata(self):
    helper = self.live.checker
    policy = helper.policy
    settings = SimpleNamespace(max_tokens=64, vision_batch=1, byte_pixels=True, cache_rotary_inputs=False)
    anchor = helper.adapter.config.traj_ids['future_start']
    for row in self.live.dataset.rows:
      arrays = self.live.dataset.load(row)
      pad = lambda p, k, n: policy.padded_inputs(p, k, n, policy.CHECKPOINT_DRAFT_IDS, anchor)
      prepared, packed, upload, _ = self.live.helpers.prepare_reference(helper.adapter, arrays, settings, helper.compact, pad, 2048)
      self.assertEqual(packed[0].shape[1], 2048)
      self.assertEqual(arrays['image_frames'].shape, (10, 1, 3, 432, 768))
      self.assertEqual(arrays['nav_text'], [row['navigation_text']])
      self.assertEqual(upload[2].shape, (1, 2048))
      self.assertEqual(upload[-1].shape, (1, 64, 2))
      self.__class__.padding.append(dict(query_ns=row['query_ns'], actual_tokens=prepared['actual_prompt_length'],
        padded_tokens=2048, eligible=bool(prepared['prefill_draft_eligible']), full_frames=10))

  def test_actual_consumer_arrays_changes_signatures_and_completion(self):
    for name in ('test_actual_consumer_arrays_and_canonical_output_are_unchanged',
                 'test_changed_navigation_history_and_content_are_current',
                 'test_incompatible_input_signature_reallocates_once',
                 'test_failed_completion_never_accepts_output_or_reuses_owner',
                 'test_changed_snapshot_after_preflight_is_rejected'):
      with self.subTest(name=name):
        getattr(self.live, name)()

  def test_actual_target_verified_acceptance_fallback_and_exact_comparison(self):
    self.live.test_actual_verified_speculation_acceptance_and_fallback()
    self.live.test_exact_comparison_checks_every_output_and_verification_field()

  def test_current_factory_order_with_existing_mocked_initializers(self):
    self.live.test_original_launcher_hooks_reach_current_factory_once()
    self.assertTrue(all(not any('vision-calibration' in value for value in e['events']) for e in self.live.events))

  def test_new_import_hook_composes_with_actual_speculation_and_shuffle_hooks(self):
    events = []
    recorded = copy.deepcopy(self.live.normalized)
    recorded['environment'] = {}  # Never apply GPU environment in these CPU checks.
    real_spec = importlib.util.spec_from_file_location
    class Decoder:
      def __init__(self, *a, **kw):
        events.append('decoder-base')
        self.prefill_draft_records = []
    def prefill(*a): events.append('verified-prefill-install')
    def decoder(*a): events.append('verified-decoder-install')
    def attention(model, stage, variant, waves):
      self.assertEqual((stage, variant, waves), ('prefill', 'shuffle-fma', 8))
      events.append('shuffle-fma')
    supplied = {
      'online-attention.py': SimpleNamespace(install=lambda *a, **kw: events.append('base-attention')),
      'padded-prefill.py': SimpleNamespace(install=forbidden, pad_inputs=forbidden),
      'lookup-decode.py': SimpleNamespace(LookupDecoder=Decoder),
      'agent-speculation-prefill.py': SimpleNamespace(install_prefill=prefill, install_decoder=decoder),
      'agent-attention-rowsoftmax.py': SimpleNamespace(install=attention)}
    def spec(name, location, *a, **kw):
      replacement = supplied.get(Path(location).name)
      if replacement is None: return real_spec(name, location, *a, **kw)
      class Loader:
        def create_module(self, value): return None
        def exec_module(self, value): value.__dict__.update(vars(replacement))
      return importlib.machinery.ModuleSpec(name, Loader())
    def target(path, run_name):
      self.assertEqual((Path(path), run_name), (EXP/'benchmark-pipeline.py', '__main__'))
      pipeline = self.policy.local('benchmark-pipeline')
      effective = pipeline.parse_args()
      self.policy.selected_args(effective)
      pipeline.local_module('online-attention').install(object(), expert=False)
      padding = pipeline.local_module('padded-prefill')
      padding.install(object())
      pipeline.local_module('lookup-decode').LookupDecoder(object())
      # Loading replay under both nested hooks must retain the new normalizer.
      self.assertEqual(self.policy.local('agent-live-replay')._dynamic_policy['policy'], self.policy.POLICY)
      return {}
    argv = ['launcher', '--invocation', '/virtual/hooked.json', '--output', '/virtual/out.json', '--mode', 'candidate',
            '--prefill-attention', 'shuffle-fma', '--pad-prompt-to', '2048', '--warmup', '2', '--runs', '62']
    read, exists, write = Path.read_text, Path.exists, Path.write_text
    with patch.object(importlib.util, 'spec_from_file_location', spec), self.policy.imports(), \
         patch.object(sys, 'argv', argv), patch.object(runpy, 'run_path', target), patch.dict(os.environ), \
         patch.object(Path, 'read_text', lambda p, *a, **kw: json.dumps(recorded) if str(p) == '/virtual/hooked.json' else read(p, *a, **kw)), \
         patch.object(Path, 'exists', lambda p: False if str(p).startswith('/virtual/') else exists(p)), \
         patch.object(Path, 'write_text', lambda p, *a, **kw: 0 if str(p).startswith('/virtual/') else write(p, *a, **kw)):
      self.policy.local('agent-speculation-run').main()
    self.assertEqual(events, ['base-attention', 'shuffle-fma', 'verified-prefill-install', 'decoder-base', 'verified-decoder-install'])
    self.__class__.hook_events = events

  def fake_gate(self):
    sources = {p.name: self.policy.sha(p) for p in [*EXP.glob('*.py'), *EXP.glob('*.c'), EXP.parent/'model.py']}
    return dict(mode='consumer', gpu_executed=True, predictions_qualified=True, decoder='speculative', prefill_attention='shuffle-fma',
      comparison=dict(exact=True, queries=64, differences=[]), initialized=dict(source_sha256=sources),
      protocol=copy.deepcopy(self.protocol), rows=[dict(completed=True)]*64)

  def test_gate_rejects_old_recipe_partial_sync_and_source_drift(self):
    self.policy.qualified_gate(self.fake_gate())  # Explicit CPU fixture, never written as a qualifying report.
    old = json.loads((WORK/'live-speculative-consumer/report.json').read_text())
    with self.assertRaises(ValueError): self.policy.qualified_gate(old)
    for key, value in (('gpu_executed', False), ('decoder', 'ordinary'), ('prefill_attention', 'baseline')):
      bad = self.fake_gate()
      bad[key] = value
      with self.assertRaises(ValueError): self.policy.qualified_gate(bad)
    for change in ('fixed', 'padding', 'sources', 'sync', 'short', 'snapshot'):
      bad = self.fake_gate()
      if change == 'fixed': bad['protocol']['flags']['vision_calibration'] = '/old/scales.json'
      elif change == 'padding': bad['protocol']['flags']['pad_prompt_to'] = 2112
      elif change == 'sources': bad['initialized']['source_sha256']['model.py'] = 'changed'
      elif change == 'sync': bad['rows'][0] = dict(completed=False)
      elif change == 'short': bad['rows'].pop()
      else: bad['protocol']['dynamic_policy']['snapshot_manifest_sha256'] = 'old'
      with self.assertRaises(ValueError): self.policy.qualified_gate(bad)
    # The original paced gate must independently reject a changed normalized recipe.
    gate = self.fake_gate()
    gate['protocol']['flags']['steps'] = 2
    read = Path.read_text
    with self.policy.imports(), patch.object(Path, 'read_text', lambda p, *a, **kw:
      json.dumps(gate) if str(p) == '/virtual/gate.json' else read(p, *a, **kw)):
      paced = self.policy.local('agent-paced-replay')
      cli = SimpleNamespace(consumer_parity_report=Path('/virtual/gate.json'), invocation=ARGS.snapshot/'pipeline-invocation.json', output=Path('/virtual/out.json'))
      with self.assertRaisesRegex(ValueError, 'Initialization recipe differs'):
        paced.model_run(cli, self.pace.trace)

  def test_10hz_ceiling_with_actual_20hz_trace_and_elapsed_work(self):
    self.pace.test_elapsed_service_selects_newest_real_20hz_queries()
    for result in self.pace.scenarios.values():
      self.assertLessEqual(result.summary['completed_queries'], 80)
      self.assertEqual(result.summary['queued_prediction_snapshots'], 0)
      self.assertGreater(result.summary['unqueued_obsolete_road_queries'], 0)
    self.pace.test_stalled_camera_does_not_republish_a_plan()
    self.pace.test_failed_sync_and_budget_never_count_completion()
    self.pace.test_no_catchup_after_slow_inference_exceeds_source_end()
    self.pace.test_future_state_values_cannot_change_current_arrays()

  def test_10hz_selected_arrays_equal_original_causal_helpers(self):
    self.pace.test_prepared_arrays_match_canonical_helpers_at_selected_queries()

  def test_raw_cache_is_complete_and_matches_original_native_trace(self):
    cached = self.pace.paced.RGB(WORK/'agent-paced-replay-cache-35-43/rgb', self.pace.trace.frames)
    self.assertEqual(len(self.pace.trace.frames), 450)
    self.assertEqual(self.manifest['raw_cache']['frame_metadata_sha256'], self.pace.trace.frame_digest)
    self.assertEqual(cached.reads, 0)  # Integrity reads are separate from active delivery reads.

  def test_guarded_entry_preserves_frozen_main_and_requires_10hz(self):
    run = self.policy.local('agent-paced-dynamic-run')
    seen = []
    def selected(name):
      seen.append(name)
      return SimpleNamespace(main=lambda options: seen.append(options))
    with patch.object(run, 'policy_module', return_value=self.policy), \
         patch.object(self.policy, 'local', side_effect=selected), patch.object(self.policy, 'qualified_gate'):
      run.main(['--task', 'replay', '--mode', 'reference', '--invocation', str(ARGS.snapshot/'pipeline-invocation.json'),
        '--decoder', 'speculative', '--prefill-attention', 'shuffle-fma', '--output', '/virtual/report.json'])
      self.assertEqual(seen[0], 'agent-live-replay')
      self.assertNotIn('--task', seen[1])
      self.assertIn('shuffle-fma', seen[1])
      args = ['--task', 'paced', '--mode', 'run', '--invocation', str(ARGS.snapshot/'pipeline-invocation.json'),
        '--hz', '5', '--start-seconds', '35', '--seconds', '8', '--inference-budget-seconds', '180',
        '--consumer-parity-report', str(WORK/'live-speculative-consumer/report.json'), '--output', '/virtual/report.json']
      with self.assertRaisesRegex(ValueError, 'requested 10 Hz'): run.main(args)
      args[args.index('--hz')+1] = '10'
      run.main(args)
      self.assertEqual(seen[2], 'agent-paced-replay')
      self.assertEqual(seen[3][seen[3].index('--hz')+1], '10')

  def test_parent_queue_is_plan_only_and_orders_exact_gates(self):
    queue = load(Path('/tmp/opencode/queue-alpamayo-paced-dynamic.py'))
    with patch.object(queue.subprocess, 'run', side_effect=forbidden), redirect_stdout(io.StringIO()):
      queue.main(['--snapshot', str(ARGS.snapshot)])
    commands = queue.commands(ARGS.snapshot)
    self.assertEqual(len(commands), 3)
    self.assertIn('10', commands[2])
    self.assertIn('180', commands[2])
    self.assertNotIn('prepare-cache', repr(commands))
    self.assertEqual(commands[0][commands[0].index('mode')+1], 'reference')
    self.assertEqual(commands[1][commands[1].index('mode')+1], 'consumer')
    events = []
    fake = SimpleNamespace(verify_snapshot=lambda: self.manifest, sha=lambda p: queue.WRAPPER_SHA,
                           qualified_gate=lambda *a, **kw: events.append('gate'))
    read, exists = Path.read_text, Path.exists
    def read_report(path, *a, **kw):
      if path in [WORK/n/'report.json' for n in queue.NAMES[:2]]: return json.dumps({'gpu_executed': True})
      return read(path, *a, **kw)
    with patch.object(queue, 'load_policy', return_value=fake), patch.object(queue, 'validate', return_value=self.manifest), \
         patch.object(Path, 'read_text', read_report), \
         patch.object(Path, 'exists', lambda p: False if p in [WORK/n for n in queue.NAMES] else exists(p)), \
         patch.object(queue.subprocess, 'run', side_effect=lambda command, **kw: events.append(command[2])), redirect_stdout(io.StringIO()):
      queue.main(['--snapshot', str(ARGS.snapshot), '--execute'])  # Every child is a CPU mock.
    self.assertEqual(events, [*queue.NAMES[:2], 'gate', queue.NAMES[2]])


def main():
  global ARGS, EXP
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--snapshot', type=Path, default=WORK/'agent-paced-dynamic-source')
  parser.add_argument('--output', type=Path, required=True)
  ARGS = parser.parse_args()
  EXP = ARGS.snapshot/'alpamayo/experiments'
  sys.path[:0] = [str(EXP), str(EXP.parent)]
  from tinygrad import Device
  with patch.object(type(Device), '__getitem__', forbidden), patch.object(socket.socket, 'connect', forbidden), \
       patch.object(socket.socket, 'connect_ex', forbidden), patch.object(socket, 'create_connection', forbidden):
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Checks))
  report = dict(cpu_only=True, gpu_executed=False, model_executed=False, native_device_operations=False,
    tests=result.testsRun, failures=len(result.failures), errors=len(result.errors),
    snapshot_manifest_sha256=Checks.policy.sha(ARGS.snapshot/'manifest.json'), policy=Checks.policy.POLICY,
    original_queries=64, actual_padding=Checks.padding, nested_hook_order=Checks.hook_events,
    canonical_upload_arrays=Checks.live.helpers.Checks.upload_arrays_checked,
    initializer_events=Checks.live.events, canonical_10hz_array_parity=Checks.pace.parity,
    requested_hz=10, real_trace_fake_services={str(k):{name:v.summary[name] for name in (
      'completed_queries', 'admitted_road_queries', 'unqueued_obsolete_road_queries', 'unselected_arrived_tail', 'bridge_counts')}
      for k,v in Checks.pace.scenarios.items()},
    limitations='Fake GPU outputs and injected service clocks; no measured model or native performance. New actual dynamic GPU parity is still required.')
  with ARGS.output.open('x') as stream:
    json.dump(report, stream, indent=2)
    stream.write('\n')
  if not result.wasSuccessful():
    raise SystemExit(1)


if __name__ == '__main__':
  main()
