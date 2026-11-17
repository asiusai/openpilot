"""Freeze the original live recipe and a fresh accepted/fallback parity pair."""
import argparse
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import importlib.util
spec = importlib.util.spec_from_file_location('exact_live_plan_common',HERE/'agent-exact-live-common.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def prepare(output, write=True, cpu_check=None):
  if output.exists() or output.parent != HERE or not output.name.startswith('agent-exact-live-'):
    raise ValueError('Use a fresh owned exact live protocol directory')
  sys.path[:0] = [str(m.c.TINYGRAD),str(m.ROOT/'experiments'),str(m.ROOT),
    '/tmp/opencode/openpilot-alpamayo','/home/john/asius/openpilot/opendbc_repo']
  evidence = m.evidence()
  policy = m.load(m.ROOT/'experiments/agent-paced-dynamic-policy.py','exact_live_plan_policy')
  with policy.imports(): replay = m.load(m.ROOT/'experiments/agent-live-replay.py','exact_live_plan_replay')
  full = replay.Dataset(m.WORK/'quality-samples.json',replay.SOURCE/'manifest.json')
  subset = m.selected(full)
  pipeline = m.load(m.ROOT/'experiments/benchmark-pipeline.py','exact_live_plan_pipeline')
  _,_,original_protocol = replay.normalize(m.INVOCATION,full,pipeline,HERE/'agent-exact-live-unused-output.json')
  combined_leaf = m.read(HERE/'agent-exact-combined-protocol-v1/quality-pipeline.json')
  combined_args = vars(pipeline.parse_args(combined_leaf['command'][2:]))
  original_flags = original_protocol['flags']
  differences = {key:dict(live=original_flags[key],combined=replay.tree(value)) for key,value in combined_args.items()
                 if key != 'output' and replay.tree(value) != original_flags[key]}
  if set(differences) != {'pipeline_inputs','freeze_warmup_gc','reduced_vocab'}:
    raise ValueError(f'Unexpected combined/live recipe differences: {differences}')
  if m.sha(original_flags['reduced_vocab']) != m.sha(combined_args['reduced_vocab']): raise ValueError('Vocabulary bytes differ')
  files = dict(m.read(m.COMBINED)['files'])
  def pin(path): files[str(Path(path).resolve())] = m.sha(path)
  pin(m.COMBINED)
  old_manifest = m.ROOT.parent/'manifest.json'
  for name in m.read(old_manifest)['files']: pin(old_manifest.parent/name)
  pin(old_manifest)
  identical, different = [], []
  for live in m.ROOT.rglob('*'):
    if not live.is_file() or live.suffix not in ('.py','.c','.json'): continue
    qualified = m.c.MLP/live.relative_to(m.ROOT)
    if qualified.is_file():
      if m.sha(live) == m.sha(qualified): identical.append(dict(live=str(live),qualified=str(qualified),sha256=m.sha(live)))
      else: different.append(dict(path=str(live.relative_to(m.ROOT)),live_sha256=m.sha(live),qualified_sha256=m.sha(qualified)))
  allowed = {'experiments/agent-fused-mlp-check.py','experiments/agent-fused-mlp-run.py','experiments/agent-fused-mlp.py',
             'experiments/agent-packed-epilogue-check.py','experiments/agent-vision-tiles-check.py'}
  if {r['path'] for r in different} != allowed: raise ValueError('Unexpected shared numerical foundation differences')
  for path in m.STARTUP.glob('*.py'): pin(path)
  for name in ('agent-startup-spec-reference','agent-startup-spec-candidate','agent-startup-spec-paced',
               'agent-exact-combined-quality-reference','agent-exact-combined-quality-candidate',
               'agent-exact-combined-native-reference','agent-exact-combined-native-candidate'):
    for path in (m.WORK/name).glob('*.json'): pin(path)
  for path in (m.WORK/'quality-samples.json',replay.SOURCE/'manifest.json',replay.SOURCE/'render-calibrations.json',
               m.RGB.parent/'paced-cache.json'): pin(path)
  inventory = m.read(m.RGB.parent/'paced-cache.json')
  if inventory['frames'] != 450 or len(inventory['files']) != 450: raise ValueError('Require original complete 450-frame RGB cache')
  for name,digest in inventory['files'].items():
    if m.sha(m.RGB/name) != digest: raise ValueError('Raw paced RGB changed')
    pin(m.RGB/name)
  for path in HERE.glob('agent-exact-live-*.py'): pin(path)
  pin(HERE/'agent-startup-spec-check.py')
  pin('/tmp/opencode/queue-alpamayo-exact-live.py')
  manifest = dict(schema=1,evidence=evidence,files=files,identical_foundations=identical,differing_archived_sources=different,
    full_samples=full.samples,full_dataset=full.provenance,selected_samples=subset.samples,selected_dataset=subset.provenance,
    original_protocol=original_protocol,combined_recipe_differences=differences,
    selection=dict(indices=list(m.INDICES),count=2,initial_lookup_history='empty',expected_acceptance=[True,False],
      previous_64_reference_reused=False,reason='Fresh matched pair preserves the selected initial history and every input audit'),
    pace=dict(start_seconds=35,seconds=8,hz=10,inference_budget_seconds=180,raw_frames=str(m.RGB),
      original_received_floor_20hz=True,queued_snapshots=0,maximum_active=1,automatic_after_parity=False))
  if write:
    if cpu_check is None: raise ValueError('Require completed source-matched CPU controls')
    check=m.read(cpu_check)
    if not check['passed'] or check['gpu_executed'] or check['actual_gpu_parity']:
      raise ValueError('CPU controls did not pass')
    for path,digest in check['source_sha256'].items():
      if m.sha(path) != digest: raise ValueError('CPU controls used different source')
    pin(cpu_check)
    manifest.update(cpu_controls=str(cpu_check),expected_transforms=check['transforms'])
    output.mkdir()
    m.write_new(output/'manifest.json',manifest)
    m.verify(manifest)
    print(json.dumps(dict(manifest=str(output/'manifest.json'),sha256=m.sha(output/'manifest.json'),gpu_executed=False)))
  return manifest


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--output',type=Path,required=True)
  parser.add_argument('--cpu-check',type=Path,required=True)
  args=parser.parse_args()
  prepare(args.output.resolve(),cpu_check=args.cpu_check.resolve())
