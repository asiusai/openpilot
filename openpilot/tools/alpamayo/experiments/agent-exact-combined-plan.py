"""Freeze composed protocols and audit original evidence without device access."""
import argparse
import copy
import importlib.util
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('combined_common_plan', HERE/'agent-exact-combined-common.py')
c = importlib.util.module_from_spec(spec)
spec.loader.exec_module(c)


def source_tree(root):
  return {str(p.relative_to(root)):c.sha(p) for p in root.rglob('*')
          if p.is_file() and p.suffix in ('.py', '.c', '.json') and '__pycache__' not in p.parts}


def prepare(output, cpu_check):
  if output.exists(): raise FileExistsError('Use a new protocol directory')
  if output.parent != HERE or not output.name.startswith('agent-exact-combined-'):
    raise ValueError('Use the owned experiment prefix')
  evidence = c.evidence()
  checks = c.read(cpu_check)
  if not checks['passed'] or checks['gpu_executed'] or checks['real_gpu_graph_validation']:
    raise ValueError('Require completed CPU-only composition controls')
  for path, digest in checks['source_sha256'].items():
    if c.sha(path) != digest: raise ValueError(f'CPU controls used different source: {path}')
  left, right = source_tree(c.MLP), source_tree(c.VISION)
  identical = sorted(k for k in left.keys() & right.keys() if left[k] == right[k])
  different = {k:dict(mlp=left[k], vision=right[k]) for k in sorted(left.keys() & right.keys()) if left[k] != right[k]}
  if any(not name.startswith('experiments/agent-') for name in different):
    raise ValueError('Shared canonical foundations differ')
  output.mkdir()
  manifest = dict(schema=1, canonical=str(c.MLP), vision_hook=str(c.VISION/'experiments/agent-vision-tiles.py'),
    evidence=evidence, cpu_controls=str(cpu_check), files={}, relocations=[], stages={},
    audit=dict(identical_files=identical, differing_sources=different,
      only_mlp=sorted(left.keys()-right.keys()), only_vision=sorted(right.keys()-left.keys()),
      original_vision_padding=2112, original_mlp_padding=2048,
      original_mlp_quality_group_limit=1, combined_quality_group_limit=2,
      combined_padding=2048, combined_gpu_qualified=False,
      source_policy='Use frozen MLP foundations and frozen vision hook; preserve all differing sources and original evidence'))
  def pin(path):
    path = Path(path).resolve()
    manifest['files'][str(path)] = c.sha(path)
  pin(cpu_check)
  for root, tree in ((c.MLP, left), (c.VISION, right)):
    for name in tree: pin(root/name)
  # Retain original MLP source pins, including the pinned tinygrad foundations.
  for stage in ('quality', 'native'):
    old = c.MLP.parent/(stage+'-manifest.json')
    saved = c.read(old)
    for path, digest in saved['files'].items():
      if c.sha(path) != digest: raise ValueError(f'Original MLP manifest changed: {path}')
      pin(path)
    pin(old)
  for stage in ('quality', 'native'):
    for stem, suffix in ((f'agent-fused-mlp-2048-{stage}-candidate','agent-fused-mlp'),
                         (f'vision-tiles-dynamic-{stage}-candidate','agent-vision-tiles')):
      report_path = c.WORK/stem/'report.json'
      status_path = c.WORK/stem/('report.'+suffix+'.json')
      report, status = c.read(report_path), c.read(status_path)
      count = 64 if stage == 'quality' else 36
      if len(report['warmup'])+len(report['runs']) != count: raise ValueError('Prior route coverage differs')
      if suffix == 'agent-fused-mlp':
        if not status['passed'] or not status['paired_exact']['float_bits_exact']: raise ValueError('MLP prior route gate failed')
        if report['pad_prompt_to'] != 2048: raise ValueError('Unexpected original MLP padding')
      else:
        if not status['exactness_validated'] or not status['comparison']['all_exact']: raise ValueError('Vision prior route gate failed')
        if report['pad_prompt_to'] != 2112: raise ValueError('Reaudit changed original vision padding')
      pin(report_path); pin(status_path)
  for name in ('agent-warp-fast-sampler.py', 'agent-warp-fast-kernel.c', 'agent-warp-fast-workers.c',
               'agent-warp-fast-validated-portable-v6.json', 'sources.json', 'comparison.json'):
    pin(c.SAMPLER/name)
  for path, digest in c.read(c.SAMPLER/'sources.json').items():
    if c.sha(path) != digest: raise ValueError(f'Original sampler source record changed: {path}')
    pin(path)
  pin(c.WORK/'agent-fused-mlp-2048-micro/report.json')
  for path in evidence['vision']['reports']: pin(path['report'])
  pin(c.VISION.parent/'coverage.json')
  # Also retain each original microcheck's exact helper-source records.
  for path, digest in c.read(evidence['mlp']['report'])['source_sha256'].items():
    if c.sha(path) != digest: raise ValueError('Original MLP micro source mismatch')
    pin(path)
  for entry in evidence['vision']['reports']:
    for path, digest in c.read(entry['report'])['sources'].items():
      if c.sha(path) != digest: raise ValueError('Original vision micro source mismatch')
      pin(path)
  for stage, counts in (('quality', (2,62)), ('native', (12,24))):
    outer_path = c.VISION.parent/(stage+'.json')
    leaf_path = c.VISION.parent/(stage+'-nested.json')
    outer, leaf = c.read(outer_path), c.read(leaf_path)
    pin(outer_path); pin(leaf_path)
    def relocate(record):
      result = copy.deepcopy(record)
      for i, value in enumerate(result['command']):
        if value.startswith(str(c.VISION)+'/'):
          old = Path(value); new = c.MLP/old.relative_to(c.VISION)
          if c.sha(old) != c.sha(new): raise ValueError(f'Relocation requires identical bytes: {old}')
          manifest['relocations'].append({'from':str(old), 'to':str(new), 'sha256':c.sha(old)})
          result['command'][i] = str(new)
      result['environment']['PYTHONPATH'] = f'{c.TINYGRAD}:{c.MLP}/experiments:{c.MLP}'
      result['environment'].update(PYTHONDONTWRITEBYTECODE='1', OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1')
      return result
    outer, leaf = relocate(outer), relocate(leaf)
    cmd = leaf['command']
    if c.option(cmd,'pixel-scale') != '1' or c.option(cmd,'steps') != '3' or c.option(cmd,'vision-group-limit') != '2':
      raise ValueError('Canonical full-image three-step recipe changed')
    banned = ('--vision-calibration','--capture-vision-scales','--front-history-indices','--history-pixel-scale',
              '--history-feature-scale','--trajectory-solver','--row-quantize','--fused-qkv')
    if any(flag in cmd for flag in banned): raise ValueError('Unexpected numerical or history override')
    cmd[cmd.index('--output')+1] = str(output/(stage+'-placeholder.json'))
    leaf_out = output/(stage+'-pipeline.json')
    c.write_new(leaf_out, leaf); pin(leaf_out)
    cmd = outer['command']
    cmd[cmd.index('--invocation')+1] = str(leaf_out)
    cmd[cmd.index('--output')+1] = str(output/(stage+'-placeholder.json'))
    for key, value in (('warmup',str(counts[0])),('runs',str(counts[1])),('pad-prompt-to','2048')):
      flag = '--'+key
      if flag in cmd: cmd[cmd.index(flag)+1] = value
      else: cmd.extend((flag,value))
    if c.option(cmd,'mode') != 'candidate' or c.option(cmd,'prefill-attention') != 'shuffle-fma':
      raise ValueError('Canonical speculation or attention changed')
    outer_out = output/(stage+'-invocation.json')
    c.write_new(outer_out, outer); pin(outer_out)
    manifest['stages'][stage] = dict(invocation=str(outer_out), warmup=counts[0], runs=counts[1])
    leaf_cmd = leaf['command']
    samples = leaf_cmd[leaf_cmd.index('--samples')+1:leaf_cmd.index('--initial-noise')]
    for sample in samples:
      for path in Path(sample).glob('*.npy'): pin(path)
    for key in ('initial-noise','reduced-vocab','prefill-calibration','native-manifest','native-calibrations'):
      if '--'+key in leaf_cmd: pin(c.option(leaf_cmd,key))
    if stage == 'native':
      fixture = c.read(c.option(leaf_cmd,'native-manifest'))
      rows = {row['sample']:row for row in fixture['schedule']}
      raw = Path(c.option(leaf_cmd,'native-frame-cache'))
      for sample in samples:
        for frame in rows[sample]['frames']: pin(raw/str(frame['view'])/f'{frame["video_index"]}.npy')
  for path in HERE.glob('agent-exact-combined-*.py'): pin(path)
  pin('/tmp/opencode/run-alpamayo-speed-trial.py')
  pin('/mnt/nvme-500/tmp/alpamayo-calibrated-two-six-20261116/matched-warm/2cam-invocation.json')
  pin('/tmp/opencode/queue-alpamayo-exact-combined.py')
  c.write_new(output/'manifest.json', manifest)
  c.verify(manifest)
  print(f'Prepared {output}/manifest.json; GPU execution remains pending')


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--output', type=Path, required=True)
  parser.add_argument('--cpu-check', type=Path, required=True)
  args = parser.parse_args()
  prepare(args.output.resolve(), args.cpu_check.resolve())
