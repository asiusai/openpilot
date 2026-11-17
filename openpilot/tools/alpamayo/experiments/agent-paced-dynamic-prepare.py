"""CPU-only immutable snapshot/recipe builder. No child process or model execution."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parent
WORK = Path('/mnt/nvme-500/tmp/alpamayo-speed-20261116')
POLICY = 'dynamic-vision-pad2048-nominal-verified-prefill-shuffle-fma-v1'
PYTHONPATH_SUFFIX = '/tmp/opencode/openpilot-alpamayo:/home/john/asius/openpilot/opendbc_repo'


def sha(path):
  return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
  with path.open('x') as stream:
    json.dump(value, stream, indent=2)
    stream.write('\n')


def option(command, name, value):
  if command.count(name) != 1:
    raise ValueError(f'Expected one existing {name}')
  command[command.index(name)+1] = str(value)


def build(destination):
  destination = destination.resolve()
  if destination.parent != WORK or not destination.name.startswith('agent-paced-dynamic-') or destination.exists():
    raise ValueError('Use a new agent-paced-dynamic directory on the 500 GB SSD')
  usage = shutil.disk_usage(destination.parent)
  if usage.free-64*1024**2 < max(usage.total*.1, 5*1024**3):
    raise ValueError('Preserve both 10% and 5 GiB reserves')
  outer_path = WORK/'combined-prefill-quality/invocation.json'
  outer = json.loads(outer_path.read_text())
  nested_path = Path(outer['command'][outer['command'].index('--invocation')+1])
  nested = json.loads(nested_path.read_text())
  if outer['command'][outer['command'].index('--mode')+1] != 'candidate' or outer['command'][outer['command'].index('--prefill-attention')+1] != 'shuffle-fma':
    raise ValueError('Combined source must use nominal candidate prefill and shuffle-FMA')
  if '--audit' in outer['command']:
    raise ValueError('An audit invocation cannot qualify throughput')
  base = WORK/'live-parity-source/alpamayo'
  prior = json.loads((WORK/'live-speculative-consumer/report.json').read_text())
  if prior['comparison']['exact'] is not True or prior['comparison']['queries'] != 64:
    raise ValueError('Missing current factory extraction parity baseline')
  for name, digest in prior['initialized']['source_sha256'].items():
    path = base/'model.py' if name == 'model.py' else base/'experiments'/name
    if sha(path) != digest:
      raise ValueError(f'Qualified current base source changed: {name}')
  destination.mkdir()
  source = destination/'alpamayo'
  shutil.copytree(base, source, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
  exp = source/'experiments'
  for path in [ROOT/'agent-paced-replay.py', ROOT/'agent-paced-replay-check.py', *ROOT.glob('agent-paced-dynamic-*.py')]:
    if (exp/path.name).exists():
      raise FileExistsError(path.name)
    shutil.copyfile(path, exp/path.name)
  recipe = copy.deepcopy(nested)
  cmd = recipe['command']
  old_source = Path(cmd[1]).parent.parent
  cmd[1] = str(exp/'benchmark-pipeline.py')
  option(cmd, '--reduced-vocab', exp/'coc_vocab.json')
  option(cmd, '--output', WORK/'agent-paced-dynamic-reference/report.benchmark.json')
  old_pad = cmd[cmd.index('--pad-prompt-to')+1]
  option(cmd, '--pad-prompt-to', 2048)
  if cmd.count('--vision-calibration') != 1:
    raise ValueError('Expected the combined fixed-vision calibration to remove')
  index = cmd.index('--vision-calibration')
  old_vision = cmd[index+1]
  del cmd[index:index+2]
  path_parts = recipe['environment']['PYTHONPATH'].split(':')
  if str(old_source) not in path_parts:
    raise ValueError('Expected the nested combined source in PYTHONPATH')
  path_parts = [str(source) if p == str(old_source) else p for p in path_parts]
  recipe['environment']['PYTHONPATH'] = ':'.join(path_parts)+':'+PYTHONPATH_SUFFIX
  save(destination/'pipeline-invocation.json', recipe)
  combined = copy.deepcopy(outer)
  combined['command'][1] = str(exp/'agent-speculation-run.py')
  option(combined['command'], '--invocation', destination/'pipeline-invocation.json')
  option(combined['command'], '--output', WORK/'agent-paced-dynamic-combined/report.json')
  if '--pad-prompt-to' in combined['command']:
    option(combined['command'], '--pad-prompt-to', 2048)
  else:
    combined['command'].extend(['--pad-prompt-to', '2048'])  # The original launcher otherwise defaults to 2112.
  combined['environment']['PYTHONPATH'] = recipe['environment']['PYTHONPATH']
  save(destination/'combined-invocation.json', combined)
  differences = []
  for path in source.rglob('*'):
    old = old_source/path.relative_to(source)
    if path.is_file() and old.is_file() and sha(path) != sha(old):
      differences.append(dict(path=str(path.relative_to(source)), old_sha256=sha(old), current_sha256=sha(path)))
  cache_inventory = WORK/'agent-paced-replay-cache-35-43/paced-cache.json'
  cache = json.loads(cache_inventory.read_text())
  if cache['frames'] != 450 or len(cache['files']) != 450:
    raise ValueError('Expected the complete existing 450-frame cache; do not decode again')
  value = dict(schema=1, policy=POLICY, source_root=str(source),
    parent_queue=dict(path='/tmp/opencode/queue-alpamayo-paced-dynamic.py', sha256=sha('/tmp/opencode/queue-alpamayo-paced-dynamic.py')),
    raw_cache=dict(inventory=str(cache_inventory), inventory_sha256=sha(cache_inventory), frames=450,
                   frame_metadata_sha256=cache['frame_metadata_sha256']),
    origin=dict(outer_path=str(outer_path), outer_sha256=sha(outer_path), nested_path=str(nested_path), nested_sha256=sha(nested_path),
                qualified_current_source=str(base), prior_parity_sha256=sha(WORK/'live-speculative-consumer/report.json')),
    intentional_changes=[dict(flag='--vision-calibration', before=old_vision, after=None),
                         dict(flag='--pad-prompt-to', before=int(old_pad), after=2048),
                         dict(paths='Relocate all model sources, reduced vocabulary, nested invocation, outputs, and PYTHONPATH'),
                         dict(pythonpath_additions=PYTHONPATH_SUFFIX.split(':'), reason='native route and vehicle-history helpers')],
    launcher_overrides=dict(mode='candidate', prefill_attention='shuffle-fma', pad_prompt_to=2048, warmup=2, runs=62),
    serial_normalization_disables=['--pipeline-inputs', '--freeze-warmup-gc'],
    replay=dict(requested_hz=10, camera_hz=20, source_start_seconds=35, source_seconds=8, hard_inference_budget_seconds=180),
    current_base_differs_from_older_combined_source=differences,
    timing='RECORDED REPLAY only; no native device latency or performance assertion',
    files={str(p.relative_to(destination)): sha(p) for p in sorted(destination.rglob('*')) if p.is_file()})
  save(destination/'manifest.json', value)
  for path in destination.rglob('*'):
    path.chmod(0o555 if path.is_dir() else 0o444)
  destination.chmod(0o555)
  print(json.dumps(dict(snapshot=str(destination), manifest_sha256=sha(destination/'manifest.json'), files=len(value['files'])), indent=2))
  return value


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--snapshot', type=Path, default=WORK/'agent-paced-dynamic-source')
  build(parser.parse_args().snapshot)
