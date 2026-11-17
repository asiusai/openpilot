"""CPU-only freeze of the dynamic-vision protocol, sources, evidence, and schedules."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import shutil

HERE = Path(__file__).resolve().parent
WORK = Path('/mnt/nvme-500/tmp/alpamayo-speed-20261116')


def load(name):
  spec = importlib.util.spec_from_file_location(name.replace('-', '_'), HERE/(name+'.py'))
  module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
  return module


def main():
  p = argparse.ArgumentParser(description=__doc__)
  p.add_argument('--directory', type=Path, required=True)
  p.add_argument('--invocation', type=Path, default=WORK/'dynamic-vision-full-route/invocation.json')
  p.add_argument('--evidence', type=Path, default=WORK/'agent-decode-gemm-rounding-retry/report.json')
  p.add_argument('--choices', default='{"16,2048,2048":"1x2"}')
  p.add_argument('--quality-samples', type=Path, help='JSON list; defaults to the recorded 64-sample quality schedule')
  p.add_argument('--turn-samples', type=Path, help='JSON list; defaults to 16 warmup and 24 measured right-turn samples at indices 430:470')
  args = p.parse_args()
  os.environ.update(DEV='CPU', ALLOW_DEVICE_USAGE='0', BEAM='0', JITBEAM='0', PARALLEL='0', PROFILE='0', DEBUG='0')
  route, hook = load('agent-decode-fast-route'), load('agent-decode-fast-hook')
  chain = route.inspect(args.invocation)
  choices = json.loads(args.choices)
  permitted, admitted = hook.evidence(args.evidence, choices, chain[-1]['environment'])
  check = hook.load('agent-decode-gemm')
  evidence = json.loads(args.evidence.read_text())
  terminal = chain[-1]['command']
  for row in evidence['cases']:
    if tuple(row['shape_mnk']) not in permitted: continue
    actual = check.model_fixture(row['case'], Path(chain[-1]['environment']['ALPAMAYO_FP8_CACHE']),
      Path(route.option(terminal, '--checkpoint')), Path(route.option(terminal, '--reduced-vocab')))['identity']
    route.require(actual == row['model_identity'], 'Current selected cache record differs from GPU evidence')
  quality_source = json.loads((WORK/'combined-source/full-fixed-vision-quality-invocation.json').read_text())['command']
  quality = json.loads(args.quality_samples.read_text()) if args.quality_samples else quality_source[
    quality_source.index('--samples')+1:quality_source.index('--initial-noise')]
  full = terminal[terminal.index('--samples')+1:terminal.index('--initial-noise')]
  turns = json.loads(args.turn_samples.read_text()) if args.turn_samples else full[430:470]
  route.require(len(quality) == 64 and len(turns) == 40, 'Require 64 quality samples and 16+24 turn samples')
  route.require(all(Path(sample).is_dir() for sample in quality+turns), 'Missing sample')
  directory = args.directory.resolve()
  route.require(not directory.exists(), 'Use a new frozen directory')
  directory.mkdir(parents=True)
  frozen = directory/'alpamayo'; exp = frozen/'experiments'; exp.mkdir(parents=True)
  roots = {Path(row['command'][0]).parent.parent for row in chain}
  route.require(len(roots) == 1, 'Mixed source trees need an explicit reviewed freeze')
  source = roots.pop()
  # Copy byte-identical source files, not generated artifacts or dependencies.
  for folder, target in ((source, frozen), (source/'experiments', exp)):
    for path in folder.iterdir():
      if path.is_file() and (path.suffix in ('.py', '.c', '.so') or path.name == 'coc_vocab.json'):
        shutil.copyfile(path, target/path.name)
  supplements = ['agent-decode-gemm.py', 'agent-decode-gemm-hook.py', 'agent-decode-gemm-rounding.py',
                 'agent-decode-fast-hook.py', 'agent-decode-fast-route.py', 'agent-decode-fast-prepare.py', 'agent-decode-fast-check.py']
  for name in supplements: shutil.copyfile(HERE/name, exp/name)
  shutil.copyfile(args.evidence, directory/'evidence.json')
  for relative, expected in evidence['provenance']['sources'].items():
    route.require(route.sha((frozen/relative).read_bytes()) == expected, f'Evidence source differs: {relative}')
  # Freeze small external numerical inputs. Camera frames and weights stay at
  # their recorded paths; their contents are not copied or claimed fully hashed.
  external = {}
  for flag in ('--prefill-calibration', '--initial-noise', '--native-manifest', '--native-calibrations'):
    if value := route.option(terminal, flag): external[value] = route.sha(Path(value).read_bytes())
  config = Path(route.option(terminal, '--checkpoint'))/'config.json'
  external[str(config)] = route.sha(config.read_bytes())
  index = Path(chain[-1]['environment']['ALPAMAYO_FP8_CACHE'])/'index.json'
  external[str(index)] = route.sha(index.read_bytes())
  family_files = {str(path): route.sha(path.read_bytes()) for path in frozen.rglob('*') if path.is_file()}
  family = route.sha(json.dumps(dict(files=family_files, evidence=admitted['sha256'], choices=choices), sort_keys=True).encode())
  for label, samples, warmup, audit in [('quality', quality, 2, True), ('turn', turns, 16, False)]:
    sample_path = directory/f'{label}-samples.json'
    sample_path.write_text(json.dumps(samples, indent=2)+'\n')
    target_paths = [directory/f'{label}-invocation-{i}.json' for i in range(len(chain))]
    for i, row in enumerate(chain):
      command = row['command'].copy()
      command[0] = str(exp/Path(command[0]).name)
      for j, value in enumerate(command):
        if value.startswith(str(source)+'/'): command[j] = str(frozen/Path(value).relative_to(source))
      route.replace(command, '--output', directory/f'{label}-unused-output.json')
      route.replace(command, '--warmup', warmup); route.replace(command, '--runs', len(samples)-warmup)
      if Path(command[0]).name == 'agent-speculation-run.py':
        route.replace(command, '--pad-prompt-to', 2048)
        route.replace(command, '--samples-json', sample_path)
      if i < len(chain)-1: route.replace(command, '--invocation', target_paths[i+1])
      else:
        command[command.index('--samples')+1:command.index('--initial-noise')] = samples
        route.replace(command, '--pad-prompt-to', 2048)
      environment = dict(row['environment'], PYTHONPATH=f'/tmp/opencode/alpamayo-asius-20261114/tinygrad:{frozen}',
        CACHEDB=str(directory/'cache.db'), LLVMOPT=None, BEAM='0', JITBEAM='0', PARALLEL='0', PROFILE='0', DEBUG='0',
        PYTHONDONTWRITEBYTECODE='1', ALLOW_DEVICE_USAGE=None)
      target_paths[i].write_text(json.dumps(dict(command=[row.get('python', '/tmp/opencode/alpamayo-asius-20261114/upstream/.venv/bin/python'),
        *command], environment=environment), indent=2)+'\n')
    frozen_chain = route.inspect(target_paths[0])
    files = {str(path): route.sha(path.read_bytes()) for path in frozen.rglob('*') if path.is_file()}
    files.update({str(path): route.sha(path.read_bytes()) for path in [directory/'evidence.json', sample_path, *target_paths]})
    files.update(external)
    protocol = dict(invocation=str(target_paths[0]), chain=frozen_chain, files=files,
      evidence=str(directory/'evidence.json'), choices=choices, samples=samples, warmup=warmup, runs=len(samples)-warmup,
      audit=audit, family_sha256=family, source_invocations=[dict(path=r['path'], sha256=r['sha256']) for r in chain],
      policy=dict(vision='dynamic', language='fixed', pad=2048, speculation='nominal', attention='shuffle-fma',
        full_images=True, full_history=True, steps=3, solver='euler'),
      limits=['Camera file contents and full weight cache are not rehashed',
              'Layer-0 fixture equality does not establish all-layer activation equality',
              'Nominal prefill acceptance bypasses this decoder hook'])
    (directory/f'{label}-protocol.json').write_text(json.dumps(protocol, indent=2)+'\n')
    route.verify_protocol(protocol)
  from tinygrad import Device
  route.require(not any('AMD' in value for value in Device._opened_devices), 'Unexpected GPU access')
  print(json.dumps(dict(gpu_executed=False, directory=str(directory), selected=choices, matrix_evidence=admitted)))


if __name__ == '__main__': main()
