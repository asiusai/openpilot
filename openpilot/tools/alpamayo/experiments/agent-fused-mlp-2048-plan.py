"""CPU-only preparation of frozen dynamic-vision 2048 qualification commands."""
import argparse
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import shlex

HERE = Path(__file__).resolve().parent
WORK = Path('/mnt/nvme-500/tmp/alpamayo-speed-20261116')
TG = Path('/tmp/opencode/alpamayo-asius-20261114/tinygrad')
WRAPPER = '/tmp/opencode/run-alpamayo-speed-trial.py'


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path, obj):
  assert not path.exists(), f'Preserve existing artifact: {path}'
  path.write_text(json.dumps(obj, indent=2)+'\n')


def runner(path):
  spec = importlib.util.spec_from_file_location('agent_fused_2048_plan_runner',path)
  module = importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def set_option(command, name, value):
  flag = '--'+name
  assert command.count(flag) <= 1
  if flag in command: command[command.index(flag)+1] = str(value)
  else: command.extend([flag,str(value)])


def prepare(directory, source):
  assert not directory.exists();directory.mkdir(parents=True)
  source = source.resolve();snapshot = directory/'alpamayo';snapshot.mkdir();(snapshot/'experiments').mkdir()
  pinned = {};original = {}
  for base,dest in ((source,snapshot),(source/'experiments',snapshot/'experiments')):
    files = sorted({*base.glob('*.py'),*base.glob('*.c')})
    if base.name == 'experiments': files += [base/'coc_vocab.json']
    for path in files:
      data = path.read_bytes();copy = dest/path.name;copy.write_bytes(data)
      pinned[str(copy)] = hashlib.sha256(data).hexdigest();original[str(path)] = pinned[str(copy)]
  assert all(sha(path) == digest for path,digest in original.items()), 'Source changed during snapshot'
  factory = snapshot/'experiments/benchmark-pipeline.py';tree = ast.parse(factory.read_text())
  functions = {node.name:node for node in tree.body if isinstance(node,ast.FunctionDef)}
  assert {'initialize','main'} <= functions.keys()
  assert any(isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id == 'initialize' for n in ast.walk(functions['main']))
  pairs = {}
  for label,template,warmup,runs in (
    ('quality',WORK/'agent-validation-source/full-padded-quality-invocation.json',2,62),
    ('native',WORK/'full-native-5hz-fixed-vision/invocation.json',12,24)):
    saved = json.loads(template.read_text());command = list(saved['command']);python = command[0]
    assert Path(command[1]).name == 'benchmark-pipeline.py'
    samples = command[command.index('--samples')+1:command.index('--initial-noise')]
    assert len(samples) >= warmup+runs
    command[1] = str(factory)
    for name,value in [('output',directory/(label+'-placeholder.json')),('warmup',warmup),('runs',runs),
                       ('pad-prompt-to',2048),('reduced-vocab',snapshot/'experiments/coc_vocab.json')]: set_option(command,name,value)
    if '--vision-calibration' in command:
      index = command.index('--vision-calibration');del command[index:index+2]
    assert '--capture-vision-scales' not in command
    env = dict(saved['environment']);env.update(PYTHONPATH=f'{TG}:{snapshot}',PROFILE='0',PYTHONDONTWRITEBYTECODE='1')
    leaf = directory/(label+'-pipeline-invocation.json')
    write(leaf,{'command':command,'environment':env,'derived_from':str(template),'derived_from_sha256':sha(template),
                'selected_recipe':{'vision':'dynamic','pad_prompt_to':2048,'original_sample_order_preserved':True}})
    outer = directory/(label+'-invocation.json')
    outer_command = [python,str(snapshot/'experiments/agent-speculation-run.py'),'--output',str(directory/(label+'-placeholder.json')),
      '--invocation',str(leaf),'--mode','candidate','--prefill-attention','shuffle-fma','--pad-prompt-to','2048',
      '--warmup',str(warmup),'--runs',str(runs)]
    write(outer,{'command':outer_command,'environment':env})
    pairs[label] = {'invocation':str(outer),'leaf_invocation':str(leaf),'warmup':warmup,'runs':runs,
      'predictions':warmup+runs,'selected_samples':samples[:warmup+runs],
      'manifest':str(directory/(label+'-manifest.json'))}
  micro = WORK/'agent-fused-mlp-2048-micro/report.json'
  frozen_exp = snapshot/'experiments'
  commands = [['python3',WRAPPER,'agent-fused-mlp-2048-micro','--program',str(frozen_exp/'agent-packed-epilogue-check.py'),
    '--timeout-seconds','300','--set','mode','gpu','--set','rows','2048','--set','variants','transpose4x4',
    '--set','rounds','3','--flag','skip-old-fused','--env','PYTHONDONTWRITEBYTECODE','1',
    '--env','PYTHONPATH',f'{TG}:{snapshot}']]
  commands.append([python,str(frozen_exp/'agent-fused-mlp-2048-plan.py'),'--mode','qualify','--protocol',str(directory/'protocol.json')])
  for label,pair in pairs.items():
    for mode in ('reference','candidate'):
      command = ['python3',WRAPPER,f'agent-fused-mlp-2048-{label}-{mode}','--program',str(frozen_exp/'agent-fused-mlp-run.py'),
        '--timeout-seconds','900','--set','mode',mode,'--set','manifest',pair['manifest'],'--env','PYTHONDONTWRITEBYTECODE','1']
      if mode == 'candidate': command.extend(['--set','reference',str(WORK/f'agent-fused-mlp-2048-{label}-reference/report.json')])
      commands.append(command)
  for pair in pairs.values():
    for field in ('invocation','leaf_invocation'): pinned[pair[field]] = sha(pair[field])
  protocol = {'schema':1,'gpu_qualified':False,'gpu_executed_by_prepare':False,'variant':'transpose4x4','rows':2048,
    'vision_policy':'dynamic','prefill_attention':'shuffle-fma','nominal_prefill_speculation':True,
    'source_root':str(snapshot),'source_files':pinned,'original_source_hashes':original,
    'factory_initialize_sha256':sha(factory),'pairs':pairs,'micro_report':str(micro),'commands':commands,
    'qualification_policy':'Do not create runnable pair manifests until the bounded 2048 GPU report passes source-bound gates.'}
  write(directory/'protocol.json',protocol)
  script = directory/'parent-commands.sh'
  script.write_text('#!/usr/bin/env bash\nset -euo pipefail\n\n'+'\n\n'.join(shlex.join(command) for command in commands)+'\n')
  print(json.dumps({'protocol':str(directory/'protocol.json'),'parent_commands':str(script),'gpu_executed':False,'pairs':pairs},indent=2))


def qualify(path):
  protocol = json.loads(path.read_text());assert protocol['schema'] == 1 and protocol['rows'] == 2048 and protocol['vision_policy'] == 'dynamic'
  for source,digest in protocol['source_files'].items(): assert sha(source) == digest, f'Frozen source/input changed: {source}'
  exp = Path(protocol['source_root'])/'experiments';module = runner(exp/'agent-fused-mlp-run.py')
  evidence = module.micro_gate(protocol['micro_report'],(2048,))
  manifests = {label:module.make_manifest(pair['invocation'],protocol['micro_report']) for label,pair in protocol['pairs'].items()}
  for label,manifest in manifests.items():
    module.verify_manifest(manifest)
    assert manifest['micro']['qualified_rows'] == [2048]
    assert all('--vision-calibration' not in row['recorded']['command'] for row in manifest['chain'])
    assert not Path(protocol['pairs'][label]['manifest']).exists()
  for label,manifest in manifests.items(): write(Path(protocol['pairs'][label]['manifest']),manifest)
  write(path.with_name('qualification.json'),{'protocol_sha256':sha(path),'gpu_micro_gate':evidence,
    'manifest_sha256':{label:sha(pair['manifest']) for label,pair in protocol['pairs'].items()},'pipeline_quality_or_speed_claim':False})
  print(json.dumps({'micro_passed':True,'manifests':{label:pair['manifest'] for label,pair in protocol['pairs'].items()}}))


def main():
  p = argparse.ArgumentParser(description=__doc__);p.add_argument('--mode',choices=('prepare','qualify'),required=True)
  p.add_argument('--output',type=Path);p.add_argument('--protocol',type=Path)
  p.add_argument('--source-root',type=Path,default=HERE.parent);args = p.parse_args()
  if args.mode == 'prepare': assert args.output is not None and args.protocol is None;prepare(args.output.resolve(),args.source_root)
  else: assert args.protocol is not None and args.output is None;qualify(args.protocol)


if __name__ == '__main__': main()
