#!/usr/bin/env python3
"""CPU-only preparation of a frozen, matched INT4 numerical route experiment.

This script does not import torch/tinygrad, launch a model, fit scales, or edit the
reviewed bundle. GPU commands are written for the parent's serial lock runner.
"""
import argparse
import ast
import copy
import importlib.util
import json
import os
from pathlib import Path
import shlex
import sys

EXP = Path(__file__).resolve().parent
WORK = Path('/mnt/nvme-500/tmp/alpamayo-speed-20261116')
PY = Path('/tmp/opencode/alpamayo-asius-20261114/upstream/.venv/bin/python')
LOCK_RUNNER = Path('/tmp/opencode/run-alpamayo-speed-trial.py')
GATE = WORK / 'agent-int4-scaled-actual-serial-fixed/report.json'
FAIR = WORK / 'agent-int4-fair-2112-buffer/report.json'
BUNDLE = EXP / 'agent-int4model-artifacts/peroutput-bundle'
MOTION = Path('/mnt/nvme-256/footage/f9ad9a96/exports/0000008e--321b2ddc03/alpamayo-4hz-calibrated-two-six/manifest.json')
INDICES = [0, 4, 8, 12, 16, 20, 24, 28]


def load(path):
  spec = importlib.util.spec_from_file_location(path.stem.replace('-', '_'), path)
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module


def read(path):
  return json.loads(Path(path).read_text())


def snapshot(destination, ptq):
  # Active original pipeline sources, runner, and optional native timing inputs.
  # Recursive local imports add dependencies; unrelated experiments/artifacts stay out.
  names = '''benchmark-pipeline benchmark-fast fast-inputs restore-precision prefill-kernels
    bf16-linear fast-quantize quantize-reduction reduced-vocab expert-kernels online-attention
    fused-norm quantize-packed padded-prefill compact-inputs projector-kernels batched-vision
    vision-cache draft-decode lookup-decode block-decode usb-readback wgp_compiler image-gemm
    stream-inputs native_warp calibration_trials compare-speed agent-int4model agent-int4model-run'''.split()
  pending = [EXP / (name + '.py') for name in names]
  gate = read(GATE)
  tested = {Path(name).name: (Path(name), digest) for name, digest in gate['source_hashes'].items()}
  pending += [EXP / name for name in tested if (EXP / name).is_file()]
  records = {}
  while pending:
    original = pending.pop().resolve()
    relative = original.relative_to(EXP.parent)
    target = destination / relative
    if str(target) in records:
      continue
    source = tested.get(original.name, (original, None))[0]
    data = source.read_bytes()
    digest = ptq.sha_file(source)
    if original.name in tested and digest != tested[original.name][1]:
      raise ValueError(f'Tested kernel source changed: {source}')
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open('xb') as stream:
      stream.write(data)
    if ptq.sha_file(target) != digest:
      raise ValueError(f'Snapshot copy mismatch: {target}')
    records[str(target)] = dict(source=str(source), sha256=digest, bytes=len(data))
    if original.suffix != '.py':
      continue
    for node in ast.walk(ast.parse(data)):
      candidates = []
      if isinstance(node, ast.Import):
        candidates = [alias.name.split('.')[0] + '.py' for alias in node.names]
      elif isinstance(node, ast.ImportFrom) and node.module:
        candidates = [node.module.split('.')[0] + '.py']
      elif original.name not in ('benchmark-pipeline.py', 'agent-int4model-run.py', 'agent-int4-scaled-check.py'):
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and len(node.value) < 100:
          if '\n' not in node.value and '/' not in node.value and node.value.endswith(('.py', '.c')):
            candidates = [node.value]
      for name in candidates:
        for root in (original.parent, EXP.parent, EXP):
          dependency = root / name
          if dependency.is_file():
            pending.append(dependency)
            break
  return records


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--output-dir', type=Path, default=WORK / 'agent-int4-route-prepared')
  args = parser.parse_args()
  root = args.output_dir.resolve()
  if not root.name.startswith('agent-int4-route-'):
    raise ValueError('Output must use the owned agent-int4-route- prefix')
  root.mkdir(parents=True, exist_ok=False)
  ptq = load(EXP / 'agent-int4model.py')
  protected = [EXP / 'agent-int4model.py', BUNDLE / 'manifest.json',
               EXP / 'agent-int4model-artifacts/reference-provenance.json',
               EXP / 'agent-int4model-reference-settings.json']
  before = {str(p): ptq.sha_file(p) for p in protected}
  source_root = root / 'source/alpamayo'
  copied = snapshot(source_root, ptq)
  frozen = source_root / 'experiments'
  runner = load(frozen / 'agent-int4model-run.py')
  original_invocation = WORK / 'full-padded-quality/invocation.json'
  invocation = read(original_invocation)
  blocks, _ = runner.option_blocks(invocation['command'])
  if '--vision-calibration' in blocks or '--capture-vision-scales' in blocks:
    raise ValueError('Dynamic vision is mandatory')
  if runner.value(blocks, 'pad-prompt-to') != '2112' or runner.value(blocks, 'steps') != '3':
    raise ValueError('Require the original 2112-row, three-Euler invocation')
  allowed, reserved = (read(WORK / name) for name in ('calibration-samples.json', 'holdout-samples.json'))
  if len(allowed) != 32 or len(reserved) != 32 or set(allowed) & set(reserved):
    raise ValueError('Unexpected calibration/reserved split')
  if blocks['--samples'] != allowed + reserved:
    raise ValueError('Expected four-window schedule with calibration first and reserved second')
  ptq.json_write(root / 'screen8-samples.json', [allowed[i] for i in INDICES])
  ptq.json_write(root / 'quality64-samples.json', blocks['--samples'])
  settings = read(EXP / 'agent-int4model-reference-settings.json')
  settings.update(purpose='Matched initial numerical route experiment; no activation fitting or collection',
                  reference_recipe=str(original_invocation), vision_fp8_activation_scales=None,
                  language_fp8_activation_scales=runner.value(blocks, 'prefill-calibration'),
                  timing_is_invalid=True, activation_study_pending=True)
  ptq.json_write(root / 'reference-settings.json', settings)

  # Preserve every recorded numerical flag. Only source location/PYTHONPATH change.
  def freeze_invocation(path, name):
    saved = read(path)
    saved['command'][1] = str(frozen / 'benchmark-pipeline.py')
    env = saved['environment']
    env['PYTHONPATH'] = os.pathsep.join(str(source_root) if Path(p).resolve() == EXP.parent else p
                                      for p in env['PYTHONPATH'].split(os.pathsep))
    ptq.json_write(root / name, saved)
    return saved

  freeze_invocation(original_invocation, 'quality-invocation.json')
  timing_source = WORK / 'full-native-5hz-padded/invocation.json'
  timing = freeze_invocation(timing_source, 'timing-invocation.json')
  timing_blocks, _ = runner.option_blocks(timing['command'])
  if '--vision-calibration' in timing_blocks:
    raise ValueError('Separate timing must also use dynamic vision')
  selection = ptq.CalibrationSelection(WORK / 'calibration-samples.json', WORK / 'holdout-samples.json')
  reference = ptq.reference_identity(Path(runner.value(blocks, 'checkpoint')),
    Path(invocation['environment']['ALPAMAYO_FP8_CACHE']), [Path(p) for p in sorted(copied)],
    settings, [Path(runner.value(blocks, 'prefill-calibration'))])
  provenance = root / 'reference-provenance.json'
  ptq.json_write(provenance, dict(reference=reference, selection=selection.provenance(),
    calibration_inputs=[selection.identity(sample) for sample in selection.allowed],
    preparation=dict(original_invocation=str(original_invocation), original_sha256=ptq.sha_file(original_invocation),
      previous_reference_unchanged=before, gpu_started=False, activation_collection=False)))
  fair = read(FAIR)
  if fair.get('cpu_only') is not False or not fair.get('complete') or not fair.get('all_pass'):
    raise ValueError('Expected completed parent fair timing report')
  # The strict runner gate separately checks actual-shape correctness and tested sources.
  common = ['--reference-provenance', str(provenance), '--bundle', str(BUNDLE)]
  stages = {
    'screen8': ['--invocation', str(root / 'quality-invocation.json'), '--samples-json', str(root / 'screen8-samples.json'), '--warmup', '0', '--runs', '8'],
    'quality64': ['--invocation', str(root / 'quality-invocation.json'), '--samples-json', str(root / 'quality64-samples.json'), '--warmup', '0', '--runs', '64'],
    'timing': ['--invocation', str(root / 'timing-invocation.json')],
  }
  commands, plans = {}, {}
  for stage, scheduling in stages.items():
    for mode in ('reference', 'candidate'):
      name = f'agent-int4-route-{stage}-{mode}'
      cli = ['--mode', mode, *common, *scheduling]
      if mode == 'candidate':
        cli += ['--kernel-report', str(GATE), '--kernel-mode', 'cu']
      # Reuse no saved controls during CPU preflight. Candidate GPU execution must pair.
      parsed = runner.parser().parse_args([*cli, '--output', str(root / f'{stage}-{mode}-preflight.json'), '--preflight'])
      verified = runner.preflight(parsed)
      ptq.json_write(parsed.output, verified.plan)
      plans[stage + '-' + mode] = verified.plan
      del verified
      if mode == 'candidate':
        cli += ['--compare-reference', str(WORK / f'agent-int4-route-{stage}-reference/report.json')]
      command = ['python3', str(LOCK_RUNNER), name, '--timeout-seconds', '1800', '--program', str(frozen / 'agent-int4model-run.py')]
      for i in range(0, len(cli), 2):
        command += ['--set', cli[i].removeprefix('--'), cli[i + 1]]
      commands[stage + '-' + mode] = command
    if plans[stage + '-reference']['protocol'] != plans[stage + '-candidate']['protocol']:
      raise ValueError(f'Unmatched reference/candidate protocol for {stage}')
    if plans[stage + '-reference']['schedule'] != plans[stage + '-candidate']['schedule']:
      raise ValueError(f'Unmatched reference/candidate schedule for {stage}')
  if any(key == 'torch' or key == 'tinygrad' or key.startswith(('torch.', 'tinygrad.')) for key in sys.modules):
    raise RuntimeError('CPU preparation imported a GPU model dependency')
  if {str(p): ptq.sha_file(p) for p in protected} != before:
    raise ValueError('Protected exporter, bundle manifest, or previous provenance changed')
  for stage in ('screen8', 'quality64'):
    commands[stage + '-compare'] = [str(PY), str(frozen / 'compare-speed.py'),
      '--reference', str(WORK / f'agent-int4-route-{stage}-reference/report.json'),
      '--report', str(WORK / f'agent-int4-route-{stage}-candidate/report.json'),
      '--manifest', str(MOTION), '--output', str(WORK / f'agent-int4-route-{stage}-comparison.json')]
    for a, b in ((35, 43), (112, 120), (164, 172), (228, 236)):
      commands[stage + '-compare'] += ['--window', str(a), str(b)]
  ptq.json_write(root / 'commands.json', commands)
  for stage in stages:
    text = '#!/bin/sh\nset -eu\n# Parent only: serial lock, device identity, timeout, and ACL restoration.\n'
    if stage != 'screen8':
      text += '# Run only after the preceding numerical screen and reasoning/turn review pass.\n'
    for key in (stage + '-reference', stage + '-candidate', stage + '-compare'):
      if key in commands:
        text += shlex.join(commands[key]) + '\n'
    (root / f'{stage}-parent.sh').write_text(text)
  manifest = dict(cpu_only=True, complete=True, all_pass=True, gpu_started=False,
    no_torch_or_tinygrad_import=True, source_files=copied, bundle_path=str(BUNDLE),
    bundle_identity=plans['screen8-candidate']['bundle'], reference_identity=reference['identity_sha256'],
    kernel_gate=plans['screen8-candidate']['kernel_gate'],
    fair_timing=dict(path=str(FAIR), sha256=ptq.sha_file(FAIR), all_pass=fair['all_pass'],
      limitation='Synthetic BF16 ReLU2 activations and one actual layer; not route accuracy or whole-pipeline timing'),
    calibration_indices=INDICES, screen8_samples=[allowed[i] for i in INDICES],
    quality64=dict(calibration=32, reserved=32, windows=[[35,43],[112,120],[164,172],[228,236]]),
    preflights={key:dict(queries=len(plan['schedule']), warmup=plan['warmup'], runs=plan['runs'],
      collection_profiling=plan['collection_profiling'], profiling=plan['profiling']) for key,plan in plans.items()},
    protected_unchanged=before, prepare_source_sha256=ptq.sha_file(Path(__file__)),
    parent_lock_runner=dict(path=str(LOCK_RUNNER), sha256=ptq.sha_file(LOCK_RUNNER)),
    actual_captured_activation_study='pending', accuracy='unverified', full_pipeline_sub_100ms='unverified',
    note='No fitting; no reserved sample contents read by this preparation. Initial eight queries are numerical screening only.')
  ptq.json_write(root / 'cpu-preflight.json', manifest)
  print(json.dumps({k:v for k,v in manifest.items() if k in ('cpu_only','complete','all_pass','preflights','reference_identity','accuracy')}, indent=2))


if __name__ == '__main__':
  main()
