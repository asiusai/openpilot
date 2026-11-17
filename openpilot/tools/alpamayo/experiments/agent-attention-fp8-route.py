"""Parent-only baseline calibration and numerical FP8-prefill route replay.

Reuse the complete saved invocation. Calibration timing includes observer work.
Candidate timing includes quantization but excludes the calibration observer.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace


def load(name):
  spec = importlib.util.spec_from_file_location(name.replace('-', '_'), Path(__file__).with_name(name+'.py'))
  result = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(result)
  return result


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--output', type=Path, required=True)
  parser.add_argument('--invocation', type=Path, required=True)
  parser.add_argument('--mode', choices=('baseline', 'calibrate', 'candidate'), required=True)
  parser.add_argument('--scales', type=Path)
  parser.add_argument('--reference', type=Path)
  parser.add_argument('--manifest', type=Path, help='Recorded-motion manifest for invocations without --native-manifest')
  args = parser.parse_args()
  assert args.mode != 'candidate' or (args.scales and args.reference)
  invocation = json.loads(args.invocation.read_text())
  command = invocation['command'][1:].copy()
  assert Path(command[0]).name == 'benchmark-pipeline.py'
  assert '--online-attention' in command and '--online-expert' not in command
  manifest = args.manifest or (Path(command[command.index('--native-manifest')+1]) if '--native-manifest' in command else None)
  assert args.mode != 'candidate' or manifest, 'Recorded-motion comparison requires --manifest or a native invocation'
  command[command.index('--output')+1] = str(args.output)
  for name, value in invocation['environment'].items():
    if name.startswith('ALPAMAYO_'):
      if value is None:
        os.environ.pop(name, None)
      else:
        os.environ[name] = value
  candidate, benchmark = load('agent-attention-fp8'), load('benchmark-pipeline')
  calibration = json.loads(args.scales.read_text()) if args.scales else None
  loader = benchmark.local_module
  hooked, observers = [], []
  def local_module(name):
    module = loader(name)
    if name == 'online-attention':
      original_install = module.install
      def install(model, *a, **kw):
        original_install(model, *a, **kw)
        if args.mode == 'calibrate':
          observers.append(candidate.install_capture(model))
        elif args.mode == 'candidate':
          candidate.install_prefill(model, calibration['scales'])
        hooked.append(args.mode)
      module.install = install
    return module
  benchmark.local_module = local_module
  provenance = dict(mode=args.mode, numerical_candidate=args.mode == 'candidate', quality_validated=False,
    source_invocation=str(args.invocation), source_invocation_sha256=hashlib.sha256(args.invocation.read_bytes()).hexdigest(),
    effective_command=command, timing_has_calibration_observer=args.mode == 'calibrate')
  sys.argv = command
  benchmark.main()
  assert hooked == [args.mode]
  report = json.loads(args.output.read_text())
  samples = [row['sample'] for row in report['warmup']+report['runs']]
  if args.mode == 'calibrate':
    assert len(observers) == 1
    result = candidate.captured_scales(observers[0])
    result.update(samples=samples, invocation=str(args.invocation),
      invocation_sha256=provenance['source_invocation_sha256'])
    destination = args.output.with_suffix('.agent-attention-fp8-scales.json')
    destination.write_text(json.dumps(result, indent=2)+'\n')
    provenance['scales'] = str(destination)
  if args.mode == 'candidate':
    provenance['scales'] = str(args.scales)
    provenance['scales_sha256'] = hashlib.sha256(args.scales.read_bytes()).hexdigest()
    provenance['samples_also_used_for_calibration'] = sorted(set(samples) & set(calibration.get('samples', [])))
    reference = json.loads(args.reference.read_text())
    reference_samples = {Path(row['sample']).name for row in reference['warmup']+reference['runs']}
    assert all(Path(sample).name in reference_samples for sample in samples), 'Missing paired reference samples'
    comparison = args.output.with_suffix('.agent-attention-fp8-quality.json')
    load('compare-speed').evaluate(SimpleNamespace(reference=args.reference, report=args.output,
      manifest=manifest, output=comparison, window=[]))
    measured = json.loads(comparison.read_text())
    assert measured['matched']['samples'] == len(set(samples)), 'Incomplete recorded-motion coverage'
    provenance['quality_comparison'] = str(comparison)
    provenance['quality_summary'] = measured['matched']
  report['agent_attention_fp8'] = provenance
  args.output.write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
  main()
