#!/usr/bin/env python3
"""CPU-only bundle export, verification, and collection-provenance preparation."""
import argparse
import importlib.util
import json
from pathlib import Path

spec = importlib.util.spec_from_file_location('agent_int4model', Path(__file__).with_name('agent-int4model.py'))
ptq = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ptq)


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  commands = parser.add_subparsers(dest='command', required=True)
  export = commands.add_parser('export')
  export.add_argument('--checkpoint', required=True, type=Path)
  export.add_argument('--output', required=True, type=Path)
  export.add_argument('--mode', choices=('scalar', 'peroutput'), default='peroutput')
  export.add_argument('--layers', type=int, nargs='+')
  verify = commands.add_parser('verify')
  verify.add_argument('bundle', type=Path)
  reference = commands.add_parser('reference')
  reference.add_argument('--checkpoint', required=True, type=Path)
  reference.add_argument('--fp8-cache', required=True, type=Path)
  reference.add_argument('--settings', required=True, type=Path)
  reference.add_argument('--sources', required=True, type=Path, nargs='+')
  reference.add_argument('--calibration-files', type=Path, nargs='*', default=[])
  reference.add_argument('--calibration-samples', required=True, type=Path)
  reference.add_argument('--reserved-samples', required=True, type=Path)
  reference.add_argument('--output', required=True, type=Path)
  args = parser.parse_args()
  if args.command == 'export':
    manifest = ptq.export_bundle(args.checkpoint, args.output, args.mode, args.layers)
    print(json.dumps(dict(complete=manifest['complete'], layers=len(manifest['records']),
                         identity_sha256=manifest['identity_sha256'], output=str(args.output)), indent=2))
  elif args.command == 'verify':
    manifest, weights = ptq.load_bundle(args.bundle)
    print(json.dumps(dict(complete=True, layers=len(weights), identity_sha256=manifest['identity_sha256']), indent=2))
  else:
    if args.output.exists():
      raise ValueError('Reference output already exists')
    settings = json.loads(args.settings.read_text())
    selection = ptq.CalibrationSelection(args.calibration_samples, args.reserved_samples)
    reference = ptq.reference_identity(args.checkpoint, args.fp8_cache, args.sources, settings, args.calibration_files)
    result = dict(reference=reference, selection=selection.provenance(),
                  calibration_inputs=[selection.identity(sample) for sample in selection.allowed])
    ptq.json_write(args.output, result)
    print(json.dumps(dict(output=str(args.output), reference_sha256=reference['identity_sha256'],
                         calibration_samples=len(selection.allowed), reserved_inputs_read=False), indent=2))


if __name__ == '__main__':
  main()
