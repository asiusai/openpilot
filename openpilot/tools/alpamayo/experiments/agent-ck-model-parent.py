"""Frozen bridge for one normal serialized 180-second parent prerequisite."""
import argparse
import hashlib
import json
from pathlib import Path
import runpy
import sys

RUNNER=Path('/tmp/opencode/openpilot-alpamayo/openpilot/tools/alpamayo/experiments/agent-ck-model-run.py')
ROOT=Path('/mnt/nvme-500/tmp/alpamayo-speed-20261116/agent-ck-model-work')
PINS={RUNNER:'2e725bb47be492f414a4761605b032d127bfae17e529a02bf767fa7a79fe1b05',
  ROOT/'prepared-final/protocol.json':'3f1ab36c32f0cca3ea9468e4745c5f2f258375061688e0df0023233f6e589f48'}

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def main():
  parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
  parser.add_argument('--verify-only',action='store_true');args=parser.parse_args()
  assert not sys.flags.optimize and all(sha(p)==s for p,s in PINS.items())
  original=ROOT/'parent-result.json'
  assert args.output.parent.is_dir() and not args.output.exists() and args.output.resolve()!=original.resolve()
  assert not original.exists() and not (ROOT/'parent-artifacts').exists()
  assert not (ROOT/'benchmark-unused.json').exists() and not (ROOT/'benchmark-unused.speculation.json').exists()
  command=[str(RUNNER)]+([] if args.verify_only else ['--parent-run'])
  saved=sys.argv
  try:
    sys.argv=command;runpy.run_path(str(RUNNER),run_name='__main__')
  finally:
    sys.argv=saved;assert all(sha(p)==s for p,s in PINS.items())
    if not args.verify_only and original.exists():
      report=json.loads(original.read_text())
      report['parent_bridge']={'original_report':str(original),'original_sha256':sha(original),'bridge_sha256':sha(__file__),
        'runner':str(RUNNER),'scope':'First-live-layer prerequisite only; no route integration or speed claim.'}
      with args.output.open('x') as stream: stream.write(json.dumps(report,indent=2,allow_nan=False)+'\n')
  if args.verify_only:
    assert not original.exists() and not args.output.exists()
    print(json.dumps({'bridge_cpu_preflight':'passed','gpu_executed':False,'runpy_target':str(RUNNER),
      'exclusive_original':str(original),'exclusive_wrapper_output':str(args.output)}))

if __name__=='__main__': main()
