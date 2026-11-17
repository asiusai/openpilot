"""Pinned entry for one normal serialized 180-second lifetime retry."""
import argparse
import hashlib
import json
from pathlib import Path
import runpy
import sys

EXP=Path('/tmp/opencode/openpilot-alpamayo/openpilot/tools/alpamayo/experiments')
ROOT=Path('/mnt/nvme-500/tmp/alpamayo-speed-20261116/agent-ck-lifetime-work')
RUNNER=EXP/'agent-ck-lifetime-run.py'
PINS={RUNNER:'f22fc16e7eff0a54a3a9e83e29bfa50293744d16c9134c83481b31bef0b8a1a4',
  EXP/'agent-ck-lifetime-overlay.py':'b5329d391317627d214c7c68fd43e5bfbb16c5b3a6bf7b77f8ff1e431a7f69a3',
  ROOT/'prepared/protocol.json':'6bed7f76cdcd5e610cef2f1e618b59e57e65826f7d5a2836d0a70d2fe69aa2ea'}

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def main():
  parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
  parser.add_argument('--verify-only',action='store_true');args=parser.parse_args()
  assert not sys.flags.optimize and all(sha(p)==s for p,s in PINS.items())
  original=ROOT/'parent-result.json'
  assert args.output.parent.is_dir() and not args.output.exists() and args.output.resolve()!=original.resolve()
  assert not original.exists() and not (ROOT/'parent-artifacts').exists()
  assert not (ROOT/'benchmark-unused.json').exists() and not (ROOT/'benchmark-unused.speculation.json').exists()
  saved=sys.argv
  try:
    sys.argv=[str(RUNNER)]+([] if args.verify_only else ['--parent-run'])
    runpy.run_path(str(RUNNER),run_name='__main__')
  finally:
    sys.argv=saved;assert all(sha(p)==s for p,s in PINS.items())
    if not args.verify_only and original.exists():
      report=json.loads(original.read_text())
      report['parent_bridge']={'original_report':str(original),'original_sha256':sha(original),
        'bridge_sha256':sha(__file__),'repair':'One output helper and adapter owner per frozen runner; ACTIVE and ABI scope unchanged.'}
      with args.output.open('x') as stream: stream.write(json.dumps(report,indent=2,allow_nan=False)+'\n')
  if args.verify_only:
    assert not original.exists() and not args.output.exists()
    print(json.dumps({'bridge_cpu_preflight':'passed','gpu_executed':False,'runpy_target':str(RUNNER)}))

if __name__=='__main__': main()
