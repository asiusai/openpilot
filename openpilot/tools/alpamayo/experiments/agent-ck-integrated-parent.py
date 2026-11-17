"""Pinned bridge for one arm under the parent's serialized 600-second wrapper."""
import argparse
import hashlib
import json
from pathlib import Path
import runpy
import sys

EXP=Path('/tmp/opencode/openpilot-alpamayo/openpilot/tools/alpamayo/experiments')
ROOT=Path('/mnt/nvme-500/tmp/alpamayo-speed-20261116/agent-ck-integrated-work')
RUNNER=EXP/'agent-ck-integrated-run.py'
PINS={RUNNER:'328bfa9ec5ac2f181e7e89d384e7913a89f3641573aab465075c340a79f6870a',
  ROOT/'prepared/protocol.json':'1b8797fc5596a6eeff39c17e028fde513a31d211678220ea171999fdc5c2da8e'}

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def main():
  p=argparse.ArgumentParser();p.add_argument('--arm',choices=('baseline','candidate'),required=True)
  p.add_argument('--output',type=Path,required=True);p.add_argument('--verify-only',action='store_true');args=p.parse_args()
  assert not sys.flags.optimize and all(sha(path)==digest for path,digest in PINS.items())
  original=ROOT/args.arm/'report.json'
  assert args.output.parent.is_dir() and not args.output.exists() and args.output.resolve()!=original.resolve()
  assert not (ROOT/args.arm).exists(),'The selected arm has already been attempted'
  argv=sys.argv
  try:
    sys.argv=[str(RUNNER),'--arm',args.arm]+([] if args.verify_only else ['--parent-run'])
    runpy.run_path(str(RUNNER),run_name='__main__')
  finally:
    sys.argv=argv;assert all(sha(path)==digest for path,digest in PINS.items())
    if not args.verify_only and original.exists():
      report=json.loads(original.read_text())
      report['parent_bridge']={'original_report':str(original),'original_sha256':sha(original),'bridge_sha256':sha(__file__)}
      with args.output.open('x') as stream: stream.write(json.dumps(report,indent=2,allow_nan=False)+'\n')
  if args.verify_only:
    assert not original.exists() and not args.output.exists()
    print(json.dumps({'runpy_bridge_verified':True,'gpu_executed':False,'arm':args.arm}))

if __name__=='__main__': main()
