"""Frozen bridge for one serialized 180-second storage-contract retry."""
import argparse
import hashlib
import json
from pathlib import Path
import runpy
import sys

EXP=Path('/tmp/opencode/openpilot-alpamayo/openpilot/tools/alpamayo/experiments')
RUNNER=EXP/'agent-ck-storage-run.py'
ROOT=Path('/mnt/nvme-500/tmp/alpamayo-speed-20261116/agent-ck-storage-work')
PINS={RUNNER:'2ffe7dce2e3b42777290f5f16c758846197d44c091fa2ad2f07efa906ac371e3',
  EXP/'agent-ck-storage-overlay.py':'09046c4ecb79b67a4b6b46ecbbfe90ba7535a4439d302d234ce9225b7777a79c',
  ROOT/'prepared/protocol.json':'3a4a2c80f94c10720d646fd7785a41ae6c4d5e02e8061d43e08b276169385df9'}

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
      report['parent_bridge']={'original_report':str(original),'original_sha256':sha(original),
        'bridge_sha256':sha(__file__),'runner':str(RUNNER),'storage_field':'offset_bytes',
        'scope':'Unchanged first-live-layer prerequisite; metadata contract correction only.'}
      with args.output.open('x') as stream: stream.write(json.dumps(report,indent=2,allow_nan=False)+'\n')
  if args.verify_only:
    assert not original.exists() and not args.output.exists()
    print(json.dumps({'bridge_cpu_preflight':'passed','gpu_executed':False,'runpy_target':str(RUNNER),
      'frozen_runner':str(EXP/'agent-ck-model-run.py'),'exclusive_original':str(original),'exclusive_wrapper_output':str(args.output)}))

if __name__=='__main__': main()
