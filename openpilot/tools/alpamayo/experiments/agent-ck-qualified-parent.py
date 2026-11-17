"""Small bridge for the parent's existing serial GPU wrapper. No locking or ACL bypass."""
import argparse
import hashlib
import json
from pathlib import Path
import runpy
import sys

RUNNER=Path('/tmp/opencode/openpilot-alpamayo/openpilot/tools/alpamayo/experiments/agent-ck-qualified-run.py')
ROOT=Path('/mnt/nvme-500/tmp/alpamayo-speed-20261116/agent-ck-qualified-work')
PINS={RUNNER:'52f55ea50702c01157a1e7051341c6122a53868a7921a76c27c7acc655d1c566',
      ROOT/'prepared-final/protocol.json':'d84e251bd733ea279953c63fedb0c25c1a7c335562af379473073914ee3f4b2e'}


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
  parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
  parser.add_argument('--verify-only',action='store_true');args=parser.parse_args()
  assert not sys.flags.optimize and all(sha(p)==s for p,s in PINS.items())
  original=ROOT/'parent-result.json'
  assert not args.output.exists() and args.output.resolve()!=original.resolve()
  assert args.output.parent.is_dir() and not original.exists() and not (ROOT/'parent-programs').exists()
  command=[str(RUNNER)] if args.verify_only else [str(RUNNER),'--parent-run','--output',str(original)]
  old_argv=sys.argv[:];sys.path.insert(0,str(RUNNER.parent))
  try:
    sys.argv=command;runpy.run_path(str(RUNNER),run_name='__main__')
  finally:
    sys.argv=old_argv
    assert all(sha(p)==s for p,s in PINS.items())
    if not args.verify_only and original.exists():
      result=json.loads(original.read_text())
      result['parent_bridge']={'original_report':str(original),'original_sha256':sha(original),'adapter_sha256':sha(__file__),
        'command':command,'original_gates_unchanged':True}
      with args.output.open('x') as stream: stream.write(json.dumps(result,indent=2,allow_nan=False)+'\n')
  if args.verify_only:
    assert not original.exists() and not args.output.exists()
    print(json.dumps({'bridge_cpu_preflight':'passed','gpu_executed':False,'runpy_target':str(RUNNER),
      'helper_directory':str(RUNNER.parent),'fixed_original_output':str(original),'exclusive_bridge_output':str(args.output)}))


if __name__=='__main__': main()
