"""One fresh chronological quality arm; parent owns the 600-second serialized launch."""
import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path

s=importlib.util.spec_from_file_location('ck_quality64_common',Path(__file__).with_name('agent-ck-quality64-common.py'))
C=importlib.util.module_from_spec(s);s.loader.exec_module(C)
R=C.runner(C)

def main():
  assert not sys.flags.optimize
  p=argparse.ArgumentParser();p.add_argument('--arm',choices=('baseline','candidate'),required=True)
  p.add_argument('--output',type=Path,required=True);p.add_argument('--parent-run',action='store_true');a=p.parse_args()
  plan=C.verify();folder=C.ROOT/a.arm;original=folder/'report.json'
  assert not folder.exists() and a.output.parent.is_dir() and not a.output.exists() and a.output.resolve()!=original.resolve()
  if not a.parent_run:
    print(json.dumps({'verified':True,'gpu_executed':False,'arm':a.arm,'protocol_sha256':C.sha(C.PREPARED/'protocol.json')}));return
  folder.mkdir();R.configure(a.arm);os.environ['ALLOW_DEVICE_USAGE']='1'
  try:
    report=R.run(plan,a.arm,folder);C.validate_arm(report)
  finally:
    if original.exists():
      r=json.loads(original.read_text());r['parent_bridge']={'original_report':str(original),'original_sha256':C.sha(original),
        'entry_sha256':C.sha(__file__)};C.write(a.output,r)

if __name__=='__main__': main()
