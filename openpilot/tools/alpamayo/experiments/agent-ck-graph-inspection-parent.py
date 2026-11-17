"""Fresh 600-second arm entry; execution requires the parent's serialized wrapper."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import runpy
import sys

EXP=Path(__file__).parent
spec=importlib.util.spec_from_file_location('ck_graph_overlay',EXP/'agent-ck-graph-inspection-overlay.py')
O=importlib.util.module_from_spec(spec);spec.loader.exec_module(O)

def main():
  p=argparse.ArgumentParser();p.add_argument('--arm',choices=('baseline','candidate'),required=True)
  p.add_argument('--output',type=Path,required=True);p.add_argument('--verify-only',action='store_true');a=p.parse_args()
  assert not sys.flags.optimize
  plan=O.C.verify();directory=O.ROOT/a.arm;original=directory/'report.json'
  assert not directory.exists(),'This retry arm was already attempted'
  assert a.output.parent.is_dir() and not a.output.exists() and a.output.resolve()!=original.resolve()
  if a.verify_only:
    print(json.dumps({'verified':True,'gpu_executed':False,'arm':a.arm,'protocol':O.C.sha(O.PREPARED/'protocol.json')}));return
  directory.mkdir();O.R.configure(a.arm);os.environ['ALLOW_DEVICE_USAGE']='1'
  try: O.R.run(plan,a.arm,directory)
  finally:
    O.C.verify()
    if original.exists():
      report=json.loads(original.read_text())
      report['parent_bridge']={'original_report':str(original),'original_sha256':O.C.sha(original),'bridge_sha256':O.C.sha(__file__)}
      O.C.write(a.output,report)

if __name__=='__main__': main()
