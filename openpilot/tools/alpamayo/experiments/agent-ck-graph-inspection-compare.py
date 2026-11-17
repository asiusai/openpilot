"""Run the unchanged fresh-pair accuracy comparison using this retry's output paths."""
import argparse
import importlib.util
from pathlib import Path

s=importlib.util.spec_from_file_location('ck_graph_overlay',Path(__file__).with_name('agent-ck-graph-inspection-overlay.py'))
O=importlib.util.module_from_spec(s);s.loader.exec_module(O)
if __name__=='__main__':
  p=argparse.ArgumentParser();p.add_argument('--baseline',type=Path,required=True);p.add_argument('--candidate',type=Path,required=True)
  p.add_argument('--output',type=Path,required=True);a=p.parse_args()
  if not O.comparator().compare(a.baseline,a.candidate,a.output): raise SystemExit(1)
