"""Write a separate CPU-only receipt. Never rewrites the failed GPU report."""
import argparse
import importlib.util
from pathlib import Path

s=importlib.util.spec_from_file_location('ck_selected_recovery',Path(__file__).with_name('agent-ck-selected-recovery-common.py'))
R=importlib.util.module_from_spec(s);s.loader.exec_module(R)

if __name__=='__main__':
  p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
  assert not a.output.exists() and a.output.resolve().is_relative_to(R.ROOT)
  value=R.recover();R.S.write(a.output,value)
  print({'recovery_complete':True,'exact64':value['exact']['exact'],'original_completed':False,'gpu_executed_by_recovery':False})
