"""Normal parent speed-trial --output adapter; canonical output and caches stay on500GB."""
import argparse,importlib.util,sys
from pathlib import Path
from unittest.mock import patch
HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('nearest_storage_parent_run',HERE/'agent-nearest-storage-run.py')
run=importlib.util.module_from_spec(spec);spec.loader.exec_module(run)

def main():
  parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
  run.verify_frozen()
  expected=run.WORK/'agent-nearest-storage-parent/report.json'
  if args.output!=expected or not args.output.parent.is_dir():raise ValueError('Use the single new normal parent output directory')
  run.reserve(args.output.parent);run.reserve(run.DEFAULT.parent)
  if args.output.exists() or run.DEFAULT.parent.exists():raise FileExistsError('Preserve all old and new run evidence')
  command=[str(HERE/'agent-nearest-storage-run.py'),'--execute-gpu','--output',str(run.DEFAULT)]
  try:
    with patch.object(sys,'argv',command):run.main()
  finally:
    run.verify_frozen()
    if run.DEFAULT.is_file():
      report=run.read(run.DEFAULT)
      report['parent_bridge']=dict(original_report=str(run.DEFAULT),original_sha256=run.sha(run.DEFAULT),
        adapter_sha256=run.sha(__file__),storage_protocol_sha256=run.sha(run.PROTOCOL),argv=command)
      run.save(args.output,report)

if __name__=='__main__':main()
